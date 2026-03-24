"""Веб-інтерфейс для семантичного пошуку."""

import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from functools import lru_cache

import faiss
import numpy as np
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, Response
from werkzeug.utils import secure_filename

# Додаємо src/ до sys.path, щоб імпортувати сусідні модулі
sys.path.insert(0, str(Path(__file__).parent))
from embedding_models import load_model_from_artifacts
from multi_criteria import run_selection, DEFAULT_WEIGHTS, CRITERIA

# ── Шляхи ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DOMAINS_ROOT = ROOT / "data" / "domains"

DOMAINS_CONFIG = [
    {"id": "tech",    "label": "Техніка / IT"},
    {"id": "legal",   "label": "Юридична база"},
    {"id": "medical", "label": "Медицина"},
]
DEFAULT_DOMAIN = "tech"


def get_domain() -> str:
    """Повертає поточний домен з URL-параметра або форми."""
    d = request.args.get("domain") or request.form.get("domain") or DEFAULT_DOMAIN
    valid = {dom["id"] for dom in DOMAINS_CONFIG}
    return d if d in valid else DEFAULT_DOMAIN


def domain_paths(domain_id: str) -> dict:
    """Повертає всі шляхи для заданого домену."""
    base = DOMAINS_ROOT / domain_id
    return {
        "raw":               base / "raw",
        "chunks":            base / "chunks.jsonl",
        "benchmark_queries": base / "benchmark" / "queries.jsonl",
        "benchmark_qrels":   base / "benchmark" / "qrels.jsonl",
        "artifacts_root":    ROOT / "artifacts" / domain_id,
        "results_dir":       ROOT / "results" / "benchmark" / domain_id,
    }

app = Flask(__name__, template_folder="templates")
app.secret_key = "kb-search-secret"

@app.route("/favicon.ico")
def favicon():
    return Response(status=204)

import datetime as _dt

@app.template_filter("datetimeformat")
def datetimeformat(ts):
    return _dt.datetime.fromtimestamp(float(ts)).strftime("%d.%m.%Y %H:%M")


# ── Утиліти──────────────────────────────────────────────────────────────────
_TYPE_TO_DISPLAY = {
    "sbert":    "SBERT",
    "bert":     "BERT",
    "tfidf":    "TF-IDF",
    "word2vec": "Word2Vec",
    "fasttext": "FastText",
    "glove":    "GloVe",
}


# Known sub-strings in model_name that identify the model type when stored as raw model_name
_MODEL_NAME_TO_TYPE: dict[str, str] = {
    "paraphrase-multilingual": "sbert",
    "sentence-transformers":   "sbert",
    "bert-base-multilingual":  "bert",
    "bert-large":              "bert",
    "roberta":                 "bert",
}


def _prettify_benchmark_data(data: dict) -> dict:
    """Замінює сирі model_name у завантажених benchmark-результатах на читабельні."""
    if not data or "models" not in data:
        return data
    for m in data["models"]:
        raw_name = m.get("model_name", "")
        artifacts_dir = m.get("artifacts_dir", "")
        # Determine model type from artifacts_dir folder name first (most reliable)
        dir_name = Path(artifacts_dir).name.lower() if artifacts_dir else ""
        model_type = dir_name if dir_name in _TYPE_TO_DISPLAY else None
        # Fallback: infer from raw_name substrings
        if model_type is None:
            for substr, mtype in _MODEL_NAME_TO_TYPE.items():
                if substr in raw_name.lower():
                    model_type = mtype
                    break
        # Fallback: raw_name itself might be a type key (e.g. "tfidf", "word2vec")
        if model_type is None and raw_name.lower() in _TYPE_TO_DISPLAY:
            model_type = raw_name.lower()
        if model_type is None:
            continue  # can't determine, leave as-is
        display = _TYPE_TO_DISPLAY[model_type]
        # Skip if already prettified (evaluate_benchmark.py stores formatted names)
        if raw_name.startswith(display):
            continue
        # For SBERT/BERT include the specific model identifier
        if model_type in {"sbert", "bert"} and raw_name and raw_name.lower() != model_type:
            m["model_name"] = f"{display} ({raw_name})"
        else:
            m["model_name"] = display
    return data


def discover_models(domain_id: str = DEFAULT_DOMAIN) -> list[dict]:
    """Повертає список доступних моделей для заданого домену."""
    artifacts_root = domain_paths(domain_id)["artifacts_root"]
    models = []
    if not artifacts_root.exists():
        return models
    for model_dir in sorted(artifacts_root.iterdir()):
        if not model_dir.is_dir():
            continue
        index_path = model_dir / "faiss.index"
        meta_path = model_dir / "meta.jsonl"
        if index_path.exists() and meta_path.exists():
            model_json = model_dir / "model.json"
            label = model_dir.name
            if model_json.exists():
                try:
                    cfg = json.loads(model_json.read_text(encoding="utf-8"))
                    mt = cfg.get("model_type", label)
                    mn = cfg.get("model_name")
                    display = _TYPE_TO_DISPLAY.get(mt, mt.upper())
                    label = display + (f" ({mn})" if mn else "")
                except Exception:
                    pass
            models.append({"id": model_dir.name, "label": label})
    return models


def load_latest_benchmark(domain_id: str = DEFAULT_DOMAIN) -> dict | None:
    """Завантажує найновіший файл benchmark_results_*.json для домену або None."""
    results_dir = domain_paths(domain_id)["results_dir"]
    if not results_dir.exists():
        return None
    files = sorted(results_dir.glob("benchmark_results_*.json"))
    if not files:
        return None
    try:
        return _prettify_benchmark_data(json.loads(files[-1].read_text(encoding="utf-8")))
    except Exception:
        return None


def load_meta(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# Кешуємо завантажені індекси/моделі у пам'яті
_cache: dict = {}


def get_index_and_model(model_id: str, domain_id: str = DEFAULT_DOMAIN):
    cache_key = f"{domain_id}/{model_id}"
    if cache_key in _cache:
        return _cache[cache_key]
    artifacts_dir = domain_paths(domain_id)["artifacts_root"] / model_id
    index = faiss.read_index(str(artifacts_dir / "faiss.index"))
    meta = load_meta(artifacts_dir / "meta.jsonl")
    model, model_cfg = load_model_from_artifacts(artifacts_dir=artifacts_dir, fallback_model_name="paraphrase-multilingual-MiniLM-L12-v2")
    _cache[cache_key] = (index, meta, model)
    return index, meta, model


def load_all_chunks(domain_id: str = DEFAULT_DOMAIN) -> list[dict]:
    chunks_path = domain_paths(domain_id)["chunks"]
    if not chunks_path.exists():
        return []
    return load_meta(chunks_path)


def do_search(query: str, model_id: str, top_k: int, domain_id: str = DEFAULT_DOMAIN) -> list[dict]:
    index, meta, model = get_index_and_model(model_id, domain_id)
    query_vector = model.encode_queries([query]).astype("float32")
    scores, indices = index.search(query_vector, top_k)
    results = []
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
        if idx < 0 or idx >= len(meta):
            continue
        record = meta[idx]
        snippet = record["text"][:400]
        if len(record["text"]) > 400:
            snippet += "…"
        results.append(
            {
                "rank": rank,
                "score": float(score),
                "source": record["source"],
                "chunk_id": record["chunk_id"],
                "snippet": snippet,
                "full_text": record["text"],
                "full_length": len(record["text"]),
            }
        )
    return results


# ── Маршрути ──────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    domain = get_domain()
    models = discover_models(domain)
    benchmark = load_latest_benchmark(domain)

    # Сортуємо моделі за nDCG з бенчмарку (від найкращої до найгіршої)
    if benchmark and benchmark.get("models"):
        ndcg_map = {
            m["artifacts_dir"].split("\\")[-1].split("/")[-1]: m["ndcg_at_k"]
            for m in benchmark["models"]
        }
        models = sorted(models, key=lambda m: ndcg_map.get(m["id"], -1), reverse=True)
        # Додаємо nDCG до кожної моделі для відображення в дропдауні
        for m in models:
            ndcg = ndcg_map.get(m["id"])
            if ndcg is not None:
                m["ndcg"] = ndcg
    query_text = request.args.get("q", "").strip()
    model_id = request.args.get("model", models[0]["id"] if models else "")
    top_k = int(request.args.get("top_k", 5))

    results = []
    error = None
    if query_text and model_id:
        try:
            results = do_search(query_text, model_id, top_k, domain)
        except Exception as exc:
            error = str(exc)

    return render_template(
        "index.html",
        models=models,
        query=query_text,
        selected_model=model_id,
        top_k=top_k,
        results=results,
        error=error,
        benchmark=benchmark,
        current_domain=domain,
        domains=DOMAINS_CONFIG,
    )


@app.route("/documents")
def documents():
    from collections import defaultdict

    domain = get_domain()
    chunks = load_all_chunks(domain)
    source_filter = request.args.get("source", "").strip()

    # Унікальні джерела
    all_sources = sorted({c["source"] for c in chunks})

    working = [c for c in chunks if c["source"] == source_filter] if source_filter else chunks

    # Групуємо по джерелу
    groups_dict: dict[str, list] = defaultdict(list)
    for c in working:
        groups_dict[c["source"]].append(c)

    source_groups = [
        {"source": s, "chunks": groups_dict[s], "count": len(groups_dict[s])}
        for s in sorted(groups_dict)
    ]

    total_chunks = sum(g["count"] for g in source_groups)

    return render_template(
        "documents.html",
        source_groups=source_groups,
        total_sources=len(source_groups),
        total_chunks=total_chunks,
        all_sources=all_sources,
        source_filter=source_filter,
        current_domain=domain,
        domains=DOMAINS_CONFIG,
    )


ALLOWED_EXTENSIONS = {".txt", ".md", ".rst", ".pdf", ".docx"}

# ── Chunking state ──────────────────────────────────────────────────────────
_chunk_job: dict = {"running": False, "result": None, "error": None}
_chunk_lock = threading.Lock()


def list_raw_files(domain_id: str = DEFAULT_DOMAIN) -> list[dict]:
    """Повертає список файлів у data/domains/{domain}/raw/."""
    raw_dir = domain_paths(domain_id)["raw"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for p in sorted(raw_dir.rglob("*")):
        if p.is_file():
            rel = p.relative_to(raw_dir)
            stat = p.stat()
            files.append({
                "name": str(rel).replace("\\", "/"),
                "size": stat.st_size,
                "mtime": stat.st_mtime,
            })
    return files


@app.route("/raw")
def raw_files():
    domain = get_domain()
    dp = domain_paths(domain)
    files = list_raw_files(domain)
    chunks_path = dp["chunks"]
    chunk_exists = chunks_path.exists()
    chunk_count = 0
    if chunk_exists:
        with chunks_path.open("r", encoding="utf-8") as fh:
            chunk_count = sum(1 for line in fh if line.strip())
    return render_template(
        "raw.html",
        files=files,
        chunk_exists=chunk_exists,
        chunk_count=chunk_count,
        current_domain=domain,
        domains=DOMAINS_CONFIG,
    )


@app.route("/raw/upload", methods=["POST"])
def raw_upload():
    domain = get_domain()
    raw_dir = domain_paths(domain)["raw"]
    uploaded = request.files.getlist("files")
    if not uploaded:
        flash("Файли не вибрані.", "error")
        return redirect(url_for("raw_files", domain=domain))
    raw_dir.mkdir(parents=True, exist_ok=True)
    saved, skipped = 0, 0
    for f in uploaded:
        if not f.filename:
            continue
        ext = Path(f.filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            skipped += 1
            continue
        dest = raw_dir / secure_filename(f.filename)
        f.save(str(dest))
        saved += 1
    if saved:
        flash(f"Збережено {saved} файл(ів)." + (f" Пропущено {skipped} (непідтримуваний формат)." if skipped else ""), "success")
    else:
        flash("Жодного файлу не збережено. Підтримуються: .txt .md .rst .pdf .docx", "error")
    return redirect(url_for("raw_files", domain=domain))


@app.route("/raw/delete", methods=["POST"])
def raw_delete():
    domain = get_domain()
    raw_dir = domain_paths(domain)["raw"]
    name = request.form.get("name", "").strip()
    if not name:
        return jsonify({"ok": False, "error": "Не вказано ім'я файлу"}), 400
    path = (raw_dir / name).resolve()
    try:
        path.relative_to(raw_dir.resolve())
    except ValueError:
        return jsonify({"ok": False, "error": "Недозволений шлях"}), 400
    if not path.exists():
        return jsonify({"ok": False, "error": "Файл не знайдено"}), 404
    path.unlink()
    try:
        path.parent.relative_to(raw_dir.resolve())
        if path.parent != raw_dir.resolve() and not any(path.parent.iterdir()):
            path.parent.rmdir()
    except Exception:
        pass
    return jsonify({"ok": True})


@app.route("/raw/chunk", methods=["POST"])
def raw_chunk():
    global _chunk_job
    with _chunk_lock:
        if _chunk_job["running"]:
            return jsonify({"ok": False, "error": "Чанкінг вже виконується"}), 409

    domain = get_domain()
    dp = domain_paths(domain)
    min_words = int(request.form.get("min_words", 300))
    max_words = int(request.form.get("max_words", 800))
    overlap   = int(request.form.get("overlap",   80))

    sys.path.insert(0, str(Path(__file__).parent))
    from ingest_chunk import ingest_chunks  # noqa: E402

    raw_dir    = dp["raw"]
    chunks_out = dp["chunks"]

    with _chunk_lock:
        _chunk_job = {"running": True, "result": None, "error": None}

    def _run():
        global _chunk_job
        try:
            count = ingest_chunks(
                input_dir=raw_dir,
                output_file=chunks_out,
                min_words=min_words,
                max_words=max_words,
                overlap=overlap,
            )
            with _chunk_lock:
                _chunk_job = {"running": False, "result": count, "error": None}
        except Exception as exc:
            with _chunk_lock:
                _chunk_job = {"running": False, "result": None, "error": str(exc)}

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return jsonify({"ok": True, "message": "Чанкінг запущено"})


@app.route("/raw/chunk/status")
def raw_chunk_status():
    with _chunk_lock:
        job = dict(_chunk_job)
    return jsonify(job)


# ── Build index ───────────────────────────────────────────────────────────────
BUILD_SCRIPT = Path(__file__).parent / "build_index.py"

MODEL_DEFS = [
    {
        "id": "sbert",
        "label": "SBERT (paraphrase-multilingual-MiniLM-L12-v2)",
        "desc": "Sentence-Transformers — найкраща якість для семантичного пошуку",
        "params": [
            {"name": "model",      "label": "Назва моделі",  "type": "text",   "default": "paraphrase-multilingual-MiniLM-L12-v2", "arg": "--model"},
            {"name": "batch_size", "label": "Batch size",    "type": "number", "default": 64,   "arg": "--batch-size"},
        ],
    },
    {
        "id": "tfidf",
        "label": "TF-IDF",
        "desc": "Класична частотна модель, швидка та інтерпретована",
        "params": [
            {"name": "tfidf_max_features", "label": "Max features", "type": "number", "default": 50000, "arg": "--tfidf-max-features"},
            {"name": "tfidf_ngram_min",    "label": "N-gram min",   "type": "number", "default": 1,     "arg": "--tfidf-ngram-min"},
            {"name": "tfidf_ngram_max",    "label": "N-gram max",   "type": "number", "default": 2,     "arg": "--tfidf-ngram-max"},
        ],
    },
    {
        "id": "word2vec",
        "label": "Word2Vec",
        "desc": "Навчається на ваших чанках, середня якість",
        "params": [
            {"name": "gensim_vector_size", "label": "Vector size", "type": "number", "default": 300, "arg": "--gensim-vector-size"},
            {"name": "gensim_window",      "label": "Window",      "type": "number", "default": 5,   "arg": "--gensim-window"},
            {"name": "gensim_min_count",   "label": "Min count",   "type": "number", "default": 2,   "arg": "--gensim-min-count"},
            {"name": "gensim_epochs",      "label": "Epochs",      "type": "number", "default": 10,  "arg": "--gensim-epochs"},
        ],
    },
    {
        "id": "fasttext",
        "label": "FastText",
        "desc": "Як Word2Vec, але з підтримкою підслів (добре для морфології)",
        "params": [
            {"name": "gensim_vector_size", "label": "Vector size", "type": "number", "default": 300, "arg": "--gensim-vector-size"},
            {"name": "gensim_window",      "label": "Window",      "type": "number", "default": 5,   "arg": "--gensim-window"},
            {"name": "gensim_min_count",   "label": "Min count",   "type": "number", "default": 2,   "arg": "--gensim-min-count"},
            {"name": "gensim_epochs",      "label": "Epochs",      "type": "number", "default": 10,  "arg": "--gensim-epochs"},
        ],
    },
    {
        "id": "bert",
        "label": "BERT (bert-base-multilingual-cased)",
        "desc": "Повноцінний трансформер — повільно, але потужно",
        "params": [
            {"name": "bert_model",      "label": "Назва моделі",  "type": "text",   "default": "bert-base-multilingual-cased", "arg": "--bert-model"},
            {"name": "bert_batch_size", "label": "Batch size",    "type": "number", "default": 32,  "arg": "--bert-batch-size"},
            {"name": "bert_max_length", "label": "Max length",   "type": "number", "default": 256, "arg": "--bert-max-length"},
        ],
    },
]

_build_jobs: dict = {}   # model_id -> {running, done, exit_code, log, started_at}
_build_lock = threading.Lock()


def _artifact_info(model_id: str, domain_id: str = DEFAULT_DOMAIN) -> dict:
    """Стан artifacts/{domain}/{model_id}/ — чи існує індекс та скільки векторів."""
    d = domain_paths(domain_id)["artifacts_root"] / model_id
    index_path = d / "faiss.index"
    meta_path  = d / "meta.jsonl"
    if not index_path.exists():
        return {"exists": False}
    chunk_count = 0
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as fh:
            chunk_count = sum(1 for l in fh if l.strip())
    mtime = index_path.stat().st_mtime
    index_size = index_path.stat().st_size
    # Collect present artifact files
    _candidate_files = [
        "faiss.index", "vectors.npy", "model.json", "meta.jsonl",
        "tfidf.joblib", "gensim.model",
    ]
    artifact_files = [f for f in _candidate_files if (d / f).exists()]
    return {
        "exists": True,
        "chunk_count": chunk_count,
        "mtime": mtime,
        "mtime_fmt": _dt.datetime.fromtimestamp(mtime).strftime("%d.%m.%Y %H:%M"),
        "index_size": index_size,
        "index_size_fmt": f"{index_size / 1024:.1f} KB" if index_size < 1024 * 1024 else f"{index_size / 1024 / 1024:.1f} MB",
        "artifact_files": artifact_files,
    }


@app.route("/build")
def build_index_page():
    domain = get_domain()
    dp = domain_paths(domain)
    models_info = []
    for m in MODEL_DEFS:
        info = dict(m)
        info["artifact"] = _artifact_info(m["id"], domain)
        job_key = f"{domain}/{m['id']}"
        with _build_lock:
            info["job"] = dict(_build_jobs.get(job_key, {}))
        models_info.append(info)
    chunks_path = dp["chunks"]
    chunks_exist = chunks_path.exists()
    chunk_count = 0
    if chunks_exist:
        with chunks_path.open("r", encoding="utf-8") as fh:
            chunk_count = sum(1 for l in fh if l.strip())
    return render_template(
        "build.html",
        models=models_info,
        chunks_exist=chunks_exist,
        chunk_count=chunk_count,
        current_domain=domain,
        domains=DOMAINS_CONFIG,
    )


@app.route("/build/run", methods=["POST"])
def build_run():
    data = request.get_json(force=True)
    domain = data.get("domain", DEFAULT_DOMAIN)
    if domain not in {d["id"] for d in DOMAINS_CONFIG}:
        domain = DEFAULT_DOMAIN
    dp = domain_paths(domain)
    selected = data.get("models", [])   # list of {id, params{}}
    if not selected:
        return jsonify({"ok": False, "error": "Не вибрано жодної моделі"}), 400
    if not dp["chunks"].exists():
        return jsonify({"ok": False, "error": "chunks.jsonl не знайдено — спочатку запустіть чанкінг"}), 400

    param_map = {m["id"]: {p["name"]: p["arg"] for p in m["params"]} for m in MODEL_DEFS}

    started = []
    for item in selected:
        mid = item["id"]
        job_key = f"{domain}/{mid}"
        with _build_lock:
            if _build_jobs.get(job_key, {}).get("running"):
                continue   # вже запущено
            _build_jobs[job_key] = {"running": True, "done": False, "exit_code": None, "log": [], "started_at": time.time()}

        artifacts_dir = str(dp["artifacts_root"] / mid)
        cmd = [
            sys.executable,
            str(BUILD_SCRIPT),
            "--model-type", mid,
            "--chunks", str(dp["chunks"]),
            "--artifacts", artifacts_dir,
        ]
        for param_name, param_val in item.get("params", {}).items():
            arg_flag = param_map.get(mid, {}).get(param_name)
            if arg_flag and param_val != "":
                cmd += [arg_flag, str(param_val)]

        def _run(model_id=mid, job_k=job_key, command=cmd):
            try:
                proc = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    cwd=str(ROOT),
                )
                for line in proc.stdout:
                    line = line.rstrip()
                    if line:
                        with _build_lock:
                            _build_jobs[job_k]["log"].append(line)
                proc.wait()
                with _build_lock:
                    _build_jobs[job_k]["running"] = False
                    _build_jobs[job_k]["done"] = True
                    _build_jobs[job_k]["exit_code"] = proc.returncode
            except Exception as exc:
                with _build_lock:
                    _build_jobs[job_k]["running"] = False
                    _build_jobs[job_k]["done"] = True
                    _build_jobs[job_k]["exit_code"] = -1
                    _build_jobs[job_k]["log"].append(f"ERROR: {exc}")

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        started.append(mid)
        # invalidate search cache for this model+domain
        _cache.pop(f"{domain}/{mid}", None)

    return jsonify({"ok": True, "started": started})


@app.route("/build/status")
def build_status():
    with _build_lock:
        snapshot = {k: dict(v) for k, v in _build_jobs.items()}
    return jsonify(snapshot)


# ── Benchmark ────────────────────────────────────────────────────────────
BENCHMARK_SCRIPT = Path(__file__).parent / "evaluate_benchmark.py"

_bench_job: dict = {"running": False, "done": False, "exit_code": None, "log": [], "result_file": None}
_bench_lock = threading.Lock()


def _load_benchmark_results(domain_id: str = DEFAULT_DOMAIN) -> list[dict]:
    """Повертає всі файли benchmark_results_*.json для домену, відсортовані від найновішого."""
    results_dir = domain_paths(domain_id)["results_dir"]
    if not results_dir.exists():
        return []
    files = sorted(results_dir.glob("benchmark_results_*.json"), reverse=True)
    results = []
    for f in files:
        try:
            data = _prettify_benchmark_data(json.loads(f.read_text(encoding="utf-8")))
            data["_filename"] = f.name
            results.append(data)
        except Exception:
            pass
    return results


@app.route("/benchmark")
def benchmark_page():
    domain = get_domain()
    dp = domain_paths(domain)
    all_results = _load_benchmark_results(domain)
    latest = all_results[0] if all_results else None
    queries_exist = dp["benchmark_queries"].exists() and dp["benchmark_qrels"].exists()
    artifacts_exist = bool(discover_models(domain))
    with _bench_lock:
        job = dict(_bench_job)
    return render_template(
        "benchmark.html",
        latest=latest,
        all_results=all_results,
        queries_exist=queries_exist,
        artifacts_exist=artifacts_exist,
        job=job,
        current_domain=domain,
        domains=DOMAINS_CONFIG,
    )


@app.route("/benchmark/run", methods=["POST"])
def benchmark_run():
    with _bench_lock:
        if _bench_job.get("running"):
            return jsonify({"ok": False, "error": "Бенчмарк вже виконується"}), 409

    domain = get_domain()
    dp = domain_paths(domain)
    if not dp["benchmark_queries"].exists() or not dp["benchmark_qrels"].exists():
        return jsonify({"ok": False, "error": "queries.jsonl або qrels.jsonl не знайдено"}), 400

    top_k = int(request.form.get("top_k", 10))
    dp["results_dir"].mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(BENCHMARK_SCRIPT),
        "--queries",        str(dp["benchmark_queries"]),
        "--qrels",          str(dp["benchmark_qrels"]),
        "--artifacts-root", str(dp["artifacts_root"]),
        "--top-k",          str(top_k),
        "--output",         str(dp["results_dir"]),
    ]

    with _bench_lock:
        _bench_job.update({"running": True, "done": False, "exit_code": None, "log": [], "result_file": None})

    results_dir = dp["results_dir"]

    def _run():
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=str(ROOT),
            )
            for line in proc.stdout:
                line = line.rstrip()
                if line:
                    with _bench_lock:
                        _bench_job["log"].append(line)
            proc.wait()
            # знаходимо найновіший результат
            result_file = None
            if results_dir.exists():
                files = sorted(results_dir.glob("benchmark_results_*.json"))
                if files:
                    result_file = files[-1].name
            with _bench_lock:
                _bench_job["running"]     = False
                _bench_job["done"]        = True
                _bench_job["exit_code"]   = proc.returncode
                _bench_job["result_file"] = result_file
        except Exception as exc:
            with _bench_lock:
                _bench_job["running"]   = False
                _bench_job["done"]      = True
                _bench_job["exit_code"] = -1
                _bench_job["log"].append(f"ERROR: {exc}")

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"ok": True})


@app.route("/benchmark/status")
def benchmark_status():
    with _bench_lock:
        job = dict(_bench_job)
    return jsonify(job)


@app.route("/benchmark/result")
def benchmark_result():
    """Повертає JSON найновішого benchmark result для live-оновлення UI."""
    domain = get_domain()
    results = _load_benchmark_results(domain)
    if not results:
        return jsonify(None)
    return jsonify(results[0])


@app.route("/results/benchmark/<filename>")
def benchmark_result_file(filename: str):
    """Повертає JSON конкретного файлу результатів benchmark."""
    domain = get_domain()
    results_dir = domain_paths(domain)["results_dir"]
    path = (results_dir / filename).resolve()
    try:
        path.relative_to(results_dir.resolve())
    except ValueError:
        return jsonify({"error": "forbidden"}), 403
    if not path.exists() or path.suffix != ".json":
        return jsonify({"error": "not found"}), 404
    return jsonify(_prettify_benchmark_data(json.loads(path.read_text(encoding="utf-8"))))


# ── Multi-criteria model selection ──────────────────────────────────────────

_VIEWABLE_ARTIFACT_FILES = {"model.json", "meta.jsonl"}
_ARTIFACT_PREVIEW_LIMIT  = 30   # max chunks shown from meta.jsonl

@app.route("/build/artifact-view")
def build_artifact_view():
    """Return text content of a viewable artifact file (model.json or meta.jsonl)."""
    domain   = request.args.get("domain", DEFAULT_DOMAIN)
    model_id = request.args.get("model", "")
    filename = request.args.get("file", "")
    if filename not in _VIEWABLE_ARTIFACT_FILES:
        return jsonify({"error": "forbidden"}), 403
    artifacts_root = domain_paths(domain)["artifacts_root"]
    path = (artifacts_root / model_id / filename).resolve()
    try:
        path.relative_to(artifacts_root.resolve())
    except ValueError:
        return jsonify({"error": "forbidden"}), 403
    if not path.exists():
        return jsonify({"error": "not found"}), 404
    raw = path.read_text(encoding="utf-8")
    if filename == "meta.jsonl":
        lines = [l for l in raw.splitlines() if l.strip()]
        total = len(lines)
        rows = []
        for line in lines[:_ARTIFACT_PREVIEW_LIMIT]:
            try:
                obj = json.loads(line)
                rows.append({"chunk_id": obj.get("chunk_id", ""), "source": obj.get("source", ""), "text": obj.get("text", "")[:200]})
            except Exception:
                rows.append({"chunk_id": "", "source": "", "text": line[:200]})
        return jsonify({"type": "meta", "rows": rows, "total": total, "shown": len(rows)})
    # model.json
    try:
        return jsonify({"type": "json", "content": json.loads(raw)})
    except Exception:
        return jsonify({"type": "text", "content": raw})


@app.route("/benchmark/selection")
def benchmark_selection():
    domain = get_domain()
    data = load_latest_benchmark(domain)
    selection = None
    if data and data.get("models"):
        selection = run_selection(data["models"])
    return render_template(
        "selection.html",
        selection=selection,
        benchmark_meta=data,
        criteria=CRITERIA,
        default_weights=DEFAULT_WEIGHTS,
        current_domain=domain,
        domains=DOMAINS_CONFIG,
    )


@app.route("/benchmark/selection/compute", methods=["POST"])
def benchmark_selection_compute():
    """Recompute model ranking with custom weights (received as JSON)."""
    weights_raw = request.get_json(force=True) or {}
    valid_keys = {c["key"] for c in CRITERIA}
    weights: dict[str, float] = {}
    for k, v in weights_raw.items():
        if k in valid_keys:
            try:
                weights[k] = max(0.0, min(1.0, float(v)))
            except (TypeError, ValueError):
                pass
    domain = get_domain()
    data = load_latest_benchmark(domain)
    if not data or not data.get("models"):
        return jsonify({"error": "No benchmark data available"}), 404
    result = run_selection(data["models"], weights)
    return jsonify(result)


if __name__ == "__main__":
    print("Starting server at http://127.0.0.1:5000")
    app.run(debug=True, host="127.0.0.1", port=5000)
