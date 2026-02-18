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
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from werkzeug.utils import secure_filename

# Додаємо src/ до sys.path, щоб імпортувати сусідні модулі
sys.path.insert(0, str(Path(__file__).parent))
from embedding_models import load_model_from_artifacts

# ── Шляхи ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
ARTIFACTS_ROOT = ROOT / "artifacts"
CHUNKS_PATH = ROOT / "data" / "chunks.jsonl"
BENCHMARK_RESULTS_DIR = ROOT / "results" / "benchmark"
RAW_DIR = ROOT / "data" / "raw"

app = Flask(__name__, template_folder="templates")
app.secret_key = "kb-search-secret"

import datetime as _dt

@app.template_filter("datetimeformat")
def datetimeformat(ts):
    return _dt.datetime.fromtimestamp(float(ts)).strftime("%d.%m.%Y %H:%M")


# ── Утиліти──────────────────────────────────────────────────────────────────
def discover_models() -> list[dict]:
    """Повертає список доступних моделей із папки artifacts/."""
    models = []
    if not ARTIFACTS_ROOT.exists():
        return models
    for model_dir in sorted(ARTIFACTS_ROOT.iterdir()):
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
                    label = f"{mt}" + (f" — {mn}" if mn else "")
                except Exception:
                    pass
            models.append({"id": model_dir.name, "label": label})
    return models


def load_latest_benchmark() -> dict | None:
    """Завантажує найновіший файл benchmark_results_*.json або None."""
    if not BENCHMARK_RESULTS_DIR.exists():
        return None
    files = sorted(BENCHMARK_RESULTS_DIR.glob("benchmark_results_*.json"))
    if not files:
        return None
    try:
        return json.loads(files[-1].read_text(encoding="utf-8"))
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


def get_index_and_model(model_id: str):
    if model_id in _cache:
        return _cache[model_id]
    artifacts_dir = ARTIFACTS_ROOT / model_id
    index = faiss.read_index(str(artifacts_dir / "faiss.index"))
    meta = load_meta(artifacts_dir / "meta.jsonl")
    model, model_cfg = load_model_from_artifacts(artifacts_dir=artifacts_dir, fallback_model_name="paraphrase-multilingual-MiniLM-L12-v2")
    _cache[model_id] = (index, meta, model)
    return index, meta, model


def load_all_chunks() -> list[dict]:
    if not CHUNKS_PATH.exists():
        return []
    return load_meta(CHUNKS_PATH)


def do_search(query: str, model_id: str, top_k: int) -> list[dict]:
    index, meta, model = get_index_and_model(model_id)
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
    models = discover_models()
    benchmark = load_latest_benchmark()

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
    query = request.args.get("q", "").strip()
    model_id = request.args.get("model", models[0]["id"] if models else "")
    top_k = int(request.args.get("top_k", 5))

    results = []
    error = None
    if query and model_id:
        try:
            results = do_search(query, model_id, top_k)
        except Exception as exc:
            error = str(exc)

    return render_template(
        "index.html",
        models=models,
        query=query,
        selected_model=model_id,
        top_k=top_k,
        results=results,
        error=error,
        benchmark=benchmark,
    )


@app.route("/documents")
def documents():
    from collections import defaultdict

    chunks = load_all_chunks()
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
    )


ALLOWED_EXTENSIONS = {".txt", ".md", ".rst", ".pdf", ".docx"}

# ── Chunking state ──────────────────────────────────────────────────────────
_chunk_job: dict = {"running": False, "result": None, "error": None}
_chunk_lock = threading.Lock()


def list_raw_files() -> list[dict]:
    """Повертає список файлів у data/raw/."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    files = []
    for p in sorted(RAW_DIR.rglob("*")):
        if p.is_file():
            rel = p.relative_to(RAW_DIR)
            stat = p.stat()
            files.append({
                "name": str(rel).replace("\\", "/"),
                "size": stat.st_size,
                "mtime": stat.st_mtime,
            })
    return files


@app.route("/raw")
def raw_files():
    files = list_raw_files()
    chunk_exists = CHUNKS_PATH.exists()
    chunk_count = 0
    if chunk_exists:
        with CHUNKS_PATH.open("r", encoding="utf-8") as fh:
            chunk_count = sum(1 for line in fh if line.strip())
    return render_template(
        "raw.html",
        files=files,
        chunk_exists=chunk_exists,
        chunk_count=chunk_count,
    )


@app.route("/raw/upload", methods=["POST"])
def raw_upload():
    uploaded = request.files.getlist("files")
    if not uploaded:
        flash("Файли не вибрані.", "error")
        return redirect(url_for("raw_files"))
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    saved, skipped = 0, 0
    for f in uploaded:
        if not f.filename:
            continue
        ext = Path(f.filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            skipped += 1
            continue
        dest = RAW_DIR / secure_filename(f.filename)
        f.save(str(dest))
        saved += 1
    if saved:
        flash(f"Збережено {saved} файл(ів)." + (f" Пропущено {skipped} (непідтримуваний формат)." if skipped else ""), "success")
    else:
        flash("Жодного файлу не збережено. Підтримуються: .txt .md .rst .pdf .docx", "error")
    return redirect(url_for("raw_files"))


@app.route("/raw/delete", methods=["POST"])
def raw_delete():
    name = request.form.get("name", "").strip()
    if not name:
        return jsonify({"ok": False, "error": "Не вказано ім'я файлу"}), 400
    path = (RAW_DIR / name).resolve()
    try:
        path.relative_to(RAW_DIR.resolve())
    except ValueError:
        return jsonify({"ok": False, "error": "Недозволений шлях"}), 400
    if not path.exists():
        return jsonify({"ok": False, "error": "Файл не знайдено"}), 404
    path.unlink()
    try:
        path.parent.relative_to(RAW_DIR.resolve())
        if path.parent != RAW_DIR.resolve() and not any(path.parent.iterdir()):
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

    min_words = int(request.form.get("min_words", 300))
    max_words = int(request.form.get("max_words", 800))
    overlap   = int(request.form.get("overlap",   80))

    sys.path.insert(0, str(Path(__file__).parent))
    from ingest_chunk import ingest_chunks  # noqa: E402

    with _chunk_lock:
        _chunk_job = {"running": True, "result": None, "error": None}

    def _run():
        global _chunk_job
        try:
            count = ingest_chunks(
                input_dir=RAW_DIR,
                output_file=CHUNKS_PATH,
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
        "label": "SBERT",
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
        "label": "BERT",
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


def _artifact_info(model_id: str) -> dict:
    """Стан artifacts/{model_id}/ — чи існує індекс та скільки векторів."""
    d = ARTIFACTS_ROOT / model_id
    index_path = d / "faiss.index"
    meta_path  = d / "meta.jsonl"
    if not index_path.exists():
        return {"exists": False}
    chunk_count = 0
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as fh:
            chunk_count = sum(1 for l in fh if l.strip())
    mtime = index_path.stat().st_mtime
    return {
        "exists": True,
        "chunk_count": chunk_count,
        "mtime": mtime,
        "mtime_fmt": _dt.datetime.fromtimestamp(mtime).strftime("%d.%m.%Y %H:%M"),
    }


@app.route("/build")
def build_index_page():
    models_info = []
    for m in MODEL_DEFS:
        info = dict(m)
        info["artifact"] = _artifact_info(m["id"])
        with _build_lock:
            info["job"] = dict(_build_jobs.get(m["id"], {}))
        models_info.append(info)
    chunks_exist = CHUNKS_PATH.exists()
    chunk_count = 0
    if chunks_exist:
        with CHUNKS_PATH.open("r", encoding="utf-8") as fh:
            chunk_count = sum(1 for l in fh if l.strip())
    return render_template(
        "build.html",
        models=models_info,
        chunks_exist=chunks_exist,
        chunk_count=chunk_count,
    )


@app.route("/build/run", methods=["POST"])
def build_run():
    data = request.get_json(force=True)
    selected = data.get("models", [])   # list of {id, params{}}
    if not selected:
        return jsonify({"ok": False, "error": "Не вибрано жодної моделі"}), 400
    if not CHUNKS_PATH.exists():
        return jsonify({"ok": False, "error": "chunks.jsonl не знайдено — спочатку запустіть чанкінг"}), 400

    param_map = {m["id"]: {p["name"]: p["arg"] for p in m["params"]} for m in MODEL_DEFS}

    started = []
    for item in selected:
        mid = item["id"]
        with _build_lock:
            if _build_jobs.get(mid, {}).get("running"):
                continue   # вже запущено
            _build_jobs[mid] = {"running": True, "done": False, "exit_code": None, "log": [], "started_at": time.time()}

        artifacts_dir = str(ARTIFACTS_ROOT / mid)
        cmd = [
            sys.executable,
            str(BUILD_SCRIPT),
            "--model-type", mid,
            "--chunks", str(CHUNKS_PATH),
            "--artifacts", artifacts_dir,
        ]
        for param_name, param_val in item.get("params", {}).items():
            arg_flag = param_map.get(mid, {}).get(param_name)
            if arg_flag and param_val != "":
                cmd += [arg_flag, str(param_val)]

        def _run(model_id=mid, command=cmd):
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
                            _build_jobs[model_id]["log"].append(line)
                proc.wait()
                with _build_lock:
                    _build_jobs[model_id]["running"] = False
                    _build_jobs[model_id]["done"] = True
                    _build_jobs[model_id]["exit_code"] = proc.returncode
            except Exception as exc:
                with _build_lock:
                    _build_jobs[model_id]["running"] = False
                    _build_jobs[model_id]["done"] = True
                    _build_jobs[model_id]["exit_code"] = -1
                    _build_jobs[model_id]["log"].append(f"ERROR: {exc}")

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        started.append(mid)
        # invalidate search cache for this model
        _cache.pop(mid, None)

    return jsonify({"ok": True, "started": started})


@app.route("/build/status")
def build_status():
    with _build_lock:
        snapshot = {k: dict(v) for k, v in _build_jobs.items()}
    return jsonify(snapshot)


# ── Benchmark ────────────────────────────────────────────────────────────
BENCHMARK_SCRIPT  = Path(__file__).parent / "evaluate_benchmark.py"
QUERIES_PATH      = ROOT / "data" / "benchmark" / "queries.jsonl"
QRELS_PATH        = ROOT / "data" / "benchmark" / "qrels.jsonl"
BENCHMARK_OUT_DIR = ROOT / "results" / "benchmark"

_bench_job: dict = {"running": False, "done": False, "exit_code": None, "log": [], "result_file": None}
_bench_lock = threading.Lock()


def _load_benchmark_results() -> list[dict]:
    """Повертає всі файли benchmark_results_*.json, відсортовані від найновішого."""
    if not BENCHMARK_OUT_DIR.exists():
        return []
    files = sorted(BENCHMARK_OUT_DIR.glob("benchmark_results_*.json"), reverse=True)
    results = []
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            data["_filename"] = f.name
            results.append(data)
        except Exception:
            pass
    return results


@app.route("/benchmark")
def benchmark_page():
    all_results = _load_benchmark_results()
    latest = all_results[0] if all_results else None
    queries_exist = QUERIES_PATH.exists() and QRELS_PATH.exists()
    artifacts_exist = bool(discover_models())
    with _bench_lock:
        job = dict(_bench_job)
    return render_template(
        "benchmark.html",
        latest=latest,
        all_results=all_results,
        queries_exist=queries_exist,
        artifacts_exist=artifacts_exist,
        job=job,
    )


@app.route("/benchmark/run", methods=["POST"])
def benchmark_run():
    with _bench_lock:
        if _bench_job.get("running"):
            return jsonify({"ok": False, "error": "Бенчмарк вже виконується"}), 409

    if not QUERIES_PATH.exists() or not QRELS_PATH.exists():
        return jsonify({"ok": False, "error": "queries.jsonl або qrels.jsonl не знайдено"}), 400

    top_k = int(request.form.get("top_k", 10))

    cmd = [
        sys.executable,
        str(BENCHMARK_SCRIPT),
        "--queries",       str(QUERIES_PATH),
        "--qrels",         str(QRELS_PATH),
        "--artifacts-root", str(ARTIFACTS_ROOT),
        "--top-k",         str(top_k),
        "--output",        str(BENCHMARK_OUT_DIR),
    ]

    with _bench_lock:
        _bench_job.update({"running": True, "done": False, "exit_code": None, "log": [], "result_file": None})

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
            if BENCHMARK_OUT_DIR.exists():
                files = sorted(BENCHMARK_OUT_DIR.glob("benchmark_results_*.json"))
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
    results = _load_benchmark_results()
    if not results:
        return jsonify(None)
    return jsonify(results[0])


@app.route("/results/benchmark/<filename>")
def benchmark_result_file(filename: str):
    """Повертає JSON конкретного файлу результатів benchmark."""
    path = (BENCHMARK_OUT_DIR / filename).resolve()
    try:
        path.relative_to(BENCHMARK_OUT_DIR.resolve())
    except ValueError:
        return jsonify({"error": "forbidden"}), 403
    if not path.exists() or path.suffix != ".json":
        return jsonify({"error": "not found"}), 404
    return jsonify(json.loads(path.read_text(encoding="utf-8")))


if __name__ == "__main__":
    print("Starting server at http://127.0.0.1:5000")
    app.run(debug=True, host="127.0.0.1", port=5000)
