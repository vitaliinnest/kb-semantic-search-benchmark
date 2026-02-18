"""Веб-інтерфейс для семантичного пошуку."""

import json
import sys
from pathlib import Path
from functools import lru_cache

import faiss
import numpy as np
from flask import Flask, render_template, request, jsonify

# Додаємо src/ до sys.path, щоб імпортувати сусідні модулі
sys.path.insert(0, str(Path(__file__).parent))
from embedding_models import load_model_from_artifacts

# ── Шляхи ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
ARTIFACTS_ROOT = ROOT / "artifacts"
CHUNKS_PATH = ROOT / "data" / "chunks.jsonl"

app = Flask(__name__, template_folder="templates")


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
    )


@app.route("/documents")
def documents():
    chunks = load_all_chunks()
    page = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 20))
    source_filter = request.args.get("source", "").strip()

    # Унікальні джерела
    all_sources = sorted({c["source"] for c in chunks})

    if source_filter:
        chunks = [c for c in chunks if c["source"] == source_filter]

    total = len(chunks)
    total_pages = max(1, (total + per_page - 1) // per_page)
    page = max(1, min(page, total_pages))
    start = (page - 1) * per_page
    page_chunks = chunks[start : start + per_page]

    return render_template(
        "documents.html",
        chunks=page_chunks,
        page=page,
        per_page=per_page,
        total=total,
        total_pages=total_pages,
        all_sources=all_sources,
        source_filter=source_filter,
    )


if __name__ == "__main__":
    print("Starting server at http://127.0.0.1:5000")
    app.run(debug=True, host="127.0.0.1", port=5000)
