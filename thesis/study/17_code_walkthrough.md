# 17. Прохід по коду

> **TL;DR:** Огляд основних Python-модулів у `src/`. Що вони роблять, де знаходяться ключові функції, який data flow. Корисно для технічних питань комісії «покажіть, як це реалізовано».

---

## Зміст

1. [Структура коду](#1-структура-коду)
2. [`src/app.py` — Flask web UI](#2-srcapppy)
3. [`src/embedding_models.py` — класи моделей](#3-srcembedding_modelspy)
4. [`src/build_index.py` — побудова індексу](#4-srcbuild_indexpy)
5. [`src/evaluate_benchmark.py` — обчислення метрик](#5-srcevaluate_benchmarkpy)
6. [`src/multi_criteria.py` — MCDA логіка](#6-srcmulti_criteriapy)
7. [`src/ingest_chunk.py` — чанкінг](#7-srcingest_chunkpy)
8. [Data flow повний pipeline](#8-data-flow-повний-pipeline)
9. [Як запустити локально](#9-як-запустити-локально)
10. [Перевірочні питання](#10-перевірочні-питання)

---

## 1. Структура коду

```
src/
├── app.py                   ← Flask web UI (головний entry-point)
├── embedding_models.py      ← Класи моделей (E5, BGE-M3, Nomic, Qwen3, BM25)
├── build_index.py           ← Побудова FAISS-індексу
├── evaluate_benchmark.py    ← Обчислення nDCG/MRR/Recall/P@10
├── multi_criteria.py        ← Pareto + лінійна згортка
├── ingest_chunk.py          ← Чанкінг документів
└── templates/               ← HTML-шаблони Flask
    ├── base.html
    ├── index.html           ← /
    ├── documents.html       ← /documents
    ├── raw.html             ← /raw
    ├── build.html           ← /build
    ├── benchmark.html       ← /benchmark
    ├── selection.html       ← /benchmark/selection
    ├── benchmark_explorer.html
    └── benchmark_explorer_query.html

scripts/                     ← Утиліти
├── run_all_benchmarks.py    ← Запустити evaluate на всіх доменах
└── build_indexes.py         ← Побудувати індекси для всіх моделей

data/domains/<domain>/       ← Дані
├── raw/                     ← Оригінальні документи (PDF, DOCX, ...)
├── chunks.jsonl             ← Чанки після чанкінгу
└── benchmark/
    ├── queries.jsonl
    └── qrels.jsonl

artifacts/<domain>/<model>/  ← Індекси
├── faiss.index              ← FAISS-індекс
├── meta.jsonl               ← Метадані чанків
└── model.json               ← Конфігурація моделі

results/benchmark/<domain>/  ← Результати оцінки
└── benchmark_results_*.json
```

---

## 2. `src/app.py`

Головний файл — Flask додаток. ~1170 рядків.

### Ключові endpoints

| Endpoint | HTTP | Що робить |
|---|---|---|
| `/` | GET | Головна — пошук |
| `/api/search` | GET | JSON API для пошуку |
| `/documents` | GET | Перегляд чанків |
| `/raw` | GET | Файли для upload |
| `/raw/upload` | POST | Upload файлів |
| `/raw/chunk` | POST | Запустити чанкінг |
| `/build` | GET | Сторінка побудови індексів |
| `/build/run` | POST | Запустити build |
| `/build/status` | GET | Статус build job |
| `/benchmark` | GET | Сторінка metric таблиці |
| `/benchmark/run` | POST | Запустити evaluation |
| `/benchmark/selection` | GET | MCDA сторінка |
| `/benchmark/selection/compute` | POST | Перерахувати score з новими вагами |
| `/benchmark/explorer` | GET | Список запитів |
| `/benchmark/explorer/<qid>` | GET | Деталі запиту |

### Ключові функції

#### `do_search(query, model_id, top_k, domain)` — головний пошук

```python
def do_search(query, model_id, top_k, domain):
    artifacts_dir = _get_artifacts_dir(model_id, domain)
    # BM25 path
    if (artifacts_dir / "bm25.pkl").exists():
        bm25, meta = get_bm25_model(model_id, domain)
        ranked_idx = bm25.search(query, top_k)
        ranked = [(meta[i], score) for i, score in ranked_idx]
        return _format_search_results(ranked)
    # FAISS path
    index, meta, model = get_index_and_model(model_id, domain)
    query_vector = model.encode_queries([query]).astype("float32")
    scores, indices = index.search(query_vector, top_k)
    ranked = [
        (meta[idx], float(score))
        for score, idx in zip(scores[0], indices[0])
        if 0 <= idx < len(meta)
    ]
    return _format_search_results(ranked)
```

**Що робить:**
1. Перевіряє, чи це BM25 (по `bm25.pkl` файлу) — окремий шлях
2. Інакше — embedding: завантажує FAISS-індекс, модель, кодує запит, шукає топ-K
3. Повертає `[(chunk, score), ...]`

#### `get_index_and_model(model_id, domain)` — кешоване завантаження

```python
def get_index_and_model(model_id, domain_id):
    cache_key = f"{domain_id}/{model_id}"
    if cache_key in _cache:
        return _cache[cache_key]
    artifacts_dir = _get_artifacts_dir(model_id, domain_id)
    index = faiss.read_index(str(artifacts_dir / "faiss.index"))
    meta = load_meta(artifacts_dir / "meta.jsonl")
    model, model_cfg = load_model_from_artifacts(
        artifacts_dir=artifacts_dir,
        fallback_model_name="BAAI/bge-m3"
    )
    _cache[cache_key] = (index, meta, model)
    return index, meta, model
```

**Що робить:**
1. Кеш per `domain/model_id`
2. Якщо немає в кеші — завантажуємо FAISS, meta, model
3. Перший запит — повільний (5-30 с завантаження моделі); наступні — миттєві

#### `_prettify_benchmark_data(data)` — нормалізація імен моделей

Замінює сирі `model_name` (наприклад `BAAI/bge-m3`) на читабельні (`BGE-M3`).

---

## 3. `src/embedding_models.py`

Класи моделей. ~270 рядків.

### Базова структура

```python
class BaseEmbeddingModel:
    def encode_passages(self, texts: list[str]) -> np.ndarray:
        """Кодує документи. Повертає (N, dim) array."""
        raise NotImplementedError
    
    def encode_queries(self, queries: list[str]) -> np.ndarray:
        """Кодує запити. Повертає (Q, dim) array."""
        raise NotImplementedError
```

### `class E5EmbeddingModel`

```python
class E5EmbeddingModel:
    def __init__(self, model_name, max_seq_length=512, batch_size=64):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.model.max_seq_length = max_seq_length
        self.batch_size = batch_size
    
    def encode_passages(self, texts):
        prefixed = [f"passage: {t}" for t in texts]
        return self.model.encode(
            prefixed,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False
        )
    
    def encode_queries(self, queries):
        prefixed = [f"query: {q}" for q in queries]
        return self.model.encode(
            prefixed,
            batch_size=self.batch_size,
            normalize_embeddings=True
        )
```

**Особливість:** автоматично додає префікси `query:` і `passage:`. **Це критично**.

### `class SbertEmbeddingModel`

Для **BGE-M3** і **Qwen3-Embedding**. Прямий wrapper над `SentenceTransformer`. Без префіксів.

```python
class SbertEmbeddingModel:
    def __init__(self, model_name, max_seq_length=256, batch_size=32):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.model.max_seq_length = max_seq_length
        self.batch_size = batch_size
    
    def encode_passages(self, texts):
        return self.model.encode(texts, batch_size=self.batch_size,
                                 normalize_embeddings=True)
    
    def encode_queries(self, queries):
        return self.model.encode(queries, batch_size=self.batch_size,
                                 normalize_embeddings=True)
```

### `class NomicEmbeddingModel`

Особливість: `trust_remote_code=True` (Nomic потребує).

```python
class NomicEmbeddingModel:
    def __init__(self, model_name, max_seq_length=512):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(
            model_name,
            trust_remote_code=True
        )
        self.model.max_seq_length = max_seq_length
    
    def encode_passages(self, texts):
        return self.model.encode(texts, normalize_embeddings=True)
    
    def encode_queries(self, queries):
        return self.model.encode(queries, normalize_embeddings=True)
```

### `class BM25RetrievalModel`

Окрема логіка — не через embeddings.

```python
class BM25RetrievalModel:
    def __init__(self, corpus_texts):
        from rank_bm25 import BM25Okapi
        tokenized_corpus = [doc.lower().split() for doc in corpus_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
    
    def search(self, query, top_k):
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        # Топ-K індекси
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [(i, scores[i]) for i in top_indices]
```

### `load_model_from_artifacts()` — factory

```python
def load_model_from_artifacts(artifacts_dir, fallback_model_name):
    cfg_path = artifacts_dir / "model.json"
    cfg = json.loads(cfg_path.read_text())
    model_type = cfg["model_type"]
    model_name = cfg["model_name"]
    params = cfg.get("params", {})
    
    if model_type == "e5":
        return E5EmbeddingModel(model_name, **params), cfg
    elif model_type == "sbert":
        return SbertEmbeddingModel(model_name, **params), cfg
    elif model_type == "nomic":
        return NomicEmbeddingModel(model_name, **params), cfg
    # ... etc
```

---

## 4. `src/build_index.py`

Будує FAISS-індекс або BM25 артефакт. ~200 рядків.

### Загальна структура

```python
def main():
    args = parse_args()
    chunks = load_chunks(args.chunks)
    
    if args.model_type == "bm25":
        build_bm25_index(chunks, args.artifacts)
    else:
        build_faiss_index(chunks, args)


def build_faiss_index(chunks, args):
    # 1. Завантажуємо модель
    model = load_model(args.model_type, args.model)
    
    # 2. Кодуємо чанки
    vectors = model.encode_passages([c["text"] for c in chunks])
    
    # 3. Створюємо FAISS-індекс
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    
    # 4. Зберігаємо
    os.makedirs(args.artifacts, exist_ok=True)
    faiss.write_index(index, f"{args.artifacts}/faiss.index")
    save_meta(chunks, f"{args.artifacts}/meta.jsonl")
    save_model_config(args, f"{args.artifacts}/model.json")
```

### CLI

```bash
python src/build_index.py \
    --model-type sbert \
    --model BAAI/bge-m3 \
    --chunks data/domains/tech/chunks.jsonl \
    --artifacts artifacts/tech/bge-m3 \
    --batch-size 32
```

---

## 5. `src/evaluate_benchmark.py`

Обчислює метрики. ~400 рядків.

### Pipeline

```python
def main():
    queries = load_queries(args.queries)
    qrels = load_qrels(args.qrels)
    
    for model_dir in args.model_artifacts:
        model, index, meta = load(model_dir)
        per_query_results = []
        for q in queries:
            top_k = search(q["text"], model, index, meta, k=10)
            metrics = compute_metrics(top_k, qrels[q["query_id"]])
            per_query_results.append(metrics)
        
        aggregated = aggregate(per_query_results)
        ci = bootstrap_ci(per_query_results)
        save_results(aggregated, ci)
```

### `compute_ndcg(retrieved, qrels, k=10)`

```python
def compute_ndcg(retrieved, relevant_chunk_ids, k=10):
    """retrieved: список chunk_ids у порядку видачі."""
    dcg = 0
    for i, chunk_id in enumerate(retrieved[:k], start=1):
        if chunk_id in relevant_chunk_ids:
            dcg += 1 / math.log2(i + 1)
    
    # IDCG: ідеальне ранжування — всі релевантні зверху
    idcg = sum(
        1 / math.log2(i + 1)
        for i in range(1, min(len(relevant_chunk_ids), k) + 1)
    )
    return dcg / idcg if idcg > 0 else 0
```

### `bootstrap_ci(per_query_values, n_iter=2000, ci=0.95)`

```python
def bootstrap_ci(values, n_iter=2000, ci=0.95):
    means = []
    for _ in range(n_iter):
        sample = np.random.choice(values, size=len(values), replace=True)
        means.append(np.mean(sample))
    lower = np.percentile(means, (1-ci)/2 * 100)
    upper = np.percentile(means, (1+ci)/2 * 100)
    return lower, upper
```

---

## 6. `src/multi_criteria.py`

MCDA логіка. ~300 рядків.

### `normalize_metric(values, direction)`

```python
def normalize_metric(values, direction="max"):
    """Min-max normalization to [0, 1]."""
    min_v, max_v = min(values), max(values)
    if max_v == min_v:
        return [1.0] * len(values)
    if direction == "max":
        return [(v - min_v) / (max_v - min_v) for v in values]
    else:  # min
        return [(max_v - v) / (max_v - min_v) for v in values]
```

### `pareto_front(alternatives)` — Парето-фільтр

```python
def pareto_front(alternatives, criteria_directions):
    pareto = []
    for i, a in enumerate(alternatives):
        is_dominated = False
        for j, b in enumerate(alternatives):
            if i == j: continue
            if dominates(b, a, criteria_directions):
                is_dominated = True
                break
        if not is_dominated:
            pareto.append(i)
    return pareto


def dominates(b, a, directions):
    at_least_one_better = False
    for j, dir_ in enumerate(directions):
        b_val, a_val = b[j], a[j]
        if dir_ == "max":
            if b_val < a_val: return False
            if b_val > a_val: at_least_one_better = True
        else:  # min
            if b_val > a_val: return False
            if b_val < a_val: at_least_one_better = True
    return at_least_one_better
```

### `run_selection(models, weights)` — повний MCDA

```python
def run_selection(models, weights):
    # 1. Extract metric values
    criteria = ["ndcg_at_k", "mrr_at_k", "recall_at_k",
                "precision_at_k", "avg_latency_ms"]
    directions = ["max", "max", "max", "max", "min"]
    
    # 2. Normalize each criterion
    normalized = {}
    for j, crit in enumerate(criteria):
        values = [m[crit] for m in models]
        normalized[crit] = normalize_metric(values, directions[j])
    
    # 3. Pareto
    points = [[m[c] for c in criteria] for m in models]
    pareto_idx = pareto_front(points, directions)
    
    # 4. Linear additive score
    scores = []
    for i, m in enumerate(models):
        score = sum(
            weights[c] * normalized[c][i]
            for c in criteria
        )
        scores.append({
            "model_name": m["model_name"],
            "score": score,
            "is_pareto": i in pareto_idx,
            # ... etc
        })
    
    # 5. Sort by score
    scores.sort(key=lambda x: x["score"], reverse=True)
    return scores
```

---

## 7. `src/ingest_chunk.py`

Чанкінг документів. ~200 рядків.

### `ingest_chunks(input_dir, output_file, min_words, max_words, overlap)`

```python
def ingest_chunks(input_dir, output_file, min_words=300, max_words=800, overlap=80):
    all_chunks = []
    for file_path in Path(input_dir).rglob("*"):
        if file_path.suffix.lower() == ".pdf":
            text = extract_pdf_text(file_path)
        elif file_path.suffix.lower() == ".docx":
            text = extract_docx_text(file_path)
        elif file_path.suffix.lower() in (".txt", ".md"):
            text = file_path.read_text(encoding="utf-8")
        else:
            continue
        
        chunks = sliding_window_chunks(text, min_words, max_words, overlap)
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "doc_id": file_path.stem,
                "chunk_id": f"{file_path.stem}_chunk_{i:04d}",
                "text": chunk,
                "source": file_path.name,
            })
    
    # Save as JSONL
    with open(output_file, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    
    return len(all_chunks)


def sliding_window_chunks(text, min_w, max_w, overlap):
    words = text.split()
    chunks = []
    i = 0
    target = (min_w + max_w) // 2  # ~600
    while i < len(words):
        chunk_end = min(i + target, len(words))
        chunk = words[i:chunk_end]
        if len(chunk) >= min_w:
            chunks.append(" ".join(chunk))
        i = chunk_end - overlap
        if chunk_end == len(words):
            break
    return chunks
```

---

## 8. Data flow повний pipeline

```
┌──────────────────────────────────────────────────────────────┐
│ КРОК 1: DOCUMENTS → CHUNKS                                   │
│                                                               │
│ data/domains/tech/raw/*.pdf                                  │
│      ↓ src/ingest_chunk.py                                   │
│ data/domains/tech/chunks.jsonl                               │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│ КРОК 2: CHUNKS → EMBEDDINGS + INDEX                          │
│                                                               │
│ chunks.jsonl                                                 │
│      ↓ src/build_index.py --model-type sbert                 │
│      ↓ (модель: BGE-M3 / E5 / nomic / Qwen3)                 │
│ artifacts/tech/bge-m3/                                       │
│   ├── faiss.index    ← вектори                              │
│   ├── meta.jsonl     ← метадані                             │
│   └── model.json     ← конфіг                               │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│ КРОК 3a: ONLINE SEARCH                                        │
│                                                               │
│ HTTP GET /?q=...&model=bge-m3&domain=tech                    │
│      ↓ src/app.py:do_search()                                │
│      ↓ model.encode_queries([q]) → query vector              │
│      ↓ faiss.read_index, index.search(qv, k=10)              │
│ JSON results to UI                                           │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│ КРОК 3b: BENCHMARK EVALUATION                                │
│                                                               │
│ queries.jsonl + qrels.jsonl                                  │
│      ↓ src/evaluate_benchmark.py                             │
│      ↓ Для кожного запиту: пошук + обчислення метрик          │
│      ↓ Bootstrap CI                                          │
│ results/benchmark/tech/benchmark_results_*.json              │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│ КРОК 4: MCDA SELECTION                                       │
│                                                               │
│ benchmark_results_*.json                                     │
│      ↓ src/multi_criteria.py:run_selection()                 │
│      ↓ Normalize → Pareto → Linear additive                  │
│ Рекомендована модель + scores                                │
│                                                               │
│ UI: /benchmark/selection?domain=tech                         │
└──────────────────────────────────────────────────────────────┘
```

---

## 9. Як запустити локально

### Initial setup

```bash
# Створи venv
python -m venv .venv
.venv/Scripts/activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Встанови залежності
pip install -r requirements.txt
```

### Запуск Flask

```bash
.venv/Scripts/python.exe src/app.py
# Відкрий http://localhost:5000
```

### Запуск benchmark для одного домена

```bash
.venv/Scripts/python.exe src/evaluate_benchmark.py \
    --queries data/domains/tech/benchmark/queries.jsonl \
    --qrels   data/domains/tech/benchmark/qrels.jsonl \
    --model-artifacts artifacts/tech/bge-m3 artifacts/tech/e5-base \
    --top-k 10 \
    --output results/benchmark/tech/
```

### Запустити всі benchmark одразу

```bash
.venv/Scripts/python.exe scripts/run_all_benchmarks.py
# Запустить evaluate для всіх 3 доменів × всіх 5 моделей
# Згенерує RESULTS.md
```

### Побудувати один індекс

```bash
.venv/Scripts/python.exe src/build_index.py \
    --model-type sbert \
    --model BAAI/bge-m3 \
    --chunks data/domains/tech/chunks.jsonl \
    --artifacts artifacts/tech/bge-m3 \
    --batch-size 32
```

---

## 10. Перевірочні питання

### Q1. Як працює `do_search`?

**A:**
1. Перевіряє, чи це BM25-модель (через `bm25.pkl` файл)
2. Якщо BM25 — викликає `bm25.get_scores(tokens)`, повертає топ-K за scores
3. Якщо embedding — завантажує FAISS-індекс, model, кодує запит через `model.encode_queries`, шукає топ-K через `index.search(qv, k)`
4. Повертає `[(chunk, score), ...]`

### Q2. Чому в коді є `_cache` для models?

**A:** Завантаження embedding моделі — повільний процес (5-30 секунд для BGE-M3). Тому ми **кешуємо** model + index per `domain/model_id`. Перший пошук — повільний, наступні — миттєві. Це **критично** для UX.

### Q3. Як ти би додав 5-ту модель?

**A:**
1. Створив би клас у `src/embedding_models.py`, наприклад `MyEmbeddingModel`
2. Додав би тип в `load_model_from_artifacts()`
3. Створив би `model.json` з `model_type="my"`, `model_name="..."`, params
4. Запустив би `build_index.py --model-type my ...`
5. Запустив би `evaluate_benchmark.py --model-artifacts artifacts/.../my`

Без зміни решти коду.

### Q4. Чому `meta.jsonl`, а не SQLite або базу даних?

**A:** Простота і **read-only** характер. JSONL легко читати рядково, легко версіонувати в git, не потребує тяжких залежностей. Для тисяч-десятків тисяч чанків — достатньо. Для мільйонів — варто перейти на SQLite або справжню БД.

### Q5. Як працює кеш для BM25?

**A:** Окрема функція `get_bm25_model()` з власним `_cache` ключем `bm25/{domain}/{model_id}`. Логіка та ж — перший виклик завантажує `bm25.pkl`, наступні — з кешу.

### Q6. Що робить `_prettify_benchmark_data`?

**A:** Маскує сирі назви моделей (типу `BAAI/bge-m3`) на читабельні (`BGE-M3`). Логіка: бере `artifacts_dir` (наприклад `artifacts/tech/bge-m3/`), бере останню частину пути (`bge-m3`), знаходить у `_TYPE_TO_DISPLAY` мапінгу → `"BGE-M3"`. Це для красивого UI.

### Q7. Чому код розкиданий по кільком файлам, а не один великий?

**A:** **Separation of concerns**:
- `app.py` — HTTP/Flask логіка
- `embedding_models.py` — model abstraction
- `build_index.py` — pipeline для індексації
- `evaluate_benchmark.py` — оцінка
- `multi_criteria.py` — MCDA

Кожен файл — окрема відповідальність. Легко тестувати, читати, змінювати. Класичний modular design.

### Q8. Як працює `/benchmark/run` (запуск benchmark з UI)?

**A:** Endpoint запускає `subprocess.Popen(...)` для `evaluate_benchmark.py` у фоновому потоці. Зберігає state у глобальному `_bench_job`. Frontend через `/benchmark/status` GET polls статус. Це **простий** background job без Celery/RQ — підходить для single-user dev environment.

### Q9. Як організовано MCDA розрахунок з custom weights?

**A:** Endpoint `/benchmark/selection/compute` POST з JSON body, де ключі — імена критеріїв, значення — ваги (0-1). Endpoint викликає `run_selection(models, weights)`, повертає результат як JSON. Frontend (selection.html) робить AJAX-виклик і оновлює таблицю — без перезавантаження сторінки.

### Q10. Який найбільший hot path у коді?

**A:** **`encode_queries` / `encode_passages`** — це **виклик нейромережі**, найдорожча операція. Для BGE-M3 на CPU — 200+ мс на запит. Для batch індексації 1000 чанків — кілька хвилин. Це **bottleneck** системи. У production варто:
- Кешувати encoded vectors для повторюваних запитів
- Використовувати GPU
- Розглянути smaller models (E5-base замість BGE-M3) для real-time

---

## Що далі

Код зрозумілий. Останній файл — **прохід по сторінках застосунку**. Корисно перед демо. [18_app_walkthrough.md](18_app_walkthrough.md).
