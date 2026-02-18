import argparse
import json
import math
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import faiss
import numpy as np

# Fix Windows console encoding
if sys.platform == "win32":
	import codecs
	sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")
	sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, "strict")

from embedding_models import load_model_from_artifacts

logging.basicConfig(
	level=logging.INFO,
	format="%(asctime)s - %(levelname)s - %(message)s",
)


@dataclass
class QueryItem:
	query_id: str
	query: str


@dataclass
class QrelItem:
	query_id: str
	chunk_id: str | None
	doc_id: str | None
	relevance: int


@dataclass
class QueryEvalResult:
	query_id: str
	recall_at_k: float
	mrr_at_k: float
	ndcg_at_k: float
	retrieved: int
	relevant_total: int


@dataclass
class ModelEvalResult:
	model_name: str
	artifacts_dir: str
	queries_evaluated: int
	recall_at_k: float
	mrr_at_k: float
	ndcg_at_k: float
	query_results: list[QueryEvalResult]


def load_jsonl(path: Path) -> list[dict]:
	records: list[dict] = []
	with path.open("r", encoding="utf-8") as handle:
		for line in handle:
			line = line.strip()
			if not line:
				continue
			records.append(json.loads(line))
	return records


def load_queries(path: Path) -> list[QueryItem]:
	records = load_jsonl(path)
	queries: list[QueryItem] = []
	for row in records:
		query_id = str(row.get("query_id", "")).strip()
		query = str(row.get("query", "")).strip()
		if not query_id or not query:
			continue
		queries.append(QueryItem(query_id=query_id, query=query))
	return queries


def load_qrels(path: Path) -> dict[str, list[QrelItem]]:
	records = load_jsonl(path)
	result: dict[str, list[QrelItem]] = defaultdict(list)
	for row in records:
		query_id = str(row.get("query_id", "")).strip()
		if not query_id:
			continue
		chunk_id_raw = row.get("chunk_id")
		doc_id_raw = row.get("doc_id")
		relevance_raw = row.get("relevance", 0)
		chunk_id = str(chunk_id_raw).strip() if chunk_id_raw is not None else None
		doc_id = str(doc_id_raw).strip() if doc_id_raw is not None else None
		if chunk_id == "":
			chunk_id = None
		if doc_id == "":
			doc_id = None
		try:
			relevance = int(relevance_raw)
		except (TypeError, ValueError):
			relevance = 0
		result[query_id].append(
			QrelItem(
				query_id=query_id,
				chunk_id=chunk_id,
				doc_id=doc_id,
				relevance=relevance,
			)
		)
	return result


def load_meta(path: Path) -> list[dict]:
	return load_jsonl(path)


def dcg_at_k(relevances: list[int], k: int) -> float:
	total = 0.0
	for idx, rel in enumerate(relevances[:k], start=1):
		if rel <= 0:
			continue
		total += (2**rel - 1) / math.log2(idx + 1)
	return total


def match_relevance(qrels: list[QrelItem], chunk_id: str, doc_id: str) -> int:
	best = 0
	for item in qrels:
		if item.relevance <= 0:
			continue
		if item.chunk_id and item.chunk_id == chunk_id:
			best = max(best, item.relevance)
		elif item.doc_id and item.doc_id == doc_id:
			best = max(best, item.relevance)
	return best


def count_relevant_targets(qrels: list[QrelItem]) -> int:
	targets: set[tuple[str, str]] = set()
	for item in qrels:
		if item.relevance <= 0:
			continue
		if item.chunk_id:
			targets.add(("chunk", item.chunk_id))
		elif item.doc_id:
			targets.add(("doc", item.doc_id))
	return len(targets)


def count_matched_targets(
	retrieved_meta: list[dict],
	qrels: list[QrelItem],
	k: int,
) -> int:
	matched: set[tuple[str, str]] = set()
	for record in retrieved_meta[:k]:
		chunk_id = str(record.get("chunk_id", ""))
		doc_id = str(record.get("source", ""))
		for item in qrels:
			if item.relevance <= 0:
				continue
			if item.chunk_id and item.chunk_id == chunk_id:
				matched.add(("chunk", item.chunk_id))
			elif item.doc_id and item.doc_id == doc_id:
				matched.add(("doc", item.doc_id))
	return len(matched)


def build_output_paths(output_arg: Path) -> tuple[Path, Path]:
	timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
	if output_arg.suffix.lower() == ".txt":
		text_path = output_arg
		json_path = output_arg.with_suffix(".json")
		return text_path, json_path

	output_dir = output_arg / "benchmark"
	text_path = output_dir / f"benchmark_results_{timestamp}.txt"
	json_path = output_dir / f"benchmark_results_{timestamp}.json"
	counter = 2
	while text_path.exists() or json_path.exists():
		text_path = output_dir / f"benchmark_results_{timestamp}-{counter}.txt"
		json_path = output_dir / f"benchmark_results_{timestamp}-{counter}.json"
		counter += 1
	return text_path, json_path


def discover_artifacts_dirs(artifacts_root: Path) -> list[Path]:
	dirs: list[Path] = []
	if (artifacts_root / "faiss.index").exists() and (artifacts_root / "meta.jsonl").exists():
		dirs.append(artifacts_root)

	for child in sorted(artifacts_root.iterdir() if artifacts_root.exists() else []):
		if not child.is_dir():
			continue
		if (child / "faiss.index").exists() and (child / "meta.jsonl").exists():
			dirs.append(child)
	return dirs


def evaluate_model(
	artifacts_dir: Path,
	queries: list[QueryItem],
	qrels_by_query: dict[str, list[QrelItem]],
	top_k: int,
	fallback_model_name: str,
) -> ModelEvalResult:
	index = faiss.read_index(str(artifacts_dir / "faiss.index"))
	meta = load_meta(artifacts_dir / "meta.jsonl")
	model, config = load_model_from_artifacts(
		artifacts_dir=artifacts_dir,
		fallback_model_name=fallback_model_name,
	)

	model_name = config.model_name or config.model_type
	query_results: list[QueryEvalResult] = []

	for query_item in queries:
		qrels = qrels_by_query.get(query_item.query_id, [])
		if not qrels:
			continue

		query_vector = model.encode_queries([query_item.query]).astype("float32")
		scores, indices = index.search(query_vector, top_k)

		retrieved_meta: list[dict] = []
		retrieved_rels: list[int] = []
		for idx in indices[0]:
			if idx < 0 or idx >= len(meta):
				continue
			record = meta[idx]
			retrieved_meta.append(record)
			chunk_id = str(record.get("chunk_id", ""))
			doc_id = str(record.get("source", ""))
			retrieved_rels.append(match_relevance(qrels, chunk_id=chunk_id, doc_id=doc_id))

		relevant_total = count_relevant_targets(qrels)
		matched_total = count_matched_targets(retrieved_meta, qrels, top_k)
		recall_at_k = (matched_total / relevant_total) if relevant_total > 0 else 0.0

		mrr_at_k = 0.0
		for rank, rel in enumerate(retrieved_rels[:top_k], start=1):
			if rel > 0:
				mrr_at_k = 1.0 / rank
				break

		ideal_rels = sorted([max(0, item.relevance) for item in qrels], reverse=True)
		dcg = dcg_at_k(retrieved_rels, top_k)
		idcg = dcg_at_k(ideal_rels, top_k)
		ndcg_at_k = (dcg / idcg) if idcg > 0 else 0.0

		query_results.append(
			QueryEvalResult(
				query_id=query_item.query_id,
				recall_at_k=recall_at_k,
				mrr_at_k=mrr_at_k,
				ndcg_at_k=ndcg_at_k,
				retrieved=len(retrieved_meta),
				relevant_total=relevant_total,
			)
		)

	queries_evaluated = len(query_results)
	if queries_evaluated == 0:
		return ModelEvalResult(
			model_name=model_name,
			artifacts_dir=str(artifacts_dir),
			queries_evaluated=0,
			recall_at_k=0.0,
			mrr_at_k=0.0,
			ndcg_at_k=0.0,
			query_results=[],
		)

	return ModelEvalResult(
		model_name=model_name,
		artifacts_dir=str(artifacts_dir),
		queries_evaluated=queries_evaluated,
		recall_at_k=float(np.mean([x.recall_at_k for x in query_results])),
		mrr_at_k=float(np.mean([x.mrr_at_k for x in query_results])),
		ndcg_at_k=float(np.mean([x.ndcg_at_k for x in query_results])),
		query_results=query_results,
	)


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="Оцінка якості retrieval по benchmark (Recall@k, MRR@k, nDCG@k)."
	)
	parser.add_argument("--queries", default="data/benchmark/queries.jsonl", help="Файл з benchmark queries")
	parser.add_argument("--qrels", default="data/benchmark/qrels.jsonl", help="Файл з benchmark qrels")
	parser.add_argument("--artifacts-root", default="artifacts", help="Коренева папка артефактів моделей")
	parser.add_argument(
		"--model-artifacts",
		nargs="*",
		help="Явний список папок артефактів моделей (якщо не задано, буде авто-пошук у --artifacts-root)",
	)
	parser.add_argument("--top-k", type=int, default=10, help="k для Recall/MRR/nDCG")
	parser.add_argument(
		"--output",
		default="results",
		help="Папка результатів (буде створено benchmark_results_YYYY-MM-DD_HH-MM-SS.txt/.json) або конкретний .txt файл",
	)
	parser.add_argument(
		"--model",
		default="paraphrase-multilingual-MiniLM-L12-v2",
		help="Fallback модель для випадку відсутнього model.json",
	)
	return parser


def main() -> None:
	args = build_arg_parser().parse_args()

	queries_path = Path(args.queries)
	qrels_path = Path(args.qrels)
	if not queries_path.exists():
		raise SystemExit(f"Файл queries не знайдено: {queries_path}")
	if not qrels_path.exists():
		raise SystemExit(f"Файл qrels не знайдено: {qrels_path}")

	queries = load_queries(queries_path)
	qrels_by_query = load_qrels(qrels_path)
	if not queries:
		raise SystemExit("У файлі queries немає валідних записів")

	if args.model_artifacts:
		artifacts_dirs = [Path(x) for x in args.model_artifacts]
	else:
		artifacts_dirs = discover_artifacts_dirs(Path(args.artifacts_root))

	if not artifacts_dirs:
		raise SystemExit("Не знайдено жодної папки артефактів моделі для оцінки")

	logging.info(f"Queries: {len(queries)} | Models: {len(artifacts_dirs)} | top_k={args.top_k}")

	model_results: list[ModelEvalResult] = []
	for artifacts_dir in artifacts_dirs:
		logging.info(f"Оцінка моделі в: {artifacts_dir}")
		result = evaluate_model(
			artifacts_dir=artifacts_dir,
			queries=queries,
			qrels_by_query=qrels_by_query,
			top_k=args.top_k,
			fallback_model_name=args.model,
		)
		model_results.append(result)

	text_path, json_path = build_output_paths(Path(args.output))
	text_path.parent.mkdir(parents=True, exist_ok=True)

	text_lines = []
	text_lines.append("Оцінка benchmark retrieval")
	text_lines.append(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
	text_lines.append(f"Queries: {len(queries)} | top_k={args.top_k}")
	text_lines.append("=" * 90)
	text_lines.append(
		f"{'Model':<24} {'Queries':<8} {'Recall@k':<12} {'MRR@k':<12} {'nDCG@k':<12} {'Artifacts':<20}"
	)
	text_lines.append("-" * 90)
	for row in model_results:
		text_lines.append(
			f"{row.model_name:<24} {row.queries_evaluated:<8} {row.recall_at_k:<12.4f} {row.mrr_at_k:<12.4f} {row.ndcg_at_k:<12.4f} {row.artifacts_dir}"
		)

	text_path.write_text("\n".join(text_lines) + "\n", encoding="utf-8")

	json_payload = {
		"timestamp": datetime.now().isoformat(),
		"top_k": args.top_k,
		"queries_count": len(queries),
		"queries_file": str(queries_path),
		"qrels_file": str(qrels_path),
		"models": [
			{
				"model_name": row.model_name,
				"artifacts_dir": row.artifacts_dir,
				"queries_evaluated": row.queries_evaluated,
				"recall_at_k": row.recall_at_k,
				"mrr_at_k": row.mrr_at_k,
				"ndcg_at_k": row.ndcg_at_k,
				"query_results": [
					{
						"query_id": q.query_id,
						"recall_at_k": q.recall_at_k,
						"mrr_at_k": q.mrr_at_k,
						"ndcg_at_k": q.ndcg_at_k,
						"retrieved": q.retrieved,
						"relevant_total": q.relevant_total,
					}
					for q in row.query_results
				],
			}
			for row in model_results
		],
	}
	json_path.write_text(json.dumps(json_payload, ensure_ascii=False, indent=2), encoding="utf-8")

	print("\n" + "=" * 90)
	print("✅ Benchmark evaluation completed")
	print(f"📝 TXT: {text_path.absolute()}")
	print(f"📊 JSON: {json_path.absolute()}")
	print("=" * 90)


if __name__ == "__main__":
	main()
