import argparse
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def load_meta(path: Path) -> list[dict]:
	records: list[dict] = []
	with path.open("r", encoding="utf-8") as handle:
		for line in handle:
			line = line.strip()
			if not line:
				continue
			records.append(json.loads(line))
	return records


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Query FAISS index.")
	parser.add_argument("query", help="Query text.")
	parser.add_argument("--artifacts", default="artifacts")
	parser.add_argument("--model", default="paraphrase-multilingual-MiniLM-L12-v2")
	parser.add_argument("--top-k", type=int, default=5, help="Number of most relevant chunks to return")
	parser.add_argument("--full-text", action="store_true", help="Show full chunk text (not just snippet)")
	parser.add_argument("--json", action="store_true", help="Output results as JSON")
	parser.add_argument("--max-snippet", type=int, default=300, help="Max characters in snippet (if not --full-text)")
	return parser


def main() -> None:
	args = build_arg_parser().parse_args()
	artifacts_dir = Path(args.artifacts)
	index_path = artifacts_dir / "faiss.index"
	meta_path = artifacts_dir / "meta.jsonl"

	index = faiss.read_index(str(index_path))
	meta = load_meta(meta_path)

	model = SentenceTransformer(args.model)
	query_vector = model.encode(
		[args.query], convert_to_numpy=True, normalize_embeddings=True
	)
	query_vector = np.array(query_vector).astype("float32")

	scores, indices = index.search(query_vector, args.top_k)
	
	results = []
	for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
		if idx < 0 or idx >= len(meta):
			continue
		record = meta[idx]
		
		if args.full_text:
			text = record["text"]
		else:
			text = record["text"][:args.max_snippet]
			if len(record["text"]) > args.max_snippet:
				text += "..."
		
		result = {
			"rank": rank,
			"score": float(score),
			"source": record["source"],
			"chunk_id": record["chunk_id"],
			"text": text,
			"full_length": len(record["text"])
		}
		results.append(result)
	
	if args.json:
		print(json.dumps(results, ensure_ascii=False, indent=2))
	else:
		print(f"\nЗнайдено {len(results)} найбільш релевантних чанків:\n")
		for r in results:
			print(f"#{r['rank']} | Score: {r['score']:.4f} | {r['source']} ({r['chunk_id']})")
			print(f"Довжина: {r['full_length']} символів")
			print(f"\n{r['text']}\n")
			print("-" * 80)


if __name__ == "__main__":
	main()
