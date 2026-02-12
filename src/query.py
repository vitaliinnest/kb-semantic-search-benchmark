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
	parser.add_argument("--model", default="all-MiniLM-L6-v2")
	parser.add_argument("--top-k", type=int, default=5)
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
	).astype("float32")

	scores, indices = index.search(query_vector, args.top_k)
	for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
		if idx < 0 or idx >= len(meta):
			continue
		record = meta[idx]
		snippet = record["text"][:300]
		print(f"#{rank} score={float(score):.4f} source={record['source']}")
		print(snippet)
		print("-")


if __name__ == "__main__":
	main()
