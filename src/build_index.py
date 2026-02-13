import argparse
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def load_chunks(path: Path) -> list[dict]:
	records: list[dict] = []
	with path.open("r", encoding="utf-8") as handle:
		for line in handle:
			line = line.strip()
			if not line:
				continue
			records.append(json.loads(line))
	return records


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Побудова FAISS індексу з чанків.")
	parser.add_argument("--chunks", default="data/chunks.jsonl", help="Шлях до файлу з чанками")
	parser.add_argument("--artifacts", default="artifacts", help="Папка для збереження індексу")
	parser.add_argument("--model", default="paraphrase-multilingual-MiniLM-L12-v2", help="Назва моделі для векторизації")
	parser.add_argument("--batch-size", type=int, default=64, help="Розмір батчу для обробки")
	return parser


def main() -> None:
	args = build_arg_parser().parse_args()
	chunks_path = Path(args.chunks)
	artifacts_dir = Path(args.artifacts)
	artifacts_dir.mkdir(parents=True, exist_ok=True)

	chunks = load_chunks(chunks_path)
	if not chunks:
		raise SystemExit("Чанки не знайдено. Спочатку запустіть ingest_chunk.py")

	texts = [chunk["text"] for chunk in chunks]
	model = SentenceTransformer(args.model)
	vectors = []

	for start in tqdm(range(0, len(texts), args.batch_size), desc="Векторизація"):
		batch = texts[start : start + args.batch_size]
		batch_vectors = model.encode(
			batch, convert_to_numpy=True, normalize_embeddings=True
		)
		vectors.append(batch_vectors)

	vectors_np = np.vstack(vectors).astype("float32")

	np.save(artifacts_dir / "vectors.npy", vectors_np)
	with (artifacts_dir / "meta.jsonl").open("w", encoding="utf-8") as handle:
		for chunk in chunks:
			handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")

	index = faiss.IndexFlatIP(vectors_np.shape[1])
	index.add(vectors_np)
	faiss.write_index(index, str(artifacts_dir / "faiss.index"))

	print(f"Збережено {len(chunks)} векторів у {artifacts_dir}")


if __name__ == "__main__":
	main()
