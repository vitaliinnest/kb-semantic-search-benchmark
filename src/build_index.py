import argparse
import json
import logging
import sys
from pathlib import Path

import faiss
import numpy as np
from tqdm import tqdm

# Fix Windows console encoding
if sys.platform == "win32":
	import codecs
	sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")
	sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, "strict")

from embedding_models import ModelConfig

logging.basicConfig(
	level=logging.INFO,
	format="%(asctime)s - %(levelname)s - %(message)s",
)


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
	parser.add_argument(
		"--model-type",
		choices=["sbert", "e5", "nomic", "openai", "bm25"],
		default="sbert",
		help="Тип моделі для векторизації",
	)
	parser.add_argument(
		"--model",
		default="BAAI/bge-m3",
		help="Назва моделі SBERT/E5/nomic/BGE (HuggingFace repo або локальний шлях)",
	)
	parser.add_argument("--openai-api-key", default=None, help="OpenAI API key для text-embedding-*")
	parser.add_argument("--max-seq-length", type=int, default=None, help="Максимальна довжина послідовності (токенів) для sentence-transformers")
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

	if args.model_type == "sbert":
		from sentence_transformers import SentenceTransformer

		model = SentenceTransformer(args.model)
		if args.max_seq_length:
			model.max_seq_length = args.max_seq_length
		vectors = []
		for start in tqdm(range(0, len(texts), args.batch_size), desc="Векторизація"):
			batch = texts[start : start + args.batch_size]
			batch_vectors = model.encode(
				batch, convert_to_numpy=True, normalize_embeddings=True
			)
			vectors.append(batch_vectors)
		vectors_np = np.vstack(vectors).astype("float32")
		config = ModelConfig(model_type="sbert", model_name=args.model, params={"max_seq_length": args.max_seq_length})
	elif args.model_type == "e5":
		from embedding_models import E5EmbeddingModel

		model_name = args.model if args.model != "paraphrase-multilingual-MiniLM-L12-v2" else "intfloat/multilingual-e5-base"
		model = E5EmbeddingModel(model_name, max_seq_length=args.max_seq_length)
		vectors_np = model.encode_documents(texts)
		config = ModelConfig(model_type="e5", model_name=model_name, params={"max_seq_length": args.max_seq_length})
	elif args.model_type == "nomic":
		from embedding_models import NomicEmbeddingModel

		model_name = args.model if args.model != "paraphrase-multilingual-MiniLM-L12-v2" else "nomic-ai/nomic-embed-text-v1.5"
		model = NomicEmbeddingModel(model_name, max_seq_length=args.max_seq_length)
		vectors_np = model.encode_documents(texts)
		config = ModelConfig(model_type="nomic", model_name=model_name, params={"max_seq_length": args.max_seq_length})
	elif args.model_type == "openai":
		import os
		from embedding_models import OpenAIEmbeddingModel

		model_name = args.model if args.model != "BAAI/bge-m3" else "text-embedding-3-large"
		api_key = args.openai_api_key or os.environ.get("OPENAI_API_KEY")
		if not api_key:
			raise SystemExit("OpenAI model requires --openai-api-key or OPENAI_API_KEY env var")
		model = OpenAIEmbeddingModel(model_name, api_key=api_key)
		vectors_np = model.encode_documents(texts)
		config = ModelConfig(model_type="openai", model_name=model_name, params={})
	elif args.model_type == "bm25":
		from embedding_models import BM25RetrievalModel, save_bm25_model

		bm25_model = BM25RetrievalModel(texts)
		save_bm25_model(bm25_model, artifacts_dir / "bm25.pkl")
		with (artifacts_dir / "meta.jsonl").open("w", encoding="utf-8") as handle:
			for chunk in chunks:
				handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")
		config = ModelConfig(model_type="bm25", model_name="BM25 (Okapi)", params={})
		config.save(artifacts_dir / "model.json")
		print(f"BM25: проіндексовано {len(chunks)} документів у {artifacts_dir}")
		return
	else:
		raise SystemExit(f"Невідомий тип моделі: {args.model_type}")

	np.save(artifacts_dir / "vectors.npy", vectors_np)
	with (artifacts_dir / "meta.jsonl").open("w", encoding="utf-8") as handle:
		for chunk in chunks:
			handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")

	config.save(artifacts_dir / "model.json")

	index = faiss.IndexFlatIP(vectors_np.shape[1])
	index.add(vectors_np)
	faiss.write_index(index, str(artifacts_dir / "faiss.index"))

	print(f"Збережено {len(chunks)} векторів у {artifacts_dir}")


if __name__ == "__main__":
	main()
