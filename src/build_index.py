import argparse
import json
import logging
from pathlib import Path

import faiss
import numpy as np
from tqdm import tqdm

from embedding_models import (
	ModelConfig,
	build_gensim_model,
	build_tfidf_model,
	save_gensim_model,
	save_tfidf_model,
	tokenize,
)

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
		choices=["sbert", "tfidf", "word2vec", "fasttext", "glove", "bert"],
		default="sbert",
		help="Тип моделі для векторизації",
	)
	parser.add_argument(
		"--model",
		default="paraphrase-multilingual-MiniLM-L12-v2",
		help="Назва моделі SBERT",
	)
	parser.add_argument(
		"--bert-model",
		default="bert-base-multilingual-cased",
		help="Назва моделі BERT",
	)
	parser.add_argument("--bert-batch-size", type=int, default=32, help="Розмір батчу для BERT")
	parser.add_argument("--bert-max-length", type=int, default=256, help="Макс. довжина послідовності BERT")
	parser.add_argument("--glove-path", help="Шлях до GloVe/word2vec векторів")
	parser.add_argument(
		"--glove-binary",
		action="store_true",
		help="Формат GloVe у binary word2vec",
	)
	parser.add_argument(
		"--glove-has-header",
		action="store_true",
		help="GloVe файл із заголовком (word2vec формат)",
	)
	parser.add_argument("--tfidf-max-features", type=int, default=50000, help="Макс. кількість ознак TF-IDF")
	parser.add_argument("--tfidf-ngram-min", type=int, default=1, help="Мін. розмір n-грами TF-IDF")
	parser.add_argument("--tfidf-ngram-max", type=int, default=2, help="Макс. розмір n-грами TF-IDF")
	parser.add_argument("--gensim-vector-size", type=int, default=300, help="Розмірність векторів Word2Vec/FastText")
	parser.add_argument("--gensim-window", type=int, default=5, help="Контекстне вікно Word2Vec/FastText")
	parser.add_argument("--gensim-min-count", type=int, default=2, help="Мін. частота токенів Word2Vec/FastText")
	parser.add_argument("--gensim-epochs", type=int, default=10, help="Кількість епох Word2Vec/FastText")
	parser.add_argument("--gensim-workers", type=int, default=4, help="Кількість потоків Word2Vec/FastText")
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
		vectors = []
		for start in tqdm(range(0, len(texts), args.batch_size), desc="Векторизація"):
			batch = texts[start : start + args.batch_size]
			batch_vectors = model.encode(
				batch, convert_to_numpy=True, normalize_embeddings=True
			)
			vectors.append(batch_vectors)
		vectors_np = np.vstack(vectors).astype("float32")
		config = ModelConfig(model_type="sbert", model_name=args.model, params={})
	elif args.model_type == "tfidf":
		model = build_tfidf_model(
			texts,
			max_features=args.tfidf_max_features,
			ngram_range=(args.tfidf_ngram_min, args.tfidf_ngram_max),
		)
		vectors_np = model.encode_documents(texts)
		config = ModelConfig(
			model_type="tfidf",
			model_name=None,
			params={
				"max_features": args.tfidf_max_features,
				"ngram_min": args.tfidf_ngram_min,
				"ngram_max": args.tfidf_ngram_max,
			},
		)
		save_tfidf_model(model, artifacts_dir / "tfidf.joblib")
	elif args.model_type in {"word2vec", "fasttext"}:
		sentences = [tokenize(text) for text in texts]
		model = build_gensim_model(
			args.model_type,
			sentences,
			vector_size=args.gensim_vector_size,
			window=args.gensim_window,
			min_count=args.gensim_min_count,
			epochs=args.gensim_epochs,
			workers=args.gensim_workers,
		)
		vectors_np = model.encode_documents(texts)
		config = ModelConfig(
			model_type=args.model_type,
			model_name=None,
			params={
				"vector_size": args.gensim_vector_size,
				"window": args.gensim_window,
				"min_count": args.gensim_min_count,
				"epochs": args.gensim_epochs,
				"workers": args.gensim_workers,
			},
		)
		save_gensim_model(model, artifacts_dir / "gensim.model")
	elif args.model_type == "glove":
		if not args.glove_path:
			raise SystemExit("Для GloVe потрібен --glove-path")
		from embedding_models import load_glove_model

		no_header = not args.glove_has_header
		model = load_glove_model(
			Path(args.glove_path),
			binary=args.glove_binary,
			no_header=no_header,
		)
		vectors_np = model.encode_documents(texts)
		config = ModelConfig(
			model_type="glove",
			model_name=None,
			params={
				"glove_path": args.glove_path,
				"glove_binary": args.glove_binary,
				"glove_no_header": no_header,
			},
		)
	elif args.model_type == "bert":
		from embedding_models import BertEmbeddingModel

		model = BertEmbeddingModel(
			args.bert_model,
			batch_size=args.bert_batch_size,
			max_length=args.bert_max_length,
		)
		vectors_np = model.encode_documents(texts)
		config = ModelConfig(
			model_type="bert",
			model_name=args.bert_model,
			params={
				"batch_size": args.bert_batch_size,
				"max_length": args.bert_max_length,
			},
		)
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
