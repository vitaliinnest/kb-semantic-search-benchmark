import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
	norms = np.linalg.norm(vectors, axis=1, keepdims=True)
	norms = np.where(norms == 0.0, 1.0, norms)
	return vectors / norms


@dataclass
class ModelConfig:
	model_type: str
	model_name: str | None = None
	params: dict | None = None

	@classmethod
	def load(cls, path: Path) -> "ModelConfig":
		data = json.loads(path.read_text(encoding="utf-8"))
		return cls(
			model_type=data["model_type"],
			model_name=data.get("model_name"),
			params=data.get("params", {}),
		)

	def save(self, path: Path) -> None:
		payload = {
			"model_type": self.model_type,
			"model_name": self.model_name,
			"params": self.params or {},
		}
		path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class BaseEmbeddingModel:
	def encode_documents(self, texts: list[str]) -> np.ndarray:
		raise NotImplementedError

	def encode_queries(self, texts: list[str]) -> np.ndarray:
		return self.encode_documents(texts)


class SbertEmbeddingModel(BaseEmbeddingModel):
	def __init__(self, model_name: str, max_seq_length: int | None = None) -> None:
		from sentence_transformers import SentenceTransformer

		self.model_name = model_name
		self.model = SentenceTransformer(model_name)
		if max_seq_length:
			self.model.max_seq_length = max_seq_length

	def encode_documents(self, texts: list[str]) -> np.ndarray:
		vectors = self.model.encode(
			texts, convert_to_numpy=True, normalize_embeddings=True
		)
		return np.asarray(vectors, dtype="float32")


class E5EmbeddingModel(BaseEmbeddingModel):
	"""Microsoft E5 model — requires 'query: '/'passage: ' prefixes."""

	def __init__(self, model_name: str, max_seq_length: int | None = None) -> None:
		from sentence_transformers import SentenceTransformer

		self.model_name = model_name
		self.model = SentenceTransformer(model_name)
		if max_seq_length:
			self.model.max_seq_length = max_seq_length

	def encode_documents(self, texts: list[str]) -> np.ndarray:
		prefixed = ["passage: " + t for t in texts]
		vectors = self.model.encode(prefixed, convert_to_numpy=True, normalize_embeddings=True)
		return np.asarray(vectors, dtype="float32")

	def encode_queries(self, texts: list[str]) -> np.ndarray:
		prefixed = ["query: " + t for t in texts]
		vectors = self.model.encode(prefixed, convert_to_numpy=True, normalize_embeddings=True)
		return np.asarray(vectors, dtype="float32")


class NomicEmbeddingModel(BaseEmbeddingModel):
	"""nomic-embed-text — requires trust_remote_code and task prefixes."""

	def __init__(self, model_name: str, max_seq_length: int | None = None) -> None:
		from sentence_transformers import SentenceTransformer

		self.model_name = model_name
		self.model = SentenceTransformer(model_name, trust_remote_code=True)
		if max_seq_length:
			self.model.max_seq_length = max_seq_length

	def encode_documents(self, texts: list[str]) -> np.ndarray:
		prefixed = ["search_document: " + t for t in texts]
		vectors = self.model.encode(prefixed, convert_to_numpy=True, normalize_embeddings=True)
		return np.asarray(vectors, dtype="float32")

	def encode_queries(self, texts: list[str]) -> np.ndarray:
		prefixed = ["search_query: " + t for t in texts]
		vectors = self.model.encode(prefixed, convert_to_numpy=True, normalize_embeddings=True)
		return np.asarray(vectors, dtype="float32")


class BM25RetrievalModel:
	"""BM25 (Okapi) lexical retrieval — does not use neural embeddings or FAISS."""

	def __init__(self, corpus_texts: list[str]) -> None:
		import re
		from rank_bm25 import BM25Okapi

		self._re = re.compile(r"\w+", re.UNICODE)
		tokenized = [self._re.findall(t.lower()) for t in corpus_texts]
		self.bm25 = BM25Okapi(tokenized)

	def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
		"""Return (corpus_index, score) pairs sorted by score descending."""
		tokens = self._re.findall(query.lower())
		scores = self.bm25.get_scores(tokens)
		top_indices = np.argsort(scores)[::-1][:top_k]
		return [(int(i), float(scores[i])) for i in top_indices]


def save_bm25_model(model: BM25RetrievalModel, path: Path) -> None:
	import pickle
	with open(path, "wb") as fh:
		pickle.dump(model.bm25, fh)


def load_bm25_model(path: Path, corpus_texts: list[str]) -> BM25RetrievalModel:
	"""Load a pre-built BM25 index; corpus_texts are needed for tokenization state."""
	import pickle
	import re
	obj = BM25RetrievalModel.__new__(BM25RetrievalModel)
	obj._re = re.compile(r"\w+", re.UNICODE)
	with open(path, "rb") as fh:
		obj.bm25 = pickle.load(fh)
	return obj


def load_model_from_artifacts(
	artifacts_dir: Path,
	fallback_model_name: str,
) -> tuple[BaseEmbeddingModel, ModelConfig]:
	config_path = artifacts_dir / "model.json"
	if not config_path.exists():
		logging.info("model.json not found, using SBERT fallback")
		model = SbertEmbeddingModel(fallback_model_name)
		config = ModelConfig(model_type="sbert", model_name=fallback_model_name, params={})
		return model, config

	config = ModelConfig.load(config_path)
	_msl_raw = config.params.get("max_seq_length", 0) if config.params else 0
	_msl = int(_msl_raw) if _msl_raw is not None else 0
	if config.model_type == "sbert":
		model_name = config.model_name or fallback_model_name
		return SbertEmbeddingModel(model_name, max_seq_length=_msl or None), config
	if config.model_type == "e5":
		model_name = config.model_name or "intfloat/multilingual-e5-base"
		return E5EmbeddingModel(model_name, max_seq_length=_msl or None), config
	if config.model_type == "nomic":
		model_name = config.model_name or "nomic-ai/nomic-embed-text-v1.5"
		return NomicEmbeddingModel(model_name, max_seq_length=_msl or None), config
	raise ValueError(f"Unknown model type in artifacts: {config.model_type}")
