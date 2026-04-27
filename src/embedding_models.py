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


class OpenAIEmbeddingModel(BaseEmbeddingModel):
	"""OpenAI text-embedding API (text-embedding-3-large etc.)."""

	def __init__(self, model_name: str, api_key: str | None = None) -> None:
		import os
		from openai import OpenAI

		self.model_name = model_name
		key = api_key or os.environ.get("OPENAI_API_KEY")
		if not key:
			raise ValueError("OPENAI_API_KEY env variable is not set")
		self.client = OpenAI(api_key=key)

	def _embed_batch(self, texts: list[str]) -> np.ndarray:
		response = self.client.embeddings.create(model=self.model_name, input=texts)
		vecs = [item.embedding for item in response.data]
		return l2_normalize(np.array(vecs, dtype="float32"))

	def encode_documents(self, texts: list[str]) -> np.ndarray:
		batch_size = 512
		parts = []
		for i in range(0, len(texts), batch_size):
			parts.append(self._embed_batch(texts[i : i + batch_size]))
		return np.vstack(parts).astype("float32")


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
	if config.model_type == "openai":
		params = config.params or {}
		model_name = config.model_name or "text-embedding-3-large"
		return OpenAIEmbeddingModel(model_name, api_key=params.get("api_key")), config

	raise ValueError(f"Unknown model type in artifacts: {config.model_type}")
