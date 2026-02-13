import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
	return _TOKEN_RE.findall(text.lower())


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
	def __init__(self, model_name: str) -> None:
		from sentence_transformers import SentenceTransformer

		self.model_name = model_name
		self.model = SentenceTransformer(model_name)

	def encode_documents(self, texts: list[str]) -> np.ndarray:
		vectors = self.model.encode(
			texts, convert_to_numpy=True, normalize_embeddings=True
		)
		return np.asarray(vectors, dtype="float32")


class TfidfEmbeddingModel(BaseEmbeddingModel):
	def __init__(self, vectorizer) -> None:
		self.vectorizer = vectorizer

	def encode_documents(self, texts: list[str]) -> np.ndarray:
		matrix = self.vectorizer.transform(texts)
		vectors = matrix.toarray().astype("float32")
		return l2_normalize(vectors)


class GensimEmbeddingModel(BaseEmbeddingModel):
	def __init__(self, model, vector_size: int) -> None:
		self.model = model
		self.vector_size = vector_size

	def _embed_tokens(self, tokens: Iterable[str]) -> np.ndarray:
		vectors = []
		for token in tokens:
			if token in self.model.wv:
				vectors.append(self.model.wv[token])
		if not vectors:
			return np.zeros(self.vector_size, dtype="float32")
		return np.mean(vectors, axis=0).astype("float32")

	def encode_documents(self, texts: list[str]) -> np.ndarray:
		rows = [self._embed_tokens(tokenize(text)) for text in texts]
		vectors = np.vstack(rows).astype("float32")
		return l2_normalize(vectors)


class GloveEmbeddingModel(BaseEmbeddingModel):
	def __init__(self, keyed_vectors) -> None:
		self.keyed_vectors = keyed_vectors
		self.vector_size = keyed_vectors.vector_size

	def _embed_tokens(self, tokens: Iterable[str]) -> np.ndarray:
		vectors = []
		for token in tokens:
			if token in self.keyed_vectors:
				vectors.append(self.keyed_vectors[token])
		if not vectors:
			return np.zeros(self.vector_size, dtype="float32")
		return np.mean(vectors, axis=0).astype("float32")

	def encode_documents(self, texts: list[str]) -> np.ndarray:
		rows = [self._embed_tokens(tokenize(text)) for text in texts]
		vectors = np.vstack(rows).astype("float32")
		return l2_normalize(vectors)


class BertEmbeddingModel(BaseEmbeddingModel):
	def __init__(self, model_name: str, batch_size: int, max_length: int) -> None:
		from transformers import AutoModel, AutoTokenizer

		self.model_name = model_name
		self.batch_size = batch_size
		self.max_length = max_length
		self.tokenizer = AutoTokenizer.from_pretrained(model_name)
		self.model = AutoModel.from_pretrained(model_name)
		self.model.eval()

	def _mean_pooling(self, last_hidden_state, attention_mask):
		import torch

		mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
		sum_embeddings = torch.sum(last_hidden_state * mask, dim=1)
		sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
		return sum_embeddings / sum_mask

	def encode_documents(self, texts: list[str]) -> np.ndarray:
		import torch

		vectors = []
		for start in range(0, len(texts), self.batch_size):
			batch = texts[start : start + self.batch_size]
			encoded = self.tokenizer(
				batch,
				padding=True,
				truncation=True,
				max_length=self.max_length,
				return_tensors="pt",
			)
			with torch.no_grad():
				output = self.model(**encoded)
				pooled = self._mean_pooling(output.last_hidden_state, encoded["attention_mask"])
			vectors.append(pooled.cpu().numpy())
		vectors_np = np.vstack(vectors).astype("float32")
		return l2_normalize(vectors_np)


def build_tfidf_model(texts: list[str], max_features: int | None, ngram_range: tuple[int, int]):
	from sklearn.feature_extraction.text import TfidfVectorizer

	vectorizer = TfidfVectorizer(
		max_features=max_features,
		ngram_range=ngram_range,
		lowercase=True,
	)
	vectorizer.fit(texts)
	return TfidfEmbeddingModel(vectorizer)


def build_gensim_model(
	model_type: str,
	sentences: list[list[str]],
	vector_size: int,
	window: int,
	min_count: int,
	epochs: int,
	workers: int,
):
	if model_type == "word2vec":
		from gensim.models import Word2Vec

		model = Word2Vec(
			sentences=sentences,
			vector_size=vector_size,
			window=window,
			min_count=min_count,
			workers=workers,
			epochs=epochs,
		)
	elif model_type == "fasttext":
		from gensim.models import FastText

		model = FastText(
			sentences=sentences,
			vector_size=vector_size,
			window=window,
			min_count=min_count,
			workers=workers,
			epochs=epochs,
		)
	else:
		raise ValueError(f"Unknown gensim model type: {model_type}")

	return GensimEmbeddingModel(model, vector_size)


def save_tfidf_model(model: TfidfEmbeddingModel, path: Path) -> None:
	from joblib import dump

	dump(model.vectorizer, path)


def load_tfidf_model(path: Path) -> TfidfEmbeddingModel:
	from joblib import load

	vectorizer = load(path)
	return TfidfEmbeddingModel(vectorizer)


def save_gensim_model(model: GensimEmbeddingModel, path: Path) -> None:
	model.model.save(str(path))


def load_gensim_model(model_type: str, path: Path) -> GensimEmbeddingModel:
	if model_type == "word2vec":
		from gensim.models import Word2Vec

		model = Word2Vec.load(str(path))
		vector_size = model.vector_size
		return GensimEmbeddingModel(model, vector_size)
	if model_type == "fasttext":
		from gensim.models import FastText

		model = FastText.load(str(path))
		vector_size = model.vector_size
		return GensimEmbeddingModel(model, vector_size)
	raise ValueError(f"Unknown gensim model type: {model_type}")


def load_glove_model(path: Path, binary: bool, no_header: bool) -> GloveEmbeddingModel:
	from gensim.models import KeyedVectors

	keyed_vectors = KeyedVectors.load_word2vec_format(
		str(path),
		binary=binary,
		no_header=no_header,
	)
	return GloveEmbeddingModel(keyed_vectors)


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
	if config.model_type == "sbert":
		model_name = config.model_name or fallback_model_name
		return SbertEmbeddingModel(model_name), config
	if config.model_type == "tfidf":
		model_path = artifacts_dir / "tfidf.joblib"
		return load_tfidf_model(model_path), config
	if config.model_type in {"word2vec", "fasttext"}:
		model_path = artifacts_dir / "gensim.model"
		return load_gensim_model(config.model_type, model_path), config
	if config.model_type == "glove":
		params = config.params or {}
		model_path = Path(params.get("glove_path", ""))
		if not model_path.exists():
			raise FileNotFoundError("GloVe model file not found. Check model.json params.")
		binary = bool(params.get("glove_binary", False))
		no_header = bool(params.get("glove_no_header", True))
		return load_glove_model(model_path, binary=binary, no_header=no_header), config
	if config.model_type == "bert":
		params = config.params or {}
		model_name = config.model_name or fallback_model_name
		batch_size = int(params.get("batch_size", 32))
		max_length = int(params.get("max_length", 256))
		return BertEmbeddingModel(model_name, batch_size=batch_size, max_length=max_length), config

	raise ValueError(f"Unknown model type in artifacts: {config.model_type}")
