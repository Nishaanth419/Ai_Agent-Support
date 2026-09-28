"""
Text-embedding abstraction for hybrid ticket search.

Mirrors the src/llm/ pattern (an ABC plus a real provider implementation
plus a deterministic fake for tests): search_tickets and the reindexing
pipeline depend only on the Embedder interface, so tests never need a
model download or network access.
"""

import hashlib
import math
from abc import ABC, abstractmethod


class Embedder(ABC):
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Length of the vectors this embedder produces."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts, returning one vector per input text."""


class SentenceTransformerEmbedder(Embedder):
    """Real embedder backed by a local sentence-transformers model.

    The model is loaded lazily on first use (not in __init__) so that
    importing this module, or constructing this class, never pays the
    model-load cost -- useful for code paths that may end up only ever
    using FakeEmbedder (e.g. the whole test suite).
    """

    def __init__(self, model_name: str):
        self._model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
        return self._model

    @property
    def dimension(self) -> int:
        return self._get_model().get_embedding_dimension()

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self._get_model().encode(texts, convert_to_numpy=True)
        return [vector.tolist() for vector in vectors]


class FakeEmbedder(Embedder):
    """Deterministic, offline embedder for tests.

    Hashes each word into a fixed-size bag-of-words vector and L2-normalizes
    it, so cosine similarity behaves sensibly for tests that check relative
    ranking (shared words -> higher similarity) without downloading any
    model or making any network call.

    Uses hashlib.md5 rather than the builtin hash(): Python salts str
    hashing per-process via PYTHONHASHSEED, which would make embeddings
    (and therefore test assertions on similarity/ranking) nondeterministic
    across runs and CI machines.
    """

    def __init__(self, dimension: int = 32):
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        # Split on whitespace rather than \w+ so a punctuation-only token
        # (e.g. "%") is still its own token when mixed with real words,
        # instead of silently vanishing from the embedding -- a real
        # embedder never drops punctuation like that.
        words = text.lower().split() or [text.lower()]
        vector = [0.0] * self._dimension
        for word in words:
            index = int(hashlib.md5(word.encode()).hexdigest(), 16) % self._dimension
            vector[index] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]
