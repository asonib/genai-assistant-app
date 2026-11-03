import logging
import os
import re
import threading
from typing import Dict, List, Sequence, Tuple

import numpy as np

from app.config.app_config import AIConfigConstants
from app.exceptions.custom_exceptions import VectorStoreException
from app.services.interfaces.vector_store_interface import IVectorStore

logger = logging.getLogger(__name__)

_NAMESPACE_RE = re.compile(r"^[A-Za-z0-9_-]+$")


class FaissVectorStore(IVectorStore):
    """
    FAISS-backed vector store.

    * Index type: IndexIDMap2(IndexFlatIP(dim)) — exact search, inner product on
      L2-normalised vectors (= cosine similarity), with our own int64 ids
      (DocumentChunk primary keys) so hits map straight back to DB rows and
      re-ingestion can remove stale vectors with remove_ids().
    * One index per namespace (one namespace per user), persisted to
      <VECTOR_STORE_DIR>/<namespace>.faiss after every write.
    * Indexes are cached in memory; a process-wide lock serialises access.
    """

    _cache: Dict[str, object] = {}
    _lock = threading.RLock()

    def __init__(self, base_dir: str, dimension: int = AIConfigConstants.EMBEDDING_DIMENSION):
        self._base_dir = str(base_dir)
        self._dimension = dimension

    @property
    def backend_name(self) -> str:
        return AIConfigConstants.VECTOR_STORE_BACKEND

    # ---- internals -------------------------------------------------------
    @staticmethod
    def _faiss():
        try:
            import faiss  # noqa: WPS433 (lazy import keeps Django start-up light)
        except ImportError as exc:
            raise VectorStoreException(f"faiss is not installed: {exc}")
        return faiss

    def _path(self, namespace: str) -> str:
        if not _NAMESPACE_RE.match(namespace):
            raise VectorStoreException(f"Invalid vector namespace '{namespace}'.")
        return os.path.join(self._base_dir, f"{namespace}.faiss")

    def _cache_key(self, namespace: str) -> str:
        return f"{self._base_dir}::{namespace}"

    def _load(self, namespace: str, create: bool):
        key = self._cache_key(namespace)
        index = self._cache.get(key)
        if index is not None:
            return index
        faiss = self._faiss()
        path = self._path(namespace)
        try:
            if os.path.exists(path):
                index = faiss.read_index(path)
                if index.d != self._dimension:
                    raise VectorStoreException(
                        f"Index '{namespace}' has dimension {index.d}, expected {self._dimension}. "
                        f"Delete {path} and re-ingest."
                    )
            elif create:
                index = faiss.IndexIDMap2(faiss.IndexFlatIP(self._dimension))
            else:
                return None
        except VectorStoreException:
            raise
        except Exception as exc:
            logger.exception("Failed to load FAISS index")
            raise VectorStoreException(f"Failed to load vector index: {exc}")
        self._cache[key] = index
        return index

    def _persist(self, namespace: str, index) -> None:
        faiss = self._faiss()
        os.makedirs(self._base_dir, exist_ok=True)
        path = self._path(namespace)
        tmp = path + ".tmp"
        try:
            faiss.write_index(index, tmp)
            os.replace(tmp, path)  # atomic swap so a crash never leaves a half-written index
        except Exception as exc:
            logger.exception("Failed to persist FAISS index")
            raise VectorStoreException(f"Failed to persist vector index: {exc}")

    # ---- IVectorStore ----------------------------------------------------
    def add(self, namespace: str, ids: Sequence[int], vectors: np.ndarray) -> None:
        if len(ids) == 0:
            return
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape != (len(ids), self._dimension):
            raise VectorStoreException(
                f"Expected vectors of shape ({len(ids)}, {self._dimension}), got {vectors.shape}."
            )
        id_array = np.asarray(ids, dtype=np.int64)
        with self._lock:
            index = self._load(namespace, create=True)
            try:
                index.add_with_ids(vectors, id_array)
            except Exception as exc:
                raise VectorStoreException(f"Failed to add vectors: {exc}")
            self._persist(namespace, index)

    def remove(self, namespace: str, ids: Sequence[int]) -> int:
        if len(ids) == 0:
            return 0
        with self._lock:
            index = self._load(namespace, create=False)
            if index is None:
                return 0
            try:
                removed = int(index.remove_ids(np.asarray(ids, dtype=np.int64)))
            except Exception as exc:
                raise VectorStoreException(f"Failed to remove vectors: {exc}")
            if removed:
                self._persist(namespace, index)
            return removed

    def search(self, namespace: str, query_vector: np.ndarray, top_k: int) -> List[Tuple[int, float]]:
        with self._lock:
            index = self._load(namespace, create=False)
            if index is None or index.ntotal == 0:
                return []
            query = np.ascontiguousarray(query_vector, dtype=np.float32).reshape(1, -1)
            if query.shape[1] != self._dimension:
                raise VectorStoreException(
                    f"Query vector has dimension {query.shape[1]}, expected {self._dimension}."
                )
            k = min(top_k, index.ntotal)
            try:
                scores, ids = index.search(query, k)
            except Exception as exc:
                raise VectorStoreException(f"Vector search failed: {exc}")
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]

    def count(self, namespace: str) -> int:
        with self._lock:
            index = self._load(namespace, create=False)
            return 0 if index is None else int(index.ntotal)
