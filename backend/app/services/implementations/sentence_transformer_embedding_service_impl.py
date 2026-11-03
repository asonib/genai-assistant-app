import logging
import threading
from typing import List

import numpy as np

from app.config.app_config import AIConfigConstants
from app.exceptions.custom_exceptions import EmbeddingServiceException
from app.services.interfaces.embedding_service_interface import IEmbeddingService

logger = logging.getLogger(__name__)


class SentenceTransformerEmbeddingService(IEmbeddingService):
    """
    Local embeddings via sentence-transformers (default: all-MiniLM-L6-v2, 384-d).
    The model is loaded lazily on first use and shared process-wide.
    Vectors are L2-normalised so inner product == cosine similarity in FAISS.
    """

    _models = {}
    _lock = threading.Lock()

    def __init__(self, model_name: str = AIConfigConstants.EMBEDDING_MODEL_NAME,
                 dimension: int = AIConfigConstants.EMBEDDING_DIMENSION,
                 batch_size: int = AIConfigConstants.EMBEDDING_BATCH_SIZE,
                 cache_folder: str = None):
        self._model_name = model_name
        self._dimension = dimension
        self._batch_size = batch_size
        self._cache_folder = cache_folder

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def _get_model(self):
        model = self._models.get(self._model_name)
        if model is not None:
            return model
        with self._lock:
            model = self._models.get(self._model_name)
            if model is None:
                try:
                    from sentence_transformers import SentenceTransformer
                    logger.info("Loading embedding model %s", self._model_name)
                    model = SentenceTransformer(self._model_name, cache_folder=self._cache_folder)
                except Exception as exc:  # ImportError, download/network errors, ...
                    logger.exception("Could not load embedding model")
                    raise EmbeddingServiceException(f"Could not load embedding model '{self._model_name}': {exc}")
                actual_dim = model.get_sentence_embedding_dimension()
                if actual_dim != self._dimension:
                    raise EmbeddingServiceException(
                        f"Embedding dimension mismatch: model produces {actual_dim}, config expects {self._dimension}."
                    )
                self._models[self._model_name] = model
        return model

    def _encode(self, texts: List[str]) -> np.ndarray:
        model = self._get_model()
        try:
            vectors = model.encode(
                texts,
                batch_size=self._batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except Exception as exc:
            logger.exception("Embedding failed")
            raise EmbeddingServiceException(f"Embedding failed: {exc}")
        return np.asarray(vectors, dtype=np.float32)

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dimension), dtype=np.float32)
        return self._encode(list(texts))

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([text])[0]
