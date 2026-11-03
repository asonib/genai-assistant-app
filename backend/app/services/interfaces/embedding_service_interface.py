from abc import ABC, abstractmethod
from typing import List

import numpy as np


class IEmbeddingService(ABC):
    """Contract for turning text into dense vectors."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        pass

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """Return a float32 array of shape (len(texts), dimension), L2-normalised."""
        pass

    @abstractmethod
    def embed_query(self, text: str) -> np.ndarray:
        """Return a float32 array of shape (dimension,), L2-normalised."""
        pass
