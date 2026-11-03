from abc import ABC, abstractmethod
from typing import List, Sequence, Tuple

import numpy as np


class IVectorStore(ABC):
    """
    Contract for a vector index.

    Vectors are partitioned by `namespace` (we use one namespace per user) so a
    search can never return another tenant's chunks.
    """

    @property
    @abstractmethod
    def backend_name(self) -> str:
        pass

    @abstractmethod
    def add(self, namespace: str, ids: Sequence[int], vectors: np.ndarray) -> None:
        pass

    @abstractmethod
    def remove(self, namespace: str, ids: Sequence[int]) -> int:
        """Remove vectors by id. Returns the number of vectors removed."""
        pass

    @abstractmethod
    def search(self, namespace: str, query_vector: np.ndarray, top_k: int) -> List[Tuple[int, float]]:
        """Return up to top_k (id, similarity_score) pairs, best first."""
        pass

    @abstractmethod
    def count(self, namespace: str) -> int:
        pass
