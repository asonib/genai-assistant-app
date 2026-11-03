from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import List, Optional, Sequence


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: int
    document_id: int
    document_title: str
    chunk_index: int
    chunk_text: str
    score: float

    @property
    def citation_tag(self) -> str:
        return f"[{self.document_id}:{self.chunk_index}]"

    def to_dict(self) -> dict:
        return asdict(self)


class IRetrievalService(ABC):
    """Semantic search over all of a user's ingested documents (multi-doc index)."""

    @abstractmethod
    def search(self, user_id: int, query: str, top_k: int,
               document_ids: Optional[Sequence[int]] = None) -> List[RetrievedChunk]:
        pass
