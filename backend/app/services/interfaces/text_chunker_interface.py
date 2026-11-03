from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class TextChunk:
    index: int
    text: str
    start_char: int
    end_char: int


class ITextChunker(ABC):
    @property
    @abstractmethod
    def chunk_size(self) -> int:
        pass

    @property
    @abstractmethod
    def chunk_overlap(self) -> int:
        pass

    @abstractmethod
    def split(self, text: str) -> List[TextChunk]:
        pass
