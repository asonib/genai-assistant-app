from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ExtractedText:
    text: str
    file_type: str            # "pdf" | "txt"
    page_count: Optional[int] = None


class ITextExtractor(ABC):
    """Turns an uploaded file's bytes into plain text."""

    @abstractmethod
    def supports(self, extension: str) -> bool:
        pass

    @abstractmethod
    def extract(self, filename: str, data: bytes) -> ExtractedText:
        pass
