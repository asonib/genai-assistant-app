import re
from typing import List, Sequence

from app.config.app_config import AIConfigConstants
from app.services.interfaces.text_chunker_interface import ITextChunker, TextChunk


class RecursiveCharacterTextChunker(ITextChunker):
    """
    Fixed-size character windows with overlap, snapped to natural boundaries.

    For every window of `chunk_size` characters we look backwards (but never
    further than half the window) for the strongest available separator:
    paragraph break -> line break -> sentence end -> whitespace. The next
    window starts `chunk_overlap` characters before the previous end, moved
    forward to the beginning of a word so chunks never start mid-word.
    """

    DEFAULT_SEPARATORS = ("\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ")

    def __init__(self, chunk_size: int = AIConfigConstants.CHUNK_SIZE,
                 chunk_overlap: int = AIConfigConstants.CHUNK_OVERLAP,
                 separators: Sequence[str] = DEFAULT_SEPARATORS):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size // 2:
            raise ValueError("chunk_overlap must be >= 0 and smaller than half of chunk_size")
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._separators = tuple(separators)

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    @property
    def chunk_overlap(self) -> int:
        return self._chunk_overlap

    @staticmethod
    def normalize(text: str) -> str:
        text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t\f\v]+", " ", text)      # collapse horizontal whitespace
        text = re.sub(r" *\n *", "\n", text)          # trim spaces around newlines
        text = re.sub(r"\n{3,}", "\n\n", text)        # at most one blank line
        return text.strip()

    def _find_break(self, text: str, start: int, end: int) -> int:
        min_break = start + self._chunk_size // 2
        for sep in self._separators:
            idx = text.rfind(sep, min_break, end)
            if idx != -1:
                return idx + len(sep)
        return end  # no separator: hard cut

    @staticmethod
    def _snap_to_word_start(text: str, pos: int, limit: int) -> int:
        if pos <= 0 or text[pos - 1].isspace():
            return pos
        nxt = pos
        while nxt < limit and not text[nxt].isspace():
            nxt += 1
        return nxt + 1 if nxt < limit else pos

    def split(self, text: str) -> List[TextChunk]:
        text = self.normalize(text)
        n = len(text)
        chunks: List[TextChunk] = []
        start = 0
        while start < n:
            end = min(start + self._chunk_size, n)
            if end < n:
                end = self._find_break(text, start, end)

            raw = text[start:end]
            stripped = raw.strip()
            if stripped:
                lead = len(raw) - len(raw.lstrip())
                s = start + lead
                chunks.append(TextChunk(index=len(chunks), text=stripped, start_char=s, end_char=s + len(stripped)))

            if end >= n:
                break
            next_start = self._snap_to_word_start(text, end - self._chunk_overlap, end)
            # end - start >= chunk_size/2 > overlap, so this always advances
            start = next_start if next_start > start else end
        return chunks
