import io
import logging

from app.exceptions.custom_exceptions import TextExtractionException
from app.services.interfaces.text_extractor_interface import ExtractedText, ITextExtractor

logger = logging.getLogger(__name__)


class PdfTxtTextExtractor(ITextExtractor):
    """
    .txt -> decoded as UTF-8 (BOM-aware), falling back to cp1252/latin-1.
    .pdf -> text layer extracted page by page with `pypdf`. Scanned (image-only)
            PDFs have no text layer and are rejected with a clear 422.
    """

    _SUPPORTED = {".pdf", ".txt"}

    def supports(self, extension: str) -> bool:
        return extension.lower() in self._SUPPORTED

    def extract(self, filename: str, data: bytes) -> ExtractedText:
        lower = filename.lower()
        if lower.endswith(".txt"):
            return ExtractedText(text=self._decode_text(data), file_type="txt")
        if lower.endswith(".pdf"):
            return self._extract_pdf(filename, data)
        raise TextExtractionException(f"No extractor for '{filename}'.")

    @staticmethod
    def _decode_text(data: bytes) -> str:
        for encoding in ("utf-8-sig", "cp1252"):
            try:
                return data.decode(encoding)
            except UnicodeDecodeError:
                continue
        return data.decode("latin-1")

    @staticmethod
    def _extract_pdf(filename: str, data: bytes) -> ExtractedText:
        if not data.startswith(b"%PDF"):
            raise TextExtractionException(f"'{filename}' is not a valid PDF file.")
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise TextExtractionException(f"pypdf is not installed: {exc}")
        try:
            reader = PdfReader(io.BytesIO(data))
            if reader.is_encrypted:
                try:
                    reader.decrypt("")
                except Exception:
                    raise TextExtractionException(f"'{filename}' is password protected.")
            pages = []
            for number, page in enumerate(reader.pages, start=1):
                page_text = (page.extract_text() or "").strip()
                if page_text:
                    # Page markers survive chunking and help users locate sources.
                    pages.append(f"[Page {number}]\n{page_text}")
        except TextExtractionException:
            raise
        except Exception as exc:
            logger.warning("PDF extraction failed for %s: %s", filename, exc)
            raise TextExtractionException(f"Could not read PDF '{filename}': {exc}")
        return ExtractedText(text="\n\n".join(pages), file_type="pdf", page_count=len(reader.pages))
