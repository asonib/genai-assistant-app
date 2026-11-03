from abc import ABC, abstractmethod


class IDocumentIngestionService(ABC):
    @abstractmethod
    def ingest_document(self, user, document_id) -> dict:
        """
        Chunk the document, embed every chunk and upsert the vectors into the
        user's index. Re-ingesting replaces the previous chunks.
        """
        pass

    @abstractmethod
    def remove_document_vectors(self, user, document_id) -> int:
        pass
