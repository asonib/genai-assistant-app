from abc import ABC, abstractmethod


class IDocumentService(ABC):
    @abstractmethod
    def upload_documents(self, user, files, title=None) -> list:
        """Validate, extract, persist and ingest uploaded files. Returns per-file results."""
        pass

    @abstractmethod
    def list_documents(self, user):
        pass

    @abstractmethod
    def get_document(self, user, document_id):
        pass

    @abstractmethod
    def delete_document(self, user, document_id) -> None:
        pass

    @abstractmethod
    def reingest_document(self, user, document_id) -> dict:
        pass
