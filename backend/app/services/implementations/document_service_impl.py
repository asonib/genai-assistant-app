import hashlib
import logging
import os

from django.db import transaction
from rest_framework.exceptions import APIException, ValidationError

from app.config.app_config import AppConfigConstants
from app.exceptions.custom_exceptions import (
    EmptyDocumentException,
    PayloadTooLargeException,
    ResourceNotFoundException,
    UnsupportedFileTypeException,
)
from app.models.document import Document
from app.services.interfaces.document_service_interface import IDocumentService
from app.services.interfaces.ingestion_service_interface import IDocumentIngestionService
from app.services.interfaces.text_extractor_interface import ITextExtractor

logger = logging.getLogger(__name__)


class DocumentServiceImpl(IDocumentService):
    """
    Upload pipeline: validate -> extract text (PDF/TXT) -> persist Document ->
    ingest (chunk + embed + FAISS). Every file in a multi-file upload is
    validated up front so a bad file rejects the batch before any work.
    """

    def __init__(self, text_extractor: ITextExtractor, ingestion_service: IDocumentIngestionService):
        self.text_extractor = text_extractor
        self.ingestion_service = ingestion_service

    # ---- validation ---------------------------------------------------------
    def _validate(self, file_obj):
        name = os.path.basename(file_obj.name or '')
        _, ext = os.path.splitext(name)
        if ext.lower() not in AppConfigConstants.ALLOWED_EXTENSIONS or not self.text_extractor.supports(ext):
            raise UnsupportedFileTypeException(
                f"'{name}': unsupported file type. Allowed: {', '.join(AppConfigConstants.ALLOWED_EXTENSIONS)}")
        if file_obj.size > AppConfigConstants.MAX_FILE_SIZE_BYTES:
            raise PayloadTooLargeException(
                f"'{name}' is {file_obj.size} bytes; the limit is {AppConfigConstants.MAX_FILE_SIZE_BYTES} bytes.")
        if file_obj.size == 0:
            raise EmptyDocumentException(f"'{name}' is empty.")
        return name

    # ---- IDocumentService -----------------------------------------------------
    def upload_documents(self, user, files, title=None) -> list:
        if not files:
            raise ValidationError({'files': ['Attach at least one PDF or TXT file.']})
        if len(files) > AppConfigConstants.MAX_FILES_PER_UPLOAD:
            raise ValidationError({'files': [f'At most {AppConfigConstants.MAX_FILES_PER_UPLOAD} files per upload.']})
        names = [self._validate(f) for f in files]

        results = []
        for file_obj, name in zip(files, names):
            extracted = self.text_extractor.extract(name, file_obj.read())
            text = extracted.text.strip()
            if not text:
                raise EmptyDocumentException(
                    f"'{name}' has no extractable text (scanned PDFs without a text layer are not supported).")
            if len(text) > AppConfigConstants.MAX_DOCUMENT_CHARS:
                raise PayloadTooLargeException(f"'{name}' contains too much text to index.")

            content_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
            duplicate = Document.objects.filter(owner=user, content_hash=content_hash).first()
            if duplicate is not None:
                results.append({'document': duplicate, 'duplicate': True, 'ingestion': None, 'error': None})
                continue

            doc_title = (title if (title and len(files) == 1) else os.path.splitext(name)[0])[:255]
            with transaction.atomic():
                document = Document.objects.create(
                    owner=user, title=doc_title, original_filename=name[:255], file_type=extracted.file_type,
                    file_size=file_obj.size, page_count=extracted.page_count, content=text,
                    content_hash=content_hash,
                )
            ingestion, error = None, None
            try:
                ingestion = self.ingestion_service.ingest_document(user, document.pk)
            except APIException as exc:
                # The document is kept (status=failed) so the user can retry via /reingest.
                logger.warning("Ingestion failed for document %s: %s", document.pk, exc.detail)
                error = str(exc.detail)
            document.refresh_from_db()
            results.append({'document': document, 'duplicate': False, 'ingestion': ingestion, 'error': error})
        return results

    def list_documents(self, user):
        return Document.objects.filter(owner=user).defer('content')

    def get_document(self, user, document_id):
        try:
            # 404 (not 403) for other users' documents so ids cannot be probed.
            return Document.objects.get(id=document_id, owner=user)
        except Document.DoesNotExist:
            raise ResourceNotFoundException('Document not found.')

    def delete_document(self, user, document_id) -> None:
        document = self.get_document(user, document_id)
        try:
            self.ingestion_service.remove_document_vectors(user, document.pk)
        except APIException:
            # Stale vectors are harmless: retrieval drops hits whose chunk row is gone.
            logger.warning('Could not remove vectors for document %s', document.pk, exc_info=True)
        document.delete()

    def reingest_document(self, user, document_id) -> dict:
        self.get_document(user, document_id)
        return self.ingestion_service.ingest_document(user, document_id)
