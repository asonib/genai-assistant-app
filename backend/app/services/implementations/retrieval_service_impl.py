import logging
from typing import List, Optional, Sequence

from app.config.app_config import AIConfigConstants
from app.models.document_chunk import DocumentChunk
from app.services.implementations.ingestion_service_impl import user_namespace
from app.services.interfaces.embedding_service_interface import IEmbeddingService
from app.services.interfaces.retrieval_service_interface import IRetrievalService, RetrievedChunk
from app.services.interfaces.vector_store_interface import IVectorStore

logger = logging.getLogger(__name__)


class VectorRetrievalService(IRetrievalService):
    """
    Dense retrieval over the user's FAISS index, which holds the chunks of *all*
    their documents (multi-document indexing). Hits are mapped back to
    DocumentChunk rows through the shared id (vector id == chunk pk); hits whose
    row no longer exists (stale vectors) are dropped.
    """

    def __init__(self, embedding_service: IEmbeddingService, vector_store: IVectorStore,
                 min_similarity: float = AIConfigConstants.RAG_MIN_SIMILARITY):
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.min_similarity = min_similarity

    def search(self, user_id: int, query: str, top_k: int,
               document_ids: Optional[Sequence[int]] = None) -> List[RetrievedChunk]:
        query = (query or "").strip()
        if not query or top_k <= 0:
            return []
        namespace = user_namespace(user_id)
        if self.vector_store.count(namespace) == 0:
            return []

        scope = {int(d) for d in document_ids} if document_ids else None
        fetch_k = top_k * AIConfigConstants.RAG_FILTER_OVERSAMPLE if scope else top_k
        hits = self.vector_store.search(namespace, self.embedding_service.embed_query(query), fetch_k)
        hits = [(cid, score) for cid, score in hits if score >= self.min_similarity]
        if not hits:
            return []

        qs = DocumentChunk.objects.select_related("document").filter(
            id__in=[cid for cid, _ in hits], document__owner_id=user_id)
        if scope:
            qs = qs.filter(document_id__in=scope)
        rows = {c.id: c for c in qs}

        results = []
        for cid, score in hits:
            row = rows.get(cid)
            if row is None:
                continue
            results.append(RetrievedChunk(
                chunk_id=row.id, document_id=row.document_id, document_title=row.document.title,
                chunk_index=row.chunk_index, chunk_text=row.chunk_text, score=round(float(score), 4),
            ))
            if len(results) >= top_k:
                break
        logger.debug("Retrieved %d chunks for user=%s", len(results), user_id)
        return results
