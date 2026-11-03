from django.db import models
from app.models.document import Document


class DocumentChunk(models.Model):
    """
    A single text chunk of a Document.

    The chunk's primary key is used as the vector id inside the FAISS index,
    so a search hit maps straight back to this row (and its document_id).
    """
    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name='chunks')
    chunk_index = models.PositiveIntegerField()
    chunk_text = models.TextField()
    start_char = models.PositiveIntegerField(default=0)
    end_char = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['document_id', 'chunk_index']
        constraints = [
            models.UniqueConstraint(fields=['document', 'chunk_index'], name='uniq_document_chunk_index'),
        ]

    def __str__(self):
        return f"Chunk({self.document_id}#{self.chunk_index})"
