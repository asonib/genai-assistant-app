from django.contrib.auth.models import User
from django.db import models


class Document(models.Model):
    STATUS_PENDING = 'pending'
    STATUS_INGESTED = 'ingested'
    STATUS_FAILED = 'failed'
    INGESTION_STATUS_CHOICES = [
        (STATUS_PENDING, 'Pending'),
        (STATUS_INGESTED, 'Ingested'),
        (STATUS_FAILED, 'Failed'),
    ]

    TYPE_PDF = 'pdf'
    TYPE_TXT = 'txt'
    FILE_TYPE_CHOICES = [(TYPE_PDF, 'PDF'), (TYPE_TXT, 'Plain text')]

    owner = models.ForeignKey(User, on_delete=models.CASCADE, related_name='documents')
    title = models.CharField(max_length=255)
    original_filename = models.CharField(max_length=255)
    file_type = models.CharField(max_length=8, choices=FILE_TYPE_CHOICES)
    file_size = models.PositiveIntegerField(default=0)
    page_count = models.PositiveIntegerField(null=True, blank=True)
    content = models.TextField()
    # SHA-256 of the extracted text; used to reject duplicate uploads per user.
    content_hash = models.CharField(max_length=64, db_index=True)

    ingestion_status = models.CharField(max_length=20, choices=INGESTION_STATUS_CHOICES, default=STATUS_PENDING)
    ingestion_error = models.CharField(max_length=500, blank=True, default='')
    chunk_count = models.PositiveIntegerField(default=0)
    ingested_at = models.DateTimeField(null=True, blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-uploaded_at', '-id']
        indexes = [models.Index(fields=['owner', '-uploaded_at'], name='doc_owner_uploaded_idx')]

    def __str__(self):
        return self.title
