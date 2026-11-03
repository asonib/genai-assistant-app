from rest_framework import serializers

from app.models.document import Document

_FIELDS = ['id', 'title', 'original_filename', 'file_type', 'file_size', 'page_count',
           'ingestion_status', 'ingestion_error', 'chunk_count', 'ingested_at', 'uploaded_at']


class DocumentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Document
        fields = _FIELDS
        read_only_fields = _FIELDS


class DocumentDetailSerializer(DocumentSerializer):
    content_preview = serializers.SerializerMethodField()

    class Meta(DocumentSerializer.Meta):
        fields = _FIELDS + ['content_preview']
        read_only_fields = fields

    def get_content_preview(self, obj) -> str:
        return obj.content[:2000]


class DocumentUploadRequestSerializer(serializers.Serializer):
    files = serializers.ListField(child=serializers.FileField(), help_text='One or more .pdf/.txt files.')
    title = serializers.CharField(required=False, max_length=255, help_text='Only used for single-file uploads.')


class DocumentUploadResultSerializer(serializers.Serializer):
    document = DocumentSerializer()
    duplicate = serializers.BooleanField()
    ingestion = serializers.DictField(allow_null=True)
    error = serializers.CharField(allow_null=True)
