from django.contrib import admin

from app.models import ChatMessage, Conversation, Document, DocumentChunk, UserSession


@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    list_display = ('id', 'title', 'owner', 'file_type', 'ingestion_status', 'chunk_count', 'uploaded_at')
    list_filter = ('file_type', 'ingestion_status')
    search_fields = ('title', 'original_filename')
    exclude = ('content',)


@admin.register(Conversation)
class ConversationAdmin(admin.ModelAdmin):
    list_display = ('id', 'title', 'user', 'updated_at')


@admin.register(ChatMessage)
class ChatMessageAdmin(admin.ModelAdmin):
    list_display = ('id', 'conversation', 'role', 'model_name', 'latency_ms', 'created_at')
    list_filter = ('role',)


admin.site.register(DocumentChunk)
admin.site.register(UserSession, list_display=('id', 'user', 'created_at', 'expires_at'))
