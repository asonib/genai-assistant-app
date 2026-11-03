from django.contrib.auth.models import User
from django.db import models


class Conversation(models.Model):
    """A chat thread. Its messages are the agent's long-term conversational memory."""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='conversations')
    title = models.CharField(max_length=120)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-updated_at', '-id']
        indexes = [models.Index(fields=['user', '-updated_at'], name='conv_user_updated_idx')]

    def __str__(self):
        return f"Conversation({self.pk}, {self.title!r})"


class ChatMessage(models.Model):
    ROLE_USER = 'user'
    ROLE_ASSISTANT = 'assistant'
    ROLE_CHOICES = [(ROLE_USER, 'User'), (ROLE_ASSISTANT, 'Assistant')]

    conversation = models.ForeignKey(Conversation, on_delete=models.CASCADE, related_name='messages')
    role = models.CharField(max_length=16, choices=ROLE_CHOICES)
    content = models.TextField()

    # ---- assistant-only metadata ----
    # [{"document_id", "document_title", "chunk_id", "chunk_index", "chunk_text", "score", "cited"}]
    citations = models.JSONField(default=list, blank=True)
    # [{"name", "input", "output", "is_error", "source": "mcp"|"builtin", "latency_ms"}]
    tool_calls = models.JSONField(default=list, blank=True)
    # Ordered list of LangGraph node executions, for observability in the UI.
    agent_trace = models.JSONField(default=list, blank=True)
    # Document scope the user selected for this turn (empty = all documents).
    document_scope = models.JSONField(default=list, blank=True)
    model_name = models.CharField(max_length=100, blank=True, default='')
    prompt_template = models.CharField(max_length=100, blank=True, default='')
    input_tokens = models.PositiveIntegerField(null=True, blank=True)
    output_tokens = models.PositiveIntegerField(null=True, blank=True)
    latency_ms = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['created_at', 'id']
        indexes = [models.Index(fields=['conversation', 'created_at'], name='msg_conv_created_idx')]

    def __str__(self):
        return f"ChatMessage({self.role}, conv={self.conversation_id})"
