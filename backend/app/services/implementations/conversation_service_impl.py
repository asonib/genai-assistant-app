from django.db.models import Count, Max

from app.config.app_config import AIConfigConstants
from app.exceptions.custom_exceptions import ResourceNotFoundException
from app.models.conversation import ChatMessage, Conversation
from app.services.interfaces.conversation_service_interface import IConversationService


def make_title(text: str) -> str:
    text = " ".join((text or "").split())
    limit = 60
    return text if len(text) <= limit else text[:limit - 1].rstrip() + "…"


class ConversationServiceImpl(IConversationService):
    def list_conversations(self, user):
        return (Conversation.objects.filter(user=user)
                .annotate(message_count=Count('messages'), last_message_at=Max('messages__created_at'))
                .order_by('-updated_at', '-id'))  # Meta.ordering is dropped for GROUP BY queries

    def get_conversation(self, user, conversation_id):
        try:
            return Conversation.objects.get(id=conversation_id, user=user)
        except Conversation.DoesNotExist:
            raise ResourceNotFoundException('Conversation not found.')

    def create_conversation(self, user, title):
        return Conversation.objects.create(user=user, title=make_title(title) or 'New chat')

    def rename_conversation(self, user, conversation_id, title):
        conversation = self.get_conversation(user, conversation_id)
        conversation.title = title.strip()[:AIConfigConstants.MAX_TITLE_CHARS]
        conversation.save(update_fields=['title', 'updated_at'])
        return conversation

    def delete_conversation(self, user, conversation_id) -> None:
        self.get_conversation(user, conversation_id).delete()

    def get_memory(self, conversation) -> list:
        """
        Last MEMORY_MAX_TURNS user/assistant pairs, oldest first, trimmed from the
        oldest side until the replayed text fits in MEMORY_MAX_CHARS.
        """
        limit = AIConfigConstants.MEMORY_MAX_TURNS * 2
        recent = list(conversation.messages.order_by('-created_at', '-id').values('role', 'content')[:limit])
        recent.reverse()
        while recent and sum(len(m['content']) for m in recent) > AIConfigConstants.MEMORY_MAX_CHARS:
            recent.pop(0)
        while recent and recent[0]['role'] != ChatMessage.ROLE_USER:
            recent.pop(0)
        return [{'role': m['role'], 'content': m['content']} for m in recent]
