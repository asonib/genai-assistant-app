import logging
import time

from django.db import transaction
from rest_framework.exceptions import ValidationError

from app.models.conversation import ChatMessage
from app.models.document import Document
from app.services.interfaces.agent_interface import AgentInput, IAgentRunner
from app.services.interfaces.chat_service_interface import IChatService
from app.services.interfaces.conversation_service_interface import IConversationService
from app.services.interfaces.external_tool_client_interface import IExternalToolClient

logger = logging.getLogger(__name__)


class ChatServiceImpl(IChatService):
    """
    One chat turn:
      1. resolve / create the conversation and validate the document scope,
      2. load memory (previous turns) from the DB,
      3. run the LangGraph agent,
      4. persist the user + assistant messages atomically (history).
    Messages are only persisted when the agent succeeds, so memory always
    contains complete user/assistant pairs.
    """

    def __init__(self, agent_runner: IAgentRunner, conversation_service: IConversationService,
                 tool_client: IExternalToolClient):
        self.agent_runner = agent_runner
        self.conversation_service = conversation_service
        self.tool_client = tool_client

    def _validate_scope(self, user, document_ids):
        ids = sorted({int(d) for d in (document_ids or [])})
        if not ids:
            return []
        owned = set(Document.objects.filter(owner=user, id__in=ids).values_list('id', flat=True))
        missing = [d for d in ids if d not in owned]
        if missing:
            raise ValidationError({'document_ids': [f'Unknown document id(s): {missing}']})
        return ids

    def send_message(self, user, message, conversation_id=None, document_ids=None):
        message = message.strip()
        scope = self._validate_scope(user, document_ids)
        if conversation_id:
            conversation = self.conversation_service.get_conversation(user, conversation_id)
            history = self.conversation_service.get_memory(conversation)
        else:
            conversation, history = None, []

        started = time.perf_counter()
        output = self.agent_runner.run(AgentInput(
            user_id=user.pk, question=message, history=history, document_ids=scope))
        latency_ms = int((time.perf_counter() - started) * 1000)

        with transaction.atomic():
            if conversation is None:
                conversation = self.conversation_service.create_conversation(user, message)
            user_msg = ChatMessage.objects.create(
                conversation=conversation, role=ChatMessage.ROLE_USER, content=message, document_scope=scope)
            assistant_msg = ChatMessage.objects.create(
                conversation=conversation, role=ChatMessage.ROLE_ASSISTANT, content=output.answer,
                citations=output.citations, tool_calls=output.tool_calls, agent_trace=output.trace,
                document_scope=scope, model_name=output.model_name, prompt_template=output.prompt_template,
                input_tokens=output.input_tokens, output_tokens=output.output_tokens, latency_ms=latency_ms,
            )
            conversation.save(update_fields=['updated_at'])
        # Same shape as the annotated list queryset (ConversationSerializer).
        conversation.message_count = conversation.messages.count()
        conversation.last_message_at = assistant_msg.created_at

        logger.info("chat_turn user=%s conversation=%s latency_ms=%s", user.pk, conversation.pk, latency_ms)
        return conversation, user_msg, assistant_msg

    def list_available_tools(self) -> list:
        from app.agents.graph import SEARCH_TOOL_SPEC

        tools = [{'name': SEARCH_TOOL_SPEC['name'], 'description': SEARCH_TOOL_SPEC['description'],
                  'source': 'builtin', 'input_schema': SEARCH_TOOL_SPEC['input_schema']}]
        tools += [{'name': t.name, 'description': t.description, 'source': 'mcp', 'input_schema': t.input_schema}
                  for t in self.tool_client.list_tools()]
        return tools
