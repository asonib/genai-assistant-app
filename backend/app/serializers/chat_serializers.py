from rest_framework import serializers

from app.config.app_config import AIConfigConstants
from app.models.conversation import ChatMessage, Conversation


class ChatRequestSerializer(serializers.Serializer):
    message = serializers.CharField(max_length=AIConfigConstants.MAX_MESSAGE_CHARS, trim_whitespace=True)
    conversation_id = serializers.IntegerField(required=False, allow_null=True, min_value=1)
    document_ids = serializers.ListField(child=serializers.IntegerField(min_value=1), required=False,
                                         allow_empty=True, max_length=50,
                                         help_text='Restrict retrieval to these documents (default: all).')


class ChatMessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = ChatMessage
        fields = ['id', 'role', 'content', 'citations', 'tool_calls', 'agent_trace', 'document_scope',
                  'model_name', 'prompt_template', 'input_tokens', 'output_tokens', 'latency_ms', 'created_at']
        read_only_fields = fields


class ConversationSerializer(serializers.ModelSerializer):
    message_count = serializers.IntegerField(read_only=True, default=0)
    last_message_at = serializers.DateTimeField(read_only=True, allow_null=True, default=None)

    class Meta:
        model = Conversation
        fields = ['id', 'title', 'created_at', 'updated_at', 'message_count', 'last_message_at']
        read_only_fields = fields


class ConversationDetailSerializer(serializers.ModelSerializer):
    messages = ChatMessageSerializer(many=True, read_only=True)

    class Meta:
        model = Conversation
        fields = ['id', 'title', 'created_at', 'updated_at', 'messages']
        read_only_fields = fields


class ConversationRenameSerializer(serializers.Serializer):
    title = serializers.CharField(max_length=AIConfigConstants.MAX_TITLE_CHARS, min_length=1)


class ChatResponseSerializer(serializers.Serializer):
    conversation = ConversationSerializer()
    user_message = ChatMessageSerializer()
    assistant_message = ChatMessageSerializer()


class ToolSerializer(serializers.Serializer):
    name = serializers.CharField()
    description = serializers.CharField()
    source = serializers.CharField()
    input_schema = serializers.DictField()
