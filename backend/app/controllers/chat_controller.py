from drf_spectacular.utils import OpenApiParameter, extend_schema
from rest_framework import status
from rest_framework.parsers import JSONParser
from rest_framework.response import Response
from rest_framework.views import APIView

from app.controllers.pagination import StandardResultsPagination
from app.serializers.chat_serializers import (
    ChatMessageSerializer,
    ChatRequestSerializer,
    ChatResponseSerializer,
    ConversationDetailSerializer,
    ConversationRenameSerializer,
    ConversationSerializer,
    ToolSerializer,
)
from app.services import container


class ChatController(APIView):
    """POST /api/chat — one turn through the LangGraph agent (RAG + MCP tools + memory)."""
    parser_classes = [JSONParser]

    @extend_schema(tags=['Chat'], request=ChatRequestSerializer, responses={200: ChatResponseSerializer})
    def post(self, request):
        serializer = ChatRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data
        conversation, user_msg, assistant_msg = container.build_chat_service().send_message(
            user=request.user, message=data['message'], conversation_id=data.get('conversation_id'),
            document_ids=data.get('document_ids'))
        return Response({
            'conversation': ConversationSerializer(conversation).data,
            'user_message': ChatMessageSerializer(user_msg).data,
            'assistant_message': ChatMessageSerializer(assistant_msg).data,
        }, status=status.HTTP_200_OK)


class ConversationListController(APIView):
    @extend_schema(tags=['Chat'], parameters=[
        OpenApiParameter('page', int), OpenApiParameter('page_size', int)],
        responses={200: ConversationSerializer(many=True)})
    def get(self, request):
        conversations = container.build_conversation_service().list_conversations(request.user)
        paginator = StandardResultsPagination()
        page = paginator.paginate_queryset(conversations, request, view=self)
        return paginator.get_paginated_response(ConversationSerializer(page, many=True).data)


class ConversationDetailController(APIView):
    parser_classes = [JSONParser]

    @extend_schema(tags=['Chat'], responses={200: ConversationDetailSerializer})
    def get(self, request, pk):
        conversation = container.build_conversation_service().get_conversation(request.user, pk)
        return Response(ConversationDetailSerializer(conversation).data)

    @extend_schema(tags=['Chat'], request=ConversationRenameSerializer, responses={200: ConversationSerializer})
    def patch(self, request, pk):
        serializer = ConversationRenameSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        conversation = container.build_conversation_service().rename_conversation(
            request.user, pk, serializer.validated_data['title'])
        return Response(ConversationSerializer(conversation).data)

    @extend_schema(tags=['Chat'], responses={204: None})
    def delete(self, request, pk):
        container.build_conversation_service().delete_conversation(request.user, pk)
        return Response(status=status.HTTP_204_NO_CONTENT)


class ToolListController(APIView):
    """GET /api/tools — tools the agent can use (built-in search + MCP server tools)."""

    @extend_schema(tags=['Chat'], responses={200: ToolSerializer(many=True)})
    def get(self, request):
        return Response({'results': container.build_chat_service().list_available_tools()})
