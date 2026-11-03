"""
Composition root for the service layer.

Controllers ask this module for fully-wired services instead of instantiating
infrastructure (Anthropic client, embedding model, FAISS index, MCP client,
LangGraph agent) themselves. Heavy singletons are created once per process.
Tests replace providers with `container.override(...)` / `container.reset()`.
"""
import threading

from django.conf import settings

from app.config.app_config import AIConfigConstants

_lock = threading.RLock()
_singletons = {}
_overrides = {}


def _singleton(name, factory):
    if name in _overrides:
        return _overrides[name]
    instance = _singletons.get(name)
    if instance is None:
        with _lock:
            instance = _singletons.get(name)
            if instance is None:
                instance = factory()
                _singletons[name] = instance
    return instance


def override(**providers):
    """Replace providers, e.g. override(llm_service=FakeLLM(), tool_client=FakeTools())."""
    with _lock:
        _overrides.update(providers)
        _singletons.clear()  # dependants (e.g. agent_runner) must be rebuilt


def reset():
    with _lock:
        _overrides.clear()
        _singletons.clear()


# ---- infrastructure ---------------------------------------------------------

def get_text_extractor():
    from app.services.implementations.text_extractor_impl import PdfTxtTextExtractor
    return _singleton("text_extractor", PdfTxtTextExtractor)


def get_text_chunker():
    from app.services.implementations.text_chunker_impl import RecursiveCharacterTextChunker
    return _singleton("text_chunker", lambda: RecursiveCharacterTextChunker(
        chunk_size=AIConfigConstants.CHUNK_SIZE, chunk_overlap=AIConfigConstants.CHUNK_OVERLAP))


def get_embedding_service():
    from app.services.implementations.sentence_transformer_embedding_service_impl import (
        SentenceTransformerEmbeddingService,
    )
    return _singleton("embedding_service", lambda: SentenceTransformerEmbeddingService(
        model_name=AIConfigConstants.EMBEDDING_MODEL_NAME,
        dimension=AIConfigConstants.EMBEDDING_DIMENSION,
        cache_folder=settings.EMBEDDING_CACHE_DIR,
    ))


def get_vector_store():
    from app.services.implementations.faiss_vector_store_impl import FaissVectorStore
    return _singleton("vector_store", lambda: FaissVectorStore(
        base_dir=settings.VECTOR_STORE_DIR, dimension=AIConfigConstants.EMBEDDING_DIMENSION))


def get_llm_service():
    from app.services.implementations.anthropic_llm_service_impl import AnthropicLLMService
    return _singleton("llm_service", lambda: AnthropicLLMService(
        api_key=settings.ANTHROPIC_API_KEY, model=settings.ANTHROPIC_MODEL))


def get_tool_client():
    def factory():
        from app.services.implementations.mcp_tool_client_impl import DisabledToolClient, McpToolClient
        if not settings.MCP_ENABLED:
            return DisabledToolClient()
        return McpToolClient(transport=settings.MCP_TRANSPORT, command=settings.MCP_SERVER_COMMAND,
                             args=settings.MCP_SERVER_ARGS, url=settings.MCP_SERVER_URL)
    return _singleton("tool_client", factory)


def get_retrieval_service():
    from app.services.implementations.retrieval_service_impl import VectorRetrievalService
    return _singleton("retrieval_service", lambda: VectorRetrievalService(
        embedding_service=get_embedding_service(), vector_store=get_vector_store()))


def get_agent_runner():
    from app.services.implementations.langgraph_agent_impl import LangGraphAgentRunner
    return _singleton("agent_runner", lambda: LangGraphAgentRunner(
        llm_service=get_llm_service(), retrieval_service=get_retrieval_service(),
        tool_client=get_tool_client()))


# ---- business services --------------------------------------------------------

def build_auth_service():
    from app.services.implementations.auth_service_impl import AuthServiceImpl
    return AuthServiceImpl()


def build_ingestion_service():
    from app.services.implementations.ingestion_service_impl import DocumentIngestionServiceImpl
    return DocumentIngestionServiceImpl(chunker=get_text_chunker(), embedding_service=get_embedding_service(),
                                        vector_store=get_vector_store())


def build_document_service():
    from app.services.implementations.document_service_impl import DocumentServiceImpl
    return DocumentServiceImpl(text_extractor=get_text_extractor(), ingestion_service=build_ingestion_service())


def build_conversation_service():
    from app.services.implementations.conversation_service_impl import ConversationServiceImpl
    return ConversationServiceImpl()


def build_chat_service():
    from app.services.implementations.chat_service_impl import ChatServiceImpl
    return ChatServiceImpl(agent_runner=get_agent_runner(), conversation_service=build_conversation_service(),
                           tool_client=get_tool_client())
