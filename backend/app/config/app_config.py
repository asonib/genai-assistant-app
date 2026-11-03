class AppConfigConstants:
    """Environment-agnostic constants for the web / API layer."""

    DEFAULT_PAGE_SIZE = 10
    MAX_PAGE_SIZE = 100

    # ---- Uploads (Assignment 3B: PDF / TXT ingestion via Streamlit) ----
    MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB per file
    MAX_FILES_PER_UPLOAD = 10
    ALLOWED_EXTENSIONS = [".pdf", ".txt"]
    MAX_DOCUMENT_CHARS = 2_000_000

    # ---- Auth ----
    TOKEN_EXPIRY_DAYS = 7
    MIN_PASSWORD_LENGTH = 8
    PUBLIC_URL_PREFIXES = [
        "/api/auth/register",
        "/api/auth/login",
        "/api/health",
        "/api/schema",
        "/api/docs",
        "/api/redoc",
        "/admin",
    ]

    # ---- Logging ----
    REQUEST_ID_HEADER = "X-Request-ID"


class AIConfigConstants:
    """
    Environment-agnostic constants for the AI layer (RAG + LangGraph agent).
    Secrets and environment-specific values (API key, model id, index path,
    MCP server command) live in settings.py and are read from .env.
    """

    # ---- Chunking (documented in README -> "Chunking strategy") ----
    # Characters. ~800 chars ~ 150-200 tokens: below the 256 word-piece limit of
    # all-MiniLM-L6-v2, so no chunk is truncated by the embedding model.
    CHUNK_SIZE = 800
    # ~19% overlap so a sentence straddling a boundary is whole in one chunk.
    CHUNK_OVERLAP = 150
    MAX_CHUNKS_PER_DOCUMENT = 3000

    # ---- Embeddings ----
    EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION = 384
    EMBEDDING_BATCH_SIZE = 32

    # ---- Vector store ----
    VECTOR_STORE_BACKEND = "faiss"
    VECTOR_INDEX_TYPE = "IndexIDMap2(IndexFlatIP)"  # cosine on L2-normalised vectors

    # ---- Retrieval ----
    RAG_TOP_K = 4
    RAG_MIN_SIMILARITY = 0.20
    # When the user scopes a question to specific documents we over-fetch from
    # the per-user index and filter, so top_k results survive the filter.
    RAG_FILTER_OVERSAMPLE = 8
    MAX_CITATIONS = 8

    # ---- LLM ----
    LLM_PROVIDER = "anthropic"
    LLM_MAX_TOKENS = 1500
    LLM_TIMEOUT_SECONDS = 60

    # ---- Agent (LangGraph) ----
    # Max agent<->tools round-trips per user turn (guards against tool loops).
    AGENT_MAX_TOOL_ITERATIONS = 4
    AGENT_RECURSION_LIMIT = 25
    # Name of the built-in (non-MCP) retrieval tool exposed to the agent.
    SEARCH_TOOL_NAME = "search_documents"
    MAX_TOOL_RESULT_CHARS = 6000

    # ---- Memory ----
    # Number of previous user/assistant *turns* replayed to the LLM.
    MEMORY_MAX_TURNS = 6
    # Hard cap on replayed history size (characters) to bound token usage.
    MEMORY_MAX_CHARS = 12000

    # ---- MCP (external tools) ----
    MCP_TOOLS_CACHE_SECONDS = 300
    MCP_CALL_TIMEOUT_SECONDS = 20

    # ---- Input limits ----
    MAX_MESSAGE_CHARS = 4000
    MAX_TITLE_CHARS = 120
