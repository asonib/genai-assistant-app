# GenAI Assistant (Assignment 3B)

A complete GenAI assistant, from the Streamlit UI down to the vector store and the agent backend. Users log in, upload PDF/TXT documents, and chat with an agent that answers from those documents with source citations. The agent can also call live external tools (weather and currency exchange) through a custom MCP server, and it remembers context across conversation turns.

```text
Streamlit UI                       frontend/      login/session, upload, chat, citations
   |
   v  REST (Bearer token)
Django / DRF Backend               backend/       auth · logging · history
   |
   v
LangGraph Agent                    backend/app/agents/graph.py
   |
   v
LLM API (Claude) + Vector Store (FAISS) + External Tools (MCP)    mcp_server/
```

## Feature checklist

| Feature | Where it lives |
|---|---|
| Document upload: PDF/TXT ingestion via Streamlit | `frontend/app.py` sidebar uploader -> `POST /api/documents/` -> `PdfTxtTextExtractor` (pypdf) -> `DocumentIngestionServiceImpl` |
| Multi-doc indexing: embeddings across multiple documents | One FAISS index per user holding the chunks of all their documents (`user_<id>`). Retrieval spans every document, or a selected subset (`document_ids` scope). |
| Chat interface: full conversation UI with history | `st.chat_message` / `st.chat_input`, a conversation list in the sidebar, and `GET /api/conversations/` plus `/{id}/` |
| Memory: context across turns | The last 6 turns are replayed to Claude, and the `condense_question` node rewrites follow-ups into standalone queries before retrieval |
| Agentic flow: at least one multi-step LangGraph agent | `condense_question -> retrieve -> agent <-> tools -> finalize` (StateGraph with a conditional tool loop) |
| External tool: weather, finance, or a custom MCP tool | Custom MCP server (`mcp_server/server.py`, FastMCP) with `get_weather` (Open-Meteo), `convert_currency` and `get_exchange_rates` (Frankfurter/ECB) |
| Source citations: retrieved chunks shown with answers | Every answer returns its retrieved chunks with similarity scores. Chunks the model cited inline as `[doc:chunk]` are flagged. The UI shows them in a "Sources" expander. |
| Auth: login/session management on the Streamlit side | `frontend/session.py`: login/register, token kept in `st.session_state`, `/auth/me` validation, idle timeout, expiry handling, 401 handling, logout (single session or all sessions) |
| Tests: end-to-end test of at least one agent flow | `backend/app/tests/test_agent_e2e.py`: register, upload 2 docs, agent turn with `search_documents` + MCP `get_weather`, then a follow-up that uses memory, then history, then logout |

## Repository layout

```text
genai-assistant-app/
├── backend/                         Django project (layered architecture)
│   ├── manage.py
│   ├── backend_service/             settings, urls, wsgi/asgi
│   └── app/
│       ├── agents/                  LangGraph state + graph
│       ├── controllers/             thin DRF APIViews
│       ├── services/
│       │   ├── interfaces/          ABC contracts (ILLMService, IVectorStore, IExternalToolClient, IAgentRunner, ...)
│       │   ├── implementations/     Anthropic, FAISS, sentence-transformers, MCP client, LangGraph runner, ...
│       │   └── container.py         composition root (DI)
│       ├── models/                  UserSession, Document, DocumentChunk, Conversation, ChatMessage
│       ├── middlewares/             request logging, token auth filter, DRF passthrough auth
│       ├── prompts/templates.py     versioned PromptTemplates
│       ├── serializers/  exceptions/  config/
│       └── tests/                   offline test suite (fakes for LLM, embeddings, FAISS, MCP)
├── frontend/                        Streamlit app (app.py, session.py, api_client.py)
├── .streamlit/config.toml           Streamlit config (run `streamlit run` from the repo root)
├── mcp_server/                      custom MCP server (server.py) + pure tool logic (tools.py)
├── samples/                         sample documents to upload
├── requirements.txt                 installs all three parts into one venv
└── .env.example
```

## Setup

Requires Python 3.10+ (developed against 3.13).

```bash
cd genai-assistant-app
python -m venv venv
# Windows: venv\Scripts\activate      macOS/Linux: source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # then set ANTHROPIC_API_KEY (and SECRET_KEY)

python backend/manage.py migrate
```

The first ingestion downloads the `all-MiniLM-L6-v2` embedding model (~90 MB) from Hugging Face.

### Run

Use two terminals with the venv activated in both:

```bash
# 1) backend (also spawns the MCP server on demand over stdio)
python backend/manage.py runserver 8000

# 2) UI
streamlit run frontend/app.py
```

Open http://localhost:8501, create an account, upload both files in `samples/` (`engineering-handbook.txt` and `travel-expense-policy.pdf`), and try:

- "How are billing failures handled?" (RAG answer with citations)
- "What's the weather in Pune for the next 3 days?" (MCP `get_weather`)
- "What is the per diem for a trip to the US, and how much is that in INR today?" (multi-step: document search + MCP `convert_currency`)
- "I'm flying to London next week. What's my daily allowance and should I worry about the weather?" (documents + FX + weather)
- A follow-up such as "and who has to approve that trip if it costs 2,000 USD?" (memory)

API docs: http://127.0.0.1:8000/api/docs/ (Swagger) and `/api/redoc/`.

### MCP transport options

| `MCP_TRANSPORT` | Behaviour |
|---|---|
| `stdio` (default) | The backend spawns `python mcp_server/server.py` for each tool operation. Nothing extra to run. `tools/list` is cached for 5 minutes. |
| `http` | Run `python mcp_server/server.py --transport http` (Streamable HTTP on `127.0.0.1:8765/mcp`) and set `MCP_SERVER_URL`. This avoids process start-up per call. |
| `MCP_ENABLED=False` | The agent runs with document search only. |

If the MCP server is unreachable, the agent still works: the tool list is empty, and a failing call becomes an `is_error` tool result that the model can explain, not an HTTP 500.

To inspect the server: `npx @modelcontextprotocol/inspector python mcp_server/server.py`

## The agent (LangGraph)

```text
START -> condense_question -> retrieve -> agent --(tool calls?)--> tools -> agent ... -> finalize -> END
```

| Node | What it does |
|---|---|
| `condense_question` | If there is history, Claude rewrites the follow-up into a standalone query (`condense_question_v1`). If there is no history, the step is skipped (no LLM call). |
| `retrieve` | Embeds the standalone query and searches the user's FAISS index (top-4, cosine ≥ 0.20), optionally limited to the selected documents. |
| `agent` | Claude, with native tool use. It receives the replayed memory and the turn prompt (`assistant_agent_v1`: retrieved context + question), and is offered `search_documents` (built-in) plus every tool the MCP server advertises. |
| `tools` | Runs the requested tools (`search_documents` locally, everything else through the MCP client) and returns Anthropic `tool_result` blocks. |
| `finalize` | Builds citations: all retrieved chunks, deduplicated, with chunks the answer cites as `[document_id:chunk_index]` flagged and listed first. |

Guards: at most `AGENT_MAX_TOOL_ITERATIONS = 4` tool rounds. After the last round the model is told to answer with what it has. A LangGraph recursion limit of 25 also applies. Tool output is truncated to 6,000 characters. Each node's timing and details are stored in `ChatMessage.agent_trace` and shown in the UI under "Agent steps".

Prompt-injection hygiene: user questions, replayed user turns, retrieved chunks and tool results are all wrapped in XML-style delimiters. Closing tags inside untrusted text are neutralised, and the system prompt tells the model to treat those sections as data.

## Chunking strategy

| Setting | Value | Why |
|---|---|---|
| `CHUNK_SIZE` | 800 characters | About 150–200 tokens, under the 256-token input limit of all-MiniLM-L6-v2, so nothing is silently truncated |
| `CHUNK_OVERLAP` | 150 characters (~19%) | A sentence that straddles a boundary appears whole in at least one chunk |
| Boundary snapping | paragraph -> line -> sentence -> word | Chunks end at natural breaks and never start mid-word |
| PDF | text layer via `pypdf`, with `[Page N]` markers | Page numbers survive chunking so users can find a source |

These values live only in `AIConfigConstants` (`backend/app/config/app_config.py`).

## Auth and sessions

On the backend, `TokenAuthFilterMiddleware` authenticates every non-public request before it reaches a controller. Tokens are 256-bit random values; only their SHA-256 hash is stored (`UserSession.token_hash`). Sessions expire after 7 days, and expired sessions are deleted on use. Logout revokes either the current session or all of the user's sessions. `FilterPassthroughAuthentication` passes the user through to DRF. There is no `SessionAuthentication`, so token POSTs never fail CSRF checks.

On the Streamlit side, `session.py` keeps the token only in server-side `st.session_state` (never in URLs or cookies). It validates the token with `/api/auth/me` when a session starts, logs the user out after `SESSION_IDLE_MINUTES` (default 30) of inactivity, checks the token expiry, and sends the user back to the login screen on any 401.

## Logging and history

`RequestLoggingMiddleware` assigns or propagates an `X-Request-ID` (the Streamlit client sends one with every call). It logs one structured line per request (method, path, status, duration, user id) and echoes the id in the response and in error bodies. Agent runs and tool calls are logged by `app.agents` and `app.services`. Logs go to the console and to a rotating `backend/logs/backend.log` (5 × 5 MB).

Every turn is stored as `ChatMessage` rows with role, content, citations, tool calls, agent trace, model, prompt template, tokens and latency. Turns are saved only when the agent succeeds, so memory always holds complete user/assistant pairs.

## API

| Method | Path | Description |
|---|---|---|
| GET | `/api/health/` | Public readiness probe (DB, LLM configured, MCP transport) |
| POST | `/api/auth/register/` · `/api/auth/login/` | Returns `{token, token_type, expires_at, user}` |
| POST | `/api/auth/logout/` | `{"all_sessions": false}` |
| GET | `/api/auth/me/` | Current user and session expiry |
| GET / POST | `/api/documents/` | List (paginated) / upload one or more files (multipart field `files`) |
| GET / DELETE | `/api/documents/{id}/` | Detail with content preview / delete (also removes vectors) |
| POST | `/api/documents/{id}/reingest/` | Re-chunk and re-embed |
| POST | `/api/chat/` | `{"message", "conversation_id"?, "document_ids"?}` returns the conversation plus the user and assistant messages |
| GET | `/api/conversations/` | History list (paginated) |
| GET / PATCH / DELETE | `/api/conversations/{id}/` | Messages / rename / delete |
| GET | `/api/tools/` | Built-in and MCP tools available to the agent |

Errors use one envelope: `{"detail", "code", "request_id", "errors"?}`.

## Tests

All tests run offline. Claude, the embedding model, FAISS and the MCP server are replaced with deterministic fakes (`backend/app/tests/fakes.py`) through `container.override(...)`. The real LangGraph graph, services, ORM, middleware and controllers still run.

```bash
python backend/manage.py test app            # backend: E2E agent flow, graph, auth, documents, adapters
python -m unittest discover -s mcp_server    # MCP tool logic (HTTP mocked)
python -m unittest discover -s frontend      # Streamlit API client
```

`test_agent_e2e.py` covers the whole flow: register, upload two documents into one index, then an agent turn where Claude calls `search_documents` and MCP `get_weather` and answers with a citation. It then sends a follow-up that is condensed using memory (the previous turn is replayed), and finishes by checking conversation history and that logout revokes the token. If `mcp` is installed, `test_real_stdio_server_advertises_tools` also starts the real MCP server over stdio.

## Production notes

Set `DEBUG=False`, a strong `SECRET_KEY` and `ALLOWED_HOSTS`. Run Django under a WSGI server (`waitress-serve --port 8000 backend_service.wsgi:application` from `backend/` on Windows, or gunicorn on Linux). Use `MCP_TRANSPORT=http` with the MCP server as its own service. SQLite is the default database. Pointing `DATABASES` at Postgres is a settings-only change. `.env`, `key.txt`, `db.sqlite3`, logs and FAISS indexes are all git-ignored.
