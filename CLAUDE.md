# CLAUDE.md - GenAI Assistant (Assignment 3B)

Monorepo: `backend/` (Django/DRF + LangGraph agent), `frontend/` (Streamlit), `mcp_server/` (custom MCP tools).

## Backend architecture (unchanged contract from A2/A3A)

```text
Request -> RequestLoggingMiddleware -> TokenAuthFilterMiddleware -> Controller (APIView)
        -> Service interface -> Service implementation -> Models (ORM) / AI infrastructure
```

- **Controllers** (`app/controllers/`): parse + validate with serializers, call one service, return a Response. No ORM, no transactions, no SDK clients.
- **Services**: contracts in `app/services/interfaces/` (`abc.ABC`), logic in `implementations/`. Wire everything in `app/services/container.py`; controllers call `container.build_*()` / `container.get_*()`.
- **Models** must be exported from `app/models/__init__.py`.
- **Config**: constants in `app/config/app_config.py` (`AppConfigConstants`, `AIConfigConstants`); secrets/env in `settings.py` from `.env` at the repo root.
- **Exceptions**: subclass DRF `APIException` with `status_code`/`default_detail`/`default_code`; the handler emits `{detail, code, request_id, errors?}`.

## Agent rules

- The agent is the LangGraph graph in `app/agents/graph.py`; `LangGraphAgentRunner` (IAgentRunner) is the only entry point. Node names must not collide with `AgentState` keys.
- Nodes return partial updates with *fresh* lists (no reducers, no in-place mutation).
- LLM access only through `ILLMService.generate(system, messages, tools)`. Never pass `temperature`/`top_p` (rejected by the installed anthropic SDK).
- Never send raw user input to the LLM: render via a `PromptTemplate` in `app/prompts/templates.py`; bump the name (`_v2`) when semantics change.
- External tools only via `IExternalToolClient` (MCP). Tool failures must become `is_error` tool results, never 500s. `list_tools()` must not raise.
- FAISS vector id == `DocumentChunk.pk`; one index per user (`user_<id>`). Chunk size/overlap live only in `AIConfigConstants` and are documented in README.
- Chat turns are persisted only after the agent succeeds (complete user/assistant pairs = clean memory).

## Django/DRF traps

- Keep `FilterPassthroughAuthentication` (with `authenticate_header`) as the only DRF auth class: no `SessionAuthentication` (CSRF 403s) and 401 instead of 403.
- `Meta.ordering` is ignored on aggregated (GROUP BY) querysets: add explicit `order_by` before paginating.
- Upload validation: size -> `PayloadTooLargeException` (413), extension -> `UnsupportedFileTypeException` (400), no text -> 422.

## Frontend rules

- `frontend/api_client.py` has no Streamlit imports (unit-testable). All UI calls go through `app.call(...)` so 401s route to the login screen.
- Token lives only in `st.session_state`; never put it in URLs, query params or cookies.

## Tests (must stay offline)

Use `container.override(...)` with fakes from `app/tests/fakes.py` (`ScriptedLLMService`, `FakeToolClient`, `FakeEmbeddingService`, `InMemoryVectorStore`).

```bash
python backend/manage.py test app
python -m unittest discover -s mcp_server
python -m unittest discover -s frontend
```

Secrets (`ANTHROPIC_API_KEY`) come from `.env`; never commit `.env` or `key.txt`.
