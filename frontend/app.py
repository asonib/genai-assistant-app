"""
GenAI Assistant - Streamlit UI (Assignment 3B).

    streamlit run frontend/app.py

Streamlit UI -> Django/DRF backend (auth, logging, history) -> LangGraph agent
-> Claude + FAISS vector store + MCP tools (weather, FX).
"""
import json

import streamlit as st

import session
from api_client import ApiError, AuthenticationError, BackendUnavailableError

st.set_page_config(page_title="GenAI Assistant", page_icon="💬", layout="wide")
session.init_state()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def call(fn, *args, quiet=False, **kwargs):
    """Run an API call; route 401s to the login screen and show other errors inline."""
    try:
        return fn(*args, **kwargs)
    except AuthenticationError as exc:
        session.handle_auth_error(exc)
    except BackendUnavailableError as exc:
        if not quiet:
            st.error(f"Backend unavailable: {exc.detail}")
    except ApiError as exc:
        if not quiet:
            suffix = f" (request id {exc.request_id})" if exc.request_id else ""
            st.error(f"{exc}{suffix}")
    return None


def show_flash():
    flash = st.session_state.get("flash")
    if flash:
        st.info(flash)
        st.session_state.flash = None


def quote(text: str) -> str:
    return "> " + text.strip().replace("\n", "\n> ")


# ---------------------------------------------------------------------------
# login / register
# ---------------------------------------------------------------------------
def render_auth_page():
    _, center, _ = st.columns([1, 1.4, 1])
    with center:
        st.title("GenAI Assistant")
        st.caption("Chat with your documents, plus live weather and currency tools.")
        show_flash()

        health = call(session.client().health, quiet=True)
        if health is None:
            st.warning(f"Backend not reachable at {session.BACKEND_URL}. Start it with "
                       "`python backend/manage.py runserver 8000`.")
        elif not health.get("llm_configured"):
            st.warning("Backend is up, but ANTHROPIC_API_KEY is not configured.")

        login_tab, register_tab = st.tabs(["Log in", "Create account"])
        with login_tab:
            with st.form("login_form", clear_on_submit=False):
                username = st.text_input("Username", autocomplete="username")
                password = st.text_input("Password", type="password", autocomplete="current-password")
                submitted = st.form_submit_button("Log in", type="primary", use_container_width=True)
            if submitted:
                if not username or not password:
                    st.error("Enter your username and password.")
                else:
                    data = call(session.client().login, username.strip(), password)
                    if data:
                        session.start_session(data)
                        st.rerun()

        with register_tab:
            with st.form("register_form"):
                new_username = st.text_input("Username", key="reg_username")
                email = st.text_input("Email (optional)", key="reg_email")
                new_password = st.text_input("Password", type="password", key="reg_password",
                                             help="At least 8 characters, not too common, not all digits.")
                confirm = st.text_input("Confirm password", type="password", key="reg_confirm")
                created = st.form_submit_button("Create account", type="primary", use_container_width=True)
            if created:
                if new_password != confirm:
                    st.error("Passwords do not match.")
                elif not new_username or not new_password:
                    st.error("Username and password are required.")
                else:
                    data = call(session.client().register, new_username.strip(), new_password, email.strip())
                    if data:
                        session.start_session(data)
                        st.rerun()


# ---------------------------------------------------------------------------
# sidebar: account, conversations, documents, tools
# ---------------------------------------------------------------------------
def open_conversation(conversation_id):
    st.session_state.conversation_id = conversation_id
    st.session_state.messages = []
    st.session_state.loaded_conversation_id = None


def render_sidebar(documents):
    api = session.client()
    with st.sidebar:
        user = st.session_state.user or {}
        st.markdown(f"Signed in as **{user.get('username', '?')}**")
        col_a, col_b = st.columns(2)
        if col_a.button("Log out", use_container_width=True):
            session.logout()
            st.rerun()
        if col_b.button("New chat", type="primary", use_container_width=True):
            open_conversation(None)
            st.rerun()

        # ---- conversations (history) ----
        st.subheader("Conversations")
        conversations = call(api.list_conversations) or []
        if not conversations:
            st.caption("No conversations yet.")
        for conv in conversations[:30]:
            is_current = conv["id"] == st.session_state.conversation_id
            if st.button(conv["title"], key=f"conv_{conv['id']}", use_container_width=True,
                         type="primary" if is_current else "secondary"):
                open_conversation(conv["id"])
                st.rerun()

        # ---- documents ----
        st.subheader("Documents")
        with st.form("upload_form", clear_on_submit=True):
            uploads = st.file_uploader("Upload PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True,
                                       key=f"uploader_{st.session_state.uploader_key}")
            do_upload = st.form_submit_button("Upload & index", use_container_width=True)
        if do_upload and uploads:
            with st.spinner(f"Extracting, chunking and embedding {len(uploads)} file(s)..."):
                results = call(api.upload_documents, [(f.name, f.getvalue(), f.type) for f in uploads])
            if results:
                for r in results:
                    doc = r["document"]
                    if r["duplicate"]:
                        st.toast(f"'{doc['title']}' was already uploaded.")
                    elif r["error"]:
                        st.session_state.flash = f"'{doc['title']}' was saved but indexing failed: {r['error']}"
                    else:
                        st.toast(f"Indexed '{doc['title']}' ({doc['chunk_count']} chunks).")
                st.session_state.uploader_key += 1
                st.rerun()

        if not documents:
            st.caption("No documents yet. Upload some to enable grounded answers with citations.")
        for doc in documents:
            status = doc["ingestion_status"]
            badge = {"ingested": ":green[indexed]", "failed": ":red[failed]"}.get(status, ":orange[pending]")
            with st.expander(f"{doc['title']}", expanded=False):
                st.markdown(f"{badge} · {doc['file_type'].upper()} · {doc['chunk_count']} chunks"
                            + (f" · {doc['page_count']} pages" if doc.get("page_count") else ""))
                if doc.get("ingestion_error"):
                    st.caption(doc["ingestion_error"])
                c1, c2 = st.columns(2)
                if c1.button("Re-index", key=f"reidx_{doc['id']}", use_container_width=True):
                    if call(api.reingest_document, doc["id"]):
                        st.toast(f"Re-indexed '{doc['title']}'.")
                        st.rerun()
                if c2.button("Delete", key=f"del_{doc['id']}", use_container_width=True):
                    call(api.delete_document, doc["id"])
                    st.session_state.doc_scope = [d for d in st.session_state.doc_scope if d != doc["id"]]
                    st.rerun()

        # ---- tools ----
        with st.expander("Available tools"):
            for tool in call(api.list_tools, quiet=True) or []:
                st.markdown(f"**{tool['name']}** · _{tool['source']}_  \n{tool['description'].splitlines()[0]}")


# ---------------------------------------------------------------------------
# chat
# ---------------------------------------------------------------------------
def render_assistant_details(msg):
    citations = msg.get("citations") or []
    tool_calls = msg.get("tool_calls") or []
    if citations:
        cited = sum(1 for c in citations if c.get("cited"))
        with st.expander(f"Sources: {len(citations)} retrieved chunks, {cited} cited"):
            for c in citations:
                flag = " · **cited**" if c.get("cited") else ""
                st.markdown(f"`[{c['document_id']}:{c['chunk_index']}]` **{c['document_title']}** "
                            f"· similarity {c['score']:.2f}{flag}")
                st.markdown(quote(c["chunk_text"]))
    if tool_calls:
        with st.expander(f"Tool calls ({len(tool_calls)})"):
            for t in tool_calls:
                status = ":red[error]" if t.get("is_error") else ":green[ok]"
                st.markdown(f"**{t['name']}** ({t.get('source')}) · {status} · {t.get('latency_ms', 0)} ms")
                st.json(t.get("input") or {}, expanded=False)
                try:
                    st.json(json.loads(t.get("output") or ""), expanded=False)
                except (TypeError, ValueError):
                    st.code(t.get("output") or "", language=None)
    trace = msg.get("agent_trace") or []
    if trace:
        with st.expander("Agent steps"):
            rows = [{"step": i + 1, "node": s.get("node"), "ms": s.get("ms"),
                     "details": ", ".join(f"{k}={v}" for k, v in s.items() if k not in ("node", "ms"))}
                    for i, s in enumerate(trace)]
            st.table(rows)
    meta = [msg.get("model_name"), f"{msg.get('latency_ms', 0)} ms"]
    if msg.get("input_tokens"):
        meta.append(f"{msg['input_tokens']} in / {msg.get('output_tokens') or 0} out tokens")
    st.caption(" · ".join(m for m in meta if m))


def render_message(msg):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_assistant_details(msg)


def load_current_conversation():
    conversation_id = st.session_state.conversation_id
    if conversation_id and st.session_state.get("loaded_conversation_id") != conversation_id:
        data = call(session.client().get_conversation, conversation_id)
        if data is None:
            open_conversation(None)
            return None
        st.session_state.messages = data["messages"]
        st.session_state.loaded_conversation_id = conversation_id
        st.session_state.conversation_title = data["title"]
    return conversation_id


def render_chat(documents):
    conversation_id = load_current_conversation()
    title = st.session_state.get("conversation_title") if conversation_id else "New conversation"
    st.title(title or "Conversation")
    show_flash()

    indexed = {d["id"]: d["title"] for d in documents if d["ingestion_status"] == "ingested"}
    st.session_state.doc_scope = [d for d in st.session_state.doc_scope if d in indexed]
    st.multiselect("Search scope (leave empty to search all documents)", options=list(indexed),
                   format_func=lambda d: indexed.get(d, str(d)), key="doc_scope")

    if conversation_id:
        with st.expander("Conversation settings"):
            new_title = st.text_input("Rename", value=title or "", key=f"rename_{conversation_id}")
            c1, c2 = st.columns(2)
            if c1.button("Save title", use_container_width=True) and new_title.strip():
                if call(session.client().rename_conversation, conversation_id, new_title.strip()):
                    st.session_state.loaded_conversation_id = None
                    st.rerun()
            if c2.button("Delete conversation", use_container_width=True):
                call(session.client().delete_conversation, conversation_id)
                open_conversation(None)
                st.rerun()

    if not st.session_state.messages:
        st.caption("Try: *\"Summarise the billing section\"*, *\"What's the weather in Pune this week?\"*, "
                   "or *\"Convert 250 USD to INR\"*.")
    for msg in st.session_state.messages:
        render_message(msg)

    prompt = st.chat_input("Ask about your documents, the weather or exchange rates...")
    if prompt:
        render_message({"role": "user", "content": prompt})
        with st.chat_message("assistant"):
            with st.spinner("Thinking: retrieving context and calling tools..."):
                result = call(session.client().chat, prompt, conversation_id, st.session_state.doc_scope)
        if result:
            st.session_state.messages.extend([result["user_message"], result["assistant_message"]])
            st.session_state.conversation_id = result["conversation"]["id"]
            st.session_state.loaded_conversation_id = result["conversation"]["id"]
            st.session_state.conversation_title = result["conversation"]["title"]
            st.rerun()


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
if not session.enforce_session():
    render_auth_page()
    st.stop()

docs = call(session.client().list_documents) or []
render_sidebar(docs)
render_chat(docs)
