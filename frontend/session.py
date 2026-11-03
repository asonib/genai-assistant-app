"""
Streamlit-side login/session management.

* The bearer token lives only in `st.session_state` (server-side, per browser
  tab) - never in the URL, cookies or local storage.
* The token is validated against /api/auth/me once per session start.
* Idle timeout: after SESSION_IDLE_MINUTES without interaction the user is
  logged out (backend session revoked).
* Any 401 from the backend (expired / revoked token) clears the session.
"""
import os
import time
from datetime import datetime, timezone
from typing import Optional

import streamlit as st

try:  # pick up BACKEND_URL etc. from <repo>/.env when present
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
except ImportError:
    pass

from api_client import ApiClient, ApiError, AuthenticationError

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000/api")
SESSION_IDLE_MINUTES = int(os.getenv("SESSION_IDLE_MINUTES", "30"))

_AUTH_KEYS = ("token", "user", "expires_at", "validated", "last_activity", "conversation_id",
              "messages", "doc_scope")


def init_state() -> None:
    defaults = {"token": None, "user": None, "expires_at": None, "validated": False,
                "last_activity": time.time(), "conversation_id": None, "messages": [],
                "doc_scope": [], "flash": None, "uploader_key": 0}
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def client() -> ApiClient:
    return ApiClient(BACKEND_URL, token=st.session_state.get("token"))


def is_authenticated() -> bool:
    return bool(st.session_state.get("token"))


def start_session(auth_payload: dict) -> None:
    st.session_state.token = auth_payload["token"]
    st.session_state.user = auth_payload["user"]
    st.session_state.expires_at = auth_payload.get("expires_at")
    st.session_state.validated = True
    st.session_state.last_activity = time.time()
    st.session_state.conversation_id = None
    st.session_state.messages = []


def clear_session(flash: Optional[str] = None) -> None:
    for key in _AUTH_KEYS:
        st.session_state.pop(key, None)
    init_state()
    if flash:
        st.session_state.flash = flash


def logout(all_sessions: bool = False) -> None:
    try:
        client().logout(all_sessions=all_sessions)
    except ApiError:
        pass  # token already invalid server-side: nothing to revoke
    clear_session("You have been logged out.")


def _token_expired() -> bool:
    raw = st.session_state.get("expires_at")
    if not raw:
        return False
    try:
        expires = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return False
    return datetime.now(timezone.utc) >= expires


def enforce_session() -> bool:
    """Call at the top of every run. Returns True if the user may see the app."""
    if not is_authenticated():
        return False
    if time.time() - st.session_state.last_activity > SESSION_IDLE_MINUTES * 60:
        logout()
        st.session_state.flash = f"Logged out after {SESSION_IDLE_MINUTES} minutes of inactivity."
        return False
    if _token_expired():
        clear_session("Your session has expired. Please log in again.")
        return False
    if not st.session_state.validated:
        try:
            st.session_state.user = client().me()
            st.session_state.validated = True
        except AuthenticationError:
            clear_session("Your session is no longer valid. Please log in again.")
            return False
    st.session_state.last_activity = time.time()
    return True


def handle_auth_error(exc: AuthenticationError) -> None:
    clear_session(f"Session ended: {exc.detail}")
    st.rerun()
