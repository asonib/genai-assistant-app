"""
Thin HTTP client for the Django backend. No Streamlit imports, so it can be
unit-tested and reused (CLI scripts, notebooks).
"""
import uuid
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

DEFAULT_TIMEOUT = (5, 120)  # (connect, read) seconds; agent turns can take a while


class ApiError(Exception):
    def __init__(self, status: int, detail: str, code: str = "error", errors: Optional[dict] = None,
                 request_id: Optional[str] = None):
        super().__init__(detail)
        self.status = status
        self.detail = detail
        self.code = code
        self.errors = errors or {}
        self.request_id = request_id

    def __str__(self):
        msg = self.detail
        if self.errors:
            parts = []
            for field, problems in self.errors.items():
                if isinstance(problems, (list, tuple)):
                    problems = "; ".join(str(p) for p in problems)
                parts.append(f"{field}: {problems}")
            msg = f"{msg} ({' | '.join(parts)})"
        return msg


class AuthenticationError(ApiError):
    """401: missing/invalid/expired token -> the UI must send the user back to login."""


class BackendUnavailableError(ApiError):
    """Network-level failure talking to the backend."""


class ApiClient:
    def __init__(self, base_url: str, token: Optional[str] = None, timeout=DEFAULT_TIMEOUT,
                 session: Optional[requests.Session] = None):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout
        self.session = session or requests.Session()

    # ---- plumbing -----------------------------------------------------------
    def _headers(self) -> Dict[str, str]:
        headers = {"Accept": "application/json", "X-Request-ID": uuid.uuid4().hex,
                   "User-Agent": "genai-assistant-streamlit/1.0"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _request(self, method: str, path: str, **kwargs) -> Any:
        url = f"{self.base_url}/{path.lstrip('/')}"
        try:
            response = self.session.request(method, url, headers=self._headers(), timeout=self.timeout, **kwargs)
        except requests.RequestException as exc:
            raise BackendUnavailableError(0, f"Cannot reach the backend at {self.base_url}: {exc}", "unavailable")

        if response.status_code == 204:
            return None
        try:
            body = response.json()
        except ValueError:
            body = {"detail": response.text[:300] or response.reason}

        if response.ok:
            return body
        detail = body.get("detail", response.reason) if isinstance(body, dict) else str(body)
        code = body.get("code", "error") if isinstance(body, dict) else "error"
        errors = body.get("errors") if isinstance(body, dict) else None
        request_id = response.headers.get("X-Request-ID")
        cls = AuthenticationError if response.status_code == 401 else ApiError
        raise cls(response.status_code, str(detail), code, errors, request_id)

    def _get_all_pages(self, path: str, page_size: int = 100) -> List[dict]:
        items, page = [], 1
        while True:
            data = self._request("GET", path, params={"page": page, "page_size": page_size})
            items.extend(data.get("results", []))
            if not data.get("next"):
                return items
            page += 1

    # ---- auth ---------------------------------------------------------------
    def health(self) -> dict:
        return self._request("GET", "health/")

    def register(self, username: str, password: str, email: str = "") -> dict:
        payload = {"username": username, "password": password}
        if email:
            payload["email"] = email
        data = self._request("POST", "auth/register/", json=payload)
        self.token = data["token"]
        return data

    def login(self, username: str, password: str) -> dict:
        data = self._request("POST", "auth/login/", json={"username": username, "password": password})
        self.token = data["token"]
        return data

    def logout(self, all_sessions: bool = False) -> None:
        try:
            self._request("POST", "auth/logout/", json={"all_sessions": all_sessions})
        finally:
            self.token = None

    def me(self) -> dict:
        return self._request("GET", "auth/me/")

    # ---- documents ----------------------------------------------------------
    def list_documents(self) -> List[dict]:
        return self._get_all_pages("documents/")

    def upload_documents(self, files: Iterable[Tuple[str, bytes, str]]) -> List[dict]:
        multipart = [("files", (name, data, mime or "application/octet-stream")) for name, data, mime in files]
        return self._request("POST", "documents/", files=multipart)["results"]

    def delete_document(self, document_id: int) -> None:
        self._request("DELETE", f"documents/{document_id}/")

    def reingest_document(self, document_id: int) -> dict:
        return self._request("POST", f"documents/{document_id}/reingest/")

    # ---- chat ---------------------------------------------------------------
    def chat(self, message: str, conversation_id: Optional[int] = None,
             document_ids: Optional[List[int]] = None) -> dict:
        payload: Dict[str, Any] = {"message": message}
        if conversation_id:
            payload["conversation_id"] = conversation_id
        if document_ids:
            payload["document_ids"] = list(document_ids)
        return self._request("POST", "chat/", json=payload)

    def list_conversations(self) -> List[dict]:
        return self._get_all_pages("conversations/")

    def get_conversation(self, conversation_id: int) -> dict:
        return self._request("GET", f"conversations/{conversation_id}/")

    def rename_conversation(self, conversation_id: int, title: str) -> dict:
        return self._request("PATCH", f"conversations/{conversation_id}/", json={"title": title})

    def delete_conversation(self, conversation_id: int) -> None:
        self._request("DELETE", f"conversations/{conversation_id}/")

    def list_tools(self) -> List[dict]:
        return self._request("GET", "tools/")["results"]
