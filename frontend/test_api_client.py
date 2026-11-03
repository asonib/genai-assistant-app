"""Offline tests for the Streamlit API client:  python -m unittest discover -s frontend"""
import os
import sys
import unittest
from unittest import mock

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from api_client import ApiClient, ApiError, AuthenticationError, BackendUnavailableError  # noqa: E402


def response(status, body=None, headers=None):
    r = mock.Mock()
    r.status_code = status
    r.ok = 200 <= status < 300
    r.reason = "reason"
    r.headers = headers or {}
    r.text = str(body)
    r.json.return_value = body
    return r


class ApiClientTests(unittest.TestCase):
    def setUp(self):
        self.session = mock.Mock(spec=requests.Session)
        self.api = ApiClient("http://backend/api/", session=self.session)

    def test_login_stores_token_and_sends_bearer(self):
        self.session.request.return_value = response(200, {"token": "t0k", "user": {"id": 1}})
        self.api.login("alice", "pw")
        self.assertEqual(self.api.token, "t0k")
        self.session.request.return_value = response(200, {"id": 1})
        self.api.me()
        method, url = self.session.request.call_args.args
        headers = self.session.request.call_args.kwargs["headers"]
        self.assertEqual((method, url), ("GET", "http://backend/api/auth/me/"))
        self.assertEqual(headers["Authorization"], "Bearer t0k")
        self.assertIn("X-Request-ID", headers)

    def test_401_raises_authentication_error(self):
        self.session.request.return_value = response(401, {"detail": "Token has expired.", "code": "x"})
        with self.assertRaises(AuthenticationError) as ctx:
            self.api.me()
        self.assertEqual(ctx.exception.detail, "Token has expired.")

    def test_validation_errors_are_readable(self):
        self.session.request.return_value = response(
            400, {"detail": "Validation failed.", "code": "invalid", "errors": {"password": ["Too short."]}},
            {"X-Request-ID": "rid"})
        with self.assertRaises(ApiError) as ctx:
            self.api.register("bob", "x")
        self.assertIn("password: Too short.", str(ctx.exception))
        self.assertEqual(ctx.exception.request_id, "rid")

    def test_network_error(self):
        self.session.request.side_effect = requests.ConnectionError("refused")
        with self.assertRaises(BackendUnavailableError):
            self.api.health()

    def test_pagination_is_followed(self):
        self.session.request.side_effect = [
            response(200, {"results": [{"id": 1}], "next": "p2"}),
            response(200, {"results": [{"id": 2}], "next": None}),
        ]
        self.assertEqual([d["id"] for d in self.api.list_documents()], [1, 2])

    def test_upload_and_chat_payloads(self):
        self.session.request.return_value = response(201, {"results": []})
        self.api.upload_documents([("a.pdf", b"%PDF", "application/pdf"), ("b.txt", b"hi", None)])
        files = self.session.request.call_args.kwargs["files"]
        self.assertEqual(files[0], ("files", ("a.pdf", b"%PDF", "application/pdf")))
        self.assertEqual(files[1][1][2], "application/octet-stream")

        self.session.request.return_value = response(200, {})
        self.api.chat("hello", conversation_id=3, document_ids=[1, 2])
        self.assertEqual(self.session.request.call_args.kwargs["json"],
                         {"message": "hello", "conversation_id": 3, "document_ids": [1, 2]})
        self.api.chat("hello")
        self.assertEqual(self.session.request.call_args.kwargs["json"], {"message": "hello"})

    def test_logout_clears_token_even_on_error(self):
        self.api.token = "t"
        self.session.request.return_value = response(401, {"detail": "Invalid"})
        with self.assertRaises(AuthenticationError):
            self.api.logout()
        self.assertIsNone(self.api.token)

    def test_204(self):
        self.session.request.return_value = response(204)
        self.assertIsNone(self.api.delete_document(4))


if __name__ == "__main__":
    unittest.main()
