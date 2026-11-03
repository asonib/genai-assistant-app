from datetime import timedelta

from django.utils import timezone

from app.models import UserSession
from app.models.user_session import hash_token
from app.tests.base import AgentTestCase


class AuthTests(AgentTestCase):
    def test_register_login_me_logout(self):
        token = self.register("alice")
        session = UserSession.objects.get()
        self.assertEqual(session.token_hash, hash_token(token))
        self.assertNotEqual(session.token_hash, token)  # raw token is never stored

        res = self.client.post("/api/auth/login", {"username": "alice", "password": self.password}, format="json")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["token_type"], "Bearer")
        login_token = res.json()["token"]

        self.client.credentials(HTTP_AUTHORIZATION=f"Bearer {login_token}")
        me = self.client.get("/api/auth/me/")
        self.assertEqual(me.status_code, 200)
        self.assertEqual(me.json()["username"], "alice")

        self.assertEqual(self.client.post("/api/auth/logout/", {}, format="json").status_code, 204)
        self.assertEqual(self.client.get("/api/auth/me/").status_code, 401)
        # The first (register) session is still valid: logout is per-session.
        self.client.credentials(HTTP_AUTHORIZATION=f"Token {token}")
        self.assertEqual(self.client.get("/api/auth/me/").status_code, 200)

    def test_logout_all_sessions(self):
        token = self.authenticate()
        self.client.post("/api/auth/login/", {"username": "alice", "password": self.password}, format="json")
        self.assertEqual(UserSession.objects.count(), 2)
        self.client.credentials(HTTP_AUTHORIZATION=f"Bearer {token}")
        self.client.post("/api/auth/logout/", {"all_sessions": True}, format="json")
        self.assertEqual(UserSession.objects.count(), 0)

    def test_wrong_password_and_duplicate_and_weak_password(self):
        self.register("alice")
        res = self.client.post("/api/auth/login/", {"username": "alice", "password": "nope-nope"}, format="json")
        self.assertEqual(res.status_code, 401)
        res = self.client.post("/api/auth/register/", {"username": "ALICE", "password": self.password},
                               format="json")
        self.assertEqual(res.status_code, 400)
        self.assertIn("username", res.json()["errors"])
        res = self.client.post("/api/auth/register/", {"username": "bob", "password": "12345678"}, format="json")
        self.assertEqual(res.status_code, 400)
        self.assertIn("password", res.json()["errors"])

    def test_protected_endpoints_require_token(self):
        for path in ("/api/documents/", "/api/conversations/", "/api/tools/", "/api/auth/me/"):
            res = self.client.get(path)
            self.assertEqual(res.status_code, 401, path)
            self.assertIn("request_id", res.json())
        self.client.credentials(HTTP_AUTHORIZATION="Bearer not-a-token")
        self.assertEqual(self.client.get("/api/documents/").status_code, 401)
        self.client.credentials(HTTP_AUTHORIZATION="Basic abc")
        self.assertEqual(self.client.get("/api/documents/").status_code, 401)

    def test_expired_token_is_rejected_and_deleted(self):
        self.authenticate()
        UserSession.objects.update(expires_at=timezone.now() - timedelta(seconds=1))
        res = self.client.get("/api/auth/me/")
        self.assertEqual(res.status_code, 401)
        self.assertIn("expired", res.json()["detail"])
        self.assertEqual(UserSession.objects.count(), 0)

    def test_post_without_csrf_works_for_token_auth(self):
        from rest_framework.test import APIClient
        client = APIClient(enforce_csrf_checks=True)
        res = client.post("/api/auth/register/", {"username": "dave", "password": self.password}, format="json")
        self.assertEqual(res.status_code, 201)
        client.credentials(HTTP_AUTHORIZATION=f"Bearer {res.json()['token']}")
        self.llm.queue()
        res = client.post("/api/chat/", {"message": "hi"}, format="json")
        self.assertEqual(res.status_code, 200, res.content)

    def test_health_is_public_and_request_id_is_echoed(self):
        res = self.client.get("/api/health/", HTTP_X_REQUEST_ID="abc-123")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res["X-Request-ID"], "abc-123")
        self.assertIn("mcp_transport", res.json())
        self.assertNotIn("ANTHROPIC_API_KEY", str(res.json()))
