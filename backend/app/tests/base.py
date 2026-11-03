from django.core.files.uploadedfile import SimpleUploadedFile
from rest_framework.test import APITestCase

from app.services import container
from app.tests.fakes import FakeEmbeddingService, FakeToolClient, InMemoryVectorStore, ScriptedLLMService

HANDBOOK = (
    "# Acme Handbook\n\n"
    "## Billing\nThe billing service runs nightly at 02:00 UTC and reconciles invoices with the payment "
    "gateway. Refunds above 500 USD require approval from a finance lead.\n\n"
    "## Travel\nEmployees travelling to Pune receive a daily allowance of 4000 INR. International travel "
    "must be booked through the travel desk at least 14 days in advance.\n"
)
ONCALL = (
    "On-call engineers rotate weekly. The on-call engineer must acknowledge pages within 15 minutes. "
    "Escalation goes to the engineering manager after 30 minutes without acknowledgement.\n"
)


class AgentTestCase(APITestCase):
    """Wires the real services with offline fakes for LLM, embeddings, vector store and MCP."""

    password = "Str0ng-Passw0rd!"

    def setUp(self):
        super().setUp()
        self.llm = ScriptedLLMService()
        self.tools = FakeToolClient()
        self.vectors = InMemoryVectorStore()
        container.override(llm_service=self.llm, embedding_service=FakeEmbeddingService(),
                           vector_store=self.vectors, tool_client=self.tools)
        self.addCleanup(container.reset)

    # ---- helpers ----
    def register(self, username="alice"):
        res = self.client.post("/api/auth/register/", {"username": username, "password": self.password},
                               format="json")
        self.assertEqual(res.status_code, 201, res.content)
        return res.json()["token"]

    def authenticate(self, username="alice"):
        token = self.register(username)
        self.client.credentials(HTTP_AUTHORIZATION=f"Bearer {token}")
        return token

    def upload(self, name, content, content_type="text/plain"):
        data = content.encode("utf-8") if isinstance(content, str) else content
        res = self.client.post("/api/documents/",
                               {"files": [SimpleUploadedFile(name, data, content_type=content_type)]},
                               format="multipart")
        return res

    def upload_ok(self, name, content):
        res = self.upload(name, content)
        self.assertEqual(res.status_code, 201, res.content)
        result = res.json()["results"][0]
        self.assertIsNone(result["error"])
        return result["document"]
