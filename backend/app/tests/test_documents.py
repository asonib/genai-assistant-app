import unittest

from django.core.files.uploadedfile import SimpleUploadedFile

from app.config.app_config import AppConfigConstants
from app.models import Document, DocumentChunk
from app.services import container
from app.tests.base import HANDBOOK, ONCALL, AgentTestCase

try:
    import pypdf  # noqa: F401
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False


def make_pdf(lines):
    """Build a tiny valid single-page PDF with a text layer (no external deps)."""
    text_ops = " ".join(f"({line}) Tj 0 -16 Td" for line in lines)
    stream = f"BT /F1 12 Tf 72 720 Td {text_ops} ET".encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref)
    return bytes(out)


class DocumentTests(AgentTestCase):
    def setUp(self):
        super().setUp()
        self.authenticate()

    def test_txt_upload_is_extracted_chunked_and_indexed(self):
        doc = self.upload_ok("handbook.txt", HANDBOOK)
        self.assertEqual(doc["file_type"], "txt")
        self.assertEqual(doc["ingestion_status"], "ingested")
        ids = list(DocumentChunk.objects.filter(document_id=doc["id"]).values_list("id", flat=True))
        self.assertEqual(sorted(self.vectors.data["user_1"].keys()), sorted(ids))  # vector id == chunk pk

    @unittest.skipUnless(HAS_PYPDF, "pypdf not installed")
    def test_pdf_upload_extracts_text_layer(self):
        pdf = make_pdf(["Quarterly security review policy.", "All laptops must use disk encryption."])
        res = self.upload("policy.pdf", pdf, "application/pdf")
        self.assertEqual(res.status_code, 201, res.content)
        doc = res.json()["results"][0]["document"]
        self.assertEqual((doc["file_type"], doc["page_count"], doc["ingestion_status"]), ("pdf", 1, "ingested"))
        content = Document.objects.get(pk=doc["id"]).content
        self.assertIn("[Page 1]", content)
        self.assertIn("disk encryption", content)

    def test_multi_file_upload(self):
        res = self.client.post("/api/documents/", {"files": [
            SimpleUploadedFile("a.txt", HANDBOOK.encode()), SimpleUploadedFile("b.txt", ONCALL.encode())]},
            format="multipart")
        self.assertEqual(res.status_code, 201, res.content)
        self.assertEqual([r["document"]["title"] for r in res.json()["results"]], ["a", "b"])
        self.assertEqual(self.client.get("/api/documents/").json()["count"], 2)

    def test_rejects_unsupported_type(self):
        res = self.upload("notes.docx", "hello")
        self.assertEqual(res.status_code, 400)
        self.assertEqual(res.json()["code"], "invalid_file_type")
        self.assertEqual(Document.objects.count(), 0)

    def test_bad_file_rejects_whole_batch(self):
        res = self.client.post("/api/documents/", {"files": [
            SimpleUploadedFile("ok.txt", b"fine text"), SimpleUploadedFile("bad.exe", b"MZ")]}, format="multipart")
        self.assertEqual(res.status_code, 400)
        self.assertEqual(Document.objects.count(), 0)

    def test_rejects_too_large(self):
        big = "a" * (AppConfigConstants.MAX_FILE_SIZE_BYTES + 1)
        res = self.upload("big.txt", big)
        self.assertEqual(res.status_code, 413)

    def test_rejects_fake_pdf_and_empty_file(self):
        self.assertEqual(self.upload("fake.pdf", b"not a pdf", "application/pdf").status_code, 422)
        self.assertEqual(self.upload("empty.txt", b"").status_code, 422)
        self.assertEqual(self.upload("blank.txt", b"   \n\n  ").status_code, 422)

    def test_no_files(self):
        res = self.client.post("/api/documents/", {}, format="multipart")
        self.assertEqual(res.status_code, 400)

    def test_duplicate_upload_is_detected(self):
        first = self.upload_ok("handbook.txt", HANDBOOK)
        res = self.upload("handbook-copy.txt", HANDBOOK)
        result = res.json()["results"][0]
        self.assertTrue(result["duplicate"])
        self.assertEqual(result["document"]["id"], first["id"])
        self.assertEqual(Document.objects.count(), 1)

    def test_delete_removes_vectors_and_isolation_between_users(self):
        doc = self.upload_ok("handbook.txt", HANDBOOK)
        self.client.credentials()
        self.authenticate("mallory")
        self.assertEqual(self.client.get(f"/api/documents/{doc['id']}/").status_code, 404)
        self.assertEqual(self.client.delete(f"/api/documents/{doc['id']}/").status_code, 404)

        self.client.credentials()
        token = self.client.post("/api/auth/login/", {"username": "alice", "password": self.password},
                                 format="json").json()["token"]
        self.client.credentials(HTTP_AUTHORIZATION=f"Bearer {token}")
        self.assertEqual(self.client.delete(f"/api/documents/{doc['id']}/").status_code, 204)
        self.assertEqual(self.vectors.count("user_1"), 0)
        self.assertEqual(DocumentChunk.objects.count(), 0)

    def test_failed_ingestion_keeps_document_and_reingest_recovers(self):
        from app.exceptions.custom_exceptions import EmbeddingServiceException
        from app.tests.fakes import FakeEmbeddingService

        class BrokenEmbeddings(FakeEmbeddingService):
            def embed_documents(self, texts):
                raise EmbeddingServiceException("model download failed")

        container.override(embedding_service=BrokenEmbeddings())
        res = self.upload("handbook.txt", HANDBOOK)
        self.assertEqual(res.status_code, 201)
        result = res.json()["results"][0]
        self.assertEqual(result["document"]["ingestion_status"], "failed")
        self.assertIn("model download failed", result["error"])

        container.override(embedding_service=FakeEmbeddingService())
        res = self.client.post(f"/api/documents/{result['document']['id']}/reingest/")
        self.assertEqual(res.status_code, 200, res.content)
        self.assertEqual(res.json()["status"], "ingested")
        self.assertEqual(Document.objects.get().ingestion_error, "")

    def test_detail_has_preview_but_list_does_not_leak_content(self):
        doc = self.upload_ok("handbook.txt", HANDBOOK)
        listing = self.client.get("/api/documents/").json()["results"][0]
        self.assertNotIn("content", listing)
        detail = self.client.get(f"/api/documents/{doc['id']}/").json()
        self.assertIn("Billing", detail["content_preview"])
