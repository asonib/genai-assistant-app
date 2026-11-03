"""
End-to-end agent flow through the public HTTP API (Assignment 3B "Tests").

register -> upload 2 docs -> chat turn 1 (agent calls search_documents + MCP get_weather,
answers with citations) -> chat turn 2 (memory: follow-up is condensed using history and the
previous turn is replayed) -> history endpoints -> logout revokes the token.

The real LangGraph graph, services, ORM and controllers run; only Claude, the embedding model,
FAISS and the MCP server are replaced by deterministic offline fakes.
"""
import re

from app.models import ChatMessage
from app.tests.base import HANDBOOK, ONCALL, AgentTestCase
from app.tests.fakes import text_reply, tool_reply


def _answer_citing_first_source(system, messages, tools):
    """Final agent step: cite a chunk that the tools/retrieval actually returned."""
    transcript = str(messages)
    tag = re.search(r'tag="(\[\d+:\d+\])"', transcript).group(1)
    return text_reply(f"Travellers to Pune get 4000 INR per day {tag}. It is 29.5°C and clear (via weather tool).")


class AgentEndToEndTests(AgentTestCase):
    def test_multi_step_agent_flow_with_rag_mcp_memory_and_history(self):
        self.authenticate()
        handbook = self.upload_ok("handbook.txt", HANDBOOK)
        oncall = self.upload_ok("oncall.txt", ONCALL)
        self.assertEqual(handbook["ingestion_status"], "ingested")
        self.assertGreaterEqual(handbook["chunk_count"], 1)
        # Multi-document indexing: both documents live in the same per-user index.
        self.assertEqual(self.vectors.count("user_1"),
                         handbook["chunk_count"] + oncall["chunk_count"])

        # ---- turn 1: agent -> tools (search + MCP weather) -> agent -> answer ----
        self.llm.queue(
            tool_reply(("search_documents", {"query": "Pune travel daily allowance"}),
                       ("get_weather", {"city": "Pune"}), text="Let me check."),
            _answer_citing_first_source,
        )
        res = self.client.post("/api/chat/", {"message": "I'm travelling to Pune. What allowance do I get "
                                                         "and what's the weather like?"}, format="json")
        self.assertEqual(res.status_code, 200, res.content)
        body = res.json()
        conversation_id = body["conversation"]["id"]
        answer = body["assistant_message"]

        self.assertIn("4000 INR", answer["content"])
        self.assertEqual(body["conversation"]["message_count"], 2)
        # External tool went through the MCP client.
        self.assertEqual(self.tools.calls, [("get_weather", {"city": "Pune"})])
        names = [(t["name"], t["source"], t["is_error"]) for t in answer["tool_calls"]]
        self.assertEqual(names, [("search_documents", "builtin", False), ("get_weather", "mcp", False)])
        # Source citations: retrieved chunks are returned alongside the answer, cited one flagged.
        self.assertTrue(answer["citations"])
        cited = [c for c in answer["citations"] if c["cited"]]
        self.assertEqual(len(cited), 1)
        self.assertEqual(cited[0]["document_id"], handbook["id"])
        self.assertIn("Pune", cited[0]["chunk_text"])
        self.assertEqual(answer["citations"][0], cited[0])  # cited chunks first
        # Multi-step graph trace.
        nodes = [s["node"] for s in answer["agent_trace"]]
        self.assertEqual(nodes, ["condense_question", "retrieve", "agent", "tools", "agent", "finalize"])
        self.assertEqual(answer["prompt_template"], "assistant_agent_v1")
        self.assertEqual(answer["input_tokens"], 30)

        # The model was offered the built-in search tool and the MCP tools.
        offered = [t["name"] for t in self.llm.agent_calls[0]["tools"]]
        self.assertEqual(offered, ["search_documents", "get_weather", "convert_currency"])
        # Second agent call received the tool results (Anthropic tool_result blocks).
        tool_results = self.llm.agent_calls[1]["messages"][-1]["content"]
        self.assertEqual([b["type"] for b in tool_results], ["tool_result", "tool_result"])
        self.assertIn("29.5", tool_results[1]["content"])
        # User input never reaches the model raw: it's wrapped by the prompt template.
        first_user_turn = self.llm.agent_calls[0]["messages"][0]["content"]
        self.assertIn("<user_question>", first_user_turn)
        self.assertIn("<retrieved_context>", first_user_turn)

        # ---- turn 2: memory ----
        self.llm.condense = lambda q: "What is the on-call acknowledgement time for engineers?"
        self.llm.queue(lambda s, m, t: text_reply("Pages must be acknowledged within 15 minutes."))
        res = self.client.post("/api/chat/", {"message": "And how fast must they acknowledge pages?",
                                              "conversation_id": conversation_id}, format="json")
        self.assertEqual(res.status_code, 200, res.content)
        turn2 = res.json()["assistant_message"]
        self.assertEqual(turn2["agent_trace"][0]["standalone_question"],
                         "What is the on-call acknowledgement time for engineers?")
        # Retrieval used the condensed question -> on-call document found.
        self.assertIn(oncall["id"], [c["document_id"] for c in turn2["citations"]])
        # The previous turn was replayed to the LLM as conversation memory.
        replay = self.llm.agent_calls[-1]["messages"]
        self.assertEqual([m["role"] for m in replay], ["user", "assistant", "user"])
        self.assertIn("<previous_user_message>", replay[0]["content"])
        self.assertIn("4000 INR", replay[1]["content"])

        # ---- history ----
        res = self.client.get("/api/conversations/")
        self.assertEqual(res.json()["count"], 1)
        self.assertEqual(res.json()["results"][0]["message_count"], 4)
        detail = self.client.get(f"/api/conversations/{conversation_id}/").json()
        self.assertEqual([m["role"] for m in detail["messages"]], ["user", "assistant", "user", "assistant"])
        self.assertEqual(ChatMessage.objects.count(), 4)

        # ---- logout revokes the token ----
        self.assertEqual(self.client.post("/api/auth/logout/", {}, format="json").status_code, 204)
        self.assertEqual(self.client.get("/api/conversations/").status_code, 401)

    def test_document_scope_restricts_retrieval(self):
        self.authenticate()
        handbook = self.upload_ok("handbook.txt", HANDBOOK)
        oncall = self.upload_ok("oncall.txt", ONCALL)
        self.llm.queue(text_reply("ok"))
        res = self.client.post("/api/chat/", {"message": "on-call engineers acknowledge pages Pune travel",
                                              "document_ids": [oncall["id"]]}, format="json")
        self.assertEqual(res.status_code, 200, res.content)
        docs = {c["document_id"] for c in res.json()["assistant_message"]["citations"]}
        self.assertEqual(docs, {oncall["id"]})
        self.assertNotIn(handbook["id"], docs)

    def test_scope_with_foreign_document_is_rejected(self):
        self.authenticate("bob")
        bob_doc = self.upload_ok("handbook.txt", HANDBOOK)
        self.client.credentials()
        self.authenticate("carol")
        res = self.client.post("/api/chat/", {"message": "hi", "document_ids": [bob_doc["id"]]}, format="json")
        self.assertEqual(res.status_code, 400)
        self.assertIn("document_ids", res.json()["errors"])

    def test_users_cannot_read_each_others_conversations(self):
        self.authenticate("bob")
        self.llm.queue(text_reply("hello"))
        conv_id = self.client.post("/api/chat/", {"message": "hi"}, format="json").json()["conversation"]["id"]
        self.client.credentials()
        self.authenticate("carol")
        self.assertEqual(self.client.get(f"/api/conversations/{conv_id}/").status_code, 404)
        res = self.client.post("/api/chat/", {"message": "hi", "conversation_id": conv_id}, format="json")
        self.assertEqual(res.status_code, 404)

    def test_failed_turn_is_not_persisted(self):
        from app.exceptions.custom_exceptions import LLMServiceException

        self.authenticate()

        def boom(system, messages, tools):
            raise LLMServiceException("upstream down")

        self.llm.queue(boom)
        res = self.client.post("/api/chat/", {"message": "hello"}, format="json")
        self.assertEqual(res.status_code, 502)
        self.assertEqual(res.json()["code"], "llm_upstream_error")
        self.assertEqual(ChatMessage.objects.count(), 0)

    def test_tools_endpoint_lists_builtin_and_mcp_tools(self):
        self.authenticate()
        res = self.client.get("/api/tools/")
        self.assertEqual(res.status_code, 200)
        self.assertEqual([(t["name"], t["source"]) for t in res.json()["results"]],
                         [("search_documents", "builtin"), ("get_weather", "mcp"), ("convert_currency", "mcp")])
