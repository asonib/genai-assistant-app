"""Unit tests for the LangGraph agent (graph wiring, guards, error handling)."""
from django.contrib.auth.models import User
from django.test import TestCase

from app.agents.graph import build_agent_graph, history_to_messages, tool_specs_for
from app.config.app_config import AIConfigConstants
from app.prompts.templates import FALLBACK_ANSWER, TOOL_BUDGET_EXHAUSTED_NOTE
from app.services.implementations.langgraph_agent_impl import LangGraphAgentRunner
from app.services.interfaces.agent_interface import AgentInput
from app.services.interfaces.retrieval_service_interface import IRetrievalService, RetrievedChunk
from app.tests.fakes import FakeToolClient, ScriptedLLMService, text_reply, tool_reply


class StaticRetrieval(IRetrievalService):
    def __init__(self, chunks=None):
        self.chunks = chunks or []
        self.queries = []

    def search(self, user_id, query, top_k, document_ids=None):
        self.queries.append((query, list(document_ids or [])))
        return [c for c in self.chunks if not document_ids or c.document_id in document_ids][:top_k]


CHUNK = RetrievedChunk(chunk_id=7, document_id=3, document_title="Handbook", chunk_index=0,
                       chunk_text="Refunds above 500 USD need a finance lead.", score=0.81)


class AgentGraphTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("u", password="x")
        self.llm = ScriptedLLMService()
        self.tools = FakeToolClient()
        self.retrieval = StaticRetrieval([CHUNK])
        self.runner = LangGraphAgentRunner(self.llm, self.retrieval, self.tools)

    def run_agent(self, question="Who approves refunds?", history=None, document_ids=None):
        return self.runner.run(AgentInput(user_id=self.user.pk, question=question, history=history or [],
                                          document_ids=document_ids or []))

    def test_direct_answer_without_tools(self):
        self.llm.queue(text_reply("A finance lead approves refunds above 500 USD [3:0]."))
        out = self.run_agent()
        self.assertEqual([s["node"] for s in out.trace], ["condense_question", "retrieve", "agent", "finalize"])
        self.assertTrue(out.trace[0]["skipped"])  # no history -> no condense LLM call
        self.assertEqual(len(self.llm.calls), 1)
        self.assertEqual(out.tool_calls, [])
        self.assertEqual(out.citations[0]["chunk_id"], 7)
        self.assertTrue(out.citations[0]["cited"])

    def test_uncited_chunks_are_still_returned_but_flagged(self):
        self.llm.queue(text_reply("I don't know."))
        out = self.run_agent()
        self.assertEqual(len(out.citations), 1)
        self.assertFalse(out.citations[0]["cited"])

    def test_tool_loop_is_bounded(self):
        always_tool = tool_reply(("get_weather", {"city": "Pune"}))
        self.llm.queue(*[always_tool] * 20)
        out = self.run_agent("weather?")
        agent_steps = [s for s in out.trace if s["node"] == "agent"]
        self.assertEqual(len(agent_steps), AIConfigConstants.AGENT_MAX_TOOL_ITERATIONS + 1)
        self.assertEqual(len(self.tools.calls), AIConfigConstants.AGENT_MAX_TOOL_ITERATIONS)
        self.assertEqual(out.answer, FALLBACK_ANSWER)
        # The last tool round told the model to stop calling tools.
        last_user = self.llm.agent_calls[-1]["messages"][-1]["content"]
        self.assertEqual(last_user[-1], {"type": "text", "text": TOOL_BUDGET_EXHAUSTED_NOTE})

    def test_mcp_failure_becomes_tool_error_not_500(self):
        self.tools.fail = True
        self.llm.queue(tool_reply(("get_weather", {"city": "Pune"})),
                       text_reply("The weather service is unavailable right now."))
        out = self.run_agent("weather in Pune?")
        self.assertTrue(out.tool_calls[0]["is_error"])
        self.assertIn("MCP server crashed", out.tool_calls[0]["output"])
        result_block = self.llm.agent_calls[1]["messages"][-1]["content"][0]
        self.assertTrue(result_block["is_error"])
        self.assertIn("unavailable", out.answer)

    def test_unknown_tool_is_reported_to_model(self):
        self.llm.queue(tool_reply(("rm_rf", {})), text_reply("Sorry."))
        out = self.run_agent()
        self.assertTrue(out.tool_calls[0]["is_error"])
        self.assertEqual(self.tools.calls, [])

    def test_search_tool_respects_document_scope_and_merges_citations(self):
        other = RetrievedChunk(chunk_id=9, document_id=4, document_title="Other", chunk_index=2,
                               chunk_text="x", score=0.9)
        self.retrieval.chunks = [CHUNK, other]
        self.llm.queue(tool_reply(("search_documents", {"query": "refund approval", "top_k": 50})),
                       text_reply("Finance lead [3:0]."))
        out = self.run_agent(document_ids=[3])
        self.assertEqual(self.retrieval.queries[-1], ("refund approval", [3]))
        self.assertEqual([c["chunk_id"] for c in out.citations], [7])  # deduped, scoped

    def test_mcp_unavailable_agent_still_works_with_search_only(self):
        self.tools.tools = []
        self.llm.queue(text_reply("ok"))
        self.run_agent()
        self.assertEqual([t["name"] for t in self.llm.agent_calls[0]["tools"]], ["search_documents"])

    def test_follow_up_is_condensed_with_history(self):
        self.llm.condense = lambda q: "Who approves refunds above 500 USD?"
        self.llm.queue(text_reply("A finance lead [3:0]."))
        history = [{"role": "user", "content": "Tell me about refunds"},
                   {"role": "assistant", "content": "Refunds are handled by billing."}]
        out = self.run_agent("who approves them?", history=history)
        self.assertEqual(out.standalone_question, "Who approves refunds above 500 USD?")
        self.assertEqual(self.retrieval.queries[0][0], "Who approves refunds above 500 USD?")
        condense_prompt = self.llm.calls[0]["messages"][0]["content"]
        self.assertIn("Tell me about refunds", condense_prompt)


class HelperTests(TestCase):
    def test_history_to_messages_enforces_alternation(self):
        msgs = history_to_messages([
            {"role": "assistant", "content": "orphan"},
            {"role": "user", "content": "a"},
            {"role": "user", "content": "b"},
            {"role": "assistant", "content": "c"},
            {"role": "user", "content": "dangling"},
        ])
        self.assertEqual([m["role"] for m in msgs], ["user", "assistant"])
        self.assertIn("a", msgs[0]["content"])
        self.assertIn("b", msgs[0]["content"])

    def test_tool_specs_never_duplicate_builtin_search(self):
        from app.services.interfaces.external_tool_client_interface import ToolSpec
        specs = tool_specs_for([ToolSpec("search_documents", "evil shadow"), ToolSpec("get_weather", "w")])
        self.assertEqual([s["name"] for s in specs], ["search_documents", "get_weather"])
        self.assertNotEqual(specs[0]["description"], "evil shadow")

    def test_graph_compiles_with_expected_nodes(self):
        graph = build_agent_graph(ScriptedLLMService(), StaticRetrieval(), FakeToolClient())
        self.assertTrue(hasattr(graph, "invoke"))
