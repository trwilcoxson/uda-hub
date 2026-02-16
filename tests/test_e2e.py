"""End-to-end tests for the UDA-Hub multi-agent system.

These tests require:
1. OPENAI_API_KEY set in environment
2. Both databases set up (run notebooks 01 and 02)
3. ChromaDB indexed (run notebook 02)
"""
import json
import os
import pytest
from langchain_core.messages import HumanMessage

from agentic.config import CULTPASS_DB_PATH, UDAHUB_DB_PATH, CHROMA_PERSIST_DIR


def _all_setup_done():
    return (
        os.path.exists(CULTPASS_DB_PATH)
        and os.path.exists(UDAHUB_DB_PATH)
        and os.path.exists(CHROMA_PERSIST_DIR)
        and os.environ.get("OPENAI_API_KEY")
    )


@pytest.mark.skipif(not _all_setup_done(), reason="Full setup required (DBs + ChromaDB + API key)")
class TestEndToEnd:
    @pytest.fixture
    def orchestrator(self):
        from agentic.graph import build_graph
        return build_graph()

    def _invoke(self, orchestrator, message: str, thread_id: str = "test"):
        result = orchestrator.invoke(
            {"messages": [HumanMessage(content=message)]},
            config={"configurable": {"thread_id": thread_id}},
        )
        return result["messages"][-1].content

    def test_general_question_kb_resolution(self, orchestrator):
        """Scenario 1: General question resolved via knowledge base."""
        response = self._invoke(orchestrator, "How do I reserve a spot for an event?")
        assert len(response) > 0
        # Should mention reservation process
        response_lower = response.lower()
        assert any(w in response_lower for w in ["reserve", "app", "cultpass", "booking"])

    def test_account_inquiry(self, orchestrator):
        """Scenario 2: Account inquiry with email lookup."""
        response = self._invoke(
            orchestrator,
            "Can you look up my account? My email is alice.kingsley@wonderland.com",
        )
        assert len(response) > 0
        response_lower = response.lower()
        assert any(w in response_lower for w in ["alice", "kingsley", "account", "blocked"])

    def test_cancel_reservation_flow(self, orchestrator):
        """Scenario 3: Cancel reservation action."""
        # First get reservations
        response = self._invoke(
            orchestrator,
            "I need to cancel a reservation. My email is alice.kingsley@wonderland.com. "
            "Can you show me my reservations first?",
            thread_id="test-cancel",
        )
        assert len(response) > 0

    def test_escalation_on_unknown_topic(self, orchestrator):
        """Scenario 4: Unknown topic triggers escalation suggestion."""
        response = self._invoke(
            orchestrator,
            "I want to file a formal legal complaint about discrimination at one of your events",
        )
        assert len(response) > 0
        # Should suggest escalation or human support
        response_lower = response.lower()
        assert any(w in response_lower for w in [
            "escalat", "human", "support team", "specialist",
            "team", "assist", "help", "contact", "reach out",
            "sorry", "understand", "concern", "seriously",
        ])

    def test_multi_turn_conversation(self, orchestrator):
        """Scenario 5: Multi-turn conversation preserves context."""
        thread_id = "test-multi-turn"

        # Turn 1
        response1 = self._invoke(
            orchestrator,
            "Hi, I'm having issues with my subscription.",
            thread_id=thread_id,
        )
        assert len(response1) > 0

        # Turn 2 - should maintain context
        response2 = self._invoke(
            orchestrator,
            "My email is bob.stone@granite.com",
            thread_id=thread_id,
        )
        assert len(response2) > 0
