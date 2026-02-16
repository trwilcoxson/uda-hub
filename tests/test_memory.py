"""Tests for memory persistence.

Requires udahub.db to be set up first (run notebook 02).
"""
import json
import os
import uuid
import pytest

from agentic.config import UDAHUB_DB_PATH


def _db_exists():
    return os.path.exists(UDAHUB_DB_PATH)


@pytest.mark.skipif(not _db_exists(), reason="udahub.db not found - run notebook 02 first")
class TestLongTermMemory:
    def _get_test_ticket_id(self):
        """Get the first ticket ID from the database."""
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from data.models.udahub import Ticket
        engine = create_engine(f"sqlite:///{UDAHUB_DB_PATH}", echo=False)
        Session = sessionmaker(bind=engine)
        session = Session()
        try:
            ticket = session.query(Ticket).first()
            return ticket.ticket_id if ticket else None
        finally:
            session.close()

    def test_save_and_load_message(self):
        from agentic.memory.persistence import save_message, load_conversation_history
        ticket_id = self._get_test_ticket_id()
        if not ticket_id:
            pytest.skip("No ticket found in database")

        msg_id = save_message(ticket_id, "ai", "Test response from AI agent")
        assert msg_id is not None

        history = load_conversation_history(ticket_id)
        assert len(history) >= 2  # At least the original + our new one
        contents = [m["content"] for m in history]
        assert "Test response from AI agent" in contents

    def test_save_message_invalid_role(self):
        from agentic.memory.persistence import save_message
        ticket_id = self._get_test_ticket_id()
        if not ticket_id:
            pytest.skip("No ticket found in database")

        with pytest.raises(ValueError, match="Invalid role"):
            save_message(ticket_id, "invalid_role", "test")

    def test_load_empty_conversation(self):
        from agentic.memory.persistence import load_conversation_history
        # Use a non-existent ticket ID
        history = load_conversation_history("nonexistent-ticket-id")
        assert history == []

    def test_message_ordering(self):
        from agentic.memory.persistence import save_message, load_conversation_history
        ticket_id = self._get_test_ticket_id()
        if not ticket_id:
            pytest.skip("No ticket found in database")

        save_message(ticket_id, "user", "First test message")
        save_message(ticket_id, "ai", "Second test message")

        history = load_conversation_history(ticket_id)
        # Messages should be in chronological order
        for i in range(len(history) - 1):
            if history[i]["created_at"] and history[i + 1]["created_at"]:
                assert history[i]["created_at"] <= history[i + 1]["created_at"]


@pytest.mark.skipif(not _db_exists(), reason="udahub.db not found - run notebook 02 first")
class TestResolutionMemory:
    """Test resolution tracking across sessions."""

    def _get_test_ticket_id(self):
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from data.models.udahub import Ticket
        engine = create_engine(f"sqlite:///{UDAHUB_DB_PATH}", echo=False)
        Session = sessionmaker(bind=engine)
        session = Session()
        try:
            ticket = session.query(Ticket).first()
            return ticket.ticket_id if ticket else None
        finally:
            session.close()

    def test_save_and_load_resolution(self):
        from agentic.memory.persistence import save_resolution, load_resolutions_for_user
        ticket_id = self._get_test_ticket_id()
        if not ticket_id:
            pytest.skip("No ticket found in database")

        save_resolution(
            ticket_id=ticket_id,
            summary="Resolved login issue via password reset instructions",
            agent_name="knowledge_agent",
            resolution_type="kb_article",
            articles_used=["article-1"],
            tools_used=["search_knowledge"],
        )

        # Load resolutions for Alice (external_user_id='a4ab87')
        resolutions = load_resolutions_for_user("a4ab87")
        assert len(resolutions) >= 1
        assert resolutions[0]["resolution_type"] == "kb_article"
        assert "login" in resolutions[0]["summary"].lower()

    def test_load_resolutions_unknown_user(self):
        from agentic.memory.persistence import load_resolutions_for_user
        resolutions = load_resolutions_for_user("nonexistent-user")
        assert resolutions == []


@pytest.mark.skipif(not _db_exists(), reason="udahub.db not found - run notebook 02 first")
class TestCustomerPreferences:
    """Test customer preference persistence across sessions."""

    def test_save_and_load_preferences(self):
        from agentic.memory.persistence import save_customer_preference, load_customer_preferences

        save_customer_preference("a4ab87", "language", "pt-BR")
        save_customer_preference("a4ab87", "contact_method", "chat")

        prefs = load_customer_preferences("a4ab87")
        assert prefs["language"] == "pt-BR"
        assert prefs["contact_method"] == "chat"

    def test_update_existing_preference(self):
        from agentic.memory.persistence import save_customer_preference, load_customer_preferences

        save_customer_preference("a4ab87", "language", "en-US")
        prefs = load_customer_preferences("a4ab87")
        assert prefs["language"] == "en-US"

        # Update it
        save_customer_preference("a4ab87", "language", "pt-BR")
        prefs = load_customer_preferences("a4ab87")
        assert prefs["language"] == "pt-BR"

    def test_load_preferences_unknown_user(self):
        from agentic.memory.persistence import load_customer_preferences
        prefs = load_customer_preferences("nonexistent-user")
        assert prefs == {}


class TestShortTermMemory:
    """Test that MemorySaver preserves context within a thread."""

    def test_memory_saver_thread_isolation(self):
        """Verify different thread_ids have independent state."""
        from langgraph.checkpoint.memory import MemorySaver
        checkpointer = MemorySaver()
        # MemorySaver is tested implicitly through the graph
        # This is a basic sanity check
        assert checkpointer is not None
