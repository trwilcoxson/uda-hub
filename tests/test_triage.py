"""Tests for the Triage Agent's LLM-powered classification tool.

These tests call the OpenAI API to verify that the LLM classifies
customer messages into the correct structured categories.
Requires OPENAI_API_KEY in the environment.
"""
import json
import os

import pytest

from agentic.tools.classification_tools import classify_ticket, TicketClassification


pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY required for LLM classification tests",
)


class TestClassifyTicket:
    """Verify the LLM produces correct structured classifications."""

    # --- issue_type ---

    def test_account_issue_classification(self):
        result = json.loads(classify_ticket.invoke({"message": "I can't log in to my account"}))
        assert result["issue_type"] == "account"

    def test_subscription_issue_classification(self):
        result = json.loads(classify_ticket.invoke({"message": "How do I cancel my subscription?"}))
        assert result["issue_type"] == "subscription"

    def test_reservation_issue_classification(self):
        result = json.loads(classify_ticket.invoke({"message": "I want to book a reservation for the concert"}))
        assert result["issue_type"] == "reservation"

    def test_billing_issue_classification(self):
        result = json.loads(classify_ticket.invoke({"message": "I need a refund for my last payment"}))
        assert result["issue_type"] == "billing"

    def test_technical_issue_classification(self):
        result = json.loads(classify_ticket.invoke({"message": "The app keeps crashing when I open it"}))
        assert result["issue_type"] == "technical"

    def test_general_issue_classification(self):
        result = json.loads(classify_ticket.invoke({"message": "What time does the office close?"}))
        assert result["issue_type"] == "general"

    # --- priority ---

    def test_high_priority(self):
        result = json.loads(classify_ticket.invoke({"message": "My account is blocked and I can't access anything"}))
        assert result["priority"] in ("high", "urgent")

    def test_urgent_priority(self):
        result = json.loads(classify_ticket.invoke({"message": "I need help ASAP, this is urgent!"}))
        assert result["priority"] == "urgent"

    def test_low_priority(self):
        result = json.loads(classify_ticket.invoke({"message": "How to book an experience?"}))
        assert result["priority"] == "low"

    # --- sentiment ---

    def test_frustrated_sentiment(self):
        result = json.loads(classify_ticket.invoke({"message": "This is terrible!!! Unacceptable service!!!"}))
        assert result["sentiment"] == "frustrated"

    def test_negative_sentiment(self):
        result = json.loads(classify_ticket.invoke({"message": "My subscription is not working properly"}))
        assert result["sentiment"] in ("negative", "frustrated")

    def test_positive_sentiment(self):
        result = json.loads(classify_ticket.invoke({"message": "Thank you for the great experience!"}))
        assert result["sentiment"] == "positive"

    # --- escalation ---

    def test_escalation_required(self):
        result = json.loads(classify_ticket.invoke({"message": "I want to speak to a real person immediately"}))
        assert result["requires_human"] is True

    def test_no_escalation_for_simple_query(self):
        result = json.loads(classify_ticket.invoke({"message": "How do I reserve an event?"}))
        assert result["requires_human"] is False

    # --- structural validity ---

    def test_classification_returns_valid_model(self):
        result = json.loads(classify_ticket.invoke({"message": "Help me with my subscription"}))
        classification = TicketClassification(**result)
        assert classification.issue_type in ("account", "subscription", "reservation", "billing", "technical", "general")
        assert classification.priority in ("low", "medium", "high", "urgent")
        assert classification.sentiment in ("positive", "neutral", "negative", "frustrated")

    def test_summary_is_descriptive(self):
        """The LLM should generate a meaningful summary, not a template."""
        result = json.loads(classify_ticket.invoke({
            "message": "I booked a samba class for Saturday but now I need to cancel because I'm traveling"
        }))
        summary = result["summary"].lower()
        # Should reference the actual content, not just "Reservation issue with medium priority"
        assert any(w in summary for w in ["cancel", "samba", "book", "reservation", "travel"])
