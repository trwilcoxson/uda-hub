"""Ticket classification tool using LLM-powered structured output."""
import logging
from typing import Literal

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from agentic.config import LLM_MODEL
from agentic.db import udahub_session
from agentic.logging_config import log_structured

logger = logging.getLogger(__name__)

CLASSIFICATION_PROMPT = (
    "You are a ticket classification engine for CultPass, a Brazilian cultural "
    "experiences subscription service. Analyze the customer message and produce "
    "a structured classification.\n\n"
    "Definitions:\n"
    "- issue_type: 'account' (login, password, profile, blocked), "
    "'subscription' (plan changes, pause, cancel, upgrade, downgrade), "
    "'reservation' (booking, events, experiences, spots), "
    "'billing' (refund, payment, charges, invoices), "
    "'technical' (app bugs, crashes, QR codes, errors), "
    "'general' (anything else)\n"
    "- priority: 'urgent' (needs immediate attention, safety, legal), "
    "'high' (service blocked, refund needed, access lost), "
    "'medium' (standard request), "
    "'low' (informational question, how-to)\n"
    "- sentiment: 'frustrated' (anger, exclamation, threats), "
    "'negative' (dissatisfaction, complaint), "
    "'positive' (gratitude, praise), "
    "'neutral' (factual, no strong emotion)\n"
    "- requires_human: true when the customer explicitly asks for a human/manager, "
    "mentions legal action or discrimination, or is frustrated with urgent/high priority\n"
    "- summary: a concise one-sentence summary of what the customer needs\n\n"
    "Customer message:\n{message}"
)


class TicketClassification(BaseModel):
    """Structured classification result for a support ticket."""
    issue_type: Literal["account", "subscription", "reservation", "billing", "technical", "general"] = Field(
        description="Category: account, subscription, reservation, billing, technical, or general"
    )
    priority: Literal["low", "medium", "high", "urgent"] = Field(
        description="Priority: low, medium, high, or urgent"
    )
    sentiment: Literal["positive", "neutral", "negative", "frustrated"] = Field(
        description="Customer sentiment: positive, neutral, negative, or frustrated"
    )
    requires_human: bool = Field(
        default=False,
        description="Whether this ticket should be escalated to a human agent"
    )
    summary: str = Field(
        description="Brief one-sentence summary of the customer's issue"
    )


# Module-level classifier (lazy-initialized to avoid import-time API calls)
_classifier = None


def _get_classifier():
    global _classifier
    if _classifier is None:
        _classifier = ChatOpenAI(
            model=LLM_MODEL, temperature=0
        ).with_structured_output(TicketClassification)
    return _classifier


@tool
def classify_ticket(
    message: str,
    ticket_id: str | None = None,
) -> str:
    """Classify a customer support ticket using LLM-powered analysis.

    Sends the customer message to a language model that returns a structured
    classification with issue_type, priority, sentiment, requires_human flag,
    and a natural-language summary.
    If ticket_id is provided, updates the TicketMetadata in udahub.db.

    Args:
        message: The customer's support message to classify.
        ticket_id: Optional ticket ID to update in the database.
    """
    classifier = _get_classifier()
    classification: TicketClassification = classifier.invoke(
        CLASSIFICATION_PROMPT.format(message=message)
    )

    # Update ticket metadata in DB if ticket_id provided
    if ticket_id:
        try:
            from data.models.udahub import TicketMetadata
            with udahub_session() as session:
                meta = session.query(TicketMetadata).filter_by(ticket_id=ticket_id).first()
                if meta:
                    meta.main_issue_type = classification.issue_type
                    meta.tags = f"{classification.issue_type}, {classification.priority}, {classification.sentiment}"
                    log_structured(logger, "Updated TicketMetadata in database",
                                   agent="triage_agent", action="update_metadata",
                                   ticket_id=ticket_id)
        except Exception as e:
            log_structured(logger, f"Failed to update ticket metadata: {e}",
                           agent="triage_agent", action="update_metadata_error",
                           ticket_id=ticket_id or "", level=logging.ERROR)

    log_structured(logger, "Ticket classified",
                   agent="triage_agent", action="classify_ticket",
                   ticket_id=ticket_id or "",
                   details=classification.model_dump())
    return classification.model_dump_json()
