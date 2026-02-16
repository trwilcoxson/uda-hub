"""Memory tools that agents use to access and store long-term context.

These tools wire the persistence layer into the agent decision-making loop,
enabling personalized, context-aware support across sessions.
"""
import json
import logging

from langchain_core.tools import tool

from agentic.logging_config import log_structured
from agentic.memory.persistence import (
    load_customer_preferences,
    load_resolutions_for_user,
    save_customer_preference,
    save_resolution,
)

logger = logging.getLogger(__name__)


@tool
def get_customer_context(external_user_id: str) -> str:
    """Retrieve full customer context including past interactions, resolved issues, and preferences.

    Use this tool when you have identified a returning customer to provide
    personalized, context-aware support. This loads cross-session long-term memory.

    Args:
        external_user_id: The CultPass user ID (e.g., 'a4ab87').
    """
    context = {}

    # Load past resolutions
    resolutions = load_resolutions_for_user(external_user_id)
    if resolutions:
        context["past_resolutions"] = [
            {
                "ticket_id": r["ticket_id"],
                "summary": r["summary"],
                "type": r["resolution_type"],
                "agent": r["agent"],
            }
            for r in resolutions[:5]  # Last 5 resolutions
        ]

    # Load customer preferences
    preferences = load_customer_preferences(external_user_id)
    if preferences:
        context["preferences"] = preferences

    if not context:
        context["message"] = "No prior history found for this customer. This appears to be a new customer."

    log_structured(logger, "Customer context loaded for personalization",
                   agent="memory", action="get_customer_context",
                   details={"user_id": external_user_id,
                            "resolutions_count": len(resolutions),
                            "preferences_count": len(preferences)})

    return json.dumps(context)


@tool
def record_customer_preference(external_user_id: str, key: str, value: str) -> str:
    """Save a customer preference for future personalized support.

    Call this when a customer expresses a preference during the conversation
    (e.g., preferred language, contact method, notification preferences).

    Args:
        external_user_id: The CultPass user ID.
        key: Preference key (e.g., 'language', 'contact_method', 'notification_pref').
        value: The preference value.
    """
    save_customer_preference(external_user_id, key, value)

    log_structured(logger, "Customer preference recorded",
                   agent="memory", action="record_preference",
                   details={"user_id": external_user_id, "key": key, "value": value})

    return json.dumps({
        "status": "success",
        "message": f"Saved preference '{key}' = '{value}' for user {external_user_id}",
    })


@tool
def record_resolution(
    ticket_id: str,
    summary: str,
    resolution_type: str,
    agent_name: str,
) -> str:
    """Record how a ticket was resolved for long-term learning.

    Call this after successfully resolving a customer's issue to store
    the resolution for future reference.

    Args:
        ticket_id: The ticket that was resolved.
        summary: Brief description of how the issue was resolved.
        resolution_type: One of 'kb_article', 'action', 'escalation'.
        agent_name: Which agent resolved the ticket.
    """
    save_resolution(
        ticket_id=ticket_id,
        summary=summary,
        agent_name=agent_name,
        resolution_type=resolution_type,
    )

    log_structured(logger, "Ticket resolution recorded",
                   agent="memory", action="record_resolution",
                   ticket_id=ticket_id,
                   details={"type": resolution_type, "agent": agent_name})

    return json.dumps({
        "status": "success",
        "message": f"Resolution recorded for ticket {ticket_id}",
    })
