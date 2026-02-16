"""Long-term memory persistence layer.

Stores and retrieves conversation history, ticket resolutions, and
customer preferences in udahub.db for cross-session context.
"""
import json
import logging
import uuid
from datetime import datetime, timezone

from agentic.config import ACCOUNT_ID
from agentic.db import udahub_session
from agentic.logging_config import log_structured

logger = logging.getLogger(__name__)


def save_message(ticket_id: str, role: str, content: str) -> str:
    """Save a message to the TicketMessage table for long-term persistence.

    Args:
        ticket_id: The ticket this message belongs to.
        role: One of 'user', 'agent', 'ai', 'system'.
        content: The message content.

    Returns:
        The message_id of the saved message.
    """
    from data.models.udahub import TicketMessage, RoleEnum

    role_map = {
        "user": RoleEnum.user,
        "agent": RoleEnum.agent,
        "ai": RoleEnum.ai,
        "system": RoleEnum.system,
    }

    if role not in role_map:
        raise ValueError(f"Invalid role: {role}. Must be one of {list(role_map.keys())}")

    with udahub_session() as session:
        message_id = str(uuid.uuid4())
        msg = TicketMessage(
            message_id=message_id,
            ticket_id=ticket_id,
            role=role_map[role],
            content=content,
        )
        session.add(msg)
        log_structured(logger, "Message saved to long-term memory",
                       agent="memory", action="save_message",
                       ticket_id=ticket_id,
                       details={"message_id": message_id, "role": role})
        return message_id


def load_conversation_history(ticket_id: str) -> list[dict]:
    """Load all messages for a ticket, ordered by creation time.

    Args:
        ticket_id: The ticket to load messages for.

    Returns:
        List of dicts with keys: message_id, role, content, created_at.
    """
    from data.models.udahub import TicketMessage

    with udahub_session() as session:
        messages = (
            session.query(TicketMessage)
            .filter_by(ticket_id=ticket_id)
            .order_by(TicketMessage.created_at)
            .all()
        )
        result = [
            {
                "message_id": msg.message_id,
                "role": msg.role.value if msg.role else "unknown",
                "content": msg.content,
                "created_at": msg.created_at.isoformat() if msg.created_at else None,
            }
            for msg in messages
        ]
        log_structured(logger, "Conversation history loaded",
                       agent="memory", action="load_conversation_history",
                       ticket_id=ticket_id,
                       details={"message_count": len(result)})
        return result


def save_resolution(
    ticket_id: str,
    summary: str,
    agent_name: str,
    resolution_type: str,
    articles_used: list[str] | None = None,
    tools_used: list[str] | None = None,
) -> None:
    """Save a ticket resolution record for long-term knowledge.

    Args:
        ticket_id: The resolved ticket ID.
        summary: Brief description of how the issue was resolved.
        agent_name: Which agent resolved the ticket.
        resolution_type: One of 'kb_article', 'action', 'escalation'.
        articles_used: List of article IDs that were cited.
        tools_used: List of tool names that were invoked.
    """
    from data.models.udahub import TicketResolution

    with udahub_session() as session:
        resolution = TicketResolution(
            ticket_id=ticket_id,
            resolution_summary=summary,
            resolution_agent=agent_name,
            resolution_type=resolution_type,
            articles_used=json.dumps(articles_used or []),
            tools_used=json.dumps(tools_used or []),
        )
        session.merge(resolution)
        log_structured(logger, "Ticket resolution saved",
                       agent="memory", action="save_resolution",
                       ticket_id=ticket_id,
                       details={"resolution_type": resolution_type, "agent": agent_name})


def load_resolutions_for_user(external_user_id: str) -> list[dict]:
    """Load past ticket resolutions for a user, enabling personalized support.

    Args:
        external_user_id: The CultPass user ID.

    Returns:
        List of past resolutions with summary, type, and date.
    """
    from data.models.udahub import TicketResolution, Ticket, User

    with udahub_session() as session:
        user = session.query(User).filter_by(external_user_id=external_user_id).first()
        if not user:
            return []

        resolutions = (
            session.query(TicketResolution)
            .join(Ticket, TicketResolution.ticket_id == Ticket.ticket_id)
            .filter(Ticket.user_id == user.user_id)
            .order_by(TicketResolution.created_at.desc())
            .all()
        )

        result = [
            {
                "ticket_id": res.ticket_id,
                "summary": res.resolution_summary,
                "resolution_type": res.resolution_type,
                "agent": res.resolution_agent,
                "articles_used": json.loads(res.articles_used) if res.articles_used else [],
                "tools_used": json.loads(res.tools_used) if res.tools_used else [],
                "created_at": res.created_at.isoformat() if res.created_at else None,
            }
            for res in resolutions
        ]

        log_structured(logger, "Past resolutions loaded for user",
                       agent="memory", action="load_resolutions",
                       details={"external_user_id": external_user_id, "count": len(result)})
        return result


def save_customer_preference(
    external_user_id: str,
    key: str,
    value: str,
) -> None:
    """Save or update a customer preference for personalized cross-session support.

    Args:
        external_user_id: The CultPass user ID.
        key: Preference key (e.g., 'language', 'contact_method').
        value: Preference value.
    """
    from data.models.udahub import CustomerPreference

    with udahub_session() as session:
        existing = (
            session.query(CustomerPreference)
            .filter_by(
                external_user_id=external_user_id,
                account_id=ACCOUNT_ID,
                preference_key=key,
            )
            .first()
        )

        if existing:
            existing.preference_value = value
            existing.updated_at = datetime.now(timezone.utc)
        else:
            pref = CustomerPreference(
                preference_id=str(uuid.uuid4()),
                external_user_id=external_user_id,
                account_id=ACCOUNT_ID,
                preference_key=key,
                preference_value=value,
            )
            session.add(pref)

        log_structured(logger, "Customer preference saved",
                       agent="memory", action="save_preference",
                       details={"user": external_user_id, "key": key})


def load_customer_preferences(external_user_id: str) -> dict[str, str]:
    """Load all preferences for a customer.

    Args:
        external_user_id: The CultPass user ID.

    Returns:
        Dict mapping preference_key -> preference_value.
    """
    from data.models.udahub import CustomerPreference

    with udahub_session() as session:
        prefs = (
            session.query(CustomerPreference)
            .filter_by(external_user_id=external_user_id, account_id=ACCOUNT_ID)
            .all()
        )
        result = {p.preference_key: p.preference_value for p in prefs}
        log_structured(logger, "Customer preferences loaded",
                       agent="memory", action="load_preferences",
                       details={"user": external_user_id, "count": len(result)})
        return result
