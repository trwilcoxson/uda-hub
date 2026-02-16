"""Account lookup tools for the Account Agent (read-only against cultpass.db)."""
import json
import logging

from langchain_core.tools import tool

from agentic.db import cultpass_session
from agentic.logging_config import log_structured

logger = logging.getLogger(__name__)


def _serialize_dt(obj):
    """Convert datetime to ISO string; pass through everything else."""
    return obj.isoformat() if hasattr(obj, "isoformat") else obj


@tool
def lookup_user(email: str) -> str:
    """Look up a CultPass user by their email address.

    Returns user profile information including account status.

    Args:
        email: The email address of the user to look up.
    """
    from data.models.cultpass import User

    with cultpass_session() as session:
        user = session.query(User).filter_by(email=email).first()
        if not user:
            return json.dumps({"error": f"No user found with email: {email}"})

        result = {
            "user_id": user.user_id,
            "full_name": user.full_name,
            "email": user.email,
            "is_blocked": user.is_blocked,
            "created_at": _serialize_dt(user.created_at),
        }
        log_structured(logger, "User lookup completed",
                       agent="account_agent", action="lookup_user",
                       details={"email": email, "user_id": user.user_id, "is_blocked": user.is_blocked})
        return json.dumps(result)


@tool
def get_subscription(user_id: str) -> str:
    """Get subscription details for a CultPass user.

    Args:
        user_id: The user ID to look up subscription for.
    """
    from data.models.cultpass import Subscription

    with cultpass_session() as session:
        sub = session.query(Subscription).filter_by(user_id=user_id).first()
        if not sub:
            return json.dumps({"error": f"No subscription found for user: {user_id}"})

        result = {
            "subscription_id": sub.subscription_id,
            "user_id": sub.user_id,
            "status": sub.status,
            "tier": sub.tier,
            "monthly_quota": sub.monthly_quota,
            "started_at": _serialize_dt(sub.started_at),
            "ended_at": _serialize_dt(sub.ended_at),
        }
        log_structured(logger, "Subscription lookup completed",
                       agent="account_agent", action="get_subscription",
                       details={"user_id": user_id, "status": sub.status, "tier": sub.tier})
        return json.dumps(result)


@tool
def get_reservations(user_id: str) -> str:
    """Get all reservations for a CultPass user.

    Args:
        user_id: The user ID to look up reservations for.
    """
    from data.models.cultpass import Reservation, Experience

    with cultpass_session() as session:
        reservations = session.query(Reservation).filter_by(user_id=user_id).all()
        if not reservations:
            return json.dumps({"reservations": [], "message": f"No reservations found for user: {user_id}"})

        results = []
        for res in reservations:
            exp = session.query(Experience).filter_by(experience_id=res.experience_id).first()
            results.append({
                "reservation_id": res.reservation_id,
                "experience_id": res.experience_id,
                "experience_title": exp.title if exp else "Unknown",
                "status": res.status,
                "created_at": _serialize_dt(res.created_at),
            })

        log_structured(logger, "Reservations lookup completed",
                       agent="account_agent", action="get_reservations",
                       details={"user_id": user_id, "count": len(results)})
        return json.dumps({"reservations": results})
