"""Action tools for the Action Agent (write operations against cultpass.db)."""
import json
import logging
import uuid
from datetime import datetime, timezone

from langchain_core.tools import tool

from agentic.db import cultpass_session
from agentic.logging_config import log_structured

logger = logging.getLogger(__name__)


@tool
def cancel_reservation(reservation_id: str) -> str:
    """Cancel a CultPass reservation by its ID.

    Updates the reservation status to 'cancelled'.

    Args:
        reservation_id: The ID of the reservation to cancel.
    """
    from data.models.cultpass import Reservation

    with cultpass_session() as session:
        reservation = session.query(Reservation).filter_by(
            reservation_id=reservation_id
        ).first()

        if not reservation:
            return json.dumps({"error": f"Reservation {reservation_id} not found"})

        if reservation.status == "cancelled":
            return json.dumps({"error": f"Reservation {reservation_id} is already cancelled"})

        reservation.status = "cancelled"
        reservation.updated_at = datetime.now(timezone.utc)

        log_structured(logger, "Reservation cancelled",
                       agent="action_agent", action="cancel_reservation",
                       details={"reservation_id": reservation_id})
        return json.dumps({
            "status": "success",
            "message": f"Reservation {reservation_id} has been cancelled",
            "reservation_id": reservation_id,
        })


@tool
def process_refund(reservation_id: str, reason: str = "customer_request") -> str:
    """Process a refund for a cancelled reservation.

    The reservation must be in 'cancelled' status before a refund can be processed.

    Args:
        reservation_id: The ID of the reservation to refund.
        reason: The reason for the refund.
    """
    from data.models.cultpass import Reservation

    with cultpass_session() as session:
        reservation = session.query(Reservation).filter_by(
            reservation_id=reservation_id
        ).first()

        if not reservation:
            return json.dumps({"error": f"Reservation {reservation_id} not found"})

        if reservation.status != "cancelled":
            return json.dumps({
                "error": f"Reservation must be cancelled before refund. Current status: {reservation.status}"
            })

        reservation.status = "refunded"
        reservation.updated_at = datetime.now(timezone.utc)

        refund_id = str(uuid.uuid4())[:8]
        log_structured(logger, "Refund processed",
                       agent="action_agent", action="process_refund",
                       details={"reservation_id": reservation_id, "refund_id": refund_id, "reason": reason})
        return json.dumps({
            "status": "success",
            "message": f"Refund processed for reservation {reservation_id}",
            "refund_id": refund_id,
            "reservation_id": reservation_id,
            "reason": reason,
        })


@tool
def update_subscription(user_id: str, action: str) -> str:
    """Update a user's subscription status (pause or cancel).

    Args:
        user_id: The user ID whose subscription to update.
        action: The action to perform: 'pause' or 'cancel'.
    """
    from data.models.cultpass import Subscription

    if action not in ("pause", "cancel"):
        return json.dumps({"error": f"Invalid action: {action}. Must be 'pause' or 'cancel'"})

    with cultpass_session() as session:
        sub = session.query(Subscription).filter_by(user_id=user_id).first()
        if not sub:
            return json.dumps({"error": f"No subscription found for user: {user_id}"})

        if action == "cancel":
            if sub.status == "cancelled":
                return json.dumps({"error": "Subscription is already cancelled"})
            sub.status = "cancelled"
            sub.ended_at = datetime.now(timezone.utc)
        elif action == "pause":
            if sub.status == "paused":
                return json.dumps({"error": "Subscription is already paused"})
            sub.status = "paused"

        sub.updated_at = datetime.now(timezone.utc)

        log_structured(logger, "Subscription updated",
                       agent="action_agent", action="update_subscription",
                       details={"user_id": user_id, "action": action, "new_status": sub.status})
        return json.dumps({
            "status": "success",
            "message": f"Subscription {action}d for user {user_id}",
            "user_id": user_id,
            "new_status": sub.status,
        })
