"""FastMCP server exposing CultPass account and action tools.

Run with: python -m agentic.tools.mcp_server
"""
import json
from datetime import datetime

from fastmcp import FastMCP
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from agentic.config import CULTPASS_DB_PATH

mcp = FastMCP("CultPass Support Tools")


def _get_session():
    engine = create_engine(f"sqlite:///{CULTPASS_DB_PATH}", echo=False)
    Session = sessionmaker(bind=engine)
    return Session()


def _serialize_dt(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    return obj


@mcp.tool()
def lookup_user(email: str) -> str:
    """Look up a CultPass user by email address."""
    from data.models.cultpass import User
    session = _get_session()
    try:
        user = session.query(User).filter_by(email=email).first()
        if not user:
            return json.dumps({"error": f"No user found with email: {email}"})
        return json.dumps({
            "user_id": user.user_id,
            "full_name": user.full_name,
            "email": user.email,
            "is_blocked": user.is_blocked,
            "created_at": _serialize_dt(user.created_at),
        })
    finally:
        session.close()


@mcp.tool()
def get_subscription(user_id: str) -> str:
    """Get subscription details for a CultPass user."""
    from data.models.cultpass import Subscription
    session = _get_session()
    try:
        sub = session.query(Subscription).filter_by(user_id=user_id).first()
        if not sub:
            return json.dumps({"error": f"No subscription found for user: {user_id}"})
        return json.dumps({
            "subscription_id": sub.subscription_id,
            "user_id": sub.user_id,
            "status": sub.status,
            "tier": sub.tier,
            "monthly_quota": sub.monthly_quota,
            "started_at": _serialize_dt(sub.started_at),
            "ended_at": _serialize_dt(sub.ended_at),
        })
    finally:
        session.close()


@mcp.tool()
def get_reservations(user_id: str) -> str:
    """Get all reservations for a CultPass user."""
    from data.models.cultpass import Reservation, Experience
    session = _get_session()
    try:
        reservations = session.query(Reservation).filter_by(user_id=user_id).all()
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
        return json.dumps({"reservations": results})
    finally:
        session.close()


@mcp.tool()
def cancel_reservation(reservation_id: str) -> str:
    """Cancel a CultPass reservation."""
    from data.models.cultpass import Reservation
    session = _get_session()
    try:
        reservation = session.query(Reservation).filter_by(reservation_id=reservation_id).first()
        if not reservation:
            return json.dumps({"error": f"Reservation {reservation_id} not found"})
        if reservation.status == "cancelled":
            return json.dumps({"error": "Reservation is already cancelled"})
        reservation.status = "cancelled"
        reservation.updated_at = datetime.now()
        session.commit()
        return json.dumps({"status": "success", "message": f"Reservation {reservation_id} cancelled"})
    except Exception as e:
        session.rollback()
        return json.dumps({"error": str(e)})
    finally:
        session.close()


@mcp.tool()
def update_subscription(user_id: str, action: str) -> str:
    """Update a user's subscription (pause or cancel)."""
    from data.models.cultpass import Subscription
    if action not in ("pause", "cancel"):
        return json.dumps({"error": f"Invalid action: {action}"})
    session = _get_session()
    try:
        sub = session.query(Subscription).filter_by(user_id=user_id).first()
        if not sub:
            return json.dumps({"error": f"No subscription for user: {user_id}"})
        if action == "cancel":
            sub.status = "cancelled"
            sub.ended_at = datetime.now()
        else:
            sub.status = "paused"
        sub.updated_at = datetime.now()
        session.commit()
        return json.dumps({"status": "success", "new_status": sub.status})
    except Exception as e:
        session.rollback()
        return json.dumps({"error": str(e)})
    finally:
        session.close()


if __name__ == "__main__":
    mcp.run()
