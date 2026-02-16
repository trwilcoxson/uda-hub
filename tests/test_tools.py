"""Tests for account and action tools.

These tests require cultpass.db to be set up first (run notebook 01).
"""
import json
import os
import uuid
import pytest
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from agentic.config import CULTPASS_DB_PATH


def _db_exists():
    return os.path.exists(CULTPASS_DB_PATH)


@pytest.mark.skipif(not _db_exists(), reason="cultpass.db not found - run notebook 01 first")
class TestAccountTools:
    def test_lookup_user_valid_email(self):
        from agentic.tools.account_tools import lookup_user
        result = json.loads(lookup_user.invoke({"email": "alice.kingsley@wonderland.com"}))
        assert result["full_name"] == "Alice Kingsley"
        assert result["user_id"] == "a4ab87"

    def test_lookup_user_invalid_email(self):
        from agentic.tools.account_tools import lookup_user
        result = json.loads(lookup_user.invoke({"email": "nobody@nowhere.com"}))
        assert "error" in result

    def test_get_subscription(self):
        from agentic.tools.account_tools import get_subscription
        result = json.loads(get_subscription.invoke({"user_id": "a4ab87"}))
        assert "subscription_id" in result
        assert result["user_id"] == "a4ab87"
        assert result["tier"] in ["basic", "premium"]

    def test_get_subscription_invalid_user(self):
        from agentic.tools.account_tools import get_subscription
        result = json.loads(get_subscription.invoke({"user_id": "nonexistent"}))
        assert "error" in result

    def test_get_reservations(self):
        from agentic.tools.account_tools import get_reservations
        result = json.loads(get_reservations.invoke({"user_id": "a4ab87"}))
        assert "reservations" in result
        assert len(result["reservations"]) >= 1

    def test_get_reservations_no_bookings(self):
        from agentic.tools.account_tools import get_reservations
        result = json.loads(get_reservations.invoke({"user_id": "f556c0"}))
        assert "reservations" in result


@pytest.mark.skipif(not _db_exists(), reason="cultpass.db not found - run notebook 01 first")
class TestActionTools:
    def _create_test_reservation(self):
        """Create a temporary reservation for testing."""
        from data.models.cultpass import Reservation, Experience
        engine = create_engine(f"sqlite:///{CULTPASS_DB_PATH}", echo=False)
        Session = sessionmaker(bind=engine)
        session = Session()
        try:
            exp = session.query(Experience).first()
            res_id = str(uuid.uuid4())[:6]
            reservation = Reservation(
                reservation_id=res_id,
                user_id="a4ab87",
                experience_id=exp.experience_id,
                status="reserved",
            )
            session.add(reservation)
            session.commit()
            return res_id
        finally:
            session.close()

    def test_cancel_reservation(self):
        res_id = self._create_test_reservation()
        from agentic.tools.action_tools import cancel_reservation
        result = json.loads(cancel_reservation.invoke({"reservation_id": res_id}))
        assert result["status"] == "success"

    def test_cancel_nonexistent_reservation(self):
        from agentic.tools.action_tools import cancel_reservation
        result = json.loads(cancel_reservation.invoke({"reservation_id": "fake123"}))
        assert "error" in result

    def test_process_refund_requires_cancelled(self):
        res_id = self._create_test_reservation()
        from agentic.tools.action_tools import process_refund
        result = json.loads(process_refund.invoke({"reservation_id": res_id}))
        assert "error" in result  # Not cancelled yet

    def test_process_refund_after_cancel(self):
        res_id = self._create_test_reservation()
        from agentic.tools.action_tools import cancel_reservation, process_refund
        cancel_reservation.invoke({"reservation_id": res_id})
        result = json.loads(process_refund.invoke({"reservation_id": res_id}))
        assert result["status"] == "success"
        assert "refund_id" in result

    def test_update_subscription_invalid_action(self):
        from agentic.tools.action_tools import update_subscription
        result = json.loads(update_subscription.invoke({"user_id": "a4ab87", "action": "invalid"}))
        assert "error" in result
