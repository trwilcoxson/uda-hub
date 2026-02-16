"""Shared database session factories.

Provides reusable engines and context-managed sessions for both
cultpass.db (customer data) and udahub.db (support system).
"""
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from agentic.config import CULTPASS_DB_PATH, UDAHUB_DB_PATH

_cultpass_engine = create_engine(f"sqlite:///{CULTPASS_DB_PATH}", echo=False)
_udahub_engine = create_engine(f"sqlite:///{UDAHUB_DB_PATH}", echo=False)

CultPassSession = sessionmaker(bind=_cultpass_engine)
UdaHubSession = sessionmaker(bind=_udahub_engine)


@contextmanager
def cultpass_session():
    """Yield a CultPass DB session with automatic commit/rollback."""
    session = CultPassSession()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@contextmanager
def udahub_session():
    """Yield a UDA-Hub DB session with automatic commit/rollback."""
    session = UdaHubSession()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
