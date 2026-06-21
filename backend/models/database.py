"""Database layer built on SQLAlchemy 2.0 (typed, declarative).

A synchronous engine is used for portability: the same models and queries run
unchanged against SQLite (MVP) and PostgreSQL (production) simply by changing
``DATABASE_URL``. Synchronous calls made from async code are dispatched to a
worker thread by the callers, keeping the event loop unblocked.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, String, Text, create_engine, text
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    sessionmaker,
)


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


class SessionRecord(Base):
    """Persisted conversation session and its serialised preferences."""

    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    preferences_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, onupdate=_utcnow
    )


class MessageRecord(Base):
    """A single conversation message, retained for context and audit."""

    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class CouponRecord(Base):
    """Normalised coupon persisted from seed data or scheduled refreshes."""

    __tablename__ = "coupons"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    code: Mapped[str] = mapped_column(String(64), index=True)
    title: Mapped[str] = mapped_column(String(255), default="")
    description: Mapped[str] = mapped_column(Text, default="")
    discount_type: Mapped[str] = mapped_column(String(16), default="percentage")
    discount_value: Mapped[float] = mapped_column(Float, default=0.0)
    max_discount: Mapped[float | None] = mapped_column(Float, nullable=True)
    min_spend: Mapped[float | None] = mapped_column(Float, nullable=True)
    currency: Mapped[str | None] = mapped_column(String(8), nullable=True)
    scope: Mapped[str] = mapped_column(String(16), default="all")
    provider: Mapped[str | None] = mapped_column(String(64), nullable=True)
    source_url: Mapped[str | None] = mapped_column(String(512), nullable=True)
    active: Mapped[bool] = mapped_column(default=True)
    fetched_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class Database:
    """Owns the engine and session factory for the application.

    Args:
        database_url: SQLAlchemy URL. SQLite URLs receive the connection args
            required for multi-threaded use under FastAPI/APScheduler.
    """

    def __init__(self, database_url: str) -> None:
        self.database_url = database_url
        connect_args: dict[str, object] = {}
        if database_url.startswith("sqlite"):
            connect_args["check_same_thread"] = False
            self._ensure_sqlite_dir(database_url)

        self.engine = create_engine(
            database_url,
            future=True,
            echo=False,
            pool_pre_ping=True,
            connect_args=connect_args,
        )
        self._session_factory = sessionmaker(
            bind=self.engine, expire_on_commit=False, class_=Session, future=True
        )

    @staticmethod
    def _ensure_sqlite_dir(database_url: str) -> None:
        """Create the directory backing a file-based SQLite database."""
        path = database_url.split("sqlite:///")[-1]
        if path and path != ":memory:":
            directory = os.path.dirname(os.path.abspath(path))
            os.makedirs(directory, exist_ok=True)

    def create_all(self) -> None:
        """Create any missing tables."""
        Base.metadata.create_all(self.engine)

    @contextmanager
    def session(self) -> Iterator[Session]:
        """Provide a transactional scope around a series of operations."""
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def ping(self) -> bool:
        """Return ``True`` if a trivial query succeeds against the database."""
        with self.engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
