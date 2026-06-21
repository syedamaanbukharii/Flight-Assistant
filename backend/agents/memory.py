"""Conversation memory.

Holds per-session preferences and the most recent search results so the
planner can satisfy follow-up requests such as "apply coupons" or "rank by
cheapest" without re-running upstream searches.

State lives in an in-process dictionary for fast access and is best-effort
persisted to the database (preferences and message history) so a session can
survive a process restart. The transient ``last_*`` result caches are
intentionally kept in memory only; they are derived data, not a system of
record, and are cheap to recompute.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field

from models.database import Database, MessageRecord, SessionRecord
from models.schemas import Preferences, RankedFlight, RankedHotel
from utils.logger import get_logger

_MAX_MESSAGES = 20


@dataclass
class SessionMemory:
    """In-memory state for a single conversation session."""

    session_id: str
    preferences: Preferences = field(default_factory=Preferences)
    messages: list[dict[str, str]] = field(default_factory=list)
    last_flights: list[RankedFlight] = field(default_factory=list)
    last_hotels: list[RankedHotel] = field(default_factory=list)


class MemoryStore:
    """Manages :class:`SessionMemory` objects with optional DB persistence."""

    def __init__(self, database: Database) -> None:
        self._db = database
        self._sessions: dict[str, SessionMemory] = {}
        self._lock = asyncio.Lock()
        self._logger = get_logger("agents.memory")

    async def get_or_create(self, session_id: str) -> SessionMemory:
        """Return the session memory, hydrating from the database if needed."""
        async with self._lock:
            memory = self._sessions.get(session_id)
            if memory is not None:
                return memory
            memory = await asyncio.to_thread(self._load_from_db, session_id)
            self._sessions[session_id] = memory
            return memory

    async def get(self, session_id: str) -> SessionMemory | None:
        """Return the in-memory session if present."""
        async with self._lock:
            return self._sessions.get(session_id)

    async def update_preferences(
        self, session_id: str, patch: Preferences
    ) -> Preferences:
        """Merge ``patch`` into the session preferences and persist them."""
        memory = await self.get_or_create(session_id)
        async with self._lock:
            memory.preferences = memory.preferences.merge(patch)
            preferences = memory.preferences
        await asyncio.to_thread(self._persist_preferences, session_id, preferences)
        return preferences

    async def add_message(self, session_id: str, role: str, content: str) -> None:
        """Append a message to history (capped) and persist it best-effort."""
        memory = await self.get_or_create(session_id)
        async with self._lock:
            memory.messages.append({"role": role, "content": content})
            if len(memory.messages) > _MAX_MESSAGES:
                memory.messages = memory.messages[-_MAX_MESSAGES:]
        await asyncio.to_thread(self._persist_message, session_id, role, content)

    async def set_last_flights(
        self, session_id: str, flights: list[RankedFlight]
    ) -> None:
        """Cache the most recent flight results for follow-up operations."""
        memory = await self.get_or_create(session_id)
        async with self._lock:
            memory.last_flights = list(flights)

    async def set_last_hotels(self, session_id: str, hotels: list[RankedHotel]) -> None:
        """Cache the most recent hotel results for follow-up operations."""
        memory = await self.get_or_create(session_id)
        async with self._lock:
            memory.last_hotels = list(hotels)

    # ------------------------------------------------------------------ #
    # Persistence (synchronous; always called via ``asyncio.to_thread``)
    # ------------------------------------------------------------------ #
    def _load_from_db(self, session_id: str) -> SessionMemory:
        try:
            with self._db.session() as session:
                record = session.get(SessionRecord, session_id)
                if record is None:
                    session.add(SessionRecord(id=session_id, preferences_json="{}"))
                    return SessionMemory(session_id=session_id)
                preferences = self._deserialise_preferences(record.preferences_json)
                messages = (
                    session.query(MessageRecord)
                    .filter(MessageRecord.session_id == session_id)
                    .order_by(MessageRecord.created_at.desc())
                    .limit(_MAX_MESSAGES)
                    .all()
                )
                history = [
                    {"role": m.role, "content": m.content} for m in reversed(messages)
                ]
                return SessionMemory(
                    session_id=session_id,
                    preferences=preferences,
                    messages=history,
                )
        except Exception as exc:  # noqa: BLE001 - persistence is best-effort
            self._logger.warning(
                "Could not load session from DB", extra={"error": str(exc)}
            )
            return SessionMemory(session_id=session_id)

    def _persist_preferences(self, session_id: str, preferences: Preferences) -> None:
        try:
            payload = preferences.model_dump_json()
            with self._db.session() as session:
                record = session.get(SessionRecord, session_id)
                if record is None:
                    session.add(
                        SessionRecord(id=session_id, preferences_json=payload)
                    )
                else:
                    record.preferences_json = payload
        except Exception as exc:  # noqa: BLE001 - persistence is best-effort
            self._logger.warning(
                "Could not persist preferences", extra={"error": str(exc)}
            )

    def _persist_message(self, session_id: str, role: str, content: str) -> None:
        try:
            with self._db.session() as session:
                if session.get(SessionRecord, session_id) is None:
                    session.add(SessionRecord(id=session_id, preferences_json="{}"))
                session.add(
                    MessageRecord(session_id=session_id, role=role, content=content)
                )
        except Exception as exc:  # noqa: BLE001 - persistence is best-effort
            self._logger.warning("Could not persist message", extra={"error": str(exc)})

    @staticmethod
    def _deserialise_preferences(payload: str | None) -> Preferences:
        if not payload:
            return Preferences()
        try:
            data = json.loads(payload)
        except (json.JSONDecodeError, TypeError):
            return Preferences()
        if not isinstance(data, dict):
            return Preferences()
        return Preferences(**data)
