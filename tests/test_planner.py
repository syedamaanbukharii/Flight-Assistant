"""Tests for the planner agent's offline (LLM-unconfigured) behaviour.

When no Groq key is configured the planner must fall back to a deterministic
heuristic for intent/parameter extraction and a templated natural-language
reply. The container fixture wires exactly that path with fake providers, so we
can exercise the full plan -> tool-execution -> synthesis pipeline offline.
"""

from __future__ import annotations

import datetime as dt

from app.startup import AppContainer
from models.schemas import Preferences
from utils.constants import CabinClass, Intent


def test_heuristic_plan_extracts_flight_query(container: AppContainer):
    message = "Find me Emirates Business Class from Delhi to Dubai next Friday under ₹80,000"
    decision = container.planner.heuristic_plan(message, Preferences())

    assert decision.flight_query is not None
    fq = decision.flight_query
    assert fq.origin.lower() == "delhi"
    assert fq.destination.lower() == "dubai"
    assert fq.cabin_class == CabinClass.BUSINESS
    assert any(a.lower() == "emirates" for a in fq.airlines)
    assert fq.max_price == 80000.0

    # "next Friday" should resolve to a future date that actually lands on a Friday.
    assert fq.date is not None
    parsed = dt.date.fromisoformat(fq.date)
    assert parsed.weekday() == 4  # Monday=0 ... Friday=4
    assert parsed >= dt.date.today()


async def test_handle_chitchat_returns_empty_results(container: AppContainer):
    response = await container.planner.handle("session-chitchat", "hello there!")

    assert response.intent == Intent.CHITCHAT
    assert isinstance(response.reply, str) and response.reply.strip()
    assert response.flights == []
    assert response.hotels == []
    assert response.session_id == "session-chitchat"


async def test_handle_flight_search_offline_pipeline(container: AppContainer):
    response = await container.planner.handle("session-flights", "flights from Delhi to Dubai")

    assert response.intent == Intent.FLIGHT_SEARCH
    assert isinstance(response.reply, str) and response.reply.strip()
    # Fake provider returns canned flights, so the offline pipeline ranks them.
    assert len(response.flights) >= 1
    ranks = [f.rank for f in response.flights]
    assert ranks == list(range(1, len(response.flights) + 1))
    # Results must be genuine ranked flights with a concrete final price.
    assert all(f.final_price > 0 for f in response.flights)
