"""HTTP API routes.

Thin transport layer over the agents and services. The ``/chat`` endpoint runs
the full planner pipeline (LLM optional), while the ``/search/*`` endpoints
expose deterministic search + coupon + ranking directly, without invoking the
language model. Domain errors raised here propagate to the centralised
exception handlers in :mod:`app.main`.
"""

from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, Depends, Request

from agents.tool_executor import ToolExecutor
from app.startup import AppContainer
from app.verify import run_verification
from models.schemas import (
    ChatRequest,
    ChatResponse,
    FlightSearchRequest,
    FlightSearchResponse,
    HealthResponse,
    HotelSearchRequest,
    HotelSearchResponse,
    VerifyReport,
)

router = APIRouter()


def get_container(request: Request) -> AppContainer:
    """FastAPI dependency returning the application container."""
    return request.app.state.container


@router.get("/health", response_model=HealthResponse, tags=["system"])
async def health(container: AppContainer = Depends(get_container)) -> HealthResponse:
    """Liveness probe with basic application metadata."""
    settings = container.settings
    return HealthResponse(
        app=settings.app_name,
        environment=settings.app_env,
        version=settings.app_version,
    )


@router.get("/verify", response_model=VerifyReport, tags=["system"])
async def verify(container: AppContainer = Depends(get_container)) -> VerifyReport:
    """Report configuration and dependency health for operators."""
    return await run_verification(container, container.settings)


@router.post("/chat", response_model=ChatResponse, tags=["assistant"])
async def chat(
    payload: ChatRequest, container: AppContainer = Depends(get_container)
) -> ChatResponse:
    """Handle a conversational turn through the planner agent."""
    session_id = payload.session_id or uuid4().hex
    return await container.planner.handle(session_id, payload.message)


@router.post("/search/flights", response_model=FlightSearchResponse, tags=["search"])
async def search_flights(
    payload: FlightSearchRequest, container: AppContainer = Depends(get_container)
) -> FlightSearchResponse:
    """Deterministic flight search with coupons and ranking (no LLM)."""
    flights = await container.flight_service.search(payload)
    ranked = await container.tool_executor.rank_flights(
        flights,
        sort_by=payload.sort_by,
        preferences=None,
        apply_coupons=payload.apply_coupons,
    )
    coupons = ToolExecutor.coupons_used(ranked)
    return FlightSearchResponse(
        count=len(ranked), results=ranked, coupons_applied=coupons
    )


@router.post("/search/hotels", response_model=HotelSearchResponse, tags=["search"])
async def search_hotels(
    payload: HotelSearchRequest, container: AppContainer = Depends(get_container)
) -> HotelSearchResponse:
    """Deterministic hotel search with coupons and ranking (no LLM)."""
    hotels = await container.hotel_service.search(payload)
    ranked = await container.tool_executor.rank_hotels(
        hotels,
        sort_by=payload.sort_by,
        preferences=None,
        apply_coupons=payload.apply_coupons,
    )
    coupons = ToolExecutor.coupons_used(ranked)
    return HotelSearchResponse(
        count=len(ranked), results=ranked, coupons_applied=coupons
    )
