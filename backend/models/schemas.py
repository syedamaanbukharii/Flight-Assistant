"""Pydantic v2 schemas used across the API, agents, and services.

These models form the single source of truth for data shapes. Internal domain
objects (flights, hotels, coupons, preferences) and the public API contract are
both defined here to guarantee consistency end to end.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from utils.constants import (
    CabinClass,
    CouponScope,
    DiscountType,
    Intent,
    SortBy,
    TimeWindow,
    ToolName,
)
from utils.helpers import new_request_id


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


class StrictModel(BaseModel):
    """Base model that ignores unknown fields and validates on assignment."""

    model_config = ConfigDict(extra="ignore", validate_assignment=True)


# --------------------------------------------------------------------------- #
# Preferences
# --------------------------------------------------------------------------- #
class Preferences(StrictModel):
    """Durable, per-session user preferences.

    Every field is optional; ``None`` means "no stated preference". The
    :meth:`merge` helper applies a partial update without clobbering existing
    values with ``None``.
    """

    airlines: list[str] = Field(default_factory=list)
    cabin_class: CabinClass | None = None
    max_budget: float | None = None
    currency: str | None = None
    departure_time_window: TimeWindow | None = None
    refundable: bool | None = None
    hotel_min_rating: float | None = None
    hotel_max_distance_km: float | None = None

    def merge(self, patch: "Preferences") -> "Preferences":
        """Return a new ``Preferences`` with non-empty fields from ``patch`` applied."""
        data = self.model_dump()
        patch_data = patch.model_dump()
        for key, value in patch_data.items():
            if key == "airlines":
                if value:
                    data[key] = sorted({*data.get(key, []), *value})
            elif value is not None:
                data[key] = value
        return Preferences(**data)


# --------------------------------------------------------------------------- #
# Coupons
# --------------------------------------------------------------------------- #
class Coupon(StrictModel):
    """A normalised promotional coupon."""

    code: str
    title: str = ""
    description: str = ""
    discount_type: DiscountType = DiscountType.PERCENTAGE
    discount_value: float = 0.0
    max_discount: float | None = None
    min_spend: float | None = None
    currency: str | None = None
    scope: CouponScope = CouponScope.ALL
    provider: str | None = None
    source_url: str | None = None
    active: bool = True

    @field_validator("code")
    @classmethod
    def _strip_code(cls, value: str) -> str:
        return value.strip().upper()


class AppliedCoupon(StrictModel):
    """A coupon together with the savings it produced for a specific item."""

    coupon: Coupon
    savings: float
    final_price: float


# --------------------------------------------------------------------------- #
# Flights
# --------------------------------------------------------------------------- #
class FlightResult(StrictModel):
    """A single flight itinerary returned by a flight provider."""

    id: str
    airline: str
    airline_code: str | None = None
    origin: str
    destination: str
    departure_time: str | None = None  # ISO-8601
    arrival_time: str | None = None  # ISO-8601
    duration_minutes: int | None = None
    stops: int = 0
    cabin_class: CabinClass | None = None
    price: float
    currency: str = "INR"
    refundable: bool | None = None
    baggage_allowance: str | None = None
    booking_link: str | None = None
    provider: str = "skyscanner"


class RankedFlight(FlightResult):
    """A flight enriched with ranking and coupon outcomes."""

    rank: int
    score: float
    rationale: str
    applied_coupon: AppliedCoupon | None = None
    final_price: float

    @classmethod
    def from_flight(
        cls,
        flight: FlightResult,
        *,
        rank: int,
        score: float,
        rationale: str,
        applied_coupon: AppliedCoupon | None,
    ) -> "RankedFlight":
        final_price = applied_coupon.final_price if applied_coupon else flight.price
        return cls(
            **flight.model_dump(),
            rank=rank,
            score=score,
            rationale=rationale,
            applied_coupon=applied_coupon,
            final_price=final_price,
        )


# --------------------------------------------------------------------------- #
# Hotels
# --------------------------------------------------------------------------- #
class HotelResult(StrictModel):
    """A single hotel returned by a hotel provider."""

    id: str
    name: str
    address: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    distance_km: float | None = None
    rating: float | None = None
    review_count: int | None = None
    price: float
    currency: str = "INR"
    price_per_night: bool = True
    refundable: bool | None = None
    booking_link: str | None = None
    image_url: str | None = None
    provider: str = "skyscanner"


class RankedHotel(HotelResult):
    """A hotel enriched with ranking and coupon outcomes."""

    rank: int
    score: float
    rationale: str
    applied_coupon: AppliedCoupon | None = None
    final_price: float

    @classmethod
    def from_hotel(
        cls,
        hotel: HotelResult,
        *,
        rank: int,
        score: float,
        rationale: str,
        applied_coupon: AppliedCoupon | None,
    ) -> "RankedHotel":
        final_price = applied_coupon.final_price if applied_coupon else hotel.price
        return cls(
            **hotel.model_dump(),
            rank=rank,
            score=score,
            rationale=rationale,
            applied_coupon=applied_coupon,
            final_price=final_price,
        )


# --------------------------------------------------------------------------- #
# Search requests / responses (public API)
# --------------------------------------------------------------------------- #
class FlightSearchRequest(StrictModel):
    """Request body for the flight search endpoint and tool."""

    origin: str = Field(..., min_length=2, description="City name or IATA code")
    destination: str = Field(..., min_length=2, description="City name or IATA code")
    date: str | None = Field(default=None, description="Outbound date YYYY-MM-DD")
    return_date: str | None = Field(default=None, description="Return date YYYY-MM-DD")
    cabin_class: CabinClass | None = None
    adults: int = Field(default=1, ge=1, le=9)
    max_price: float | None = Field(default=None, ge=0)
    currency: str | None = None
    airlines: list[str] = Field(default_factory=list)
    refundable: bool | None = None
    departure_time_window: TimeWindow | None = None
    sort_by: SortBy = SortBy.BEST_VALUE
    apply_coupons: bool = True


class HotelSearchRequest(StrictModel):
    """Request body for the hotel search endpoint and tool."""

    location: str = Field(..., min_length=2, description="City or area name")
    landmark: str | None = Field(
        default=None, description="Anchor landmark for distance filtering"
    )
    check_in: str | None = Field(default=None, description="Check-in date YYYY-MM-DD")
    check_out: str | None = Field(default=None, description="Check-out date YYYY-MM-DD")
    adults: int = Field(default=2, ge=1, le=16)
    max_price: float | None = Field(default=None, ge=0)
    currency: str | None = None
    min_rating: float | None = Field(default=None, ge=0, le=5)
    max_distance_km: float | None = Field(default=None, ge=0)
    sort_by: SortBy = SortBy.BEST_VALUE
    apply_coupons: bool = True


class FlightSearchResponse(StrictModel):
    """Response payload for the flight search endpoint."""

    request_id: str = Field(default_factory=new_request_id)
    count: int
    results: list[RankedFlight]
    coupons_applied: list[Coupon] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class HotelSearchResponse(StrictModel):
    """Response payload for the hotel search endpoint."""

    request_id: str = Field(default_factory=new_request_id)
    count: int
    results: list[RankedHotel]
    coupons_applied: list[Coupon] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Chat
# --------------------------------------------------------------------------- #
class ChatRequest(StrictModel):
    """Inbound chat message."""

    message: str = Field(..., min_length=1, max_length=4000)
    session_id: str | None = Field(
        default=None, description="Opaque session id; generated if omitted"
    )


class ChatResponse(StrictModel):
    """Conversational reply plus the structured data backing it."""

    request_id: str = Field(default_factory=new_request_id)
    session_id: str
    intent: Intent
    reply: str
    flights: list[RankedFlight] = Field(default_factory=list)
    hotels: list[RankedHotel] = Field(default_factory=list)
    coupons_applied: list[Coupon] = Field(default_factory=list)
    preferences: Preferences = Field(default_factory=Preferences)
    notes: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Planner contract (LLM output)
# --------------------------------------------------------------------------- #
class PlannerDecision(StrictModel):
    """Structured plan produced by the planner LLM.

    The LLM decides *what* to do; native Python performs the orchestration.
    """

    intent: Intent = Intent.CHITCHAT
    tools: list[ToolName] = Field(default_factory=list)
    flight_query: FlightSearchRequest | None = None
    hotel_query: HotelSearchRequest | None = None
    preferences: Preferences | None = None
    apply_coupons: bool = False
    sort_by: SortBy | None = None
    notes: str = ""


# --------------------------------------------------------------------------- #
# Health & verification
# --------------------------------------------------------------------------- #
class HealthResponse(StrictModel):
    """Liveness payload."""

    status: str = "ok"
    app: str
    environment: str
    version: str
    time: datetime = Field(default_factory=_utcnow)


class CheckResult(StrictModel):
    """Result of a single verification probe."""

    name: str
    status: str  # "ok" | "warn" | "fail"
    message: str


class VerifyReport(StrictModel):
    """Aggregated startup/runtime verification report."""

    status: str  # "ok" | "warn" | "fail"
    checks: list[CheckResult]
    time: datetime = Field(default_factory=_utcnow)


# --------------------------------------------------------------------------- #
# Tool execution (internal)
# --------------------------------------------------------------------------- #
class ToolResult(StrictModel):
    """Outcome of executing a single tool."""

    name: ToolName
    ok: bool
    data: Any = None
    error: str | None = None
