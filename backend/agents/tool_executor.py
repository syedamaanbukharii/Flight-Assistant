"""Tool executor.

The executor is the bridge between the planner's decisions and the concrete
services. Each public ``run_*`` method performs one logical tool invocation and
returns a :class:`ToolResult`, translating expected failures (notably an
unconfigured provider) into structured, non-fatal outcomes rather than raising.

Coupon application and ranking are deterministic and live here so the planner
never has to touch service internals: it simply asks for a flight/hotel search
or a re-rank of cached results and receives ready-to-present ranked items.
"""

from __future__ import annotations

from app.config import Settings
from models.schemas import (
    AppliedCoupon,
    Coupon,
    FlightResult,
    FlightSearchRequest,
    HotelResult,
    HotelSearchRequest,
    Preferences,
    RankedFlight,
    RankedHotel,
    ToolResult,
)
from services.coupon_service import CouponService
from services.flight_service import FlightService
from services.hotel_service import HotelService
from services.ranking_service import RankingService
from utils.constants import CouponScope, ItemKind, SortBy, ToolName
from utils.errors import AppError
from utils.logger import get_logger

# Human-readable tool catalogue injected into the planner prompt.
TOOL_SPECS: list[dict[str, str]] = [
    {
        "name": ToolName.PREFERENCE_MEMORY.value,
        "description": (
            "Persist user preferences (airlines, cabin class, budget, preferred "
            "departure time window, refundable flag, hotel rating/distance)."
        ),
    },
    {
        "name": ToolName.FLIGHT_SEARCH.value,
        "description": (
            "Search flights between an origin and destination with optional date, "
            "cabin class, budget, airline and refundable filters."
        ),
    },
    {
        "name": ToolName.HOTEL_SEARCH.value,
        "description": (
            "Search hotels in a location, optionally near a landmark, with rating "
            "and distance filters."
        ),
    },
    {
        "name": ToolName.COUPON_LOOKUP.value,
        "description": "Fetch active coupons and apply the best one to each result.",
    },
    {
        "name": ToolName.RANKING.value,
        "description": (
            "Rank results deterministically by best_value, cheapest, fastest, "
            "rating or closest."
        ),
    },
]


class ToolExecutor:
    """Executes individual tools on behalf of the planner."""

    def __init__(
        self,
        flight_service: FlightService,
        hotel_service: HotelService,
        coupon_service: CouponService,
        ranking_service: RankingService,
        settings: Settings,
    ) -> None:
        self._flights = flight_service
        self._hotels = hotel_service
        self._coupons = coupon_service
        self._ranking = ranking_service
        self._settings = settings
        self._logger = get_logger("agents.tool_executor")

    # ------------------------------------------------------------------ #
    # Flights
    # ------------------------------------------------------------------ #
    async def run_flight_search(
        self,
        request: FlightSearchRequest,
        *,
        preferences: Preferences | None = None,
        apply_coupons: bool = True,
        sort_by: SortBy = SortBy.BEST_VALUE,
    ) -> ToolResult:
        """Search, optionally apply coupons, and rank flights."""
        try:
            flights = await self._flights.search(request, preferences)
            ranked = await self.rank_flights(
                flights,
                sort_by=sort_by,
                preferences=preferences,
                apply_coupons=apply_coupons,
            )
            return ToolResult(name=ToolName.FLIGHT_SEARCH, ok=True, data=ranked)
        except AppError as exc:
            self._logger.info(
                "Flight search unavailable", extra={"error": exc.message}
            )
            return ToolResult(
                name=ToolName.FLIGHT_SEARCH, ok=False, data=[], error=exc.message
            )

    async def rank_flights(
        self,
        flights: list[FlightResult],
        *,
        sort_by: SortBy = SortBy.BEST_VALUE,
        preferences: Preferences | None = None,
        apply_coupons: bool = True,
    ) -> list[RankedFlight]:
        """Apply coupons (optional) and rank an existing flight candidate set."""
        coupons: dict[str, AppliedCoupon] = {}
        if apply_coupons and flights:
            coupons = await self._coupon_map_for_flights(flights)
        return self._ranking.rank_flights(
            flights, sort_by=sort_by, preferences=preferences, coupons=coupons
        )

    async def _coupon_map_for_flights(
        self, flights: list[FlightResult]
    ) -> dict[str, AppliedCoupon]:
        active = await self._coupons.list_active()
        mapping: dict[str, AppliedCoupon] = {}
        for flight in flights:
            best = self._coupons.select_best(
                active,
                kind=ItemKind.FLIGHT,
                base_price=flight.price,
                currency=flight.currency,
                provider=flight.airline,
            )
            if best is not None:
                mapping[flight.id] = best
        return mapping

    # ------------------------------------------------------------------ #
    # Hotels
    # ------------------------------------------------------------------ #
    async def run_hotel_search(
        self,
        request: HotelSearchRequest,
        *,
        preferences: Preferences | None = None,
        apply_coupons: bool = True,
        sort_by: SortBy = SortBy.BEST_VALUE,
    ) -> ToolResult:
        """Search, optionally apply coupons, and rank hotels."""
        try:
            hotels = await self._hotels.search(request, preferences)
            ranked = await self.rank_hotels(
                hotels,
                sort_by=sort_by,
                preferences=preferences,
                apply_coupons=apply_coupons,
            )
            return ToolResult(name=ToolName.HOTEL_SEARCH, ok=True, data=ranked)
        except AppError as exc:
            self._logger.info("Hotel search unavailable", extra={"error": exc.message})
            return ToolResult(
                name=ToolName.HOTEL_SEARCH, ok=False, data=[], error=exc.message
            )

    async def rank_hotels(
        self,
        hotels: list[HotelResult],
        *,
        sort_by: SortBy = SortBy.BEST_VALUE,
        preferences: Preferences | None = None,
        apply_coupons: bool = True,
    ) -> list[RankedHotel]:
        """Apply coupons (optional) and rank an existing hotel candidate set."""
        coupons: dict[str, AppliedCoupon] = {}
        if apply_coupons and hotels:
            coupons = await self._coupon_map_for_hotels(hotels)
        return self._ranking.rank_hotels(
            hotels, sort_by=sort_by, preferences=preferences, coupons=coupons
        )

    async def _coupon_map_for_hotels(
        self, hotels: list[HotelResult]
    ) -> dict[str, AppliedCoupon]:
        active = await self._coupons.list_active()
        mapping: dict[str, AppliedCoupon] = {}
        for hotel in hotels:
            best = self._coupons.select_best(
                active,
                kind=ItemKind.HOTEL,
                base_price=hotel.price,
                currency=hotel.currency,
                provider=None,
            )
            if best is not None:
                mapping[hotel.id] = best
        return mapping

    # ------------------------------------------------------------------ #
    # Coupons
    # ------------------------------------------------------------------ #
    async def run_coupon_lookup(
        self, scope: CouponScope | None = None
    ) -> ToolResult:
        """Return the active coupon catalogue (optionally scoped)."""
        try:
            coupons: list[Coupon] = await self._coupons.list_active(scope)
            return ToolResult(name=ToolName.COUPON_LOOKUP, ok=True, data=coupons)
        except AppError as exc:
            return ToolResult(
                name=ToolName.COUPON_LOOKUP, ok=False, data=[], error=exc.message
            )

    @staticmethod
    def coupons_used(items: list[RankedFlight] | list[RankedHotel]) -> list[Coupon]:
        """Return the distinct coupons applied across a ranked result set."""
        seen: dict[str, Coupon] = {}
        for item in items:
            applied = item.applied_coupon
            if applied is not None and applied.coupon.code not in seen:
                seen[applied.coupon.code] = applied.coupon
        return list(seen.values())
