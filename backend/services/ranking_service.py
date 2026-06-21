"""Deterministic ranking engine.

Ranking is fully deterministic and explainable — no LLM involvement. Each
candidate is scored by normalising its features to ``[0, 1]`` across the
candidate set and combining them with strategy-specific weights. A short,
human-readable rationale accompanies every ranked item.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.config import Settings
from models.schemas import (
    AppliedCoupon,
    FlightResult,
    HotelResult,
    Preferences,
    RankedFlight,
    RankedHotel,
)
from utils.constants import SortBy
from utils.helpers import normalise_min_max
from utils.logger import get_logger

# Weight presets per strategy. Keys are feature names; weights need not sum to
# one (the score is normalised by the total active weight).
_FLIGHT_WEIGHTS: dict[SortBy, dict[str, float]] = {
    SortBy.BEST_VALUE: {
        "price": 0.35,
        "savings": 0.15,
        "duration": 0.20,
        "refundable": 0.10,
        "airline": 0.10,
        "baggage": 0.10,
    },
    SortBy.CHEAPEST: {"price": 0.8, "savings": 0.2},
    SortBy.FASTEST: {"duration": 0.7, "price": 0.2, "savings": 0.1},
    SortBy.RATING: {  # flights have no rating; fall back to value-style weights
        "price": 0.4,
        "duration": 0.3,
        "refundable": 0.15,
        "airline": 0.15,
    },
    SortBy.CLOSEST: {"price": 0.5, "duration": 0.3, "savings": 0.2},
}

_HOTEL_WEIGHTS: dict[SortBy, dict[str, float]] = {
    SortBy.BEST_VALUE: {
        "price": 0.35,
        "rating": 0.30,
        "distance": 0.20,
        "savings": 0.15,
    },
    SortBy.CHEAPEST: {"price": 0.8, "savings": 0.2},
    SortBy.FASTEST: {"price": 0.6, "rating": 0.4},  # no duration for hotels
    SortBy.RATING: {"rating": 0.6, "distance": 0.2, "price": 0.2},
    SortBy.CLOSEST: {"distance": 0.6, "rating": 0.2, "price": 0.2},
}

_NEUTRAL = 0.5  # score assigned to a feature whose data is unknown


@dataclass(frozen=True)
class _Range:
    minimum: float
    maximum: float


class RankingService:
    """Scores and orders flight and hotel candidates."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._logger = get_logger("services.ranking")

    # ------------------------------------------------------------------ #
    # Flights
    # ------------------------------------------------------------------ #
    def rank_flights(
        self,
        flights: list[FlightResult],
        *,
        sort_by: SortBy = SortBy.BEST_VALUE,
        preferences: Preferences | None = None,
        coupons: dict[str, AppliedCoupon] | None = None,
    ) -> list[RankedFlight]:
        """Return flights ordered best-first with scores and rationales.

        Args:
            flights: Candidate flights (already filtered).
            sort_by: Ranking strategy.
            preferences: User preferences influencing airline matching.
            coupons: Map of flight id -> best applied coupon, if any.
        """
        if not flights:
            return []

        coupons = coupons or {}
        weights = _FLIGHT_WEIGHTS.get(sort_by, _FLIGHT_WEIGHTS[SortBy.BEST_VALUE])
        preferred_airlines = {a.lower() for a in (preferences.airlines if preferences else [])}

        final_prices = [self._final_price(f.price, coupons.get(f.id)) for f in flights]
        savings = [coupons[f.id].savings if f.id in coupons else 0.0 for f in flights]
        durations = [f.duration_minutes or 0 for f in flights]

        price_range = _Range(min(final_prices), max(final_prices))
        savings_range = _Range(min(savings), max(savings))
        duration_range = _Range(min(durations), max(durations))

        scored: list[tuple[float, str, FlightResult, AppliedCoupon | None]] = []
        for flight, final_price, saving in zip(flights, final_prices, savings):
            features = {
                "price": 1.0
                - normalise_min_max(final_price, price_range.minimum, price_range.maximum),
                "savings": normalise_min_max(saving, savings_range.minimum, savings_range.maximum),
                "duration": 1.0
                - normalise_min_max(
                    flight.duration_minutes or duration_range.maximum,
                    duration_range.minimum,
                    duration_range.maximum,
                ),
                "refundable": self._bool_feature(flight.refundable),
                "airline": 1.0 if flight.airline.lower() in preferred_airlines else _NEUTRAL,
                "baggage": 1.0 if flight.baggage_allowance else _NEUTRAL,
            }
            score = self._weighted_score(features, weights)
            rationale = self._rationale(features, weights, sort_by)
            scored.append((score, rationale, flight, coupons.get(flight.id)))

        scored.sort(key=lambda item: (-item[0], item[2].price))
        return [
            RankedFlight.from_flight(
                flight,
                rank=index + 1,
                score=round(score, 4),
                rationale=rationale,
                applied_coupon=coupon,
            )
            for index, (score, rationale, flight, coupon) in enumerate(scored)
        ]

    # ------------------------------------------------------------------ #
    # Hotels
    # ------------------------------------------------------------------ #
    def rank_hotels(
        self,
        hotels: list[HotelResult],
        *,
        sort_by: SortBy = SortBy.BEST_VALUE,
        preferences: Preferences | None = None,
        coupons: dict[str, AppliedCoupon] | None = None,
    ) -> list[RankedHotel]:
        """Return hotels ordered best-first with scores and rationales."""
        if not hotels:
            return []

        coupons = coupons or {}
        weights = _HOTEL_WEIGHTS.get(sort_by, _HOTEL_WEIGHTS[SortBy.BEST_VALUE])

        final_prices = [self._final_price(h.price, coupons.get(h.id)) for h in hotels]
        savings = [coupons[h.id].savings if h.id in coupons else 0.0 for h in hotels]
        distances = [h.distance_km for h in hotels if h.distance_km is not None]
        ratings = [h.rating for h in hotels if h.rating is not None]

        price_range = _Range(min(final_prices), max(final_prices))
        savings_range = _Range(min(savings), max(savings))
        distance_range = _Range(min(distances), max(distances)) if distances else _Range(0, 1)
        rating_range = _Range(min(ratings), max(ratings)) if ratings else _Range(0, 5)

        scored: list[tuple[float, str, HotelResult, AppliedCoupon | None]] = []
        for hotel, final_price, saving in zip(hotels, final_prices, savings):
            features = {
                "price": 1.0
                - normalise_min_max(final_price, price_range.minimum, price_range.maximum),
                "savings": normalise_min_max(saving, savings_range.minimum, savings_range.maximum),
                "rating": (
                    normalise_min_max(hotel.rating, rating_range.minimum, rating_range.maximum)
                    if hotel.rating is not None
                    else _NEUTRAL
                ),
                "distance": (
                    1.0
                    - normalise_min_max(
                        hotel.distance_km, distance_range.minimum, distance_range.maximum
                    )
                    if hotel.distance_km is not None
                    else _NEUTRAL
                ),
            }
            score = self._weighted_score(features, weights)
            rationale = self._rationale(features, weights, sort_by)
            scored.append((score, rationale, hotel, coupons.get(hotel.id)))

        scored.sort(key=lambda item: (-item[0], item[2].price))
        return [
            RankedHotel.from_hotel(
                hotel,
                rank=index + 1,
                score=round(score, 4),
                rationale=rationale,
                applied_coupon=coupon,
            )
            for index, (score, rationale, hotel, coupon) in enumerate(scored)
        ]

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    @staticmethod
    def _final_price(price: float, coupon: AppliedCoupon | None) -> float:
        return coupon.final_price if coupon else price

    @staticmethod
    def _bool_feature(value: bool | None) -> float:
        if value is True:
            return 1.0
        if value is False:
            return 0.0
        return _NEUTRAL

    @staticmethod
    def _weighted_score(features: dict[str, float], weights: dict[str, float]) -> float:
        total_weight = sum(weights.values())
        if total_weight <= 0:
            return 0.0
        weighted = sum(
            features.get(name, _NEUTRAL) * weight for name, weight in weights.items()
        )
        return weighted / total_weight

    @staticmethod
    def _rationale(
        features: dict[str, float], weights: dict[str, float], sort_by: SortBy
    ) -> str:
        """Summarise the two highest-contributing factors for transparency."""
        contributions = {
            name: features.get(name, _NEUTRAL) * weight for name, weight in weights.items()
        }
        top = sorted(contributions.items(), key=lambda kv: kv[1], reverse=True)[:2]
        labels = {
            "price": "competitive price",
            "savings": "strong coupon savings",
            "duration": "short travel time",
            "refundable": "refundable fare",
            "airline": "preferred airline",
            "baggage": "baggage included",
            "rating": "high guest rating",
            "distance": "close to your anchor point",
        }
        reasons = ", ".join(
            labels.get(name, name) for name, _ in top if features.get(name, 0) >= _NEUTRAL
        )
        prefix = {
            SortBy.BEST_VALUE: "Best overall value",
            SortBy.CHEAPEST: "Lowest cost",
            SortBy.FASTEST: "Quickest option",
            SortBy.RATING: "Top rated",
            SortBy.CLOSEST: "Closest option",
        }.get(sort_by, "Recommended")
        return f"{prefix}: {reasons}." if reasons else f"{prefix}."
