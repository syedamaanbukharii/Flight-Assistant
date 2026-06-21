"""Application-wide constants and enumerations.

Centralising these values keeps business logic free of magic strings and makes
provider-specific mappings explicit and testable.
"""

from __future__ import annotations

from enum import Enum


class CabinClass(str, Enum):
    """Normalised cabin classes used throughout the application."""

    ECONOMY = "economy"
    PREMIUM_ECONOMY = "premium_economy"
    BUSINESS = "business"
    FIRST = "first"

    @classmethod
    def from_text(cls, value: str | None) -> "CabinClass | None":
        """Best-effort parse of a free-text cabin description."""
        if not value:
            return None
        normalised = value.strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "economy": cls.ECONOMY,
            "coach": cls.ECONOMY,
            "premium": cls.PREMIUM_ECONOMY,
            "premium_economy": cls.PREMIUM_ECONOMY,
            "business": cls.BUSINESS,
            "biz": cls.BUSINESS,
            "first": cls.FIRST,
            "first_class": cls.FIRST,
        }
        return aliases.get(normalised)


class TimeWindow(str, Enum):
    """Coarse departure-time buckets used for preference filtering."""

    ANY = "any"
    MORNING = "morning"
    AFTERNOON = "afternoon"
    EVENING = "evening"
    NIGHT = "night"

    @classmethod
    def from_text(cls, value: str | None) -> "TimeWindow | None":
        if not value:
            return None
        normalised = value.strip().lower()
        for member in cls:
            if member.value in normalised:
                return member
        return None


class SortBy(str, Enum):
    """Supported ranking strategies."""

    BEST_VALUE = "best_value"
    CHEAPEST = "cheapest"
    FASTEST = "fastest"
    RATING = "rating"
    CLOSEST = "closest"


class ItemKind(str, Enum):
    """The category of a rankable item."""

    FLIGHT = "flight"
    HOTEL = "hotel"


class DiscountType(str, Enum):
    """Coupon discount mechanics."""

    PERCENTAGE = "percentage"
    FLAT = "flat"


class CouponScope(str, Enum):
    """Which result categories a coupon may apply to."""

    ALL = "all"
    FLIGHT = "flight"
    HOTEL = "hotel"


class Intent(str, Enum):
    """High-level user intents recognised by the planner."""

    FLIGHT_SEARCH = "flight_search"
    HOTEL_SEARCH = "hotel_search"
    COMBINED_SEARCH = "combined_search"
    UPDATE_PREFERENCES = "update_preferences"
    APPLY_COUPONS = "apply_coupons"
    RERANK = "rerank"
    CHITCHAT = "chitchat"


class ToolName(str, Enum):
    """Identifiers for tools the planner may invoke."""

    PREFERENCE_MEMORY = "preference_memory"
    FLIGHT_SEARCH = "flight_search"
    HOTEL_SEARCH = "hotel_search"
    COUPON_LOOKUP = "coupon_lookup"
    RANKING = "ranking"


# Inclusive hour ranges (local departure hour) for each time window.
TIME_WINDOW_HOURS: dict[TimeWindow, tuple[int, int]] = {
    TimeWindow.MORNING: (5, 11),
    TimeWindow.AFTERNOON: (12, 16),
    TimeWindow.EVENING: (17, 22),
    TimeWindow.NIGHT: (23, 4),  # wraps past midnight
}

# Map normalised cabin classes to Sky Scrapper API parameter values.
SKYSCANNER_CABIN_PARAM: dict[CabinClass, str] = {
    CabinClass.ECONOMY: "economy",
    CabinClass.PREMIUM_ECONOMY: "premium_economy",
    CabinClass.BUSINESS: "business",
    CabinClass.FIRST: "first",
}

# Default distance (km) that counts as "walking distance".
WALKING_DISTANCE_KM: float = 1.5

# Default maximum number of results returned to the user.
DEFAULT_MAX_RESULTS: int = 10

# Roles used in conversation history.
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
ROLE_SYSTEM = "system"
