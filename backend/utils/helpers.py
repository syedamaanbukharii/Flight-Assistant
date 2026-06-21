"""Pure, dependency-light helper functions.

These utilities are deliberately free of framework or service dependencies so
they can be unit-tested in isolation and reused anywhere.
"""

from __future__ import annotations

import asyncio
import functools
import math
import re
import uuid
from collections.abc import Awaitable, Callable, Iterable, Mapping
from datetime import date, datetime
from typing import Any, TypeVar

from utils.constants import TIME_WINDOW_HOURS, TimeWindow
from utils.logger import get_logger

T = TypeVar("T")

_logger = get_logger("utils.helpers")

# Multipliers for Indian-style magnitude suffixes found in budgets.
_MAGNITUDE_SUFFIXES: dict[str, float] = {
    "k": 1_000,
    "thousand": 1_000,
    "l": 100_000,
    "lac": 100_000,
    "lakh": 100_000,
    "lakhs": 100_000,
    "m": 1_000_000,
    "mn": 1_000_000,
    "million": 1_000_000,
    "cr": 10_000_000,
    "crore": 10_000_000,
}

_AMOUNT_RE = re.compile(
    r"(?P<value>\d[\d,]*\.?\d*)\s*(?P<suffix>k|thousand|l|lac|lakhs?|m|mn|million|cr|crore)?",
    re.IGNORECASE,
)


def new_request_id() -> str:
    """Return a fresh, URL-safe request identifier."""
    return uuid.uuid4().hex


def parse_currency_amount(text: str | float | int | None) -> float | None:
    """Parse a human-written monetary amount into a float.

    Handles currency symbols, thousands separators, and magnitude suffixes such
    as ``k``, ``lakh`` and ``cr``. Returns ``None`` when no number is present.

    Examples:
        ``"₹80,000"`` -> ``80000.0``
        ``"80k"`` -> ``80000.0``
        ``"1.2 lakh"`` -> ``120000.0``
    """
    if text is None:
        return None
    if isinstance(text, (int, float)):
        return float(text)

    match = _AMOUNT_RE.search(text)
    if match is None:
        return None

    raw_value = match.group("value").replace(",", "")
    try:
        value = float(raw_value)
    except ValueError:
        return None

    suffix = (match.group("suffix") or "").lower()
    multiplier = _MAGNITUDE_SUFFIXES.get(suffix, 1.0)
    return value * multiplier


def to_iso_date(value: str | date | datetime | None) -> str | None:
    """Validate and normalise a date to ``YYYY-MM-DD`` form.

    Accepts ``date``/``datetime`` objects and common string formats. Returns
    ``None`` for unparseable input rather than raising, so callers can decide
    how to handle missing dates.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()

    text = value.strip()
    if not text:
        return None

    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def parse_iso_datetime(value: str | None) -> datetime | None:
    """Parse an ISO-8601 datetime string, tolerating a trailing ``Z``."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def within_time_window(departure_iso: str | None, window: TimeWindow | None) -> bool:
    """Return whether a departure time falls inside ``window``.

    Unknown departure times or an ``ANY``/``None`` window are treated as a
    match so we never discard otherwise-valid results on missing data.
    """
    if window in (None, TimeWindow.ANY):
        return True
    parsed = parse_iso_datetime(departure_iso)
    if parsed is None:
        return True

    hour = parsed.hour
    start, end = TIME_WINDOW_HOURS[window]
    if start <= end:
        return start <= hour <= end
    # Window wraps past midnight (e.g. night 23:00 -> 04:00).
    return hour >= start or hour <= end


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points in kilometres."""
    radius = 6371.0088
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(d_lon / 2) ** 2
    )
    return radius * 2 * math.asin(math.sqrt(a))


def safe_get(data: Mapping[str, Any] | None, *keys: str, default: Any = None) -> Any:
    """Safely traverse nested mappings, returning ``default`` on any miss."""
    current: Any = data
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return current if current is not None else default


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    """Constrain ``value`` to the inclusive ``[lower, upper]`` range."""
    return max(lower, min(upper, value))


def normalise_min_max(value: float, minimum: float, maximum: float) -> float:
    """Scale ``value`` to ``[0, 1]`` given a known range.

    When ``minimum == maximum`` (a degenerate range) every value maps to ``1.0``.
    """
    if maximum <= minimum:
        return 1.0
    return clamp((value - minimum) / (maximum - minimum))


def minutes_to_human(minutes: int | None) -> str:
    """Render a minute count as ``"Xh Ym"`` (or an em dash when unknown)."""
    if minutes is None or minutes < 0:
        return "—"
    hours, mins = divmod(int(minutes), 60)
    if hours and mins:
        return f"{hours}h {mins}m"
    if hours:
        return f"{hours}h"
    return f"{mins}m"


def first_present(values: Iterable[Any]) -> Any | None:
    """Return the first non-``None`` value from ``values``."""
    for value in values:
        if value is not None:
            return value
    return None


def async_retry(
    *,
    attempts: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """Decorator adding exponential-backoff retries to an async callable.

    The final failure is re-raised so callers retain the original exception
    type. Each retry is logged with the attempt number for observability.
    """

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            delay = base_delay
            last_exc: BaseException | None = None
            for attempt in range(1, attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as exc:  # noqa: PERF203 - retry semantics
                    last_exc = exc
                    if attempt >= attempts:
                        break
                    _logger.warning(
                        "Retrying after failure",
                        extra={
                            "callable": func.__name__,
                            "attempt": attempt,
                            "max_attempts": attempts,
                            "error": str(exc),
                        },
                    )
                    await asyncio.sleep(delay)
                    delay = min(max_delay, delay * 2)
            assert last_exc is not None  # for type-checkers
            raise last_exc

        return wrapper

    return decorator
