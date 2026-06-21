"""Hotel search service.

Mirrors the flight service design: the concrete provider (RapidAPI Sky
Scrapper hotels) is hidden behind the :class:`HotelProvider` interface so it
can be swapped for any other supplier without touching business logic. The
service resolves a destination (and an optional anchor landmark), queries the
provider, computes great-circle distance from the landmark, and applies
preference-driven filters (rating, distance, budget) before returning results.
Ranking happens later in the pipeline.

As with flights, this uses an aggregator API and never scrapes hotel or
metasearch websites directly.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

import httpx

from app.config import Settings
from models.schemas import HotelResult, HotelSearchRequest, Preferences
from utils.errors import ExternalServiceError, NotConfiguredError, ValidationAppError
from utils.helpers import async_retry, haversine_km, safe_get
from utils.logger import get_logger


@dataclass(frozen=True)
class DestinationRef:
    """A resolved hotel destination or landmark reference."""

    entity_id: str
    name: str
    latitude: float | None = None
    longitude: float | None = None


class HotelProvider(abc.ABC):
    """Abstract hotel data provider."""

    @abc.abstractmethod
    async def search(self, request: HotelSearchRequest) -> list[HotelResult]:
        """Return raw hotels for a search request."""

    async def aclose(self) -> None:  # pragma: no cover - default no-op
        """Release any held resources."""


class UnconfiguredHotelProvider(HotelProvider):
    """Fallback provider used when no API credentials are configured."""

    async def search(self, request: HotelSearchRequest) -> list[HotelResult]:
        raise NotConfiguredError(
            "Hotel search is not configured. Set RAPIDAPI_KEY (and RAPIDAPI_HOST) "
            "to enable live hotel results."
        )


class SkyScrapperHotelProvider(HotelProvider):
    """Hotel provider backed by the RapidAPI Sky Scrapper API."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._logger = get_logger("services.hotel.skyscrapper")
        self._client: httpx.AsyncClient | None = None
        self._destination_cache: dict[str, DestinationRef] = {}

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=f"https://{self._settings.rapidapi_host}",
                headers={
                    "x-rapidapi-key": self._settings.rapidapi_key or "",
                    "x-rapidapi-host": self._settings.rapidapi_host,
                },
                timeout=self._settings.http_timeout_seconds,
            )
        return self._client

    @async_retry(attempts=2, base_delay=0.6, exceptions=(httpx.HTTPError,))
    async def _get(self, path: str, params: dict[str, object]) -> dict:
        client = self._get_client()
        try:
            response = await client.get(path, params=params)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ExternalServiceError(
                "The hotel provider returned an error.",
                detail={"status": exc.response.status_code, "path": path},
            ) from exc
        except httpx.HTTPError as exc:
            raise ExternalServiceError("Could not reach the hotel provider.") from exc
        return response.json()

    async def _resolve_destination(self, query: str) -> DestinationRef:
        """Resolve a place name to an entity id and coordinates."""
        key = query.strip().lower()
        if key in self._destination_cache:
            return self._destination_cache[key]

        payload = await self._get(
            "/api/v1/hotels/searchDestinationOrHotel", {"query": query}
        )
        candidates = payload.get("data") or []
        if not candidates:
            raise ValidationAppError(f"Could not find a destination for '{query}'.")

        chosen = candidates[0]
        coordinate = chosen.get("coordinates") or chosen.get("coordinate") or {}
        ref = DestinationRef(
            entity_id=str(chosen.get("entityId") or chosen.get("entity_id") or ""),
            name=str(chosen.get("entityName") or chosen.get("title") or query),
            latitude=_as_float(
                coordinate.get("latitude") if isinstance(coordinate, dict) else None
            ),
            longitude=_as_float(
                coordinate.get("longitude") if isinstance(coordinate, dict) else None
            ),
        )
        self._destination_cache[key] = ref
        return ref

    async def search(self, request: HotelSearchRequest) -> list[HotelResult]:
        destination = await self._resolve_destination(request.location)
        if not destination.entity_id:
            raise ValidationAppError(
                f"Could not resolve a hotel destination for '{request.location}'."
            )

        anchor: DestinationRef | None = None
        if request.landmark:
            try:
                anchor = await self._resolve_destination(request.landmark)
            except (ExternalServiceError, ValidationAppError) as exc:
                self._logger.warning(
                    "Could not resolve landmark; distance filtering disabled",
                    extra={"landmark": request.landmark, "error": str(exc)},
                )

        currency = request.currency or self._settings.default_currency
        params: dict[str, object] = {
            "entityId": destination.entity_id,
            "adults": request.adults,
            "currency": currency,
            "market": self._settings.default_market,
            "countryCode": self._settings.default_country,
        }
        if request.check_in:
            params["checkin"] = request.check_in
        if request.check_out:
            params["checkout"] = request.check_out

        payload = await self._get("/api/v1/hotels/searchHotels", params)
        raw_hotels = safe_get(payload, "data", "hotels", default=[]) or []

        results: list[HotelResult] = []
        for index, hotel in enumerate(raw_hotels):
            parsed = self._parse_hotel(hotel, index, request, anchor, currency)
            if parsed is not None:
                results.append(parsed)
        self._logger.info(
            "Hotel search complete",
            extra={"destination": destination.entity_id, "results": len(results)},
        )
        return results

    def _parse_hotel(
        self,
        hotel: dict,
        index: int,
        request: HotelSearchRequest,
        anchor: DestinationRef | None,
        currency: str,
    ) -> HotelResult | None:
        try:
            price = _as_float(
                safe_get(hotel, "price", "raw")
                if isinstance(hotel.get("price"), dict)
                else hotel.get("rawPrice") or hotel.get("price")
            )
            if price is None:
                return None

            coordinate = hotel.get("coordinates") or hotel.get("coordinate") or {}
            latitude = _as_float(
                coordinate.get("latitude") if isinstance(coordinate, dict) else None
            )
            longitude = _as_float(
                coordinate.get("longitude") if isinstance(coordinate, dict) else None
            )

            distance_km: float | None = None
            if (
                anchor is not None
                and anchor.latitude is not None
                and anchor.longitude is not None
                and latitude is not None
                and longitude is not None
            ):
                distance_km = round(
                    haversine_km(anchor.latitude, anchor.longitude, latitude, longitude),
                    3,
                )

            rating = _as_float(
                safe_get(hotel, "reviews", "score")
                if isinstance(hotel.get("reviews"), dict)
                else hotel.get("rating") or hotel.get("stars")
            )
            review_count = _as_int(
                safe_get(hotel, "reviews", "total")
                if isinstance(hotel.get("reviews"), dict)
                else hotel.get("reviewsCount")
            )

            return HotelResult(
                id=str(hotel.get("hotelId") or hotel.get("id") or f"hotel-{index}"),
                name=str(hotel.get("name") or "Unknown hotel"),
                address=_first_str(hotel.get("address"), hotel.get("distance")),
                latitude=latitude,
                longitude=longitude,
                distance_km=distance_km,
                rating=rating,
                review_count=review_count,
                price=price,
                currency=currency,
                price_per_night=True,
                refundable=hotel.get("isFreeCancellation")
                if isinstance(hotel.get("isFreeCancellation"), bool)
                else None,
                booking_link=self._booking_link(hotel, request),
                image_url=_first_image(hotel.get("images") or hotel.get("heroImage")),
                provider="skyscanner",
            )
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            self._logger.warning("Skipping unparseable hotel", extra={"error": str(exc)})
            return None

    @staticmethod
    def _booking_link(hotel: dict, request: HotelSearchRequest) -> str | None:
        """Return a provider booking link if present, else a search URL."""
        for key in ("url", "deeplink", "pageUrl"):
            value = hotel.get(key)
            if isinstance(value, str) and value.startswith("http"):
                return value
        location = request.location.replace(" ", "-").lower()
        return f"https://www.skyscanner.net/hotels/search?q={location}"

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


class HotelService:
    """Coordinates hotel providers and applies preference-driven filters."""

    def __init__(self, provider: HotelProvider, settings: Settings) -> None:
        self._provider = provider
        self._settings = settings
        self._logger = get_logger("services.hotel")

    async def search(
        self, request: HotelSearchRequest, preferences: Preferences | None = None
    ) -> list[HotelResult]:
        """Return filtered hotel results for ``request``.

        Preferences are merged into the request (without overriding explicit
        request fields) before filtering.
        """
        effective = self._apply_preferences(request, preferences)
        results = await self._provider.search(effective)
        filtered = [h for h in results if self._matches(h, effective)]
        filtered.sort(key=lambda h: h.price)
        limited = filtered[: self._settings.max_results]
        self._logger.info(
            "Hotels filtered",
            extra={
                "raw": len(results),
                "filtered": len(filtered),
                "returned": len(limited),
            },
        )
        return limited

    def _apply_preferences(
        self, request: HotelSearchRequest, preferences: Preferences | None
    ) -> HotelSearchRequest:
        if preferences is None:
            return request
        data = request.model_dump()
        if request.max_price is None and preferences.max_budget is not None:
            data["max_price"] = preferences.max_budget
        if request.currency is None and preferences.currency is not None:
            data["currency"] = preferences.currency
        if request.min_rating is None and preferences.hotel_min_rating is not None:
            data["min_rating"] = preferences.hotel_min_rating
        if (
            request.max_distance_km is None
            and preferences.hotel_max_distance_km is not None
        ):
            data["max_distance_km"] = preferences.hotel_max_distance_km
        return HotelSearchRequest(**data)

    @staticmethod
    def _matches(hotel: HotelResult, request: HotelSearchRequest) -> bool:
        if request.max_price is not None and hotel.price > request.max_price:
            return False
        if request.min_rating is not None:
            if hotel.rating is None or hotel.rating < request.min_rating:
                return False
        if request.max_distance_km is not None and hotel.distance_km is not None:
            if hotel.distance_km > request.max_distance_km:
                return False
        return True

    async def aclose(self) -> None:
        await self._provider.aclose()


def build_hotel_service(settings: Settings) -> HotelService:
    """Construct a :class:`HotelService` with the appropriate provider."""
    if settings.is_provider_configured:
        provider: HotelProvider = SkyScrapperHotelProvider(settings)
    else:
        provider = UnconfiguredHotelProvider()
    return HotelService(provider, settings)


# --------------------------------------------------------------------------- #
# Small parsing helpers (defensive: provider payloads vary)
# --------------------------------------------------------------------------- #
def _as_float(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _as_int(value: object) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _first_str(*values: object) -> str | None:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value
    return None


def _first_image(value: object) -> str | None:
    if isinstance(value, str) and value.startswith("http"):
        return value
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str) and item.startswith("http"):
                return item
            if isinstance(item, dict):
                url = item.get("url") or item.get("dynamic")
                if isinstance(url, str) and url.startswith("http"):
                    return url
    return None
