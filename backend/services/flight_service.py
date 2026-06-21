"""Flight search service.

The concrete provider (RapidAPI Sky Scrapper) is hidden behind the
:class:`FlightProvider` interface so it can be swapped without touching the
business logic. The service resolves airports, queries the provider, then
applies preference-driven filters (budget, airline, refundable, departure
window) before returning results. Ranking happens later in the pipeline.

This integration uses an official aggregator API; it never scrapes airline or
metasearch websites directly.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

import httpx

from app.config import Settings
from models.schemas import FlightResult, FlightSearchRequest, Preferences
from utils.constants import (
    SKYSCANNER_CABIN_PARAM,
    CabinClass,
    TimeWindow,
)
from utils.errors import ExternalServiceError, NotConfiguredError, ValidationAppError
from utils.helpers import async_retry, safe_get, within_time_window
from utils.logger import get_logger


@dataclass(frozen=True)
class AirportRef:
    """A resolved place reference used to query flights."""

    sky_id: str
    entity_id: str
    name: str


class FlightProvider(abc.ABC):
    """Abstract flight data provider."""

    @abc.abstractmethod
    async def search(self, request: FlightSearchRequest) -> list[FlightResult]:
        """Return raw flight itineraries for a search request."""

    async def aclose(self) -> None:  # pragma: no cover - default no-op
        """Release any held resources."""


class UnconfiguredFlightProvider(FlightProvider):
    """Fallback provider used when no API credentials are configured."""

    async def search(self, request: FlightSearchRequest) -> list[FlightResult]:
        raise NotConfiguredError(
            "Flight search is not configured. Set RAPIDAPI_KEY (and RAPIDAPI_HOST) "
            "to enable live flight results."
        )


class SkyScrapperFlightProvider(FlightProvider):
    """Flight provider backed by the RapidAPI Sky Scrapper API."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._logger = get_logger("services.flight.skyscrapper")
        self._client: httpx.AsyncClient | None = None
        self._airport_cache: dict[str, AirportRef] = {}

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
                "The flight provider returned an error.",
                detail={"status": exc.response.status_code, "path": path},
            ) from exc
        except httpx.HTTPError as exc:
            raise ExternalServiceError("Could not reach the flight provider.") from exc
        return response.json()

    async def _resolve_airport(self, query: str) -> AirportRef:
        key = query.strip().lower()
        if key in self._airport_cache:
            return self._airport_cache[key]

        payload = await self._get(
            "/api/v1/flights/searchAirport",
            {"query": query, "locale": self._settings.default_market},
        )
        candidates = payload.get("data") or []
        if not candidates:
            raise ValidationAppError(f"Could not find an airport for '{query}'.")

        # Prefer a CITY entity; otherwise take the first relevant result.
        chosen = next(
            (
                c
                for c in candidates
                if safe_get(c, "navigation", "entityType") == "CITY"
            ),
            candidates[0],
        )
        params = safe_get(chosen, "navigation", "relevantFlightParams", default={})
        ref = AirportRef(
            sky_id=str(params.get("skyId") or chosen.get("skyId") or query),
            entity_id=str(params.get("entityId") or chosen.get("entityId") or ""),
            name=str(safe_get(chosen, "presentation", "title", default=query)),
        )
        self._airport_cache[key] = ref
        return ref

    async def search(self, request: FlightSearchRequest) -> list[FlightResult]:
        origin = await self._resolve_airport(request.origin)
        destination = await self._resolve_airport(request.destination)

        currency = request.currency or self._settings.default_currency
        cabin = SKYSCANNER_CABIN_PARAM.get(
            request.cabin_class or CabinClass.ECONOMY, "economy"
        )
        params: dict[str, object] = {
            "originSkyId": origin.sky_id,
            "destinationSkyId": destination.sky_id,
            "originEntityId": origin.entity_id,
            "destinationEntityId": destination.entity_id,
            "cabinClass": cabin,
            "adults": request.adults,
            "sortBy": "best",
            "currency": currency,
            "market": self._settings.default_market,
            "countryCode": self._settings.default_country,
        }
        if request.date:
            params["date"] = request.date
        if request.return_date:
            params["returnDate"] = request.return_date

        payload = await self._get("/api/v2/flights/searchFlights", params)
        itineraries = safe_get(payload, "data", "itineraries", default=[]) or []

        results: list[FlightResult] = []
        for index, itinerary in enumerate(itineraries):
            parsed = self._parse_itinerary(
                itinerary, index, request, origin, destination, currency
            )
            if parsed is not None:
                results.append(parsed)
        self._logger.info(
            "Flight search complete",
            extra={
                "origin": origin.sky_id,
                "destination": destination.sky_id,
                "results": len(results),
            },
        )
        return results

    def _parse_itinerary(
        self,
        itinerary: dict,
        index: int,
        request: FlightSearchRequest,
        origin: AirportRef,
        destination: AirportRef,
        currency: str,
    ) -> FlightResult | None:
        try:
            price = safe_get(itinerary, "price", "raw")
            legs = itinerary.get("legs") or []
            if price is None or not legs:
                return None

            first_leg = legs[0]
            carrier = safe_get(
                first_leg, "carriers", "marketing", default=[{}]
            )[0]
            total_duration = sum(int(leg.get("durationInMinutes") or 0) for leg in legs)
            total_stops = sum(int(leg.get("stopCount") or 0) for leg in legs)

            refundable = safe_get(itinerary, "farePolicy", "isCancellationAllowed")

            return FlightResult(
                id=str(itinerary.get("id") or f"itinerary-{index}"),
                airline=str(carrier.get("name") or "Unknown airline"),
                airline_code=str(carrier.get("alternateId") or carrier.get("id") or "") or None,
                origin=str(safe_get(first_leg, "origin", "displayCode", default=origin.sky_id)),
                destination=str(
                    safe_get(legs[-1], "destination", "displayCode", default=destination.sky_id)
                ),
                departure_time=first_leg.get("departure"),
                arrival_time=legs[-1].get("arrival"),
                duration_minutes=total_duration or None,
                stops=total_stops,
                cabin_class=request.cabin_class,
                price=float(price),
                currency=currency,
                refundable=refundable if isinstance(refundable, bool) else None,
                baggage_allowance=None,
                booking_link=self._booking_link(request, origin, destination),
                provider="skyscanner",
            )
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            self._logger.warning(
                "Skipping unparseable itinerary", extra={"error": str(exc)}
            )
            return None

    def _booking_link(
        self, request: FlightSearchRequest, origin: AirportRef, destination: AirportRef
    ) -> str:
        """Build a Skyscanner search URL as a stable booking deep-link."""
        base = "https://www.skyscanner.net/transport/flights"
        segments = [origin.sky_id.lower(), destination.sky_id.lower()]
        if request.date:
            segments.append(request.date.replace("-", "")[2:])
        if request.return_date:
            segments.append(request.return_date.replace("-", "")[2:])
        cabin = SKYSCANNER_CABIN_PARAM.get(request.cabin_class or CabinClass.ECONOMY, "economy")
        return f"{base}/{'/'.join(segments)}/?adultsv2={request.adults}&cabinclass={cabin}"

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


class FlightService:
    """Coordinates flight providers and applies preference-driven filters."""

    def __init__(self, provider: FlightProvider, settings: Settings) -> None:
        self._provider = provider
        self._settings = settings
        self._logger = get_logger("services.flight")

    async def search(
        self, request: FlightSearchRequest, preferences: Preferences | None = None
    ) -> list[FlightResult]:
        """Return filtered flight results for ``request``.

        Preferences are merged into the request (without overriding explicit
        request fields) before filtering.
        """
        effective = self._apply_preferences(request, preferences)
        results = await self._provider.search(effective)
        filtered = [f for f in results if self._matches(f, effective)]
        filtered.sort(key=lambda f: f.price)
        limited = filtered[: self._settings.max_results]
        self._logger.info(
            "Flights filtered",
            extra={"raw": len(results), "filtered": len(filtered), "returned": len(limited)},
        )
        return limited

    def _apply_preferences(
        self, request: FlightSearchRequest, preferences: Preferences | None
    ) -> FlightSearchRequest:
        if preferences is None:
            return request
        data = request.model_dump()
        if request.cabin_class is None and preferences.cabin_class is not None:
            data["cabin_class"] = preferences.cabin_class
        if request.max_price is None and preferences.max_budget is not None:
            data["max_price"] = preferences.max_budget
        if request.currency is None and preferences.currency is not None:
            data["currency"] = preferences.currency
        if not request.airlines and preferences.airlines:
            data["airlines"] = preferences.airlines
        if request.refundable is None and preferences.refundable is not None:
            data["refundable"] = preferences.refundable
        if (
            request.departure_time_window in (None, TimeWindow.ANY)
            and preferences.departure_time_window is not None
        ):
            data["departure_time_window"] = preferences.departure_time_window
        return FlightSearchRequest(**data)

    @staticmethod
    def _matches(flight: FlightResult, request: FlightSearchRequest) -> bool:
        if request.max_price is not None and flight.price > request.max_price:
            return False
        if request.refundable is True and flight.refundable is not True:
            return False
        if request.airlines:
            wanted = {a.lower() for a in request.airlines}
            if not any(w in flight.airline.lower() for w in wanted):
                return False
        if not within_time_window(flight.departure_time, request.departure_time_window):
            return False
        return True

    async def aclose(self) -> None:
        await self._provider.aclose()


def build_flight_service(settings: Settings) -> FlightService:
    """Construct a :class:`FlightService` with the appropriate provider."""
    if settings.is_provider_configured:
        provider: FlightProvider = SkyScrapperFlightProvider(settings)
    else:
        provider = UnconfiguredFlightProvider()
    return FlightService(provider, settings)
