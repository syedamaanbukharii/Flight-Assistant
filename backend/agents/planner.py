"""Planner agent.

The planner is the conversational brain. It turns a user message into a
:class:`PlannerDecision` (intent, structured tool parameters, preference
updates), then orchestrates the tools in native Python and finally produces a
grounded natural-language reply.

Two design rules are enforced here:

1. **The LLM decides, Python executes.** The model only emits a structured
   plan and, separately, prose grounded in tool output. It never fabricates
   flights, hotels or prices — those always come from the tools.
2. **Graceful degradation.** When the LLM is unavailable or misbehaves, a
   deterministic regex-based planner and a templated responder keep the
   assistant fully functional (and make the system testable offline).
"""

from __future__ import annotations

import json
import re
from datetime import date, datetime, timedelta

from agents.memory import MemoryStore, SessionMemory
from agents.tool_executor import TOOL_SPECS, ToolExecutor
from app.config import Settings
from models.schemas import (
    ChatResponse,
    Coupon,
    FlightSearchRequest,
    HotelSearchRequest,
    PlannerDecision,
    Preferences,
    RankedFlight,
    RankedHotel,
)
from services.llm_service import LLMService
from utils.constants import (
    ROLE_ASSISTANT,
    ROLE_USER,
    WALKING_DISTANCE_KM,
    CabinClass,
    Intent,
    SortBy,
    TimeWindow,
)
from utils.errors import AppError
from utils.helpers import minutes_to_human, parse_currency_amount
from utils.logger import get_logger

_WEEKDAYS = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}

_KNOWN_AIRLINES = [
    "emirates",
    "qatar airways",
    "etihad",
    "air india",
    "indigo",
    "vistara",
    "spicejet",
    "lufthansa",
    "british airways",
    "singapore airlines",
    "cathay pacific",
    "united",
    "delta",
    "american airlines",
    "klm",
    "turkish airlines",
    "air france",
    "qantas",
]

_ROUTE_WITH_FROM_RE = re.compile(
    r"\bfrom\s+([a-z .'-]+?)\s+to\s+([a-z .'-]+?)"
    r"(?=\s+(?:next|this|on|tomorrow|today|under|below|less|in|for|by|departing|"
    r"leaving|returning|round|one|with|business|economy|first|premium|refundable|"
    r"non|cheapest|fastest|best)\b|[,.?!]|$)",
    re.I,
)
_ROUTE_BARE_RE = re.compile(
    r"\b([a-z][a-z .'-]+?)\s+to\s+([a-z .'-]+?)"
    r"(?=\s+(?:next|this|on|tomorrow|today|under|below|less|in|for|by|departing|"
    r"leaving|returning|round|one|with|business|economy|first|premium|refundable|"
    r"non|cheapest|fastest|best)\b|[,.?!]|$)",
    re.I,
)
_HOTEL_IN_RE = re.compile(r"hotels?\s+in\s+([a-z .'-]+?)(?=[,.?!]|$|\s+near\b)", re.I)
_NEAR_RE = re.compile(
    r"(?:near|close to|next to|around|within walking distance of|walking distance of|by)\s+"
    r"([a-z0-9 .'-]+?)(?=[,.?!]|$)",
    re.I,
)
_BUDGET_RE = re.compile(
    r"(?:under|below|less than|within|max(?:imum)?|up to|cheaper than|budget of)\s+"
    r"([₹$€£]?\s?[\d.,]+\s?(?:k|thousand|lakh|lakhs|lac|cr|crore|million|m)?)",
    re.I,
)
_ONLY_RE = re.compile(r"\b(?:only|just|show)\b", re.I)


class PlannerAgent:
    """Plans and fulfils a single conversational turn."""

    def __init__(
        self,
        llm_service: LLMService,
        tool_executor: ToolExecutor,
        memory_store: MemoryStore,
        settings: Settings,
    ) -> None:
        self._llm = llm_service
        self._tools = tool_executor
        self._memory = memory_store
        self._settings = settings
        self._logger = get_logger("agents.planner")

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #
    async def handle(self, session_id: str, user_message: str) -> ChatResponse:
        """Process one user message and return a full :class:`ChatResponse`."""
        memory = await self._memory.get_or_create(session_id)
        await self._memory.add_message(session_id, ROLE_USER, user_message)

        decision = await self._plan(user_message, memory)
        self._logger.info(
            "Planner decision",
            extra={"intent": decision.intent.value, "session": session_id},
        )

        preferences = memory.preferences
        if decision.preferences is not None:
            preferences = await self._memory.update_preferences(
                session_id, decision.preferences
            )

        notes: list[str] = []
        if decision.notes:
            notes.append(decision.notes)

        sort_by = decision.sort_by or SortBy.BEST_VALUE
        flights: list[RankedFlight] = []
        hotels: list[RankedHotel] = []

        ran_search = False
        if decision.flight_query is not None:
            ran_search = True
            result = await self._tools.run_flight_search(
                decision.flight_query,
                preferences=preferences,
                apply_coupons=True,
                sort_by=sort_by,
            )
            if result.ok:
                flights = result.data
                await self._memory.set_last_flights(session_id, flights)
            elif result.error:
                notes.append(result.error)

        if decision.hotel_query is not None:
            ran_search = True
            result = await self._tools.run_hotel_search(
                decision.hotel_query,
                preferences=preferences,
                apply_coupons=True,
                sort_by=sort_by,
            )
            if result.ok:
                hotels = result.data
                await self._memory.set_last_hotels(session_id, hotels)
            elif result.error:
                notes.append(result.error)

        # Follow-up operations on previously returned results.
        if not ran_search and decision.intent in (
            Intent.RERANK,
            Intent.APPLY_COUPONS,
        ):
            apply = decision.apply_coupons or decision.intent == Intent.APPLY_COUPONS
            if memory.last_flights:
                flights = await self._tools.rank_flights(
                    memory.last_flights,
                    sort_by=sort_by,
                    preferences=preferences,
                    apply_coupons=apply,
                )
                await self._memory.set_last_flights(session_id, flights)
            if memory.last_hotels:
                hotels = await self._tools.rank_hotels(
                    memory.last_hotels,
                    sort_by=sort_by,
                    preferences=preferences,
                    apply_coupons=apply,
                )
                await self._memory.set_last_hotels(session_id, hotels)
            if not memory.last_flights and not memory.last_hotels:
                notes.append(
                    "There are no earlier results to update yet — try a search first."
                )

        coupons_applied = self._collect_coupons(flights, hotels)
        reply = await self._synthesize(
            user_message, decision, flights, hotels, coupons_applied, preferences, notes
        )
        await self._memory.add_message(session_id, ROLE_ASSISTANT, reply)

        return ChatResponse(
            session_id=session_id,
            intent=decision.intent,
            reply=reply,
            flights=flights,
            hotels=hotels,
            coupons_applied=coupons_applied,
            preferences=preferences,
            notes=notes,
        )

    # ------------------------------------------------------------------ #
    # Planning
    # ------------------------------------------------------------------ #
    async def _plan(self, user_message: str, memory: SessionMemory) -> PlannerDecision:
        """Produce a plan via the LLM, falling back to heuristics on failure."""
        if not self._llm.is_configured:
            return self.heuristic_plan(user_message, memory.preferences)

        messages = self._build_planner_messages(user_message, memory)
        for attempt in range(2):
            try:
                raw = await self._llm.complete_json(messages, temperature=0.0)
                return PlannerDecision.model_validate(raw)
            except AppError as exc:
                self._logger.warning(
                    "Planner LLM call failed",
                    extra={"attempt": attempt, "error": exc.message},
                )
                break
            except Exception as exc:  # noqa: BLE001 - validation/parse issues
                self._logger.warning(
                    "Planner decision invalid; retrying",
                    extra={"attempt": attempt, "error": str(exc)},
                )
                messages = messages + [
                    {
                        "role": ROLE_USER,
                        "content": "Return ONLY a valid JSON object matching the schema.",
                    }
                ]
        return self.heuristic_plan(user_message, memory.preferences)

    def _build_planner_messages(
        self, user_message: str, memory: SessionMemory
    ) -> list[dict[str, str]]:
        today = datetime.now()
        tool_lines = "\n".join(
            f"- {spec['name']}: {spec['description']}" for spec in TOOL_SPECS
        )
        system = f"""You are the planning module of a flight and travel assistant.
Today's date is {today:%Y-%m-%d} ({today:%A}). Resolve relative dates such as
"next Friday" or "tomorrow" to absolute YYYY-MM-DD values.

Decide the user's intent and extract structured parameters. You ONLY plan; you
never invent flight, hotel or price data. Available tools:
{tool_lines}

Respond with a single JSON object using these keys:
  "intent": one of {[i.value for i in Intent]}
  "tools": array of tool names to run
  "flight_query": object|null with keys origin, destination, date, return_date,
      cabin_class (economy|premium_economy|business|first), adults, max_price,
      currency, airlines (array), refundable (bool), departure_time_window
      (any|morning|afternoon|evening|night), sort_by
  "hotel_query": object|null with keys location, landmark, check_in, check_out,
      adults, max_price, currency, min_rating, max_distance_km, sort_by
  "preferences": object|null with any stated durable preferences (airlines,
      cabin_class, max_budget, currency, departure_time_window, refundable,
      hotel_min_rating, hotel_max_distance_km)
  "apply_coupons": boolean
  "sort_by": one of {[s.value for s in SortBy]} or null
  "notes": short string, may be empty

Rules:
- Use flight_query only when the user is searching flights (needs origin and
  destination). Use hotel_query only for hotel searches (needs a location).
- For "apply coupons" or "rank/sort" requests about existing results, set the
  matching intent (apply_coupons or rerank) and leave queries null.
- Always copy stated preferences into "preferences" so they persist.
- Output JSON only, no commentary."""

        history = "\n".join(
            f"{m['role']}: {m['content']}" for m in memory.messages[-6:]
        )
        prefs = memory.preferences.model_dump(exclude_none=True)
        user = (
            f"Known preferences: {json.dumps(prefs) if prefs else 'none'}\n"
            f"Recent conversation:\n{history or '(none)'}\n\n"
            f"User message: {user_message}"
        )
        return [
            {"role": "system", "content": system},
            {"role": ROLE_USER, "content": user},
        ]

    # ------------------------------------------------------------------ #
    # Heuristic planner (LLM-free fallback)
    # ------------------------------------------------------------------ #
    def heuristic_plan(self, message: str, preferences: Preferences) -> PlannerDecision:
        """Deterministically derive a plan from the message using regexes."""
        text = message.strip()
        lower = text.lower()

        cabin = self._detect_cabin(lower)
        airlines = self._detect_airlines(lower)
        currency = self._detect_currency(lower)
        budget = self._detect_budget(text)
        window = TimeWindow.from_text(lower)
        refundable = self._detect_refundable(lower)
        sort_by = self._detect_sort(lower)

        patch = Preferences(
            airlines=airlines,
            cabin_class=cabin,
            max_budget=budget,
            currency=currency,
            departure_time_window=window if window != TimeWindow.ANY else None,
            refundable=refundable,
        )

        origin, destination = self._detect_route(text)
        hotel_location, landmark, walking = self._detect_hotel(text)

        wants_hotel = hotel_location is not None or landmark is not None
        wants_flight = origin is not None and destination is not None
        wants_coupons = bool(re.search(r"coupon|promo|discount|offer|deal", lower))
        date_value = self._detect_date(lower)

        flight_query = None
        if wants_flight:
            flight_query = FlightSearchRequest(
                origin=origin,  # type: ignore[arg-type]
                destination=destination,  # type: ignore[arg-type]
                date=date_value,
                cabin_class=cabin,
                max_price=budget,
                currency=currency,
                airlines=airlines,
                refundable=refundable,
                departure_time_window=window,
                sort_by=sort_by or SortBy.BEST_VALUE,
            )

        hotel_query = None
        if wants_hotel:
            location = hotel_location or landmark
            hotel_query = HotelSearchRequest(
                location=location,  # type: ignore[arg-type]
                landmark=landmark,
                max_price=budget,
                currency=currency,
                min_rating=preferences.hotel_min_rating,
                max_distance_km=WALKING_DISTANCE_KM if walking else None,
                sort_by=sort_by or SortBy.BEST_VALUE,
            )

        intent, tools = self._infer_intent(
            wants_flight=wants_flight,
            wants_hotel=wants_hotel,
            wants_coupons=wants_coupons,
            sort_by=sort_by,
            patch=patch,
        )

        return PlannerDecision(
            intent=intent,
            tools=tools,
            flight_query=flight_query,
            hotel_query=hotel_query,
            preferences=patch if self._has_preference(patch) else None,
            apply_coupons=wants_coupons or intent == Intent.APPLY_COUPONS,
            sort_by=sort_by,
            notes="",
        )

    # --- heuristic helpers --- #
    @staticmethod
    def _detect_cabin(lower: str) -> CabinClass | None:
        for token in ("first class", "business", "premium economy", "economy", "first"):
            if token in lower:
                return CabinClass.from_text(token)
        return None

    @staticmethod
    def _detect_airlines(lower: str) -> list[str]:
        found = [name.title() for name in _KNOWN_AIRLINES if name in lower]
        return sorted(set(found))

    @staticmethod
    def _detect_currency(lower: str) -> str | None:
        if "₹" in lower or "inr" in lower or "rupee" in lower:
            return "INR"
        if "$" in lower or "usd" in lower or "dollar" in lower:
            return "USD"
        if "€" in lower or "eur" in lower or "euro" in lower:
            return "EUR"
        if "£" in lower or "gbp" in lower or "pound" in lower:
            return "GBP"
        return None

    @staticmethod
    def _detect_budget(text: str) -> float | None:
        match = _BUDGET_RE.search(text)
        if not match:
            return None
        return parse_currency_amount(match.group(1))

    @staticmethod
    def _detect_refundable(lower: str) -> bool | None:
        if re.search(r"non[- ]?refundable", lower):
            return False
        if "refundable" in lower:
            return True
        return None

    @staticmethod
    def _detect_sort(lower: str) -> SortBy | None:
        if re.search(r"best value|value for money|recommended", lower):
            return SortBy.BEST_VALUE
        if re.search(r"cheap|lowest price|least expensive|budget", lower):
            return SortBy.CHEAPEST
        if re.search(r"fast|quick|short(est)? (flight|duration)", lower):
            return SortBy.FASTEST
        if re.search(r"best rated|highest rated|top rated|by rating", lower):
            return SortBy.RATING
        if re.search(r"closest|nearest|shortest distance", lower):
            return SortBy.CLOSEST
        return None

    @staticmethod
    def _detect_route(text: str) -> tuple[str | None, str | None]:
        match = _ROUTE_WITH_FROM_RE.search(text) or _ROUTE_BARE_RE.search(text)
        if not match:
            return None, None
        origin = match.group(1).strip(" .,'-")
        destination = match.group(2).strip(" .,'-")
        if not origin or not destination:
            return None, None
        return origin.title(), destination.title()

    @staticmethod
    def _detect_hotel(text: str) -> tuple[str | None, str | None, bool]:
        lower = text.lower()
        if "hotel" not in lower and "stay" not in lower and "accommodation" not in lower:
            # A bare "near X" without hotel intent is not a hotel search.
            if not _NEAR_RE.search(text) or "flight" in lower:
                return None, None, False
        location_match = _HOTEL_IN_RE.search(text)
        near_match = _NEAR_RE.search(text)
        walking = "walking distance" in lower
        location = location_match.group(1).strip(" .,'-").title() if location_match else None
        landmark = near_match.group(1).strip(" .,'-").title() if near_match else None
        if location is None and landmark is None and "hotel" in lower:
            return None, None, walking
        return location, landmark, walking

    def _detect_date(self, lower: str) -> str | None:
        explicit = re.search(r"(\d{4}-\d{2}-\d{2})", lower)
        if explicit:
            return explicit.group(1)
        today = date.today()
        if "day after tomorrow" in lower:
            return (today + timedelta(days=2)).isoformat()
        if "tomorrow" in lower:
            return (today + timedelta(days=1)).isoformat()
        if "today" in lower or "tonight" in lower:
            return today.isoformat()
        match = re.search(r"\b(next\s+)?(" + "|".join(_WEEKDAYS) + r")\b", lower)
        if match:
            force_next = bool(match.group(1))
            return self._next_weekday(today, _WEEKDAYS[match.group(2)], force_next).isoformat()
        return None

    @staticmethod
    def _next_weekday(base: date, weekday: int, force_next: bool) -> date:
        """Return the upcoming date for ``weekday``.

        A named weekday (with or without "next") resolves to the next future
        occurrence; if today *is* that weekday we advance a full week so the
        date is always in the future, matching everyday usage of phrases like
        "next Friday".
        """
        days_ahead = (weekday - base.weekday()) % 7
        if days_ahead == 0:
            days_ahead = 7
        return base + timedelta(days=days_ahead)

    @staticmethod
    def _has_preference(patch: Preferences) -> bool:
        return (
            bool(patch.airlines)
            or patch.cabin_class is not None
            or patch.max_budget is not None
            or patch.currency is not None
            or patch.departure_time_window is not None
            or patch.refundable is not None
            or patch.hotel_min_rating is not None
            or patch.hotel_max_distance_km is not None
        )

    @staticmethod
    def _infer_intent(
        *,
        wants_flight: bool,
        wants_hotel: bool,
        wants_coupons: bool,
        sort_by: SortBy | None,
        patch: Preferences,
    ) -> tuple[Intent, list]:
        from utils.constants import ToolName

        if wants_flight and wants_hotel:
            return Intent.COMBINED_SEARCH, [
                ToolName.FLIGHT_SEARCH,
                ToolName.HOTEL_SEARCH,
                ToolName.COUPON_LOOKUP,
                ToolName.RANKING,
            ]
        if wants_flight:
            return Intent.FLIGHT_SEARCH, [
                ToolName.FLIGHT_SEARCH,
                ToolName.COUPON_LOOKUP,
                ToolName.RANKING,
            ]
        if wants_hotel:
            return Intent.HOTEL_SEARCH, [
                ToolName.HOTEL_SEARCH,
                ToolName.COUPON_LOOKUP,
                ToolName.RANKING,
            ]
        if wants_coupons:
            return Intent.APPLY_COUPONS, [ToolName.COUPON_LOOKUP, ToolName.RANKING]
        if sort_by is not None:
            return Intent.RERANK, [ToolName.RANKING]
        if PlannerAgent._has_preference(patch):
            return Intent.UPDATE_PREFERENCES, [ToolName.PREFERENCE_MEMORY]
        return Intent.CHITCHAT, []

    # ------------------------------------------------------------------ #
    # Response synthesis
    # ------------------------------------------------------------------ #
    async def _synthesize(
        self,
        user_message: str,
        decision: PlannerDecision,
        flights: list[RankedFlight],
        hotels: list[RankedHotel],
        coupons: list[Coupon],
        preferences: Preferences,
        notes: list[str],
    ) -> str:
        """Produce the natural-language reply, grounded strictly in tool data."""
        if not self._llm.is_configured:
            return self._template_reply(decision, flights, hotels, coupons, notes)
        try:
            messages = self._build_synthesis_messages(
                user_message, decision, flights, hotels, coupons, preferences, notes
            )
            return (await self._llm.complete(messages, temperature=0.3)).strip()
        except AppError as exc:
            self._logger.warning(
                "Synthesis LLM failed; using template", extra={"error": exc.message}
            )
            return self._template_reply(decision, flights, hotels, coupons, notes)

    def _build_synthesis_messages(
        self,
        user_message: str,
        decision: PlannerDecision,
        flights: list[RankedFlight],
        hotels: list[RankedHotel],
        coupons: list[Coupon],
        preferences: Preferences,
        notes: list[str],
    ) -> list[dict[str, str]]:
        system = (
            "You are a concise, friendly travel assistant. Summarise the results "
            "for the user in clear prose. Ground every statement strictly in the "
            "DATA provided; never invent flights, hotels, prices or links. If the "
            "data is empty, say so and, where relevant, mention any notes. Mention "
            "the top options, their prices (with currency), key attributes and any "
            "coupon savings. Keep it brief and skimmable."
        )
        payload = {
            "intent": decision.intent.value,
            "preferences": preferences.model_dump(exclude_none=True),
            "flights": [self._flight_brief(f) for f in flights[:5]],
            "hotels": [self._hotel_brief(h) for h in hotels[:5]],
            "coupons_applied": [c.code for c in coupons],
            "notes": notes,
        }
        user = (
            f"User message: {user_message}\n\n"
            f"DATA (the only facts you may use):\n{json.dumps(payload, default=str)}"
        )
        return [
            {"role": "system", "content": system},
            {"role": ROLE_USER, "content": user},
        ]

    # ------------------------------------------------------------------ #
    # Deterministic templated reply (LLM-free)
    # ------------------------------------------------------------------ #
    def _template_reply(
        self,
        decision: PlannerDecision,
        flights: list[RankedFlight],
        hotels: list[RankedHotel],
        coupons: list[Coupon],
        notes: list[str],
    ) -> str:
        lines: list[str] = []
        if flights:
            lines.append(f"Here are the top {min(len(flights), 5)} flights I found:")
            for f in flights[:5]:
                lines.append(self._flight_line(f))
        if hotels:
            lines.append(f"Here are the top {min(len(hotels), 5)} hotels I found:")
            for h in hotels[:5]:
                lines.append(self._hotel_line(h))
        if coupons:
            codes = ", ".join(c.code for c in coupons)
            lines.append(f"Applied coupons where eligible: {codes}.")
        if not flights and not hotels:
            if decision.intent == Intent.CHITCHAT:
                lines.append(
                    "I can search flights and hotels, apply coupons and rank results. "
                    "Tell me where and when you'd like to travel."
                )
            else:
                lines.append("I couldn't find any results for that request.")
        lines.extend(notes)
        return "\n".join(lines)

    @staticmethod
    def _flight_line(f: RankedFlight) -> str:
        bits = [f"{f.rank}. {f.airline} {f.origin}→{f.destination}"]
        if f.duration_minutes:
            bits.append(minutes_to_human(f.duration_minutes))
        stops = "non-stop" if f.stops == 0 else f"{f.stops} stop(s)"
        bits.append(stops)
        price = f"{f.currency} {f.final_price:,.0f}"
        if f.applied_coupon:
            price += f" (saved {f.currency} {f.applied_coupon.savings:,.0f})"
        bits.append(price)
        if f.refundable:
            bits.append("refundable")
        if f.booking_link:
            bits.append(f"book: {f.booking_link}")
        return " — ".join(bits)

    @staticmethod
    def _hotel_line(h: RankedHotel) -> str:
        bits = [f"{h.rank}. {h.name}"]
        if h.rating is not None:
            bits.append(f"{h.rating:.1f}★")
        if h.distance_km is not None:
            bits.append(f"{h.distance_km:.1f} km away")
        price = f"{h.currency} {h.final_price:,.0f}/night"
        if h.applied_coupon:
            price += f" (saved {h.currency} {h.applied_coupon.savings:,.0f})"
        bits.append(price)
        if h.booking_link:
            bits.append(f"book: {h.booking_link}")
        return " — ".join(bits)

    @staticmethod
    def _flight_brief(f: RankedFlight) -> dict:
        return {
            "rank": f.rank,
            "airline": f.airline,
            "route": f"{f.origin}->{f.destination}",
            "departure": f.departure_time,
            "duration": minutes_to_human(f.duration_minutes),
            "stops": f.stops,
            "cabin": f.cabin_class.value if f.cabin_class else None,
            "price": f.final_price,
            "currency": f.currency,
            "refundable": f.refundable,
            "coupon": f.applied_coupon.coupon.code if f.applied_coupon else None,
            "savings": f.applied_coupon.savings if f.applied_coupon else 0,
            "booking_link": f.booking_link,
            "why": f.rationale,
        }

    @staticmethod
    def _hotel_brief(h: RankedHotel) -> dict:
        return {
            "rank": h.rank,
            "name": h.name,
            "rating": h.rating,
            "distance_km": h.distance_km,
            "price_per_night": h.final_price,
            "currency": h.currency,
            "coupon": h.applied_coupon.coupon.code if h.applied_coupon else None,
            "savings": h.applied_coupon.savings if h.applied_coupon else 0,
            "booking_link": h.booking_link,
            "why": h.rationale,
        }

    @staticmethod
    def _collect_coupons(
        flights: list[RankedFlight], hotels: list[RankedHotel]
    ) -> list[Coupon]:
        seen: dict[str, Coupon] = {}
        for item in (*flights, *hotels):
            applied = item.applied_coupon
            if applied is not None and applied.coupon.code not in seen:
                seen[applied.coupon.code] = applied.coupon
        return list(seen.values())
