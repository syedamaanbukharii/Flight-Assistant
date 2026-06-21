"""Shared test fixtures.

Builds a fully offline application container: the LLM is left unconfigured (so
the planner exercises its deterministic heuristic + templated-response paths)
and the flight/hotel providers are replaced with in-memory fakes returning
canned results. Coupons are loaded from the repository seed file into a
temporary SQLite database.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agents.memory import MemoryStore
from agents.planner import PlannerAgent
from agents.tool_executor import ToolExecutor
from app.config import Settings
from app.startup import AppContainer
from models.database import Database
from models.schemas import FlightResult, FlightSearchRequest, HotelResult, HotelSearchRequest
from services.coupon_service import CouponService
from services.flight_service import FlightProvider, FlightService
from services.hotel_service import HotelProvider, HotelService
from services.llm_service import LLMService
from services.ranking_service import RankingService
from utils.constants import CabinClass

_SEED_PATH = Path(__file__).resolve().parent.parent / "backend" / "data" / "coupons.seed.json"


# --------------------------------------------------------------------------- #
# Canned provider data
# --------------------------------------------------------------------------- #
def sample_flights() -> list[FlightResult]:
    return [
        FlightResult(
            id="F1",
            airline="Emirates",
            airline_code="EK",
            origin="DEL",
            destination="DXB",
            departure_time="2025-01-10T19:00:00",
            arrival_time="2025-01-10T22:20:00",
            duration_minutes=200,
            stops=0,
            cabin_class=CabinClass.BUSINESS,
            price=75000.0,
            currency="INR",
            refundable=True,
            baggage_allowance="30kg",
        ),
        FlightResult(
            id="F2",
            airline="IndiGo",
            airline_code="6E",
            origin="DEL",
            destination="DXB",
            departure_time="2025-01-10T08:00:00",
            arrival_time="2025-01-10T11:50:00",
            duration_minutes=230,
            stops=1,
            cabin_class=CabinClass.ECONOMY,
            price=45000.0,
            currency="INR",
            refundable=False,
            baggage_allowance=None,
        ),
        FlightResult(
            id="F3",
            airline="Air India",
            airline_code="AI",
            origin="DEL",
            destination="DXB",
            departure_time="2025-01-10T22:00:00",
            arrival_time="2025-01-11T01:30:00",
            duration_minutes=210,
            stops=0,
            cabin_class=CabinClass.ECONOMY,
            price=60000.0,
            currency="INR",
            refundable=True,
            baggage_allowance="25kg",
        ),
    ]


def sample_hotels() -> list[HotelResult]:
    return [
        HotelResult(
            id="H1",
            name="Palm View",
            latitude=25.197,
            longitude=55.274,
            distance_km=0.5,
            rating=4.5,
            review_count=1200,
            price=12000.0,
            currency="INR",
        ),
        HotelResult(
            id="H2",
            name="Marina Bay",
            latitude=25.08,
            longitude=55.14,
            distance_km=2.0,
            rating=4.8,
            review_count=900,
            price=18000.0,
            currency="INR",
        ),
        HotelResult(
            id="H3",
            name="Budget Inn",
            latitude=25.19,
            longitude=55.27,
            distance_km=1.0,
            rating=3.9,
            review_count=300,
            price=6000.0,
            currency="INR",
        ),
    ]


class FakeFlightProvider(FlightProvider):
    """Returns canned flights regardless of the request."""

    def __init__(self, flights: list[FlightResult] | None = None) -> None:
        self._flights = flights if flights is not None else sample_flights()

    async def search(self, request: FlightSearchRequest) -> list[FlightResult]:
        return [f.model_copy(deep=True) for f in self._flights]


class FakeHotelProvider(HotelProvider):
    """Returns canned hotels regardless of the request."""

    def __init__(self, hotels: list[HotelResult] | None = None) -> None:
        self._hotels = hotels if hotels is not None else sample_hotels()

    async def search(self, request: HotelSearchRequest) -> list[HotelResult]:
        return [h.model_copy(deep=True) for h in self._hotels]


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    db_path = tmp_path / "test.db"
    return Settings(
        groq_api_key=None,
        rapidapi_key=None,
        database_url=f"sqlite:///{db_path}",
        coupon_seed_path=str(_SEED_PATH),
        enable_scheduler=False,
        coupon_refresh_enabled=False,
        max_results=10,
        default_currency="INR",
    )


@pytest.fixture
def database(settings: Settings) -> Database:
    db = Database(settings.database_url)
    db.create_all()
    return db


@pytest.fixture
def coupon_service(settings: Settings, database: Database) -> CouponService:
    service = CouponService(settings, database)
    service.load_seed()
    return service


@pytest.fixture
def ranking_service(settings: Settings) -> RankingService:
    return RankingService(settings)


@pytest.fixture
def container(
    settings: Settings,
    database: Database,
    coupon_service: CouponService,
    ranking_service: RankingService,
) -> AppContainer:
    llm = LLMService(settings)  # unconfigured -> heuristic planner
    flight_service = FlightService(FakeFlightProvider(), settings)
    hotel_service = HotelService(FakeHotelProvider(), settings)
    memory_store = MemoryStore(database)
    tool_executor = ToolExecutor(
        flight_service=flight_service,
        hotel_service=hotel_service,
        coupon_service=coupon_service,
        ranking_service=ranking_service,
        settings=settings,
    )
    planner = PlannerAgent(llm, tool_executor, memory_store, settings)
    return AppContainer(
        settings=settings,
        database=database,
        llm_service=llm,
        flight_service=flight_service,
        hotel_service=hotel_service,
        coupon_service=coupon_service,
        ranking_service=ranking_service,
        memory_store=memory_store,
        tool_executor=tool_executor,
        planner=planner,
    )


# --------------------------------------------------------------------------- #
# HTTP client fixture (real wiring, keyless providers)
# --------------------------------------------------------------------------- #
@pytest.fixture
def client(settings: Settings):
    """A ``TestClient`` wrapping the real application.

    The app is built with the keyless ``settings`` fixture, so the lifespan
    wires *unconfigured* flight/hotel providers (``/search/*`` therefore returns
    503) while coupons still load from the seed file into the temp database.
    Using the client as a context manager runs the startup/shutdown lifespan.
    """
    from fastapi.testclient import TestClient

    from app.main import create_app

    app = create_app(settings)
    with TestClient(app) as test_client:
        yield test_client
