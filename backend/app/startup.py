"""Application startup wiring.

Constructs the dependency-injection container that owns every long-lived
collaborator (database, services, agents, scheduler). Building the object graph
in one place keeps wiring explicit, makes testing straightforward (swap a
provider, inject a fake LLM) and gives the FastAPI lifespan a single handle for
clean startup and shutdown.
"""

from __future__ import annotations

from dataclasses import dataclass

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from agents.memory import MemoryStore
from agents.planner import PlannerAgent
from agents.tool_executor import ToolExecutor
from app.config import Settings
from models.database import Database
from services.coupon_service import CouponService
from services.flight_service import FlightService, build_flight_service
from services.hotel_service import HotelService, build_hotel_service
from services.llm_service import LLMService
from services.ranking_service import RankingService
from utils.logger import get_logger

_logger = get_logger("app.startup")


@dataclass
class AppContainer:
    """Holds the application's singleton collaborators."""

    settings: Settings
    database: Database
    llm_service: LLMService
    flight_service: FlightService
    hotel_service: HotelService
    coupon_service: CouponService
    ranking_service: RankingService
    memory_store: MemoryStore
    tool_executor: ToolExecutor
    planner: PlannerAgent
    scheduler: AsyncIOScheduler | None = None

    async def aclose(self) -> None:
        """Release all held resources (HTTP clients, scheduler)."""
        if self.scheduler is not None and self.scheduler.running:
            self.scheduler.shutdown(wait=False)
        await self.flight_service.aclose()
        await self.hotel_service.aclose()
        await self.llm_service.aclose()


def build_container(settings: Settings) -> AppContainer:
    """Construct the full object graph from configuration."""
    database = Database(settings.database_url)
    llm_service = LLMService(settings)
    flight_service = build_flight_service(settings)
    hotel_service = build_hotel_service(settings)
    coupon_service = CouponService(settings, database)
    ranking_service = RankingService(settings)
    memory_store = MemoryStore(database)
    tool_executor = ToolExecutor(
        flight_service=flight_service,
        hotel_service=hotel_service,
        coupon_service=coupon_service,
        ranking_service=ranking_service,
        settings=settings,
    )
    planner = PlannerAgent(
        llm_service=llm_service,
        tool_executor=tool_executor,
        memory_store=memory_store,
        settings=settings,
    )
    _logger.info(
        "Container built",
        extra={
            "llm_configured": settings.is_llm_configured,
            "provider_configured": settings.is_provider_configured,
        },
    )
    return AppContainer(
        settings=settings,
        database=database,
        llm_service=llm_service,
        flight_service=flight_service,
        hotel_service=hotel_service,
        coupon_service=coupon_service,
        ranking_service=ranking_service,
        memory_store=memory_store,
        tool_executor=tool_executor,
        planner=planner,
    )


def init_database(container: AppContainer) -> None:
    """Create tables and load the coupon seed catalogue."""
    container.database.create_all()
    loaded = container.coupon_service.load_seed()
    _logger.info("Database initialised", extra={"coupons_seeded": loaded})


def build_scheduler(container: AppContainer) -> AsyncIOScheduler | None:
    """Create (but do not start) the background scheduler, if enabled.

    A coupon-refresh job is registered only when scheduling is enabled and at
    least one coupon source URL is configured.
    """
    settings = container.settings
    if not settings.enable_scheduler:
        _logger.info("Scheduler disabled by configuration")
        return None

    scheduler = AsyncIOScheduler(timezone="UTC")
    if settings.coupon_refresh_enabled and settings.coupon_sources_list:
        scheduler.add_job(
            container.coupon_service.refresh,
            trigger=IntervalTrigger(
                minutes=settings.coupon_refresh_interval_minutes
            ),
            id="coupon_refresh",
            name="Refresh coupon catalogue",
            replace_existing=True,
            max_instances=1,
            coalesce=True,
        )
        _logger.info(
            "Coupon refresh scheduled",
            extra={"interval_minutes": settings.coupon_refresh_interval_minutes},
        )
    else:
        _logger.info("No coupon sources configured; refresh job not scheduled")
    return scheduler
