"""Standalone coupon refresh job.

Intended to be run on a schedule (cron, systemd timer, CI) independently of the
API process. It loads configuration, seeds the catalogue, then fetches and
upserts coupons from every configured source.

Usage:
    python scripts/refresh_coupons.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Make the backend package importable when run from the repository root.
_BACKEND = Path(__file__).resolve().parent.parent / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from app.config import get_settings  # noqa: E402
from models.database import Database  # noqa: E402
from services.coupon_service import CouponService  # noqa: E402
from utils.logger import configure_logging, get_logger  # noqa: E402


async def _run() -> int:
    settings = get_settings()
    configure_logging(settings.log_level)
    logger = get_logger("scripts.refresh_coupons")

    database = Database(settings.database_url)
    database.create_all()
    service = CouponService(settings, database)

    seeded = service.load_seed()
    logger.info("Seed loaded", extra={"count": seeded})

    refreshed = await service.refresh()
    logger.info("Refresh finished", extra={"count": refreshed})

    total = service.count_active()
    logger.info("Active coupons after refresh", extra={"count": total})
    return total


def main() -> None:
    total = asyncio.run(_run())
    print(f"Active coupons: {total}")


if __name__ == "__main__":
    main()
