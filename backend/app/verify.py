"""Runtime verification.

Produces a :class:`VerifyReport` summarising whether the service is correctly
configured and its dependencies are reachable. It reports only presence/absence
and reachability — never the values of secrets — so the ``/verify`` endpoint is
safe to expose to operators.
"""

from __future__ import annotations

import asyncio

from app.config import Settings
from app.startup import AppContainer
from models.schemas import CheckResult, VerifyReport
from utils.logger import get_logger

_logger = get_logger("app.verify")

_OK = "ok"
_WARN = "warn"
_FAIL = "fail"


async def run_verification(container: AppContainer, settings: Settings) -> VerifyReport:
    """Run all probes and aggregate them into a report."""
    checks: list[CheckResult] = [
        _check_llm(settings),
        _check_provider(settings),
        await _check_database(container),
        await _check_coupons(container),
    ]
    status = _aggregate(checks)
    _logger.info("Verification complete", extra={"status": status})
    return VerifyReport(status=status, checks=checks)


def _check_llm(settings: Settings) -> CheckResult:
    if settings.is_llm_configured:
        return CheckResult(
            name="llm",
            status=_OK,
            message=f"Groq configured (model={settings.groq_model}).",
        )
    return CheckResult(
        name="llm",
        status=_WARN,
        message="GROQ_API_KEY not set; planner uses heuristic fallback.",
    )


def _check_provider(settings: Settings) -> CheckResult:
    if settings.is_provider_configured:
        return CheckResult(
            name="flight_hotel_provider",
            status=_OK,
            message=f"RapidAPI configured (host={settings.rapidapi_host}).",
        )
    return CheckResult(
        name="flight_hotel_provider",
        status=_WARN,
        message="RAPIDAPI_KEY not set; flight and hotel search are disabled.",
    )


async def _check_database(container: AppContainer) -> CheckResult:
    try:
        await asyncio.to_thread(container.database.ping)
        return CheckResult(name="database", status=_OK, message="Database reachable.")
    except Exception as exc:  # noqa: BLE001 - report any failure as a check result
        _logger.error("Database check failed", extra={"error": str(exc)})
        return CheckResult(
            name="database", status=_FAIL, message="Database is not reachable."
        )


async def _check_coupons(container: AppContainer) -> CheckResult:
    try:
        count = await asyncio.to_thread(container.coupon_service.count_active)
    except Exception as exc:  # noqa: BLE001
        _logger.error("Coupon check failed", extra={"error": str(exc)})
        return CheckResult(
            name="coupons", status=_WARN, message="Could not read the coupon catalogue."
        )
    if count > 0:
        return CheckResult(
            name="coupons", status=_OK, message=f"{count} active coupon(s) available."
        )
    return CheckResult(
        name="coupons",
        status=_WARN,
        message="No active coupons loaded; coupon application will be a no-op.",
    )


def _aggregate(checks: list[CheckResult]) -> str:
    if any(c.status == _FAIL for c in checks):
        return _FAIL
    if any(c.status == _WARN for c in checks):
        return _WARN
    return _OK
