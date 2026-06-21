"""Coupon subsystem.

Responsibilities:
    * Load a seed catalogue of coupons (operator-provided JSON).
    * Refresh coupons from configured promotional pages (httpx + BeautifulSoup,
      with an optional Playwright fallback for JS-rendered pages).
    * Persist normalised coupons to the database.
    * Apply coupons to results with deterministic, explainable savings logic.

The LLM is never used for HTML parsing; extraction is regex/DOM based.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

from app.config import Settings
from models.database import CouponRecord, Database
from models.schemas import AppliedCoupon, Coupon
from utils.constants import CouponScope, DiscountType, ItemKind
from utils.errors import ExternalServiceError
from utils.helpers import async_retry, parse_currency_amount
from utils.logger import get_logger

try:  # Playwright is optional; coupon refresh works without it.
    from playwright.async_api import async_playwright

    _PLAYWRIGHT_AVAILABLE = True
except Exception:  # noqa: BLE001 - any import failure means "unavailable"
    _PLAYWRIGHT_AVAILABLE = False


_CODE_RE = re.compile(r"\b(?:code|coupon|promo)\s*[:\-]?\s*([A-Z0-9]{4,12})\b", re.IGNORECASE)
_PERCENT_RE = re.compile(r"(\d{1,2}(?:\.\d+)?)\s*%\s*(?:off|discount)?", re.IGNORECASE)
_FLAT_RE = re.compile(r"(?:flat|save|get)?\s*[₹$]\s*([\d,]+)\s*(?:off|discount)", re.IGNORECASE)
_MIN_SPEND_RE = re.compile(
    r"min(?:imum)?\.?\s*(?:spend|order|booking)?\s*[₹$]?\s*([\d,]+)", re.IGNORECASE
)


class CouponSourceFetcher:
    """Fetches raw HTML for coupon pages."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._logger = get_logger("services.coupon.fetcher")

    @async_retry(attempts=2, base_delay=0.5, exceptions=(httpx.HTTPError,))
    async def fetch(self, url: str) -> str:
        """Return page HTML, using Playwright only when enabled and available."""
        if self._settings.coupon_use_playwright and _PLAYWRIGHT_AVAILABLE:
            return await self._fetch_with_playwright(url)
        return await self._fetch_with_httpx(url)

    async def _fetch_with_httpx(self, url: str) -> str:
        headers = {"User-Agent": "FlightAssistant/1.0 (+coupon-refresh)"}
        async with httpx.AsyncClient(
            timeout=self._settings.http_timeout_seconds, follow_redirects=True
        ) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            return response.text

    async def _fetch_with_playwright(self, url: str) -> str:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(headless=True)
            try:
                page = await browser.new_page()
                await page.goto(url, wait_until="networkidle", timeout=int(
                    self._settings.http_timeout_seconds * 1000
                ))
                return await page.content()
            finally:
                await browser.close()


class CouponParser:
    """Extracts normalised coupons from coupon-page HTML.

    The parser is intentionally generic and best-effort: it scans card-like
    blocks for a code, a discount, and (optionally) a minimum spend. Operators
    can extend ``_candidate_blocks`` selectors for site-specific structures.
    """

    def __init__(self) -> None:
        self._logger = get_logger("services.coupon.parser")

    def parse(self, html: str, source_url: str | None = None) -> list[Coupon]:
        soup = BeautifulSoup(html, "html.parser")
        coupons: dict[str, Coupon] = {}
        for block in self._candidate_blocks(soup):
            coupon = self._parse_block(block.get_text(" ", strip=True), source_url)
            if coupon is not None:
                coupons[coupon.code] = coupon
        return list(coupons.values())

    @staticmethod
    def _candidate_blocks(soup: BeautifulSoup) -> list:
        blocks = soup.select(
            "[class*=coupon], [class*=offer], [class*=deal], [class*=promo], li, article"
        )
        return blocks or [soup.body or soup]

    def _parse_block(self, text: str, source_url: str | None) -> Coupon | None:
        if not text or len(text) > 600:
            return None
        code_match = _CODE_RE.search(text)
        if code_match is None:
            return None

        percent_match = _PERCENT_RE.search(text)
        flat_match = _FLAT_RE.search(text)
        if percent_match is None and flat_match is None:
            return None

        if percent_match is not None:
            discount_type = DiscountType.PERCENTAGE
            discount_value = float(percent_match.group(1))
        else:
            discount_type = DiscountType.FLAT
            discount_value = parse_currency_amount(flat_match.group(1)) or 0.0

        if discount_value <= 0:
            return None

        min_spend_match = _MIN_SPEND_RE.search(text)
        min_spend = (
            parse_currency_amount(min_spend_match.group(1)) if min_spend_match else None
        )

        return Coupon(
            code=code_match.group(1),
            title=text[:80],
            description=text[:240],
            discount_type=discount_type,
            discount_value=discount_value,
            min_spend=min_spend,
            scope=CouponScope.ALL,
            source_url=source_url,
            active=True,
        )


class CouponService:
    """Manages the coupon catalogue and applies coupons to results."""

    def __init__(
        self,
        settings: Settings,
        database: Database,
        fetcher: CouponSourceFetcher | None = None,
        parser: CouponParser | None = None,
    ) -> None:
        self._settings = settings
        self._db = database
        self._fetcher = fetcher or CouponSourceFetcher(settings)
        self._parser = parser or CouponParser()
        self._logger = get_logger("services.coupon")

    # ------------------------------------------------------------------ #
    # Catalogue management
    # ------------------------------------------------------------------ #
    def load_seed(self) -> int:
        """Upsert coupons from the seed file. Returns the number loaded."""
        path = Path(self._settings.coupon_seed_path)
        if not path.exists():
            self._logger.info("No coupon seed file found", extra={"path": str(path)})
            return 0
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            self._logger.error("Failed to read coupon seed", extra={"error": str(exc)})
            return 0

        coupons = [Coupon(**item) for item in raw if isinstance(item, dict)]
        self._upsert(coupons)
        self._logger.info("Loaded coupon seed", extra={"count": len(coupons)})
        return len(coupons)

    async def refresh(self) -> int:
        """Fetch and persist coupons from all configured sources.

        Returns the total number of coupons upserted. Per-source failures are
        logged and skipped so one bad source cannot abort the whole refresh.
        """
        sources = self._settings.coupon_sources_list
        if not sources:
            self._logger.info("Coupon refresh skipped: no sources configured")
            return 0

        total = 0
        for url in sources:
            try:
                html = await self._fetcher.fetch(url)
                coupons = self._parser.parse(html, source_url=url)
                await asyncio.to_thread(self._upsert, coupons)
                total += len(coupons)
                self._logger.info(
                    "Refreshed coupons from source",
                    extra={"source": url, "count": len(coupons)},
                )
            except Exception as exc:  # noqa: BLE001 - isolate per-source failures
                self._logger.error(
                    "Coupon source refresh failed",
                    extra={"source": url, "error": str(exc)},
                )
        return total

    async def list_active(self, scope: CouponScope | None = None) -> list[Coupon]:
        """Return active coupons, optionally filtered by scope."""
        return await asyncio.to_thread(self._list_active_sync, scope)

    def count_active(self) -> int:
        """Return the number of active coupons (synchronous)."""
        with self._db.session() as session:
            return (
                session.query(CouponRecord).filter(CouponRecord.active.is_(True)).count()
            )

    # ------------------------------------------------------------------ #
    # Application (pure, deterministic)
    # ------------------------------------------------------------------ #
    @staticmethod
    def compute_savings(coupon: Coupon, base_price: float, currency: str) -> float | None:
        """Return the savings a coupon yields, or ``None`` if inapplicable."""
        if not coupon.active:
            return None
        if coupon.currency and currency and coupon.currency.upper() != currency.upper():
            return None
        if coupon.min_spend is not None and base_price < coupon.min_spend:
            return None

        if coupon.discount_type == DiscountType.PERCENTAGE:
            savings = base_price * (coupon.discount_value / 100.0)
        else:
            savings = coupon.discount_value

        if coupon.max_discount is not None:
            savings = min(savings, coupon.max_discount)
        savings = min(savings, base_price)
        return round(savings, 2) if savings > 0 else None

    def select_best(
        self,
        coupons: list[Coupon],
        *,
        kind: ItemKind,
        base_price: float,
        currency: str,
        provider: str | None = None,
    ) -> AppliedCoupon | None:
        """Pick the coupon producing the greatest savings for one item."""
        best: AppliedCoupon | None = None
        for coupon in coupons:
            if coupon.scope not in (CouponScope.ALL, CouponScope(kind.value)):
                continue
            if coupon.provider and provider and coupon.provider.lower() != provider.lower():
                continue
            savings = self.compute_savings(coupon, base_price, currency)
            if savings is None:
                continue
            if best is None or savings > best.savings:
                best = AppliedCoupon(
                    coupon=coupon,
                    savings=savings,
                    final_price=round(base_price - savings, 2),
                )
        return best

    # ------------------------------------------------------------------ #
    # Persistence internals
    # ------------------------------------------------------------------ #
    def _list_active_sync(self, scope: CouponScope | None) -> list[Coupon]:
        with self._db.session() as session:
            query = session.query(CouponRecord).filter(CouponRecord.active.is_(True))
            if scope is not None and scope != CouponScope.ALL:
                query = query.filter(
                    CouponRecord.scope.in_([CouponScope.ALL.value, scope.value])
                )
            return [self._to_schema(record) for record in query.all()]

    def _upsert(self, coupons: list[Coupon]) -> None:
        if not coupons:
            return
        with self._db.session() as session:
            for coupon in coupons:
                record = (
                    session.query(CouponRecord)
                    .filter(
                        CouponRecord.code == coupon.code,
                        CouponRecord.provider.is_(coupon.provider)
                        if coupon.provider is None
                        else CouponRecord.provider == coupon.provider,
                    )
                    .one_or_none()
                )
                if record is None:
                    record = CouponRecord(code=coupon.code)
                    session.add(record)
                record.title = coupon.title
                record.description = coupon.description
                record.discount_type = coupon.discount_type.value
                record.discount_value = coupon.discount_value
                record.max_discount = coupon.max_discount
                record.min_spend = coupon.min_spend
                record.currency = coupon.currency
                record.scope = coupon.scope.value
                record.provider = coupon.provider
                record.source_url = coupon.source_url
                record.active = coupon.active

    @staticmethod
    def _to_schema(record: CouponRecord) -> Coupon:
        return Coupon(
            code=record.code,
            title=record.title,
            description=record.description,
            discount_type=DiscountType(record.discount_type),
            discount_value=record.discount_value,
            max_discount=record.max_discount,
            min_spend=record.min_spend,
            currency=record.currency,
            scope=CouponScope(record.scope),
            provider=record.provider,
            source_url=record.source_url,
            active=record.active,
        )


# Surface a clear error type if a future provider integration needs it.
__all__ = [
    "CouponService",
    "CouponParser",
    "CouponSourceFetcher",
    "ExternalServiceError",
]
