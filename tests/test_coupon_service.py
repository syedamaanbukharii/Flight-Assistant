"""Tests for coupon savings maths, selection logic, and seed persistence."""

from __future__ import annotations

import pytest

from models.database import Database
from models.schemas import Coupon
from services.coupon_service import CouponService
from utils.constants import CouponScope, DiscountType, ItemKind


def _coupon(
    code: str,
    *,
    discount_type: DiscountType,
    value: float,
    max_discount: float | None = None,
    min_spend: float | None = None,
    currency: str | None = "INR",
    scope: CouponScope = CouponScope.ALL,
    provider: str | None = None,
) -> Coupon:
    return Coupon(
        code=code,
        discount_type=discount_type,
        discount_value=value,
        max_discount=max_discount,
        min_spend=min_spend,
        currency=currency,
        scope=scope,
        provider=provider,
    )


# --------------------------------------------------------------------------- #
# compute_savings (pure, static)
# --------------------------------------------------------------------------- #
def test_percentage_savings():
    c = _coupon("PCT10", discount_type=DiscountType.PERCENTAGE, value=10)
    assert CouponService.compute_savings(c, 1000.0, "INR") == pytest.approx(100.0)


def test_flat_savings():
    c = _coupon("FLAT200", discount_type=DiscountType.FLAT, value=200)
    assert CouponService.compute_savings(c, 1000.0, "INR") == pytest.approx(200.0)


def test_min_spend_not_met_yields_none():
    c = _coupon("BIG", discount_type=DiscountType.FLAT, value=200, min_spend=2000)
    assert CouponService.compute_savings(c, 1000.0, "INR") is None


def test_currency_mismatch_yields_none():
    c = _coupon("USD", discount_type=DiscountType.FLAT, value=200, currency="USD")
    assert CouponService.compute_savings(c, 1000.0, "INR") is None


def test_max_discount_caps_savings():
    c = _coupon("HALF", discount_type=DiscountType.PERCENTAGE, value=50, max_discount=100)
    # 50% of 1000 = 500, capped at 100.
    assert CouponService.compute_savings(c, 1000.0, "INR") == pytest.approx(100.0)


def test_savings_never_exceeds_base_price():
    c = _coupon("HUGE", discount_type=DiscountType.FLAT, value=2000)
    assert CouponService.compute_savings(c, 1000.0, "INR") == pytest.approx(1000.0)


def test_zero_savings_yields_none():
    c = _coupon("ZERO", discount_type=DiscountType.FLAT, value=0)
    assert CouponService.compute_savings(c, 1000.0, "INR") is None


# --------------------------------------------------------------------------- #
# select_best
# --------------------------------------------------------------------------- #
def test_select_best_picks_largest_savings(coupon_service: CouponService):
    flat = _coupon("FLAT100", discount_type=DiscountType.FLAT, value=100)
    pct = _coupon("PCT20", discount_type=DiscountType.PERCENTAGE, value=20)  # 200 on 1000
    best = coupon_service.select_best(
        [flat, pct], kind=ItemKind.FLIGHT, base_price=1000.0, currency="INR"
    )
    assert best is not None
    assert best.coupon.code == "PCT20"
    assert best.savings == pytest.approx(200.0)
    assert best.final_price == pytest.approx(800.0)


def test_select_best_respects_scope(coupon_service: CouponService):
    hotel_only = _coupon(
        "HOTELONLY", discount_type=DiscountType.FLAT, value=500, scope=CouponScope.HOTEL
    )
    assert (
        coupon_service.select_best(
            [hotel_only], kind=ItemKind.FLIGHT, base_price=1000.0, currency="INR"
        )
        is None
    )


def test_select_best_respects_provider(coupon_service: CouponService):
    ek = _coupon(
        "EKONLY", discount_type=DiscountType.FLAT, value=300, provider="Emirates"
    )
    # Provider mismatch -> excluded.
    assert (
        coupon_service.select_best(
            [ek], kind=ItemKind.FLIGHT, base_price=1000.0, currency="INR", provider="IndiGo"
        )
        is None
    )
    # Matching provider -> applied.
    match = coupon_service.select_best(
        [ek], kind=ItemKind.FLIGHT, base_price=1000.0, currency="INR", provider="Emirates"
    )
    assert match is not None
    assert match.savings == pytest.approx(300.0)


# --------------------------------------------------------------------------- #
# Seed loading + persistence (async)
# --------------------------------------------------------------------------- #
async def test_load_seed_populates_active_coupons(settings, database: Database):
    service = CouponService(settings, database)
    loaded = service.load_seed()
    assert loaded == 8
    assert service.count_active() == 8


async def test_list_active_filters_by_scope(coupon_service: CouponService):
    flight_coupons = await coupon_service.list_active(CouponScope.FLIGHT)
    codes = {c.code for c in flight_coupons}

    # Flight scope returns flight-scoped + universal coupons, never hotel-only ones.
    assert all(c.scope in (CouponScope.ALL, CouponScope.FLIGHT) for c in flight_coupons)
    assert "STAYEASY12" not in codes and "HOTEL2000" not in codes
    assert len(flight_coupons) == 6

    hotel_coupons = await coupon_service.list_active(CouponScope.HOTEL)
    assert all(c.scope in (CouponScope.ALL, CouponScope.HOTEL) for c in hotel_coupons)
    assert len(hotel_coupons) == 5
