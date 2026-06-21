"""Tests for the deterministic ranking service.

The ranking service must be fully deterministic (no LLM) and explainable. These
tests pin down the scoring behaviour for representative inputs so regressions in
the weighting maths are caught.
"""

from __future__ import annotations

import pytest

from models.schemas import AppliedCoupon, Coupon, FlightResult, HotelResult, Preferences
from services.ranking_service import RankingService
from utils.constants import CabinClass, DiscountType, SortBy


def _flight(
    flight_id: str,
    *,
    airline: str,
    price: float,
    duration: int,
    stops: int,
    refundable: bool,
    baggage: str | None,
    cabin: CabinClass,
) -> FlightResult:
    return FlightResult(
        id=flight_id,
        airline=airline,
        airline_code=airline[:2].upper(),
        origin="DEL",
        destination="DXB",
        departure_time="2025-01-10T09:00:00",
        arrival_time="2025-01-10T12:20:00",
        duration_minutes=duration,
        stops=stops,
        cabin_class=cabin,
        price=price,
        currency="INR",
        refundable=refundable,
        baggage_allowance=baggage,
    )


# Cheap, no-frills IndiGo flight vs. premium Emirates flight.
FA = _flight(
    "FA",
    airline="IndiGo",
    price=50000.0,
    duration=200,
    stops=0,
    refundable=False,
    baggage=None,
    cabin=CabinClass.ECONOMY,
)
FB = _flight(
    "FB",
    airline="Emirates",
    price=80000.0,
    duration=180,
    stops=1,
    refundable=True,
    baggage="30kg",
    cabin=CabinClass.BUSINESS,
)


@pytest.fixture
def service() -> RankingService:
    return RankingService(settings=None)  # settings unused by ranking logic


def _ranks_are_contiguous(ranked) -> bool:
    return [item.rank for item in ranked] == list(range(1, len(ranked) + 1))


def test_best_value_ties_break_on_price(service: RankingService):
    """With these inputs both flights score 0.60; the cheaper one wins the tie."""
    ranked = service.rank_flights([FB, FA], sort_by=SortBy.BEST_VALUE)
    assert _ranks_are_contiguous(ranked)
    assert ranked[0].id == "FA"  # tie-break: lower raw price first
    assert ranked[0].score == pytest.approx(0.60, abs=1e-6)
    assert ranked[1].score == pytest.approx(0.60, abs=1e-6)


def test_cheapest_prefers_lower_price(service: RankingService):
    ranked = service.rank_flights([FB, FA], sort_by=SortBy.CHEAPEST)
    assert ranked[0].id == "FA"
    assert ranked[0].score == pytest.approx(1.0, abs=1e-6)
    assert ranked[1].score == pytest.approx(0.2, abs=1e-6)


def test_airline_preference_promotes_match(service: RankingService):
    """Preferring Emirates lifts FB above FA under best-value scoring."""
    baseline = {f.id: f.score for f in service.rank_flights([FA, FB], sort_by=SortBy.BEST_VALUE)}
    prefs = Preferences(airlines=["Emirates"])
    ranked = service.rank_flights([FA, FB], sort_by=SortBy.BEST_VALUE, preferences=prefs)
    by_id = {f.id: f for f in ranked}

    assert by_id["FB"].score > baseline["FB"]  # preference raised its score
    assert by_id["FA"].score == pytest.approx(baseline["FA"], abs=1e-6)  # FA unchanged
    assert ranked[0].id == "FB"  # and that is enough to overtake FA


def test_coupon_changes_final_price_and_order(service: RankingService):
    coupon = Coupon(code="FLAT40K", discount_type=DiscountType.FLAT, discount_value=40000.0)
    applied = {"FB": AppliedCoupon(coupon=coupon, savings=40000.0, final_price=40000.0)}

    ranked = service.rank_flights([FA, FB], sort_by=SortBy.CHEAPEST, coupons=applied)
    top = ranked[0]
    assert top.id == "FB"  # 40k effective price beats FA's 50k
    assert top.final_price == pytest.approx(40000.0)
    assert top.applied_coupon is not None
    assert top.applied_coupon.savings == pytest.approx(40000.0)
    # The uncouponed flight keeps its sticker price as the final price.
    assert {f.id: f.final_price for f in ranked}["FA"] == pytest.approx(50000.0)


def test_empty_input_returns_empty(service: RankingService):
    assert service.rank_flights([]) == []
    assert service.rank_hotels([]) == []


# --------------------------------------------------------------------------- #
# Hotels
# --------------------------------------------------------------------------- #
HA = HotelResult(id="HA", name="Grand Plaza", rating=4.5, price=8000.0, distance_km=0.5)
HB = HotelResult(id="HB", name="Thrift Lodge", rating=3.5, price=6000.0, distance_km=2.0)


def test_rating_sort_prefers_higher_rating(service: RankingService):
    ranked = service.rank_hotels([HB, HA], sort_by=SortBy.RATING)
    assert _ranks_are_contiguous(ranked)
    assert ranked[0].id == "HA"
    assert ranked[0].score == pytest.approx(0.8, abs=1e-6)
    assert ranked[1].score == pytest.approx(0.2, abs=1e-6)


def test_cheapest_sort_prefers_lower_price_hotel(service: RankingService):
    ranked = service.rank_hotels([HA, HB], sort_by=SortBy.CHEAPEST)
    assert ranked[0].id == "HB"
    assert ranked[0].score == pytest.approx(1.0, abs=1e-6)
