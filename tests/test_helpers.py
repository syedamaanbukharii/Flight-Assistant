"""Tests for pure utility helpers."""

from __future__ import annotations

import pytest

from utils.constants import TimeWindow
from utils.helpers import (
    haversine_km,
    minutes_to_human,
    normalise_min_max,
    parse_currency_amount,
    to_iso_date,
    within_time_window,
)


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("₹80,000", 80000.0),
        ("$1,250.50", 1250.50),
        ("60k", 60000.0),
        ("2 lakh", 200000.0),
        ("1.5 crore", 15000000.0),
        ("free", None),
        (None, None),
        (5000, 5000.0),
    ],
)
def test_parse_currency_amount(raw, expected):
    result = parse_currency_amount(raw)
    if expected is None:
        assert result is None
    else:
        assert result == pytest.approx(expected)


def test_haversine_known_distance():
    # Delhi (DEL) to Dubai (DXB) is roughly 2,200 km.
    distance = haversine_km(28.5562, 77.1000, 25.2532, 55.3657)
    assert 2000 < distance < 2400


def test_haversine_zero():
    assert haversine_km(10.0, 20.0, 10.0, 20.0) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize(
    "iso, window, expected",
    [
        ("2025-01-10T19:00:00", TimeWindow.EVENING, True),
        ("2025-01-10T08:00:00", TimeWindow.MORNING, True),
        ("2025-01-10T08:00:00", TimeWindow.EVENING, False),
        ("2025-01-10T02:00:00", TimeWindow.NIGHT, True),  # wraps midnight
        ("2025-01-10T23:30:00", TimeWindow.NIGHT, True),
        (None, TimeWindow.EVENING, True),  # unknown time never filtered out
        ("2025-01-10T08:00:00", TimeWindow.ANY, True),
    ],
)
def test_within_time_window(iso, window, expected):
    assert within_time_window(iso, window) is expected


def test_normalise_min_max():
    assert normalise_min_max(5, 0, 10) == pytest.approx(0.5)
    assert normalise_min_max(5, 5, 5) == 1.0  # degenerate range
    assert normalise_min_max(-1, 0, 10) == 0.0  # clamped


def test_minutes_to_human():
    assert minutes_to_human(200) == "3h 20m"
    assert minutes_to_human(60) == "1h"
    assert minutes_to_human(45) == "45m"
    assert minutes_to_human(None) == "—"


def test_to_iso_date_passthrough():
    assert to_iso_date("2025-01-10") == "2025-01-10"
    assert to_iso_date(None) is None
