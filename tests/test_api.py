"""End-to-end API tests.

These run the real FastAPI app (built from the keyless ``settings`` fixture) via
``TestClient``, so the lifespan wires unconfigured providers. The expected
behaviour is graceful degradation:

* ``/health`` and ``/verify`` always succeed (``/verify`` reports ``warn``).
* ``/search/*`` return ``503 not_configured`` because no RapidAPI key is set.
* ``/chat`` always returns ``200`` — the planner falls back to heuristics and
  surfaces provider unavailability through ``notes`` rather than an error.
"""

from __future__ import annotations


def test_health_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_verify_reports_warn_when_keyless(client):
    response = client.get("/verify")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "warn"  # llm + provider unconfigured

    by_name = {check["name"]: check for check in body["checks"]}
    assert by_name["coupons"]["status"] == "ok"  # seed loaded
    assert by_name["database"]["status"] == "ok"
    assert by_name["llm"]["status"] == "warn"
    assert by_name["flight_hotel_provider"]["status"] == "warn"


def test_search_flights_requires_provider(client):
    response = client.post(
        "/search/flights", json={"origin": "Delhi", "destination": "Dubai"}
    )
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "not_configured"


def test_search_hotels_requires_provider(client):
    response = client.post("/search/hotels", json={"location": "Dubai"})
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "not_configured"


def test_search_flights_validation_error(client):
    # ``origin`` shorter than the 2-char minimum -> 422 from request validation.
    response = client.post(
        "/search/flights", json={"origin": "D", "destination": "Dubai"}
    )
    assert response.status_code == 422


def test_chat_chitchat_succeeds(client):
    response = client.post("/chat", json={"message": "hello"})
    assert response.status_code == 200
    body = response.json()
    assert body["reply"].strip()
    assert body["flights"] == []
    assert body["session_id"]  # a session id is always assigned


def test_chat_flight_search_degrades_gracefully(client):
    response = client.post(
        "/chat", json={"message": "flights from Delhi to Dubai tomorrow"}
    )
    assert response.status_code == 200
    body = response.json()
    # No provider -> no flights, but the user still gets a reply and a note.
    assert body["flights"] == []
    assert len(body["notes"]) >= 1
    assert body["reply"].strip()


def test_request_id_header_present(client):
    response = client.get("/health")
    assert response.headers.get("X-Request-ID")
