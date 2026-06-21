"""Streamlit chat frontend for the Flight & Travel Assistant.

A thin client over the FastAPI backend: it owns a session id, renders the
conversation, and displays ranked flights/hotels with expandable details and
booking links. All intelligence lives in the backend; this module only handles
presentation and transport.

Run with:
    streamlit run frontend/streamlit_app.py
"""

from __future__ import annotations

import os
import uuid
from typing import Any

import httpx
import streamlit as st

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000").rstrip("/")
REQUEST_TIMEOUT = float(os.environ.get("FRONTEND_TIMEOUT", "60"))


def _init_state() -> None:
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid.uuid4().hex
    if "messages" not in st.session_state:
        st.session_state.messages = []  # list[dict[str, Any]]


def _post_chat(message: str) -> dict[str, Any]:
    """Send a chat message to the backend and return the parsed response."""
    payload = {"message": message, "session_id": st.session_state.session_id}
    with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
        response = client.post(f"{BACKEND_URL}/chat", json=payload)
        response.raise_for_status()
        return response.json()


def _fetch_verify() -> dict[str, Any] | None:
    try:
        with httpx.Client(timeout=10) as client:
            response = client.get(f"{BACKEND_URL}/verify")
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError:
        return None


def _render_flight(flight: dict[str, Any]) -> None:
    rank = flight.get("rank", "?")
    airline = flight.get("airline", "Unknown")
    price = flight.get("final_price", flight.get("price"))
    currency = flight.get("currency", "")
    title = f"#{rank} · {airline} · {currency} {price:,.0f}"
    with st.expander(title):
        st.write(
            f"**Route:** {flight.get('origin')} → {flight.get('destination')}"
        )
        if flight.get("departure_time"):
            st.write(f"**Departs:** {flight['departure_time']}")
        if flight.get("arrival_time"):
            st.write(f"**Arrives:** {flight['arrival_time']}")
        stops = flight.get("stops", 0)
        st.write(f"**Stops:** {'Non-stop' if stops == 0 else stops}")
        if flight.get("cabin_class"):
            st.write(f"**Cabin:** {flight['cabin_class']}")
        if flight.get("refundable") is not None:
            st.write(f"**Refundable:** {'Yes' if flight['refundable'] else 'No'}")
        applied = flight.get("applied_coupon")
        if applied:
            st.success(
                f"Coupon {applied['coupon']['code']} saved "
                f"{currency} {applied['savings']:,.0f}"
            )
        if flight.get("rationale"):
            st.caption(flight["rationale"])
        if flight.get("booking_link"):
            st.link_button("Book this flight", flight["booking_link"])


def _render_hotel(hotel: dict[str, Any]) -> None:
    rank = hotel.get("rank", "?")
    name = hotel.get("name", "Hotel")
    price = hotel.get("final_price", hotel.get("price"))
    currency = hotel.get("currency", "")
    title = f"#{rank} · {name} · {currency} {price:,.0f}/night"
    with st.expander(title):
        if hotel.get("rating") is not None:
            st.write(f"**Rating:** {hotel['rating']}★")
        if hotel.get("distance_km") is not None:
            st.write(f"**Distance:** {hotel['distance_km']:.2f} km from landmark")
        if hotel.get("address"):
            st.write(f"**Address:** {hotel['address']}")
        applied = hotel.get("applied_coupon")
        if applied:
            st.success(
                f"Coupon {applied['coupon']['code']} saved "
                f"{currency} {applied['savings']:,.0f}"
            )
        if hotel.get("rationale"):
            st.caption(hotel["rationale"])
        if hotel.get("booking_link"):
            st.link_button("Book this hotel", hotel["booking_link"])


def _render_assistant_payload(payload: dict[str, Any]) -> None:
    st.markdown(payload.get("reply", ""))
    flights = payload.get("flights") or []
    hotels = payload.get("hotels") or []
    if flights:
        st.subheader("Flights")
        for flight in flights:
            _render_flight(flight)
    if hotels:
        st.subheader("Hotels")
        for hotel in hotels:
            _render_hotel(hotel)
    notes = payload.get("notes") or []
    for note in notes:
        st.info(note)


def _render_sidebar() -> None:
    with st.sidebar:
        st.header("Session")
        st.caption(f"ID: {st.session_state.session_id[:8]}…")
        if st.button("Start new session"):
            st.session_state.session_id = uuid.uuid4().hex
            st.session_state.messages = []
            st.rerun()

        st.divider()
        st.header("Backend status")
        report = _fetch_verify()
        if report is None:
            st.error("Backend unreachable.")
        else:
            status = report.get("status", "unknown")
            emoji = {"ok": "🟢", "warn": "🟡", "fail": "🔴"}.get(status, "⚪")
            st.write(f"{emoji} Overall: **{status}**")
            for check in report.get("checks", []):
                st.caption(f"{check['name']}: {check['status']} — {check['message']}")

        st.divider()
        st.header("Preferences")
        prefs = _latest_preferences()
        if prefs:
            st.json(prefs)
        else:
            st.caption("No preferences captured yet.")

        st.divider()
        st.caption("Try: ‘Emirates business class from Delhi to Dubai next Friday "
                   "under ₹80,000’, then ‘apply coupons’ or ‘rank by cheapest’.")


def _latest_preferences() -> dict[str, Any]:
    for message in reversed(st.session_state.messages):
        if message["role"] == "assistant" and message.get("payload"):
            prefs = message["payload"].get("preferences") or {}
            cleaned = {k: v for k, v in prefs.items() if v not in (None, [], {})}
            if cleaned:
                return cleaned
    return {}


def main() -> None:
    st.set_page_config(page_title="Flight & Travel Assistant", page_icon="✈️")
    _init_state()
    st.title("✈️ Flight & Travel Assistant")

    _render_sidebar()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and message.get("payload"):
                _render_assistant_payload(message["payload"])
            else:
                st.markdown(message["content"])

    prompt = st.chat_input("Ask about flights, hotels, coupons…")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching…"):
            try:
                payload = _post_chat(prompt)
            except httpx.HTTPStatusError as exc:
                payload = None
                st.error(
                    f"Backend returned {exc.response.status_code}. "
                    "Please check the server logs."
                )
            except httpx.HTTPError:
                payload = None
                st.error("Could not reach the backend. Is it running?")
        if payload is not None:
            _render_assistant_payload(payload)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": payload.get("reply", ""),
                    "payload": payload,
                }
            )


if __name__ == "__main__":
    main()
