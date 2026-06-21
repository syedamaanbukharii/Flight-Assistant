"""LLM service backed by the Groq API.

The service exposes a small, well-typed surface: free-form completion and
JSON-constrained completion. It never instantiates a client without an API key
and raises :class:`NotConfiguredError` if used while unconfigured, so the rest
of the system can degrade gracefully.
"""

from __future__ import annotations

import json
from typing import Any

from app.config import Settings
from utils.errors import LLMError, NotConfiguredError
from utils.helpers import async_retry
from utils.logger import get_logger

Message = dict[str, str]


class LLMService:
    """Thin async wrapper around the Groq chat completions API."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._logger = get_logger("services.llm")
        self._client: Any | None = None

    @property
    def is_configured(self) -> bool:
        """Return whether an API key is available."""
        return self._settings.is_llm_configured

    def _get_client(self) -> Any:
        """Lazily build the Groq async client; cache it for reuse."""
        if not self.is_configured:
            raise NotConfiguredError(
                "The language model is not configured. Set GROQ_API_KEY to enable "
                "conversational features."
            )
        if self._client is None:
            # Imported lazily so the module imports cleanly without the package
            # installed (e.g. for tooling) and so construction stays cheap.
            from groq import AsyncGroq

            self._client = AsyncGroq(
                api_key=self._settings.groq_api_key,
                timeout=self._settings.groq_timeout_seconds,
                max_retries=0,  # retries handled by ``async_retry`` for uniform logging
            )
        return self._client

    @async_retry(attempts=3, base_delay=0.5)
    async def complete(
        self,
        messages: list[Message],
        *,
        json_mode: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Return the assistant text for a chat completion.

        Args:
            messages: OpenAI-style ``{"role", "content"}`` messages.
            json_mode: Request a strict JSON object response.
            temperature: Sampling temperature; falls back to the configured value.
            max_tokens: Output token cap; falls back to the configured value.

        Raises:
            NotConfiguredError: If no API key is configured.
            LLMError: If the provider call fails or returns no content.
        """
        client = self._get_client()
        kwargs: dict[str, Any] = {
            "model": self._settings.groq_model,
            "messages": messages,
            "temperature": (
                temperature if temperature is not None else self._settings.groq_temperature
            ),
            "max_tokens": max_tokens or self._settings.groq_max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            response = await client.chat.completions.create(**kwargs)
        except NotConfiguredError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalise provider errors
            self._logger.error("LLM request failed", extra={"error": str(exc)})
            raise LLMError("The language model request failed.") from exc

        content = response.choices[0].message.content if response.choices else None
        if not content:
            raise LLMError("The language model returned an empty response.")
        return content

    async def complete_json(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Return a parsed JSON object from a JSON-mode completion.

        Raises:
            LLMError: If the response is not valid JSON.
        """
        raw = await self.complete(
            messages, json_mode=True, temperature=temperature, max_tokens=max_tokens
        )
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            self._logger.error(
                "LLM returned invalid JSON", extra={"snippet": raw[:500]}
            )
            raise LLMError("The language model returned malformed JSON.") from exc
        if not isinstance(parsed, dict):
            raise LLMError("The language model did not return a JSON object.")
        return parsed

    async def aclose(self) -> None:
        """Close the underlying HTTP client if it was created."""
        if self._client is not None:
            close = getattr(self._client, "close", None)
            if close is not None:
                try:
                    await close()
                except Exception:  # noqa: BLE001 - best-effort shutdown
                    self._logger.debug("Error while closing LLM client", exc_info=True)
            self._client = None
