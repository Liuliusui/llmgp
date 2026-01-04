"""HTTP-based LLM helper utilities."""
from __future__ import annotations

import http.client
import json
import time
import traceback
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

DEFAULT_ENDPOINT = "/v1/chat/completions"

class LLMTransportError(RuntimeError):
    """Raised when the remote LLM cannot be reached or returns an error."""

class HttpsApi:
    """Minimal HTTPS client for OpenAI-compatible chat completions."""

    def __init__(
        self,
        host: str,
        key: str,
        model: str,
        timeout: float = 20.0,
        *,
        path: str = DEFAULT_ENDPOINT,
        max_tokens: int = 10000,
        temperature: float = 0.3,
        top_p: Optional[float] = None,
    ) -> None:
        if not host:
            raise ValueError("host is required")
        self.host = host.replace("https://", "").strip("/")
        self.api_key = key
        self.model = model
        self.timeout = timeout
        self.path = path
        self.default_args = {
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if top_p is not None:
            self.default_args["top_p"] = top_p

    def draw_sample(self, prompt: str | List[Dict[str, str]], *, extra_args: Optional[Dict[str, Any]] = None) -> str:
        messages = prompt
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt.strip()}]
        payload = {
            "model": self.model,
            "messages": messages,
        }
        payload.update(self.default_args)
        if extra_args:
            payload.update(extra_args)
        data = json.dumps(payload)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "llmgp-lab/0.1",
        }
        conn = http.client.HTTPSConnection(self.host, timeout=self.timeout)
        try:
            conn.request("POST", self.path, body=data, headers=headers)
            resp = conn.getresponse()
            raw = resp.read().decode("utf-8")
            if resp.status >= 400:
                raise LLMTransportError(f"{resp.status} {resp.reason}: {raw[:200]}")
            parsed = json.loads(raw)
            return parsed["choices"][0]["message"]["content"].strip()
        finally:
            conn.close()

@dataclass
class StubLLM:
    """Deterministic fallback client for offline development."""

    responses: Sequence[str] | None = None
    pointer: int = 0

    def draw_sample(self, prompt: Any, *, extra_args: Optional[Dict[str, Any]] = None) -> str:
        if not self.responses:
            # Produce a JSON stub that downstream can parse.
            return "{}"
        resp = self.responses[self.pointer % len(self.responses)]
        self.pointer += 1
        return resp

class LLMClient:
    """High-level helper that picks HTTPS or stub backend automatically."""

    def __init__(
        self,
        host: Optional[str],
        api_key: Optional[str],
        model: Optional[str],
        timeout_ms: Optional[int] = 5000,
        *,
        mock_responses: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> None:
        self._backend: Any = None
        self._stub: StubLLM | None = None
        if mock_responses:
            self._stub = StubLLM(mock_responses)
        if host and api_key and model:
            self._backend = HttpsApi(
                host,
                api_key,
                model,
                timeout=(timeout_ms or 5000) / 1000.0,
                **{k: v for k, v in kwargs.items() if k in {"max_tokens", "temperature", "top_p"}},
            )

    def available(self) -> bool:
        return self._backend is not None or self._stub is not None

    def _active_backend(self) -> Any:
        if self._backend is not None:
            return self._backend
        if self._stub is not None:
            return self._stub
        raise LLMTransportError("LLM client not configured")

    def chat(self, prompt: str | List[Dict[str, str]], *, extra_args: Optional[Dict[str, Any]] = None) -> str:
        backend = self._active_backend()
        try:
            return backend.draw_sample(prompt, extra_args=extra_args)
        except Exception as exc:  # pragma: no cover - defensive log
            raise LLMTransportError(f"LLM request failed: {exc}") from exc

    def propose(self, system_prompt: str, user_prompt: str, *, fallback: str = "{}") -> str:
        if not self.available():
            return fallback
        messages = [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ]
        try:
            return self.chat(messages)
        except LLMTransportError:
            return fallback


