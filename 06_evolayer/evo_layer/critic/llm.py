"""Optional LLM narrative for findings.

The LLM never computes, adjusts or vetoes a number. It receives the already
computed evidence and may only add prose (``narrative``), a self-reported
confidence, and caveats. The response is schema-validated; anything else is
discarded and the loop continues without it.

Implemented against any OpenAI-compatible ``/chat/completions`` endpoint using
:mod:`urllib` so the core has no third-party HTTP dependency. Calls are cached by
content hash and budgeted per run.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from ..config import EvoConfig
from ..ids import digest
from ..store import Store

SYSTEM_PROMPT = (
    "You are the narrative writer for a robot evolution loop. You receive a finding that has "
    "already been computed statistically. Do not invent numbers, do not recompute anything, and "
    "do not recommend skipping safety review. Reply with JSON only, matching: "
    '{"narrative": string (<= 600 chars), "confidence": number in [0,1], "caveats": [string]}.'
)


class LLMUnavailable(RuntimeError):
    """The LLM is disabled, unconfigured or returned an unusable response."""


@dataclass
class Narrative:
    text: str
    confidence: float
    caveats: list[str]
    model: str
    cached: bool = False


class LLMClient:
    def __init__(self, cfg: EvoConfig, store: Store, env: dict[str, str] | None = None):
        self.cfg = cfg
        self.store = store
        import os

        self.env = dict(os.environ if env is None else env)
        self.calls = 0

    def available(self) -> bool:
        return bool(self.cfg.llm.enabled and self.cfg.llm.base_url and self.cfg.llm.model)

    def narrate(self, finding_payload: dict[str, Any]) -> Narrative:
        if not self.available():
            raise LLMUnavailable("LLM narrative is disabled (llm.enabled=false or no base_url/model)")
        if self.calls >= self.cfg.llm.max_calls_per_run:
            raise LLMUnavailable("LLM call budget for this run is exhausted")

        cache_key = "llm:" + digest({"model": self.cfg.llm.model, "finding": finding_payload})
        cached = self.store.kv_get(cache_key)
        if cached:
            data = json.loads(cached)
            return Narrative(
                text=data["narrative"],
                confidence=float(data["confidence"]),
                caveats=list(data.get("caveats") or []),
                model=self.cfg.llm.model,
                cached=True,
            )

        api_key = self.env.get(self.cfg.llm.api_key_env, "")
        if not api_key:
            raise LLMUnavailable(f"{self.cfg.llm.api_key_env} is not set")

        body = {
            "model": self.cfg.llm.model,
            "temperature": 0.2,
            "max_tokens": 400,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(finding_payload, sort_keys=True)},
            ],
        }
        request = urllib.request.Request(
            self.cfg.llm.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
            method="POST",
        )
        self.calls += 1
        try:
            with urllib.request.urlopen(request, timeout=self.cfg.llm.timeout_s) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise LLMUnavailable(f"LLM request failed: {exc}") from exc

        try:
            content = payload["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            narrative = str(parsed["narrative"]).strip()
            confidence = float(parsed.get("confidence", 0.0))
            caveats = [str(item) for item in (parsed.get("caveats") or [])]
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise LLMUnavailable(f"LLM returned an unusable response: {exc}") from exc

        if not narrative or len(narrative) > 2000:
            raise LLMUnavailable("LLM narrative empty or too long")
        if not 0.0 <= confidence <= 1.0:
            raise LLMUnavailable("LLM confidence outside [0, 1]")

        record = {"narrative": narrative, "confidence": confidence, "caveats": caveats}
        self.store.kv_set(cache_key, json.dumps(record, sort_keys=True))
        return Narrative(text=narrative, confidence=confidence, caveats=caveats, model=self.cfg.llm.model)
