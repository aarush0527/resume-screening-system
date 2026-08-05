"""Provider-agnostic LLM client.

Auto-detects which provider to use from whichever API key is set in the
environment (checked in this order: GROQ_API_KEY, ANTHROPIC_API_KEY,
OPENAI_API_KEY), or pass provider= explicitly. Groq and OpenAI both speak
the OpenAI chat-completions wire format, so they share one code path;
Anthropic gets its own since the Messages API shape differs.

Why this exists: the brief allows any of OpenAI / Anthropic / a free option
via Groq, and different builders/graders will have access to different
keys. A single missing key shouldn't block the whole pipeline.
"""
from __future__ import annotations

import json
import os
from typing import Optional


class LLMError(Exception):
    pass


class LLMClient:
    def __init__(self, provider: Optional[str] = None):
        self.provider = provider or self._detect_provider()
        self.model = self._default_model()
        self._client = self._build_client()

    def _detect_provider(self) -> str:
        if os.getenv("GROQ_API_KEY"):
            return "groq"
        if os.getenv("ANTHROPIC_API_KEY"):
            return "anthropic"
        if os.getenv("OPENAI_API_KEY"):
            return "openai"
        raise LLMError(
            "No API key found. Set exactly one of GROQ_API_KEY, "
            "ANTHROPIC_API_KEY, or OPENAI_API_KEY in your .env file "
            "(see .env.example)."
        )

    def _default_model(self) -> str:
        # Env var override always wins; these are just sane, current defaults.
        # Groq's Llama chat models (llama-3.3-70b-versatile etc.) were
        # deprecated in favor of the GPT-OSS lineup -- if this default ever
        # 404s, check https://console.groq.com/docs/models for the current
        # production model list.
        defaults = {
            "groq": "openai/gpt-oss-20b",
            "anthropic": "claude-haiku-4-5-20251001",
            "openai": "gpt-4o-mini",
        }
        env_key = {"groq": "GROQ_MODEL", "anthropic": "ANTHROPIC_MODEL", "openai": "OPENAI_MODEL"}[self.provider]
        return os.getenv(env_key, defaults[self.provider])

    def _build_client(self):
        if self.provider == "anthropic":
            import anthropic
            return anthropic.Anthropic()
        if self.provider == "groq":
            import openai
            return openai.OpenAI(
                api_key=os.getenv("GROQ_API_KEY"),
                base_url="https://api.groq.com/openai/v1",
            )
        if self.provider == "openai":
            import openai
            return openai.OpenAI()
        raise LLMError(f"Unknown provider: {self.provider}")

    def complete(self, system_prompt: str, user_prompt: str, json_mode: bool = False) -> str:
        """Returns the raw text response from the model."""
        try:
            if self.provider == "anthropic":
                resp = self._client.messages.create(
                    model=self.model,
                    max_tokens=2000,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                )
                return "".join(block.text for block in resp.content if block.type == "text")

            # groq and openai: both OpenAI-compatible chat completions
            kwargs = {}
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            resp = self._client.chat.completions.create(
                model=self.model,
                max_tokens=2000,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                **kwargs,
            )
            return resp.choices[0].message.content
        except Exception as e:
            raise LLMError(f"{self.provider} API call failed: {e}") from e

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict:
        """Calls complete() in JSON mode and parses the result. If parsing
        fails, retries once asking the model to fix its own output before
        giving up -- LLMs occasionally wrap JSON in prose or code fences
        even when asked not to."""
        raw = self.complete(system_prompt, user_prompt, json_mode=True)
        parsed = _try_parse_json(raw)
        if parsed is not None:
            return parsed

        fix_prompt = (
            "The text below was supposed to be a single valid JSON object "
            "but failed to parse. Return ONLY the corrected valid JSON, "
            "nothing else:\n\n" + raw
        )
        raw2 = self.complete(system_prompt, fix_prompt, json_mode=True)
        parsed = _try_parse_json(raw2)
        if parsed is not None:
            return parsed

        raise LLMError(f"Model did not return valid JSON after one retry. Raw output:\n{raw[:500]}")


def _try_parse_json(raw: str) -> Optional[dict]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None
