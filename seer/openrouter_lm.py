from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import httpx

import dspy


class OpenRouterLM(dspy.LM):
    """DSPy LM wrapper that strips OpenRouter-unsupported kwargs."""

    @staticmethod
    def _sanitize_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
        sanitized = dict(kwargs)
        sanitized.pop("usage", None)
        return sanitized

    def __init__(self, *args, **kwargs):
        sanitized = self._sanitize_kwargs(kwargs)
        super().__init__(*args, **sanitized)

    def forward(self, prompt: str | None = None, messages: list[dict[str, Any]] | None = None, **kwargs):
        sanitized = self._sanitize_kwargs(kwargs)
        return super().forward(prompt=prompt, messages=messages, **sanitized)


class OpenRouterClientLM(dspy.BaseLM):
    """Direct OpenRouter client that bypasses LiteLLM."""

    def __init__(self, model: str, api_key: str, api_base: str = "https://openrouter.ai/api/v1", **kwargs):
        normalized_model = model.replace("openrouter/", "", 1)
        super().__init__(model=normalized_model, model_type="chat", **kwargs)
        self.api_key = api_key
        self.api_base = api_base

    @staticmethod
    def _sanitize_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
        sanitized = dict(kwargs)
        sanitized.pop("usage", None)
        sanitized.pop("api_key", None)
        sanitized.pop("api_base", None)
        if sanitized.get("max_tokens") is None:
            sanitized.pop("max_tokens", None)
        if sanitized.get("temperature") is None:
            sanitized.pop("temperature", None)
        return sanitized

    def _to_response(self, data: dict[str, Any]):
        choices = []
        for choice in data.get("choices", []):
            message = choice.get("message", {})
            msg = SimpleNamespace(content=message.get("content", ""))
            provider_fields = message.get("provider_specific_fields")
            if provider_fields is not None:
                msg.provider_specific_fields = provider_fields
            choices.append(SimpleNamespace(message=msg, logprobs=choice.get("logprobs"), finish_reason=choice.get("finish_reason")))
        usage = data.get("usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
        response = SimpleNamespace(choices=choices, usage=usage, model=data.get("model", self.model))
        return response

    def forward(self, prompt: str | None = None, messages: list[dict[str, Any]] | None = None, **kwargs):
        payload = {
            "model": self.model,
            "messages": messages or [{"role": "user", "content": prompt or ""}],
        }
        merged = {**self.kwargs, **kwargs}
        payload.update(self._sanitize_kwargs(merged))
        headers = {"Authorization": f"Bearer {self.api_key}"}
        with httpx.Client(timeout=120) as client:
            response = client.post(f"{self.api_base}/chat/completions", json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
        return self._to_response(data)
