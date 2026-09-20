"""Bridge to the Edge0 engine.

Two transports, one interface:

``InProcessClient``  drives an engine loaded in this process (the
                     workbench's normal mode — one copy of the weights,
                     decisions and chat sharing the same engine lock).
``HttpClient``       talks to a separate ``edge0 serve`` over the
                     OpenAI-compatible API, for running the workbench on
                     a different machine from the model.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request


class InProcessClient:
    """Chat against a locally loaded Edge0 engine via its QueueServer."""

    def __init__(self, queue_server):
        self.queue_server = queue_server
        self.model_name = queue_server.model_name

    @property
    def engine(self):
        return self.queue_server.engine

    def chat(self, messages: list[dict], max_tokens: int = 1024,
             temperature: float | None = None, seed: int | None = None,
             enable_thinking: bool = False, on_token=None) -> str:
        from edge0.server.chat import ChatMessage, ChatRequest, decode_tokens

        request = ChatRequest(
            model=self.model_name,
            messages=[ChatMessage(role=m["role"], content=m["content"])
                      for m in messages],
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
            enable_thinking=enable_thinking,
        )
        tokens, _meta = self.queue_server.chat(request, on_token=on_token)
        text = decode_tokens(self.engine, tokens)
        return strip_thinking(text)


class HttpClient:
    """Chat against a remote ``edge0 serve`` instance."""

    def __init__(self, base_url: str = "http://127.0.0.1:8000/v1",
                 model: str = "edge0-35b", timeout: float = 600.0):
        self.base_url = base_url.rstrip("/")
        self.model_name = model
        self.timeout = timeout

    def chat(self, messages: list[dict], max_tokens: int = 1024,
             temperature: float | None = None, seed: int | None = None,
             enable_thinking: bool = False, on_token=None) -> str:
        payload = {
            "model": self.model_name, "messages": messages,
            "max_tokens": max_tokens, "enable_thinking": enable_thinking,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if seed is not None:
            payload["seed"] = seed
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as resp:
                body = json.loads(resp.read())
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"cannot reach Edge0 at {self.base_url}: {exc}") from exc
        if "error" in body:
            raise RuntimeError(body["error"].get("message", "unknown error"))
        content = body["choices"][0]["message"].get("content", "")
        return strip_thinking(content)


def strip_thinking(text: str) -> str:
    """Drop a reasoning block from a reply.

    The qwen template can emit ``<think>...</think>`` before the answer.
    Downstream parsers (tool envelopes, CHOICE markers) must see the
    answer only, or a reasoning block that mentions a tool name gets
    parsed as a tool call.
    """
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.strip()
