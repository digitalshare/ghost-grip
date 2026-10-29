"""Parasail: the crew's brain. CrewAI LLM, schema-locked JSON, token streaming and embeddings."""
from __future__ import annotations

import json
import types
from typing import Callable, TypeVar

import openai
from crewai import LLM
from openai import OpenAI
from pydantic import BaseModel, ValidationError

from ghost_grip import config

M = TypeVar("M", bound=BaseModel)

_client: OpenAI | None = None


def client() -> OpenAI:
    global _client
    if _client is None:
        if not config.PARASAIL_API_KEY:
            raise RuntimeError("PARASAIL_API_KEY is not set; copy .env.example to .env and fill it in.")
        _client = OpenAI(base_url=config.PARASAIL_BASE_URL, api_key=config.PARASAIL_API_KEY, max_retries=0, timeout=120)
    return _client


def _call(fn):
    def run():
        try:
            return fn()
        except openai.RateLimitError as e:
            ra = e.response.headers.get("retry-after") if e.response is not None else None
            raise config.RetryableError(str(e), float(ra) if ra else None) from e
        except (openai.APIConnectionError, openai.InternalServerError) as e:
            raise config.RetryableError(str(e)) from e
    return config.with_backoff(run)


def crew_llm(temperature: float = 0.4) -> LLM:
    """CrewAI LLM on Parasail via CrewAI's native OpenAI-compatible client (bare model id + provider)."""
    llm = LLM(model=config.PARASAIL_MODEL, provider="openai", base_url=config.PARASAIL_BASE_URL,
              api_key=config.PARASAIL_API_KEY, temperature=temperature, timeout=120, max_retries=2)
    # CrewAI assumes ~8k tokens for model ids it does not know and summarises early; Parasail's models hold far more.
    object.__setattr__(llm, "get_context_window_size", types.MethodType(lambda self: 100_000, llm))
    return llm


def _strict_schema(model: type[BaseModel]) -> dict:
    """OpenAI-style strict JSON schema: every property required, no extras, $defs inlined by reference."""
    schema = model.model_json_schema()

    def fix(node):
        if isinstance(node, dict):
            if node.get("type") == "object" and "properties" in node:
                node["required"] = list(node["properties"].keys())
                node["additionalProperties"] = False
            # drop pydantic's annotation keys (a string "title"), never a field that happens to be called title
            if isinstance(node.get("title"), str):
                node.pop("title")
            if not isinstance(node.get("default", {}), dict):
                node.pop("default")
            for v in node.values():
                fix(v)
        elif isinstance(node, list):
            for v in node:
                fix(v)
    fix(schema)
    return schema


def _thinking() -> dict:
    """Hybrid reasoning models (DeepSeek V4.x) think silently before the first token: ~45-100 s on the shot-list
    brief vs ~10 s with thinking off. The simulator's revise loop catches what a fast draft gets wrong."""
    if config.PARASAIL_THINKING:
        return {}
    return {"chat_template_kwargs": {"thinking": False, "enable_thinking": False}}


def structured(model: type[M], system: str, user: str, temperature: float = 0.3,
               on_token: Callable[[str], None] | None = None) -> M:
    """Schema-locked JSON (response_format json_schema). Streams raw tokens to on_token when given.

    Decoding is constrained server-side, but a model can still stop early or skip a field, so one
    repair round feeds the validation error back.
    """
    schema = _strict_schema(model)
    messages = [{"role": "system", "content": system + "\nReply with one JSON object matching the schema; include every field."},
                {"role": "user", "content": user}]
    for attempt in range(2):
        kwargs = dict(
            model=config.PARASAIL_MODEL, messages=messages, temperature=temperature, extra_body=_thinking(),
            response_format={"type": "json_schema", "json_schema": {"name": model.__name__, "schema": schema, "strict": True}},
        )
        if on_token is None:
            resp = _call(lambda: client().chat.completions.create(**kwargs))
            text = resp.choices[0].message.content or ""
        else:
            def stream() -> str:
                parts = []
                for chunk in client().chat.completions.create(stream=True, **kwargs):
                    delta = chunk.choices[0].delta.content if chunk.choices else None
                    if delta:
                        parts.append(delta)
                        on_token(delta)
                return "".join(parts)
            text = _call(stream)
        try:
            return model.model_validate(_tidy(json.loads(_strip_fences(text))))
        except (ValueError, ValidationError) as e:
            if attempt == 1:
                raise
            messages = messages + [{"role": "assistant", "content": text},
                                   {"role": "user", "content": f"That JSON is invalid: {str(e)[:800]}\nReturn the complete corrected JSON object."}]
    raise RuntimeError("unreachable")


def stream_text(system: str, user: str, on_token: Callable[[str], None], temperature: float = 0.7, max_tokens: int = 400) -> str:
    def run() -> str:
        parts = []
        for chunk in client().chat.completions.create(
            model=config.PARASAIL_MODEL, stream=True, temperature=temperature, max_tokens=max_tokens, extra_body=_thinking(),
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        ):
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                parts.append(delta)
                on_token(delta)
        return "".join(parts)
    return _call(run)


def embed(texts: list[str]) -> list[list[float]]:
    resp = _call(lambda: client().embeddings.create(model=config.PARASAIL_EMBED_MODEL, input=texts))
    return [d.embedding for d in resp.data]


def _tidy(value):
    """Strip stray separators some models leave on string values in constrained decoding (e.g. ',Maya')."""
    if isinstance(value, str):
        return value.strip().lstrip(",;").strip()
    if isinstance(value, list):
        return [_tidy(v) for v in value]
    if isinstance(value, dict):
        return {k: _tidy(v) for k, v in value.items()}
    return value


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        text = text.rsplit("```", 1)[0]
    # some reasoning models prefix a <think> block
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.strip()
