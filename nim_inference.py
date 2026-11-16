"""Shared streamed NIM request for evidence organization and semantic checking."""

from __future__ import annotations

from datetime import datetime, timezone
from time import monotonic
from typing import Any


MODEL = "nvidia/nemotron-3-ultra-550b-a55b"
BASE_URL = "https://integrate.api.nvidia.com/v1"
GENERATION_PARAMETERS = {
    "temperature": 1,
    "top_p": 0.95,
    "max_tokens": 16384,
    "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
    "stream": True,
}


def stream_nim_completion(client: Any, messages: list[dict[str, str]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "model": MODEL, "base_url": BASE_URL,
        "generation_parameters": GENERATION_PARAMETERS,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "api_error", "reasoning_text": "", "response_text": "",
        "finish_reason": None, "error": None, "usage": None,
    }
    reasoning, answer = [], []
    stream = None
    started = monotonic()
    try:
        stream = client.chat.completions.create(model=MODEL, messages=messages, **GENERATION_PARAMETERS)
        for chunk in stream:
            usage = getattr(chunk, "usage", None)
            if usage is not None:
                result["usage"] = usage.model_dump(mode="json")
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            if getattr(choice.delta, "reasoning_content", None):
                reasoning.append(choice.delta.reasoning_content)
            if choice.delta.content is not None:
                answer.append(choice.delta.content)
            if choice.finish_reason is not None:
                result["finish_reason"] = choice.finish_reason
        result["status"] = "response_received"
    except Exception as exc:
        result["error"] = {"type": type(exc).__name__}
        status_code = getattr(exc, "status_code", None)
        if isinstance(status_code, int):
            result["error"]["http_status"] = status_code
    finally:
        if stream is not None:
            try:
                stream.close()
            except Exception as exc:
                result["status"] = "api_error"
                if result["error"] is None:
                    result["error"] = {"type": type(exc).__name__}
        result["elapsed_seconds"] = round(monotonic() - started, 3)
        result["reasoning_text"] = "".join(reasoning)
        result["response_text"] = "".join(answer)
    if result["status"] == "response_received" and result["finish_reason"] != "stop":
        result["status"] = "incomplete_response"
        result["error"] = {"type": "IncompleteGeneration",
                           "message": "Generation did not finish with stop; see finish_reason."}
    return result
