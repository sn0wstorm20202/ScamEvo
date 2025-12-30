from __future__ import annotations

import json
from typing import Any

import httpx

from app.core.config import Settings


class OpenAIError(RuntimeError):
    pass


def _extract_json_object(text: str) -> str | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def _get_content_from_responses_api(resp_json: dict[str, Any]) -> str | None:
    if isinstance(resp_json.get("output_text"), str):
        return str(resp_json.get("output_text"))

    output = resp_json.get("output")
    if isinstance(output, list):
        parts: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for c in content:
                if not isinstance(c, dict):
                    continue
                if c.get("type") == "output_text" and isinstance(c.get("text"), str):
                    parts.append(str(c.get("text")))
        if parts:
            return "\n".join(parts)

    return None


def _get_content_from_chat_completions(resp_json: dict[str, Any]) -> str | None:
    choices = resp_json.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    if not isinstance(first, dict):
        return None
    msg = first.get("message")
    if not isinstance(msg, dict):
        return None
    content = msg.get("content")
    if isinstance(content, str):
        return content
    return None


def _parse_candidates_from_text(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []

    obj: dict[str, Any] | None = None

    try:
        obj = json.loads(text)
    except Exception:
        embedded = _extract_json_object(text)
        if embedded:
            try:
                obj = json.loads(embedded)
            except Exception:
                obj = None

    if not isinstance(obj, dict):
        return []

    raw = obj.get("candidates")
    if isinstance(raw, list):
        out: list[str] = []
        for item in raw:
            if isinstance(item, str):
                t = item.strip()
                if t:
                    out.append(t)
                continue
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                t = str(item.get("text")).strip()
                if t:
                    out.append(t)
        return out

    return []


def generate_mutations_openai(
    *,
    settings: Settings,
    base_text: str,
    num_candidates: int,
    seed: int,
    mode: str,
) -> list[str]:
    if not settings.openai_api_key:
        raise OpenAIError("Missing SCAMEVO_OPENAI_API_KEY")

    system = (
        "You are generating adversarial variants of scam SMS/email text for security research. "
        "Keep the intent (financial scam) the same while changing surface form. "
        "Output MUST be valid JSON only."
    )

    user = (
        "Return JSON in the format {\"candidates\": [\"...\", ...]}. "
        "Generate exactly {k} distinct variants. "
        "Do not include analysis or extra keys. "
        "Mode: {mode}. "
        "Base text:\n{base}\n"
    ).format(k=int(num_candidates), mode=str(mode), base=str(base_text))

    headers = {
        "Authorization": f"Bearer {settings.openai_api_key}",
        "Content-Type": "application/json",
    }

    timeout = httpx.Timeout(timeout=settings.llm_timeout_seconds)

    errors: list[str] = []

    with httpx.Client(timeout=timeout) as client:
        url_responses = f"{settings.openai_base_url}/responses"
        payload_responses: dict[str, Any] = {
            "model": settings.llm_model,
            "input": [
                {"role": "system", "content": [{"type": "text", "text": system}]},
                {"role": "user", "content": [{"type": "text", "text": user}]},
            ],
            "temperature": 0.7,
        }

        try:
            r = client.post(url_responses, headers=headers, json=payload_responses)
            if r.status_code >= 400:
                errors.append(f"responses_api_status={r.status_code}")
            else:
                content = _get_content_from_responses_api(r.json())
                if content:
                    cands = _parse_candidates_from_text(content)
                    if cands:
                        return cands
                errors.append("responses_api_parse_failed")
        except Exception as e:
            errors.append(f"responses_api_error={e}")

        url_chat = f"{settings.openai_base_url}/chat/completions"
        payload_chat: dict[str, Any] = {
            "model": settings.llm_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.7,
        }

        try:
            r = client.post(url_chat, headers=headers, json=payload_chat)
            if r.status_code >= 400:
                errors.append(f"chat_completions_status={r.status_code}")
            else:
                content = _get_content_from_chat_completions(r.json())
                if content:
                    cands = _parse_candidates_from_text(content)
                    if cands:
                        return cands
                errors.append("chat_completions_parse_failed")
        except Exception as e:
            errors.append(f"chat_completions_error={e}")

    raise OpenAIError("OpenAI generation failed: " + ";".join(errors))
