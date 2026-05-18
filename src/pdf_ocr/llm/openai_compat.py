"""OpenAI-compatible VLM client.

Works with **vLLM**, **llama.cpp (llama-server)**, and **SGLang** — all of which
expose ``/v1/chat/completions`` with image input via ``image_url`` and base64
data URIs. The only meaningful difference is JSON-schema enforcement support,
toggled via :attr:`VLMConfig.use_response_format`.
"""

from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass

from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    BadRequestError,
    InternalServerError,
    RateLimitError,
)
from PIL.Image import Image
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .schema import PAGE_LAYOUT_JSON_SCHEMA

log = logging.getLogger(__name__)


@dataclass(slots=True)
class VLMConfig:
    base_url: str
    api_key: str
    model: str
    use_response_format: bool = True
    timeout: float = 180.0
    max_retries: int = 2
    # Max tokens for the *response*. Must be ≤ (max_model_len - input_tokens).
    # 2048 is enough for ~3-4k Korean characters of OCR JSON, fits in small servers.
    max_tokens: int = 2048


def _image_to_data_uri(img: Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    mime = f"image/{fmt.lower()}"
    return f"data:{mime};base64,{b64}"


# Errors worth retrying. 4xx other than 429 are NOT here — they will fail fast.
_RETRYABLE = (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)


class VLMClient:
    """Thin async wrapper around an OpenAI-compatible chat endpoint."""

    def __init__(self, cfg: VLMConfig) -> None:
        self.cfg = cfg
        self._client = AsyncOpenAI(
            base_url=cfg.base_url,
            api_key=cfg.api_key,
            timeout=cfg.timeout,
        )

    async def aclose(self) -> None:
        await self._client.close()

    async def chat_image_json(
        self,
        image: Image,
        *,
        system: str,
        user: str,
        max_tokens: int | None = None,
    ) -> str:
        """Send (system, user+image) and return the raw JSON string the model produced.

        Caller is responsible for parsing/validating with Pydantic.
        """
        data_uri = _image_to_data_uri(image, fmt="PNG")
        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            },
        ]

        kwargs: dict = {
            "model": self.cfg.model,
            "messages": messages,
            "max_tokens": max_tokens or self.cfg.max_tokens,
            "temperature": 0.0,
        }
        if self.cfg.use_response_format:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": PAGE_LAYOUT_JSON_SCHEMA,
            }
        else:
            # llama.cpp fallback: ask for JSON object and validate downstream.
            kwargs["response_format"] = {"type": "json_object"}

        return await self._call_with_retry(**kwargs)

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(_RETRYABLE),
    )
    async def _call_with_retry(self, **kwargs) -> str:
        try:
            resp = await self._client.chat.completions.create(**kwargs)
        except BadRequestError as e:
            # Permanent client error (e.g., max_tokens > max_model_len, bad image).
            # Surface a clearer message so the user knows what to fix.
            log.error("VLM rejected the request (400): %s", e)
            raise
        content = resp.choices[0].message.content
        if not content:
            raise RuntimeError("VLM returned empty content")
        return content
