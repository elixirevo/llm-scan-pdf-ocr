"""Per-page extraction: page image → validated :class:`PageLayout`."""

from __future__ import annotations

import json
import logging

from pydantic import ValidationError

from ..llm.openai_compat import VLMClient
from ..llm.prompts import SYSTEM_PROMPT, user_prompt
from ..llm.schema import PageLayout
from ..pdf.page import Page

log = logging.getLogger(__name__)


class ExtractionError(RuntimeError):
    pass


async def extract_page(client: VLMClient, page: Page) -> PageLayout:
    """Call the VLM once for ``page`` and return a normalized PageLayout.

    Performs one transparent retry if the model returns malformed JSON or a
    schema-invalid payload — typically a tightened "JSON only" reminder is
    enough to recover. After that we raise :class:`ExtractionError`.
    """
    user = user_prompt(page.page_no)

    last_err: Exception | None = None
    for attempt in (1, 2):
        raw = await client.chat_image_json(
            page.image, system=SYSTEM_PROMPT, user=user
        )
        try:
            data = json.loads(raw)
            # Force the page_no the caller expects, in case the model omits/guesses it.
            data["page_no"] = page.page_no
            layout = PageLayout.model_validate(data).normalized()
            return layout
        except (json.JSONDecodeError, ValidationError) as e:
            last_err = e
            log.warning(
                "page %d: invalid VLM JSON (attempt %d/2): %s", page.page_no, attempt, e
            )
            # Tighten the next try.
            user = (
                user_prompt(page.page_no)
                + "\n\nIMPORTANT: Your previous reply was not valid JSON for the schema. "
                "Return JSON only, no markdown fences, no prose."
            )

    raise ExtractionError(
        f"page {page.page_no}: VLM produced invalid JSON twice: {last_err}"
    )
