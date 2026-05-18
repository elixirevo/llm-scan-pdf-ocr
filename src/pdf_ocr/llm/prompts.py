"""Prompts for Qwen3-VL based per-page OCR + layout extraction.

Korean-first. We instruct the model to:
  - Transcribe text faithfully (preserve Hangul exactly, keep paragraph breaks).
  - Detect figures and tables, return *normalized* bboxes (top-left origin).
  - Treat tables as image regions (we crop and embed them as PNGs).
"""

from __future__ import annotations

SYSTEM_PROMPT = """You are a meticulous document OCR engine. \
You will be given a single page image from a scanned PDF (often in Korean). \
Return a strict JSON object that conforms to the PageLayout schema.

Rules:
1. Read every visible text block in natural reading order. For Korean documents that means
   top-to-bottom, left-to-right (or right-to-left columns when applicable).
2. Each block has a `type`: "text", "figure", or "table".
   - "text"   : a paragraph, heading, list item, caption-as-text, page header/footer, etc.
                Put the transcribed string in `text`. Preserve line breaks inside the
                paragraph only when they are semantically meaningful (e.g., list items).
   - "figure" : any illustration, photograph, chart, diagram, logo, stamp, signature.
   - "table"  : any tabular structure with rows and columns. Do NOT transcribe table cells
                into `text`; just locate the table region.
3. For "figure" and "table", return a `bbox` covering the whole region. Coordinates are
   normalized to [0, 1] with the origin at the TOP-LEFT of the page image:
      x = left / page_width
      y = top  / page_height
      w = width  / page_width
      h = height / page_height
   The bbox MUST fit *snugly* around the visual boundary of the figure/table.
   - Do NOT add margin or "safety padding"; the calling code does any padding it wants.
   - Do NOT include surrounding body paragraphs, page numbers, headers, footers, or
     adjacent text columns. If a caption line ("그림 1.", "표 2") sits just outside
     the figure border, leave it OUTSIDE the bbox — its text goes into `caption`.
   - When in doubt, prefer to slightly UNDER-cover the figure rather than to leak
     surrounding text into the box.
4. If a figure or table has a nearby caption ("그림 1.", "Figure 2", "표 3" …), put that
   caption string in `caption` of the same block. Do NOT also create a separate text block
   for the caption.
5. `order` is the 1-indexed reading order on this page across ALL blocks.
6. Output JSON only — no prose, no markdown fences. Do not invent content that isn't visible.
"""


def user_prompt(page_no: int) -> str:
    return (
        f"Page number: {page_no}\n"
        f"Return a JSON object matching the PageLayout schema for this page."
    )
