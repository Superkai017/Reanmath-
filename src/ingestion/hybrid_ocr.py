"""Kiri-guided Groq OCR: correct Khmer from Kiri, formulas from a vision model.

Neither free engine is usable alone on this corpus:

* Kiri (``khmer_ocr``) spells Khmer correctly but deletes every formula, so a
  worked example collapses to dangling connectives.
* A vision model given only the page image transcribes formulas accurately but
  paraphrases Khmer, inventing plausible non-words and occasionally a whole
  section heading, and it silently under-transcribes dense pages.

So Kiri runs first and its transcript is passed to the vision model as the
authoritative spelling, with the page image alongside. The model's only job is
to put the formulas back in the right places. Anchoring it to real text also
stops the under-transcription: it can see what it is expected to account for,
so one call per page is enough and the page needs no tiling.

Merging the two transcripts in code was tried first and does not work: Kiri
drops words and the vision model corrupts them, so there is no stable anchor
to align on and formulas land mid-word, duplicated and out of order.
"""
from __future__ import annotations

import base64
import hashlib
import logging
import random
import re
import threading
import time
from typing import Any, Literal

from src.ingestion.khmer_ocr import KiriPageOCR
from src.ingestion.ocr import CachedPageOCR, OCREngine, OCRError, PageImage, clean_transcript

logger = logging.getLogger(__name__)

HYBRID_CACHE_VERSION = "1"

_RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}

GUIDED_PROMPT = """You are reconstructing one page of a Cambodian Grade 12 (Bac II) mathematics document.

You are given (1) the page image and (2) a Khmer-only OCR transcript of that page. The
transcript has CORRECT Khmer spelling, but every mathematical formula was deleted from it.

Your job is to output the page with the formulas put back.

STRICT RULES:
- Use the Khmer words from the transcript EXACTLY as given. Do not re-spell, translate,
  paraphrase or "improve" any Khmer word. Copy them character for character.
- The transcript may be missing Khmer words that the image shows. Add such a word only
  when it is clearly legible in the image; otherwise leave it out.
- Insert each mathematical expression, in LaTeX ($...$ inline, $$...$$ displayed), at the
  point in the text where the image shows it. Use proper commands (\\frac, \\sqrt,
  \\lim_{x \\to a}, \\int_a^b, \\vec{u}, \\overrightarrow{AB}, \\begin{cases}...\\end{cases}).
- Never put Khmer inside \\text{} unless it genuinely sits inside a formula in the image.
- Never repeat a formula. Each one appears exactly once.
- Keep the reading order of the image, top to bottom.
- Write titles as Markdown headings: # for a lesson title, ## for a section
  (និយមន័យ, ទ្រឹស្តីបទ, លំហាត់), ### for a numbered exercise. Only real titles.
- For a figure or graph, write one line: [រូបភាព: short description].
- Skip page numbers, running headers/footers and watermarks.

Output only the reconstructed page, with no preamble, commentary or code fences.

--- KHMER TRANSCRIPT (authoritative for spelling) ---
"""

# The vision model sometimes wraps a whole Khmer sentence in \text{} instead of
# leaving it as prose; unwrap those so the Khmer stays searchable.
_TEXT_WRAPPED_KHMER = re.compile(r"\\text\{([^{}$]*[\u1780-\u17FF][^{}$]*)\}")
_KHMER = re.compile(r"[\u1780-\u17FF]")


def unwrap_khmer_text(transcript: str) -> str:
    """Unwrap ``\\text{...}`` spans that are Khmer prose rather than formula labels.

    A span is left alone when it sits inside a formula that also carries maths;
    only a ``$$ \\text{...} $$`` whose entire body is Khmer is unwrapped.
    """

    def _unwrap(match: re.Match[str]) -> str:
        return match.group(1)

    def _strip_lone(match: re.Match[str]) -> str:
        body = match.group(1).strip()
        inner = _TEXT_WRAPPED_KHMER.fullmatch(body)
        return _unwrap(inner) if inner else match.group(0)

    transcript = re.sub(r"\$\$\s*(.+?)\s*\$\$", _strip_lone, transcript, flags=re.DOTALL)
    return re.sub(r"\$\s*(\\text\{[^{}$]*\})\s*\$", _strip_lone, transcript)


class HybridPageOCR(CachedPageOCR):
    """Kiri for Khmer spelling, a Groq vision model for the formulas."""

    engine: OCREngine = "hybrid"

    def __init__(
        self,
        api_key: str,
        model: str,
        kiri: KiriPageOCR,
        *,
        mode: Literal["auto", "always"] = "auto",
        min_chars: int = 20,
        cache_dir: Any = None,
        concurrency: int = 2,
        max_retries: int = 4,
        max_output_tokens: int = 8000,
        timeout: float = 300.0,
        requests_per_minute: int = 0,
        client: Any = None,
    ) -> None:
        super().__init__(mode=mode, min_chars=min_chars, cache_dir=cache_dir, concurrency=concurrency)
        if client is None:
            import groq

            client = groq.Groq(api_key=api_key, timeout=timeout)
        self._client = client
        self.model = model
        self.kiri = kiri
        self.max_retries = max(0, max_retries)
        self.max_output_tokens = max_output_tokens
        self._interval = 60.0 / requests_per_minute if requests_per_minute > 0 else 0.0
        self._next_request = 0.0
        self._pace_lock = threading.Lock()

    # -- cache ----------------------------------------------------------------

    def _cache_key(self, payload: PageImage) -> str:
        digest = hashlib.sha256()
        for part in (
            "hybrid",
            HYBRID_CACHE_VERSION,
            self.model,
            self.kiri._cache_key(payload),  # pins the Kiri model and its filtering
            payload.mime_type,
        ):
            digest.update(part.encode("utf-8") + b"\x00")
        digest.update(payload.data)
        return digest.hexdigest()

    # -- transcription --------------------------------------------------------

    def _pace(self) -> None:
        if not self._interval:
            return
        with self._pace_lock:
            now = time.monotonic()
            start = max(now, self._next_request)
            self._next_request = start + self._interval
        if start > now:
            time.sleep(start - now)

    def _request(self, payload: PageImage, khmer: str) -> tuple[str, bool]:
        encoded = base64.b64encode(payload.data).decode("ascii")
        self._pace()
        response = self._client.chat.completions.create(
            model=self.model,
            temperature=0.0,
            max_tokens=self.max_output_tokens,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": GUIDED_PROMPT + khmer},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{payload.mime_type};base64,{encoded}"},
                        },
                    ],
                }
            ],
        )
        choice = response.choices[0]
        text = choice.message.content or ""
        truncated = choice.finish_reason == "length"
        return clean_transcript(text), truncated

    def _request_with_retries(self, payload: PageImage, khmer: str) -> tuple[str, bool]:
        import groq

        for attempt in range(self.max_retries + 1):
            try:
                return self._request(payload, khmer)
            except groq.APIStatusError as exc:
                if exc.status_code not in _RETRYABLE_STATUS or attempt == self.max_retries:
                    raise OCRError(f"Groq OCR failed ({exc.status_code}): {exc}") from exc
            except (groq.APIConnectionError, TimeoutError, OSError) as exc:
                if attempt == self.max_retries:
                    raise OCRError(f"Groq OCR request failed: {exc}") from exc
            delay = min(60.0, 2.0 ** (attempt + 1)) + random.uniform(0, 1)
            logger.info("Groq OCR busy; retrying in %.0fs (attempt %d)", delay, attempt + 1)
            time.sleep(delay)
        raise AssertionError("unreachable")

    def _transcribe_uncached(self, payload: PageImage) -> tuple[str, bool]:
        khmer, _ = self.kiri.transcribe_page(payload)
        if not _KHMER.search(khmer):
            # Nothing for the vision model to anchor on: a blank page, or a page
            # whose Khmer Kiri could not read at all. Sending it unanchored would
            # invite the paraphrasing this engine exists to avoid.
            logger.info("Kiri found no Khmer on the page; skipping the guided pass")
            return khmer, False
        text, truncated = self._request_with_retries(payload, khmer)
        if not text.strip():
            return khmer, False
        return unwrap_khmer_text(text), truncated
