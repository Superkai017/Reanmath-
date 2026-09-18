"""OCR for scanned PDF pages and images.

Three engines share the page cache and concurrency logic in ``CachedPageOCR``:

* ``GeminiPageOCR`` sends each page to Gemini vision (as its embedded scan
  image, or as a one-page PDF when the page is not a single image) and gets
  back Unicode Khmer with formulas in LaTeX. Best quality, but a paid key:
  the free tier allows only 20 requests per day per model.
* ``KiriPageOCR`` (``src/ingestion/khmer_ocr.py``) runs the open-source Kiri
  OCR model locally and keeps Khmer words only. Free, but it deletes formulas.
* ``HybridPageOCR`` (``src/ingestion/hybrid_ocr.py``) feeds Kiri's Khmer to a
  Groq vision model together with the page image, so the Khmer keeps Kiri's
  spelling and the formulas come back as LaTeX. Free, and the default when a
  Groq key is set but no Gemini one.

Transcripts are cached on disk by content hash, so re-ingesting a document
never pays for the same page twice.
"""
from __future__ import annotations

import hashlib
import importlib.util
import io
import logging
import os
import random
import re
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from src.ingestion.khmer_segment import looks_garbled

if TYPE_CHECKING:
    from src.config import Settings

logger = logging.getLogger(__name__)

OCR_PROMPT_VERSION = "2"

OCR_PROMPT = """You are transcribing one scanned page of a Cambodian Grade 12 (Bac II) mathematics document.

Transcribe everything on the page, faithfully and completely:
- Write Khmer text in correct Unicode Khmer, exactly as printed. Do not translate, summarise or correct it.
- Write every mathematical expression in LaTeX: $...$ inline, $$...$$ for displayed equations.
  Use proper commands (\\frac, \\sqrt, \\lim_{x \\to a}, \\int_a^b, \\vec{u}, \\overrightarrow{AB}, \\begin{cases}...\\end{cases}).
  Khmer words inside a formula go in \\text{...}.
- Write titles as Markdown headings: # for a lesson or chapter title, ## for a section (e.g. និយមន័យ, ទ្រឹស្តីបទ, លំហាត់),
  ### for a numbered exercise or problem. Only real titles, never ordinary sentences.
- Keep the page structure: exercise and question numbers (១. ២. ក. ខ. ...), line breaks between items.
- Write tables as Markdown tables.
- For a figure or graph, write one line: [រូបភាព: short description of what it shows].
- Skip page numbers, running headers/footers and watermarks.
- If a character is illegible, write [?] in its place.

Output only the transcription, with no preamble, commentary or code fences.
If the page is blank, output nothing."""

_CODE_FENCE = re.compile(r"\A\s*```[a-zA-Z]*\s*\n(.*?)\n\s*```\s*\Z", re.DOTALL)
_IMAGE_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}
_RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}
_RETRY_HINT = re.compile(r"retry in (\d+(?:\.\d+)?)\s*s", re.IGNORECASE)

OCRMode = Literal["auto", "always", "never"]


@dataclass(frozen=True)
class PageImage:
    """What gets sent to the vision model for one page."""

    data: bytes
    mime_type: str


class OCRError(RuntimeError):
    pass


OCREngine = Literal["gemini", "kiri", "hybrid"]


def page_payload(page: Any, render_scale: float = 2.0) -> PageImage:
    """The page's single scan image if it has one, else the page rendered as PNG.

    Rendering matters for typeset pages: a PDF's text layer can hold legacy
    Khmer font encodings, and a vision model given the PDF may read that
    broken layer instead of the glyphs. The one-page PDF is the fallback when
    rendering fails.
    """
    try:
        images = list(page.images)
    except Exception:
        logger.debug("Could not list page images; sending the page as PDF", exc_info=True)
        images = []
    if len(images) == 1:
        suffix = Path(images[0].name).suffix.lower()
        if suffix in _IMAGE_TYPES and images[0].data:
            return PageImage(images[0].data, _IMAGE_TYPES[suffix])

    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_page(page)
    buffer = io.BytesIO()
    writer.write(buffer)
    pdf_bytes = buffer.getvalue()
    try:
        return PageImage(_render_png(pdf_bytes, render_scale), "image/png")
    except Exception:
        logger.debug("Could not render the page; sending it as PDF", exc_info=True)
        return PageImage(pdf_bytes, "application/pdf")


def _render_png(pdf_bytes: bytes, scale: float) -> bytes:
    import pypdfium2 as pdfium

    document = pdfium.PdfDocument(pdf_bytes)
    try:
        image = document[0].render(scale=scale).to_pil()
    finally:
        document.close()
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def clean_transcript(text: str) -> str:
    match = _CODE_FENCE.match(text)
    if match:
        text = match.group(1)
    return text.strip()


class CachedPageOCR:
    """Page cache, the auto/always decision and concurrent page transcription.

    Subclasses implement ``_cache_key`` and ``_transcribe_uncached``.
    """

    engine: OCREngine
    model: str

    def __init__(
        self,
        *,
        mode: Literal["auto", "always"] = "auto",
        min_chars: int = 20,
        cache_dir: str | Path | None = None,
        concurrency: int = 1,
    ) -> None:
        self.mode = mode
        self.min_chars = min_chars
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.concurrency = max(1, concurrency)
        self._cache_lock = threading.Lock()

    def needs_ocr(self, extracted_text: str) -> bool:
        return (
            self.mode == "always"
            or len(extracted_text.strip()) < self.min_chars
            or looks_garbled(extracted_text)
        )

    # -- cache ----------------------------------------------------------------

    def _cache_key(self, payload: PageImage) -> str:
        raise NotImplementedError

    def _legacy_cache_keys(self, payload: PageImage) -> list[str]:
        """Keys of older cache entries that are still valid transcripts."""
        return []

    def _cache_path(self, key: str) -> Path | None:
        return self.cache_dir / f"{key}.md" if self.cache_dir is not None else None

    def _read_cache(self, key: str) -> str | None:
        path = self._cache_path(key)
        if path is None or not path.exists():
            return None
        return path.read_text(encoding="utf-8")

    def _write_cache(self, key: str, text: str) -> None:
        path = self._cache_path(key)
        if path is None:
            return
        with self._cache_lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temp_name = tempfile.mkstemp(dir=path.parent, prefix=".ocr-", suffix=".md")
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    handle.write(text)
                os.replace(temp_name, path)
            except BaseException:
                Path(temp_name).unlink(missing_ok=True)
                raise

    # -- transcription ----------------------------------------------------------

    def _transcribe_uncached(self, payload: PageImage) -> tuple[str, bool]:
        """Returns (transcript, truncated); raises ``OCRError`` on failure."""
        raise NotImplementedError

    def transcribe_page(self, payload: PageImage) -> tuple[str, bool]:
        """Transcribe one page. Returns (transcript, truncated).

        Complete transcripts are cached; truncated ones never are.
        """
        key = self._cache_key(payload)
        for candidate in (key, *self._legacy_cache_keys(payload)):
            cached = self._read_cache(candidate)
            if cached is not None:
                return cached, False
        text, truncated = self._transcribe_uncached(payload)
        if not truncated:
            self._write_cache(key, text)
        return text, truncated

    def transcribe(self, payload: PageImage) -> str:
        return self.transcribe_page(payload)[0]

    def transcribe_pages(self, pages: dict[int, PageImage]) -> tuple[dict[int, str], list[str]]:
        """Transcribe several pages concurrently.

        Returns transcripts by page number, plus warnings for pages that
        failed (a failed page never aborts the whole document).
        """
        results: dict[int, str] = {}
        failures: dict[int, str] = {}
        truncated: list[int] = []
        if not pages:
            return results, []
        total = len(pages)
        with ThreadPoolExecutor(max_workers=min(self.concurrency, total)) as pool:
            futures = {pool.submit(self.transcribe_page, payload): number for number, payload in pages.items()}
            for done, future in enumerate(as_completed(futures), start=1):
                number = futures[future]
                try:
                    results[number], was_truncated = future.result()
                    if was_truncated:
                        truncated.append(number)
                    logger.info("OCR page %d done (%d/%d)", number, done, total)
                except OCRError as exc:
                    failures[number] = str(exc)
                    logger.error("OCR page %d failed: %s", number, exc)
        warnings = []
        if failures:
            listed = ", ".join(str(number) for number in sorted(failures))
            warnings.append(f"OCR failed for page(s) {listed}: {next(iter(failures.values()))}")
        if truncated:
            listed = ", ".join(str(number) for number in sorted(truncated))
            warnings.append(f"OCR transcript may be incomplete for page(s) {listed} (output limit reached)")
        return results, warnings


class GeminiPageOCR(CachedPageOCR):
    engine: OCREngine = "gemini"

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        mode: Literal["auto", "always"] = "auto",
        min_chars: int = 20,
        cache_dir: str | Path | None = None,
        concurrency: int = 4,
        max_retries: int = 4,
        thinking_level: str = "low",
        timeout: float = 300.0,
        requests_per_minute: int = 0,
        client: Any = None,
    ) -> None:
        from google.genai import errors, types

        super().__init__(mode=mode, min_chars=min_chars, cache_dir=cache_dir, concurrency=concurrency)
        self._types = types
        self._errors = errors
        if client is None:
            from google import genai

            client = genai.Client(
                api_key=api_key, http_options=types.HttpOptions(timeout=int(timeout * 1000))
            )
        self._client = client
        self.model = model
        self.max_retries = max(0, max_retries)
        self.thinking_level = thinking_level
        self.max_output_tokens = 16000
        self._interval = 60.0 / requests_per_minute if requests_per_minute > 0 else 0.0
        self._next_request = 0.0
        self._pace_lock = threading.Lock()

    def _cache_key(self, payload: PageImage, version: str = OCR_PROMPT_VERSION) -> str:
        # The model is deliberately not part of the key: a page transcribed by
        # any Gemini model is reused, so a quota-limited run can be finished with another.
        digest = hashlib.sha256()
        for part in (version, payload.mime_type):
            digest.update(part.encode("utf-8") + b"\x00")
        digest.update(payload.data)
        return digest.hexdigest()

    def _legacy_cache_keys(self, payload: PageImage) -> list[str]:
        # Version 1 differs only in how headings are marked up.
        return [self._cache_key(payload, "1")]

    def _pace(self) -> None:
        """Space requests out to stay under OCR_REQUESTS_PER_MINUTE."""
        if not self._interval:
            return
        with self._pace_lock:
            now = time.monotonic()
            start = max(now, self._next_request)
            self._next_request = start + self._interval
        if start > now:
            time.sleep(start - now)

    # -- transcription ----------------------------------------------------------

    def _request(self, payload: PageImage, max_output_tokens: int) -> tuple[str, bool]:
        """Returns (transcript, truncated)."""
        types = self._types
        config = types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=max_output_tokens,
            thinking_config=types.ThinkingConfig(thinking_level=self.thinking_level.upper()),
        )
        self._pace()
        response = self._client.models.generate_content(
            model=self.model,
            contents=[
                types.Part.from_bytes(data=payload.data, mime_type=payload.mime_type),
                OCR_PROMPT,
            ],
            config=config,
        )
        text = response.text or ""
        if not text.strip():
            block_reason = getattr(response.prompt_feedback, "block_reason", None)
            if block_reason:
                raise OCRError(f"Gemini blocked the page ({block_reason})")
        truncated = False
        if response.candidates:
            reason = response.candidates[0].finish_reason
            truncated = getattr(reason, "value", reason) == "MAX_TOKENS"
        return clean_transcript(text), truncated

    def _request_with_retries(self, payload: PageImage, max_output_tokens: int) -> tuple[str, bool]:
        errors = self._errors
        for attempt in range(self.max_retries + 1):
            wait_hint = 0.0
            try:
                return self._request(payload, max_output_tokens)
            except errors.APIError as exc:
                # Quota errors say how long to wait ("Please retry in 46.1s").
                hint = _RETRY_HINT.search(exc.message or "")
                wait_hint = min(300.0, float(hint.group(1)) + 1.0) if hint else 0.0
                retryable = (exc.code or 0) in _RETRYABLE_STATUS
                if not retryable or attempt == self.max_retries:
                    first_line = (exc.message or "").strip().splitlines()[0] if exc.message else ""
                    raise OCRError(f"Gemini OCR failed ({exc.code}): {first_line}") from exc
            except (TimeoutError, OSError) as exc:
                if attempt == self.max_retries:
                    raise OCRError(f"Gemini OCR request failed: {exc}") from exc
            delay = min(60.0, 2.0 ** (attempt + 1)) + random.uniform(0, 1)
            delay = max(delay, wait_hint)
            logger.info("Gemini OCR busy; retrying in %.0fs (attempt %d)", delay, attempt + 1)
            time.sleep(delay)
        raise AssertionError("unreachable")

    def _transcribe_uncached(self, payload: PageImage) -> tuple[str, bool]:
        # A transcript cut off by the output limit is retried once with a larger limit.
        text, truncated = self._request_with_retries(payload, self.max_output_tokens)
        if truncated:
            logger.info("OCR transcript hit the output limit; retrying with a larger limit")
            text, truncated = self._request_with_retries(payload, self.max_output_tokens * 4)
        return text, truncated


def kiri_available() -> bool:
    return importlib.util.find_spec("kiri_ocr") is not None


def resolve_ocr_engine(settings: Settings) -> OCREngine | None:
    """The engine OCR_ENGINE selects, or None when no engine is usable.

    ``auto`` prefers Gemini (it also transcribes formulas as LaTeX) and falls
    back to the local Kiri model when no Gemini key is configured.
    """
    if settings.ocr_engine == "gemini":
        if settings.gemini_api_key is None:
            raise RuntimeError("OCR_ENGINE=gemini requires GEMINI_API_KEY")
        return "gemini"
    if settings.ocr_engine == "kiri":
        if not kiri_available():
            raise RuntimeError("OCR_ENGINE=kiri requires kiri-ocr (uv add kiri-ocr)")
        return "kiri"
    if settings.ocr_engine == "hybrid":
        if not kiri_available():
            raise RuntimeError("OCR_ENGINE=hybrid requires kiri-ocr (uv add kiri-ocr)")
        if settings.groq_api_key is None:
            raise RuntimeError("OCR_ENGINE=hybrid requires GROQ_API_KEY")
        return "hybrid"
    if settings.gemini_api_key is not None:
        return "gemini"
    # Hybrid beats Kiri alone: Kiri deletes every formula, which a maths corpus
    # cannot afford, so prefer it whenever a Groq key is also configured.
    if kiri_available() and settings.groq_api_key is not None:
        return "hybrid"
    if kiri_available():
        return "kiri"
    return None


def build_ocr(settings: Settings) -> CachedPageOCR | None:
    """The configured OCR engine, or None when OCR is disabled/unavailable."""
    if settings.ocr_mode == "never":
        return None
    engine = resolve_ocr_engine(settings)
    if engine is None:
        if settings.ocr_mode == "always":
            raise RuntimeError("OCR_MODE=always needs GEMINI_API_KEY or kiri-ocr")
        logger.info("No OCR engine available; scanned pages and images will not be OCR'd")
        return None
    if engine in ("kiri", "hybrid"):
        from src.ingestion.khmer_ocr import KiriPageOCR

        kiri = KiriPageOCR(
            settings.kiri_model,
            mode=settings.ocr_mode,
            min_chars=settings.ocr_min_chars,
            cache_dir=settings.ocr_cache_dir,
            device=settings.kiri_device,
            decode_method=settings.kiri_decode_method,
            min_confidence=settings.kiri_min_confidence,
            khmer_only=settings.kiri_khmer_only,
            render_scale=settings.kiri_render_scale,
        )
        if engine == "kiri":
            return kiri

        from src.ingestion.hybrid_ocr import HybridPageOCR

        return HybridPageOCR(
            settings.groq_api_key.get_secret_value(),
            settings.hybrid_ocr_model,
            kiri,
            mode=settings.ocr_mode,
            min_chars=settings.ocr_min_chars,
            cache_dir=settings.ocr_cache_dir,
            concurrency=settings.ocr_concurrency,
            max_retries=settings.ocr_max_retries,
            max_output_tokens=settings.hybrid_ocr_max_tokens,
            timeout=settings.llm_timeout_seconds,
            requests_per_minute=settings.ocr_requests_per_minute,
        )
    return GeminiPageOCR(
        settings.gemini_api_key.get_secret_value(),
        settings.ocr_model,
        mode=settings.ocr_mode,
        min_chars=settings.ocr_min_chars,
        cache_dir=settings.ocr_cache_dir,
        concurrency=settings.ocr_concurrency,
        max_retries=settings.ocr_max_retries,
        thinking_level=settings.ocr_thinking_level,
        timeout=settings.llm_timeout_seconds,
        requests_per_minute=settings.ocr_requests_per_minute,
    )
