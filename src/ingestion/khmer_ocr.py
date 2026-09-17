"""Local Khmer OCR with Kiri OCR (https://github.com/mrrtmob/kiri-ocr).

Kiri detects text lines on a page image and recognises each one with a small
Transformer model that runs on the CPU, so no API key is needed. It reads
Khmer well but cannot transcribe formulas, so by default only Khmer words are
kept: every recognised line is reduced to its runs of Khmer script (letters,
Khmer digits and Khmer punctuation) and lines below ``min_confidence`` are
dropped (the published model scores even clean lines around 0.4-0.45, so
the default cut-off is low). Latin letters, Arabic digits and math symbols,
which Kiri tends to misread in formulas, are removed rather than indexed as
noise.

Kiri reads images from disk, so each page is written to a temporary PNG/JPEG
first. PDF pages that are not a single embedded scan are rendered with
pypdfium2.
"""
from __future__ import annotations

import hashlib
import io
import logging
import os
import re
import tempfile
import threading
import unicodedata
from typing import Any, Literal

from src.ingestion.ocr import CachedPageOCR, OCREngine, OCRError, PageImage

logger = logging.getLogger(__name__)

KIRI_CACHE_VERSION = "2"

DecodeMethod = Literal["fast", "accurate", "beam"]

_IMAGE_SUFFIXES = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp"}

# A run of Khmer script: consonants, vowels, signs, Khmer digits, Khmer
# punctuation and symbols, plus the joiners used inside Khmer words.
_KHMER_RUN = re.compile(r"[\u1780-\u17DD\u17E0-\u17E9\u17F0-\u17F9\u19E0-\u19FF\u200C\u200D]+")
# A run that carries meaning has at least one base character or Khmer digit.
_KHMER_BASE = re.compile(r"[\u1780-\u17B3\u17E0-\u17E9]")
# Dependent vowels, signs and joiners cannot start a word; OCR sometimes
# leaves them behind when the preceding (non-Khmer) glyph was dropped.
_LEADING_MARKS = re.compile(r"^[\u17B4-\u17D3\u17DD\u200C\u200D]+")
_KHMER_PUNCTUATION = re.compile(r"[\u17D4-\u17DA]+")
# Kiri often reads the Khmer sign YUUKALEAPINTU (ៈ) as an ASCII colon.
_COLON_IN_WORD = re.compile(r"(?<=[\u1780-\u17D3]):(?=[\u1780-\u17B3])")


def khmer_words_only(line: str) -> str:
    """Keep only the Khmer words of ``line``, separated by single spaces.

    Standalone Khmer punctuation (។ ៕ ...) is kept when it follows a word, so
    sentence boundaries survive for chunking.
    """
    runs: list[str] = []
    line = _COLON_IN_WORD.sub("\u17c8", unicodedata.normalize("NFC", line))
    for match in _KHMER_RUN.finditer(line):
        run = _LEADING_MARKS.sub("", match.group(0)).rstrip("\u200c\u200d")
        if _KHMER_BASE.search(run):
            runs.append(run)
        elif runs and _KHMER_PUNCTUATION.fullmatch(run):
            runs.append(run)
    return " ".join(runs)


def group_lines(results: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Group detected regions into visual lines, keeping the detector's
    reading order (the same rule Kiri's own ``extract_text`` uses)."""
    lines: list[list[dict[str, Any]]] = []
    previous_center: float | None = None
    previous_height = 0
    for result in results:
        _, y, _, height = result["box"]
        center = y + height / 2
        if (
            lines
            and previous_center is not None
            and abs(center - previous_center) < max(height, previous_height) * 0.8
        ):
            lines[-1].append(result)
        else:
            lines.append([result])
        previous_center, previous_height = center, height
    return lines


class KiriPageOCR(CachedPageOCR):
    engine: OCREngine = "kiri"

    def __init__(
        self,
        model: str = "mrrtmob/kiri-ocr",
        *,
        mode: Literal["auto", "always"] = "auto",
        min_chars: int = 20,
        cache_dir: str | os.PathLike[str] | None = None,
        device: str = "cpu",
        decode_method: DecodeMethod = "accurate",
        min_confidence: float = 0.2,
        khmer_only: bool = True,
        render_scale: float = 2.0,
        engine_factory: Any = None,
    ) -> None:
        # The model is not thread-safe and already uses every CPU core, so
        # pages are processed one at a time.
        super().__init__(mode=mode, min_chars=min_chars, cache_dir=cache_dir, concurrency=1)
        self.model = model
        self.device = device
        self.decode_method = decode_method
        self.min_confidence = min_confidence
        self.khmer_only = khmer_only
        self.render_scale = render_scale
        self._engine_factory = engine_factory
        self._engine: Any = None
        self._engine_lock = threading.Lock()

    # -- model ------------------------------------------------------------------

    def _load_engine(self) -> Any:
        if self._engine is not None:
            return self._engine
        if self._engine_factory is None:
            try:
                from kiri_ocr import OCR
            except ImportError as exc:
                raise OCRError("kiri-ocr is not installed (uv add kiri-ocr)") from exc
            factory = OCR
        else:
            factory = self._engine_factory
        logger.info("Loading Kiri OCR model %s on %s (first use downloads it)", self.model, self.device)
        try:
            # Kiri calls sys.exit() on a model/vocabulary mismatch.
            self._engine = factory(
                model_path=self.model, device=self.device, decode_method=self.decode_method
            )
        except (Exception, SystemExit) as exc:
            raise OCRError(f"Could not load Kiri OCR model '{self.model}': {exc}") from exc
        return self._engine

    def warmup(self) -> None:
        with self._engine_lock:
            self._load_engine()

    # -- cache ------------------------------------------------------------------

    def _cache_key(self, payload: PageImage) -> str:
        # Unlike Gemini transcripts, Kiri output depends on the model and the
        # filtering, so both are part of the key.
        digest = hashlib.sha256()
        for part in (
            "kiri",
            KIRI_CACHE_VERSION,
            self.model,
            self.decode_method,
            f"{self.min_confidence:.4f}",
            "khmer" if self.khmer_only else "all",
            f"{self.render_scale:.2f}",
            payload.mime_type,
        ):
            digest.update(part.encode("utf-8") + b"\x00")
        digest.update(payload.data)
        return digest.hexdigest()

    # -- transcription ----------------------------------------------------------

    def _page_image(self, payload: PageImage) -> tuple[bytes, str]:
        """Image bytes and file suffix Kiri (OpenCV) can read."""
        if payload.mime_type in _IMAGE_SUFFIXES:
            return payload.data, _IMAGE_SUFFIXES[payload.mime_type]
        if payload.mime_type != "application/pdf":
            raise OCRError(f"Kiri OCR cannot read {payload.mime_type}")
        try:
            import pypdfium2 as pdfium
        except ImportError as exc:
            raise OCRError("Rendering PDF pages for Kiri OCR requires pypdfium2") from exc
        try:
            document = pdfium.PdfDocument(payload.data)
            try:
                if len(document) == 0:
                    raise OCRError("PDF page payload is empty")
                page = document[0]
                image = page.render(scale=self.render_scale).to_pil()
            finally:
                document.close()
        except OCRError:
            raise
        except Exception as exc:
            raise OCRError(f"Could not render PDF page for Kiri OCR: {exc}") from exc
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="PNG")
        return buffer.getvalue(), ".png"

    def _recognise(self, image: bytes, suffix: str) -> list[dict[str, Any]]:
        descriptor, path = tempfile.mkstemp(prefix="kiri-", suffix=suffix)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(image)
            with self._engine_lock:
                engine = self._load_engine()
                return engine.process_document(path, mode="lines")
        except OCRError:
            raise
        except Exception as exc:
            raise OCRError(f"Kiri OCR failed: {exc}") from exc
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    def text_from_results(self, results: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        for group in group_lines([r for r in results if "box" in r]):
            words: list[str] = []
            for region in group:
                if float(region.get("confidence", 0.0)) < self.min_confidence:
                    continue
                raw = str(region.get("text") or "")
                text = khmer_words_only(raw) if self.khmer_only else " ".join(raw.split())
                if text:
                    words.append(text)
            if words:
                lines.append(" ".join(words))
        return "\n".join(lines)

    def _transcribe_uncached(self, payload: PageImage) -> tuple[str, bool]:
        image, suffix = self._page_image(payload)
        results = self._recognise(image, suffix)
        text = self.text_from_results(results)
        logger.debug("Kiri OCR: %d region(s) -> %d character(s)", len(results), len(text))
        return text, False
