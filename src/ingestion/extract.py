"""Turn plain text, Markdown and PDF files into clean text sections."""
from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from src.ingestion.latex_guard import mask_latex, unmask_latex
from src.ingestion.ocr import PageImage, page_payload

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS: dict[str, str] = {
    ".txt": "text",
    ".text": "text",
    ".md": "markdown",
    ".markdown": "markdown",
    ".pdf": "pdf",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".webp": "image",
}

IMAGE_MIME_TYPES: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}


class UnsupportedFileTypeError(ValueError):
    pass


class ExtractionError(RuntimeError):
    pass


@dataclass(frozen=True)
class Section:
    """A contiguous block of text; ``page`` is 1-based for PDFs."""

    text: str
    page: int | None = None
    ocr: bool = False


@dataclass
class ExtractedDocument:
    source: str
    format: str
    sections: list[Section]
    pages: int | None = None
    warnings: list[str] = field(default_factory=list)
    ocr_pages: int = 0

    @property
    def text(self) -> str:
        return "\n\n".join(section.text for section in self.sections)

    @property
    def characters(self) -> int:
        return sum(len(section.text) for section in self.sections)


def detect_format(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    try:
        return SUPPORTED_EXTENSIONS[suffix]
    except KeyError:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise UnsupportedFileTypeError(
            f"Unsupported file type '{suffix or filename}'. Supported: {supported}"
        ) from None


def decode_text_bytes(data: bytes) -> str:
    """Decode bytes as UTF-8/UTF-16 (BOM-aware), falling back to replacement."""
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16")
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        logger.warning("Input is not valid UTF-8; undecodable bytes were replaced")
        return data.decode("utf-8", errors="replace")


# --- Plain text -------------------------------------------------------------

def extract_plain_text(data: bytes) -> str:
    return decode_text_bytes(data)


# --- Markdown ---------------------------------------------------------------

_FRONT_MATTER = re.compile(r"\A---\s*\n.*?\n---\s*\n", re.DOTALL)
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_HTML_TAG = re.compile(r"</?[A-Za-z][A-Za-z0-9-]*(?:\s[^<>]*)?/?>")
_CODE_FENCE = re.compile(r"^(```|~~~)[^\n]*\n(.*?)^\1[ \t]*$", re.DOTALL | re.MULTILINE)
_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_REFERENCE_DEF = re.compile(r"^\s*\[[^\]]+\]:\s+\S+.*$", re.MULTILINE)
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(.*?)\s*#*\s*$", re.MULTILINE)
_SETEXT_UNDERLINE = re.compile(r"^\s{0,3}(=+|-+)\s*$", re.MULTILINE)
_BLOCKQUOTE = re.compile(r"^\s{0,3}>\s?", re.MULTILINE)
_HORIZONTAL_RULE = re.compile(r"^\s{0,3}([-*_])(\s*\1){2,}\s*$", re.MULTILINE)
_BOLD = re.compile(r"(\*\*|__)(?=\S)(.+?)(?<=\S)\1")
_ITALIC = re.compile(r"(?<![\w*])\*(?=\S)([^*\n]+?)(?<=\S)\*(?![\w*])")
_INLINE_CODE = re.compile(r"`([^`\n]+)`")
_TABLE_SEPARATOR = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$", re.MULTILINE)


def clean_markdown(markdown: str) -> str:
    """Strip Markdown syntax while keeping text, lists, tables and LaTeX.

    Formulas are masked first so link/emphasis patterns cannot touch them
    (e.g. ``[0,1](x)`` inside a formula is not a link).
    """
    masked, vault = mask_latex(markdown)
    text = _FRONT_MATTER.sub("", masked)
    text = _HTML_COMMENT.sub("", text)
    text = _CODE_FENCE.sub(lambda match: match.group(2), text)
    text = _IMAGE.sub(lambda match: match.group(1), text)
    text = _LINK.sub(lambda match: match.group(1), text)
    text = _REFERENCE_DEF.sub("", text)
    text = _HORIZONTAL_RULE.sub("", text)
    text = _HEADING.sub(lambda match: match.group(1), text)
    text = _SETEXT_UNDERLINE.sub("", text)
    text = _BLOCKQUOTE.sub("", text)
    text = _TABLE_SEPARATOR.sub("", text)
    text = _BOLD.sub(lambda match: match.group(2), text)
    text = _ITALIC.sub(lambda match: match.group(1), text)
    text = _INLINE_CODE.sub(lambda match: match.group(1), text)
    text = _HTML_TAG.sub("", text)
    return unmask_latex(text, vault)


def extract_markdown(data: bytes) -> str:
    return clean_markdown(decode_text_bytes(data))


# --- PDF --------------------------------------------------------------------

_HYPHENATED_BREAK = re.compile(r"(?<=[A-Za-z])-\n(?=[a-z])")


class PageOCR(Protocol):
    def needs_ocr(self, extracted_text: str) -> bool: ...

    def transcribe_pages(self, pages: dict[int, PageImage]) -> tuple[dict[int, str], list[str]]: ...


def _open_pdf(data: bytes):
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception as exc:
                raise ExtractionError("PDF is password protected") from exc
        return reader, list(reader.pages)
    except PdfReadError as exc:
        raise ExtractionError(f"Invalid or corrupted PDF: {exc}") from exc


def _page_text(page, number: int) -> str:
    try:
        text = page.extract_text() or ""
    except Exception:
        logger.warning("Could not extract text from PDF page %d", number, exc_info=True)
        text = ""
    return _HYPHENATED_BREAK.sub("", text)


def extract_pdf_pages(data: bytes) -> list[str]:
    """Text of each PDF page (empty string for pages with no text layer)."""
    _, pages = _open_pdf(data)
    return [_page_text(page, number) for number, page in enumerate(pages, start=1)]


def extract_pdf(data: bytes, source: str, ocr: PageOCR | None = None) -> ExtractedDocument:
    """Per-page PDF text; pages without a usable text layer go to OCR."""
    _, pages = _open_pdf(data)
    texts = {number: _page_text(page, number) for number, page in enumerate(pages, start=1)}
    warnings: list[str] = []
    ocr_pages: set[int] = set()

    if ocr is not None:
        targets = {number for number, text in texts.items() if ocr.needs_ocr(text)}
        if targets:
            payloads = {number: page_payload(pages[number - 1]) for number in sorted(targets)}
            logger.info("%s: OCR for %d of %d page(s)", source, len(payloads), len(pages))
            transcripts, ocr_warnings = ocr.transcribe_pages(payloads)
            warnings.extend(ocr_warnings)
            for number, transcript in transcripts.items():
                # OCR output is Markdown; keep the old text if OCR found nothing.
                cleaned = clean_markdown(transcript)
                if cleaned.strip():
                    texts[number] = cleaned
                    ocr_pages.add(number)

    sections = [
        Section(text, number, ocr=number in ocr_pages)
        for number, text in sorted(texts.items())
        if text.strip()
    ]
    empty = len(pages) - len(sections)
    if empty:
        hint = (
            "OCR returned no text for them"
            if ocr is not None
            else "scanned pages need OCR; set GEMINI_API_KEY or install kiri-ocr, with OCR_MODE=auto"
        )
        warnings.append(f"{empty} of {len(pages)} PDF page(s) had no text layer ({hint})")
    return ExtractedDocument(
        source, "pdf", sections, pages=len(pages), warnings=warnings, ocr_pages=len(ocr_pages)
    )


# --- Images -----------------------------------------------------------------

def extract_image(data: bytes, filename: str, ocr: PageOCR | None) -> ExtractedDocument:
    """A photo or scan of one page, transcribed by OCR as page 1."""
    if ocr is None:
        raise ExtractionError(
            "Image uploads need OCR; set GEMINI_API_KEY or install kiri-ocr, and OCR_MODE must not be 'never'"
        )
    payload = PageImage(data, IMAGE_MIME_TYPES[Path(filename).suffix.lower()])
    transcripts, warnings = ocr.transcribe_pages({1: payload})
    text = clean_markdown(transcripts.get(1, ""))
    sections = [Section(text, 1, ocr=True)] if text.strip() else []
    if not sections and not warnings:
        warnings.append("OCR found no text in the image")
    return ExtractedDocument(
        filename, "image", sections, pages=1, warnings=list(warnings), ocr_pages=len(sections)
    )


# --- Dispatch ---------------------------------------------------------------

def extract_document(data: bytes, filename: str, ocr: PageOCR | None = None) -> ExtractedDocument:
    """Extract text sections from raw file bytes, dispatching on extension.

    ``ocr`` is used for images and for PDF pages it reports as needing OCR.
    """
    file_format = detect_format(filename)
    if file_format == "pdf":
        return extract_pdf(data, filename, ocr)
    if file_format == "image":
        return extract_image(data, filename, ocr)

    text = extract_markdown(data) if file_format == "markdown" else extract_plain_text(data)
    sections = [Section(text)] if text.strip() else []
    return ExtractedDocument(filename, file_format, sections)


def extract_file(
    path: str | Path, source: str | None = None, ocr: PageOCR | None = None
) -> ExtractedDocument:
    path = Path(path)
    detect_format(path.name)
    document = extract_document(path.read_bytes(), path.name, ocr)
    document.source = source or path.name
    return document
