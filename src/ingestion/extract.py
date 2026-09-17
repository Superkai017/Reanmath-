"""Turn plain text, Markdown and PDF files into clean text sections."""
from __future__ import annotations

import io
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from src.ingestion.khmer_segment import looks_garbled
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
    """A contiguous block of text; ``page`` is 1-based for PDFs.

    ``heading`` is the path of Markdown headings the text sits under
    (``Lesson › Section › Exercise``); ``starts_heading`` is set when the
    section begins with its own heading rather than continuing the previous
    section's text (for example on the next page).
    """

    text: str
    page: int | None = None
    ocr: bool = False
    heading: str = ""
    starts_heading: bool = False


@dataclass
class ExtractedDocument:
    source: str
    format: str
    sections: list[Section]
    title: str = ""
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


HEADING_SEPARATOR = " › "
_MAX_HEADING_CHARS = 120
_ATX_HEADING = re.compile(r"^\s{0,3}(#{1,6})\s+(.*?)\s*#*\s*$")
_FENCE_LINE = re.compile(r"^\s{0,3}(```|~~~)")
_HEADING_DECORATION = re.compile(r"^[\s*_✧◆◇♦•▪■□☐✦❖\-–—]+|[\s*_:៖]+$")


class HeadingTracker:
    """Markdown heading path that carries over from one page to the next."""

    def __init__(self) -> None:
        self._stack: list[tuple[int, str]] = []

    @property
    def path(self) -> str:
        return HEADING_SEPARATOR.join(title for _, title in self._stack)

    def push(self, level: int, title: str) -> None:
        while self._stack and self._stack[-1][0] >= level:
            self._stack.pop()
        self._stack.append((level, title))


def _heading_title(raw: str) -> str:
    title = clean_markdown(raw)
    title = _HEADING_DECORATION.sub("", title).strip()
    if len(title) > _MAX_HEADING_CHARS:
        title = title[:_MAX_HEADING_CHARS].rstrip() + "…"
    return title


def split_markdown_sections(
    markdown: str, tracker: HeadingTracker, page: int | None = None, ocr: bool = False
) -> list[Section]:
    """Split Markdown at its headings into cleaned sections.

    Each section keeps its heading line as text and records the heading path
    from ``tracker``, which is updated so later pages continue the path.
    """
    blocks: list[tuple[bool, list[str]]] = [(False, [])]
    in_fence = False
    for line in markdown.split("\n"):
        if _FENCE_LINE.match(line):
            in_fence = not in_fence
        match = None if in_fence else _ATX_HEADING.match(line)
        if match and _heading_title(match.group(2)):
            blocks.append((True, [line]))
        else:
            blocks[-1][1].append(line)

    sections: list[Section] = []
    for starts_heading, lines in blocks:
        if starts_heading:
            match = _ATX_HEADING.match(lines[0])
            tracker.push(len(match.group(1)), _heading_title(match.group(2)))
        text = clean_markdown("\n".join(lines))
        if text.strip():
            sections.append(Section(text, page, ocr, tracker.path, starts_heading))
    return sections


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


_EDGE_LINES = 2
_MIN_PAGES_FOR_RUNNING_LINES = 5


def _edge_lines(text: str) -> set[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return set(lines[:_EDGE_LINES] + lines[-_EDGE_LINES:])


def remove_running_lines(pages: dict[int, str]) -> dict[int, str]:
    """Drop running headers and footers: lines at the top or bottom of a page
    that repeat on at least half of the document's pages."""
    if len(pages) < _MIN_PAGES_FOR_RUNNING_LINES:
        return pages
    counts = Counter(line for text in pages.values() for line in _edge_lines(text))
    running = {line for line, count in counts.items() if count >= len(pages) / 2}
    if not running:
        return pages

    def strip(text: str) -> str:
        lines = text.split("\n")
        for _ in range(2):  # top, then bottom (reversed)
            removed = 0
            while lines and removed < _EDGE_LINES and (not lines[0].strip() or lines[0].strip() in running):
                removed += bool(lines[0].strip())
                lines.pop(0)
            lines.reverse()
        return "\n".join(lines)

    return {number: strip(text) for number, text in pages.items()}


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
    transcripts: dict[int, str] = {}

    if ocr is not None:
        targets = {number for number, text in texts.items() if ocr.needs_ocr(text)}
        if targets:
            payloads = {number: page_payload(pages[number - 1]) for number in sorted(targets)}
            logger.info("%s: OCR for %d of %d page(s)", source, len(payloads), len(pages))
            transcripts, ocr_warnings = ocr.transcribe_pages(payloads)
            warnings.extend(ocr_warnings)
            for number, transcript in transcripts.items():
                # OCR output is Markdown; keep the old text if OCR found nothing.
                if clean_markdown(transcript).strip():
                    ocr_pages.add(number)

    # A garbled text layer (legacy Khmer font encoding) that OCR did not
    # replace is unreadable; indexing it would only add noise.
    garbled = [n for n, text in texts.items() if n not in ocr_pages and looks_garbled(text)]
    for number in garbled:
        texts[number] = ""
    if garbled:
        warnings.append(
            f"{len(garbled)} page(s) with an unreadable Khmer text layer were skipped"
            + (" because OCR did not transcribe them" if ocr is not None else "; they need OCR")
        )

    page_texts = remove_running_lines(
        {number: transcripts[number] if number in ocr_pages else text for number, text in texts.items()}
    )
    sections: list[Section] = []
    tracker = HeadingTracker()
    for number, text in sorted(page_texts.items()):
        if number in ocr_pages:
            sections.extend(split_markdown_sections(text, tracker, number, ocr=True))
        elif text.strip():
            sections.append(Section(text, number, heading=tracker.path))
    empty = len(pages) - len({section.page for section in sections}) - len(garbled)
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
    sections = split_markdown_sections(transcripts.get(1, ""), HeadingTracker(), 1, ocr=True)
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

    if file_format == "markdown":
        sections = split_markdown_sections(decode_text_bytes(data), HeadingTracker())
    else:
        text = extract_plain_text(data)
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
