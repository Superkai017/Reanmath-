"""Khmer text normalisation, word segmentation and language detection.

Khmer is written without spaces between words. For chunking we mark word
boundaries with ZERO WIDTH SPACE (U+200B), which the splitter can break on,
and strip the markers again before text is stored or shown.

Two segmentation backends are available:

* ``crf``   - khmer-nltk's CRF word tokenizer (dictionary-quality words).
* ``regex`` - orthographic cluster (syllable) segmentation, no dependencies.

Segmentation only ever operates on runs of Khmer script. LaTeX placeholder
tokens and Latin text pass through untouched, and every backend is checked to
return tokens that concatenate back to the exact input, so the text can never
be corrupted at the code point level.
"""
from __future__ import annotations

import logging
import re
import threading
import unicodedata
from typing import Literal

from src.ingestion.latex_guard import PLACEHOLDER_PATTERN, mask_latex

logger = logging.getLogger(__name__)

ZWSP = "\u200b"
ZWNJ = "\u200c"
ZWJ = "\u200d"

KHMER_RANGES = ((0x1780, 0x17FF), (0x19E0, 0x19FF))
KHMER_SENTENCE_END = ("\u17d4", "\u17d5")  # ។ ៕

# One orthographic cluster: a base (consonant or independent vowel) followed by
# subscript consonants (COENG + consonant), dependent vowels and diacritics.
_KHMER_CLUSTER = (
    r"[\u1780-\u17B3]"
    r"(?:\u17D2[\u1780-\u17B3]|[\u17B4-\u17D1\u17D3\u17DD]|[\u200C\u200D](?=[\u1780-\u17DD]))*"
)
_KHMER_DIGITS = r"[\u17E0-\u17E9\u17F0-\u17F9]+"
_KHMER_UNIT = re.compile(
    rf"{_KHMER_CLUSTER}|{_KHMER_DIGITS}|[\u17D4-\u17DC\u19E0-\u19FF]|.",
    re.DOTALL,
)
_KHMER_RUN = re.compile(r"[\u1780-\u17FF\u19E0-\u19FF\u200C\u200D]+")
_SEGMENT_SPLIT = re.compile(
    rf"({PLACEHOLDER_PATTERN.pattern})|({_KHMER_RUN.pattern})"
)
_REMOVED_CHARS = dict.fromkeys(map(ord, [ZWSP, "\ufeff", "\u00ad", "\u2060"]))
_HORIZONTAL_SPACE = re.compile(r"[^\S\n]+")
_SPACE_AROUND_NEWLINE = re.compile(r" *\n *")
_MANY_NEWLINES = re.compile(r"\n{3,}")
_WORD = re.compile(r"\\[A-Za-z]+|[A-Za-z]+|\d+(?:\.\d+)?")
_LATIN_LETTER = re.compile(r"[A-Za-z]")

SegmenterBackend = Literal["auto", "crf", "regex"]
Language = Literal["km", "en"]


def is_khmer_char(char: str) -> bool:
    code = ord(char)
    return any(start <= code <= end for start, end in KHMER_RANGES)


def contains_khmer(text: str) -> bool:
    return any(is_khmer_char(char) for char in text)


def normalize_khmer_text(text: str) -> str:
    """NFC-normalise, drop invisible characters and tidy whitespace.

    Newlines are kept (paragraph structure guides chunking); runs of spaces
    and tabs collapse to a single space. ZWNJ is removed only when it is not
    between two Khmer characters, where it can affect rendering.
    """
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.translate(_REMOVED_CHARS)
    text = re.sub(
        rf"(?<![\u1780-\u17FF]){ZWNJ}|{ZWNJ}(?![\u1780-\u17FF])", "", text
    )
    text = _HORIZONTAL_SPACE.sub(" ", text)
    text = _SPACE_AROUND_NEWLINE.sub("\n", text)
    text = _MANY_NEWLINES.sub("\n\n", text)
    return text.strip()


def regex_segment(run: str) -> list[str]:
    """Split Khmer text into orthographic clusters (syllable-like units)."""
    units: list[str] = []
    for match in _KHMER_UNIT.finditer(run):
        unit = match.group(0)
        # A stray combining mark belongs to the preceding cluster.
        if units and unicodedata.category(unit[0]) in ("Mn", "Mc") and not units[-1].isspace():
            units[-1] += unit
        else:
            units.append(unit)
    return units


class KhmerSegmenter:
    """Thread-safe Khmer word segmenter with lazy CRF model loading."""

    def __init__(self, backend: SegmenterBackend = "auto") -> None:
        self.backend = backend
        self._crf = None
        self._crf_failed = False
        self._lock = threading.Lock()

    @property
    def active_backend(self) -> Literal["crf", "regex"]:
        if self.backend == "regex":
            return "regex"
        return "crf" if self._load_crf() is not None else "regex"

    def _load_crf(self):
        if self._crf is not None or self._crf_failed:
            return self._crf
        with self._lock:
            if self._crf is None and not self._crf_failed:
                try:
                    from khmernltk import word_tokenize
                except ImportError as exc:
                    self._crf_failed = True
                    if self.backend == "crf":
                        raise RuntimeError(
                            "KHMER_SEGMENTER=crf requires khmer-nltk (uv add khmer-nltk)"
                        ) from exc
                    logger.warning("khmer-nltk not installed; using regex cluster segmentation")
                else:
                    self._crf = word_tokenize
        return self._crf

    def _segment_run(self, run: str) -> list[str]:
        if self.backend != "regex":
            tokenize = self._load_crf()
            if tokenize is not None:
                try:
                    tokens = [token for token in tokenize(run, return_tokens=True) if token]
                except Exception:  # the CRF model can fail on unusual input
                    logger.debug("CRF segmentation failed; falling back to regex", exc_info=True)
                else:
                    if "".join(tokens) == run:
                        return tokens
                    logger.debug("CRF tokens did not reconstruct the input; using regex")
        return regex_segment(run)

    def segment_pieces(self, text: str) -> list[str]:
        """Split ``text`` into pieces whose concatenation equals ``text``.

        Khmer runs are split into words; placeholders and all other text are
        returned as single pieces.
        """
        pieces: list[str] = []
        position = 0
        for match in _SEGMENT_SPLIT.finditer(text):
            if match.start() > position:
                pieces.append(text[position:match.start()])
            if match.group(1):
                pieces.append(match.group(1))
            else:
                pieces.extend(self._segment_run(match.group(2)))
            position = match.end()
        if position < len(text):
            pieces.append(text[position:])
        return pieces

    def segment(self, text: str) -> list[str]:
        """Word tokens of ``text``. Whitespace is dropped; non-Khmer text is
        split on whitespace; LaTeX placeholders stay whole."""
        tokens: list[str] = []
        for piece in self.segment_pieces(text):
            if PLACEHOLDER_PATTERN.fullmatch(piece) or contains_khmer(piece):
                if not piece.isspace():
                    tokens.append(piece)
            else:
                tokens.extend(piece.split())
        return tokens

    def insert_word_boundaries(self, text: str, marker: str = ZWSP) -> str:
        """Put ``marker`` between adjacent Khmer words.

        No marker is added next to whitespace or at the edges of the text.
        """
        pieces = self.segment_pieces(text)
        output: list[str] = []
        for index, piece in enumerate(pieces):
            output.append(piece)
            if index + 1 >= len(pieces):
                continue
            following = pieces[index + 1]
            if (
                contains_khmer(piece[-1])
                and contains_khmer(following[0])
                and not piece[-1].isspace()
            ):
                output.append(marker)
        return "".join(output)


def strip_word_boundaries(text: str, marker: str = ZWSP) -> str:
    return text.replace(marker, "")


_default_segmenters: dict[str, KhmerSegmenter] = {}
_default_lock = threading.Lock()


def get_segmenter(backend: SegmenterBackend = "auto") -> KhmerSegmenter:
    """Process-wide shared segmenter per backend (the CRF model loads once)."""
    with _default_lock:
        if backend not in _default_segmenters:
            _default_segmenters[backend] = KhmerSegmenter(backend)
        return _default_segmenters[backend]


def segment_khmer(text: str, backend: SegmenterBackend = "auto") -> list[str]:
    return get_segmenter(backend).segment(text)


def tokenize_for_search(text: str, segmenter: KhmerSegmenter | None = None) -> list[str]:
    """Lower-cased lexical tokens: Khmer words, Latin words, numbers and LaTeX
    command names (``\\int``, ``\\lim``...). Used by the hashing embedder."""
    segmenter = segmenter or get_segmenter("regex")
    masked, vault = mask_latex(normalize_khmer_text(text))
    tokens: list[str] = []
    for piece in segmenter.segment_pieces(masked):
        if piece in vault:
            tokens.extend(match.group(0).lower() for match in _WORD.finditer(vault[piece]))
        elif contains_khmer(piece):
            if not piece.isspace() and not all(ch in "\u17d4\u17d5\u17d6" for ch in piece):
                tokens.append(piece)
        else:
            tokens.extend(match.group(0).lower() for match in _WORD.finditer(piece))
    return tokens


def khmer_ratio(text: str) -> float:
    """Share of Khmer letters among Khmer + Latin letters, ignoring LaTeX."""
    masked, _ = mask_latex(text)
    masked = PLACEHOLDER_PATTERN.sub(" ", masked)
    khmer = sum(1 for char in masked if is_khmer_char(char) and char.isalpha())
    latin = len(_LATIN_LETTER.findall(masked))
    total = khmer + latin
    return khmer / total if total else 0.0


_ORPHAN_KHMER_MARK = re.compile(r"(?:^|[^\u1780-\u17FF\u200C\u200D])[\u17B6-\u17D3\u17DD]")
_BROKEN_GLYPH = re.compile(r"[\uE000-\uF8FF\uFFFD]")


def looks_garbled(text: str, threshold: float = 0.02, min_khmer: int = 20) -> bool:
    """True for a PDF text layer whose Khmer came from a legacy font encoding.

    Such layers store glyphs in visual order (``េមេរៀន`` for ``មេរៀន``) and map
    some glyphs to private-use or replacement characters. Correct Unicode never
    starts a word with a dependent vowel, sign or COENG, so the share of those
    orphaned marks, plus broken glyphs, separates the two reliably.
    """
    khmer = sum(1 for char in text if is_khmer_char(char))
    if khmer < min_khmer:
        return False
    suspicious = len(_ORPHAN_KHMER_MARK.findall(text)) + len(_BROKEN_GLYPH.findall(text))
    return suspicious / khmer > threshold


def detect_language(text: str, default: Language = "km") -> Language:
    """Detect Khmer vs English from Unicode code points.

    Libraries such as langdetect confuse Khmer with Thai and Lao, so the check
    is done on script ranges only. Text with no letters returns ``default``.
    """
    masked, _ = mask_latex(text)
    masked = PLACEHOLDER_PATTERN.sub(" ", masked)
    has_khmer = any(is_khmer_char(char) and char.isalpha() for char in masked)
    has_latin = _LATIN_LETTER.search(masked) is not None
    if not has_khmer and not has_latin:
        return default
    return "km" if khmer_ratio(text) >= 0.3 else "en"
