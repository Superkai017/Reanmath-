"""Khmer word segmentation, used for cleaning/validation of extracted text.

Note: this does NOT insert spaces into the stored chunk text (Khmer is
normally written without spaces between words) -- it's used to detect
broken sub-consonant clusters and to collapse stray whitespace safely.
"""
try:
    from khmernltk import word_tokenize as _khmer_tokenize
except ImportError:  # pragma: no cover - optional at import time
    _khmer_tokenize = None


def segment_khmer(text: str) -> list[str]:
    if _khmer_tokenize is None:
        raise RuntimeError(
            "khmer-nltk is not installed. Run: uv add khmer-nltk"
        )
    return _khmer_tokenize(text, return_tokens=True)


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace, but never touch content between
    LaTeX placeholder tokens (caller is expected to mask LaTeX first).
    """
    import re

    # Collapse 2+ spaces/tabs, but keep newlines (paragraph structure
    # matters for chunking).
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
