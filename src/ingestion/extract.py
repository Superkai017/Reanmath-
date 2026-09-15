"""Parses raw PDF/Markdown curriculum files into clean, chunked text."""
from pathlib import Path

from pypdf import PdfReader

from reanmath.ingestion.chunk import chunk_document
from reanmath.ingestion.khmer_segment import normalize_whitespace
from reanmath.ingestion.latex_guard import mask_latex, unmask_latex


def load_raw_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    if path.suffix.lower() in {".md", ".markdown", ".txt"}:
        return path.read_text(encoding="utf-8")
    raise ValueError(f"Unsupported source file type: {path.suffix}")


def extract_and_chunk(path: str | Path, grade: int = 12) -> list[dict]:
    path = Path(path)
    raw = load_raw_text(path)

    # 1. Shield LaTeX before any cleaning touches the text.
    masked, vault = mask_latex(raw)

    # 2. Clean/normalize (safe now -- LaTeX spans are opaque tokens).
    cleaned = normalize_whitespace(masked)

    # 3. Restore LaTeX before chunking so chunk text is final/readable.
    restored = unmask_latex(cleaned, vault)

    # 4. Fixed-size chunking with overlap.
    metadata = {"source": path.name, "grade": grade}
    return chunk_document(restored, metadata)
