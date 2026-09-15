"""One-off / cron CLI: (re)builds the Qdrant index from data/raw/.

Usage:
    uv run python scripts/ingest_corpus.py
"""
from pathlib import Path

from reanmath.embeddings.embedder import embed_passages
from reanmath.ingestion.extract import extract_and_chunk
from reanmath.vectorstore.qdrant_store import collection_stats, ensure_collection, upsert_chunks

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def main() -> None:
    ensure_collection()

    files = [
        p for p in RAW_DIR.iterdir()
        if p.suffix.lower() in {".pdf", ".md", ".markdown", ".txt"}
    ]
    if not files:
        print(f"No source files found in {RAW_DIR}. Drop PDFs/Markdown there first.")
        return

    total = 0
    for path in files:
        print(f"Processing {path.name} ...")
        chunks = extract_and_chunk(path)
        if not chunks:
            print(f"  no text extracted, skipping")
            continue
        vectors = embed_passages([c["text"] for c in chunks])
        n = upsert_chunks(chunks, vectors)
        total += n
        print(f"  indexed {n} chunks")

    print(f"\nDone. {len(files)} file(s) processed, {total} chunks indexed.")
    print(collection_stats())


if __name__ == "__main__":
    main()
