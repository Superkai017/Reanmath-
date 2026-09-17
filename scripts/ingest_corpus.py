"""Index every supported file under the data directory.

Usage:
    uv run python scripts/ingest_corpus.py                 # index ./data
    uv run python scripts/ingest_corpus.py --data-dir notes --reset
    uv run python scripts/ingest_corpus.py --dry-run       # extract + chunk only
    OCR_ENGINE=kiri uv run python scripts/ingest_corpus.py --only 'lessons/*.pdf'

Each file is stored under its path relative to the data directory, so
re-running the script replaces a file's chunks instead of duplicating them.

An optional ``catalog.json`` in the data directory maps those relative paths
to readable titles ({"lesson2.pdf": "មេរៀនទី២ លីមីតនៃអនុគមន៍"}). The title is
embedded with every chunk and shown to the model, which helps when file
names say nothing about the content.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_settings  # noqa: E402
from src.embeddings.embedder import build_embedder  # noqa: E402
from src.ingestion.ocr import build_ocr  # noqa: E402
from src.ingestion import (  # noqa: E402
    SUPPORTED_EXTENSIONS,
    DuplicateSourceError,
    EmptyDocumentError,
    ExtractionError,
    KhmerSegmenter,
    LatexIntegrityError,
    extract_file,
    index_document,
    prepare_document,
)
from src.vectorstore import EmbeddingMismatchError, InMemoryVectorStore  # noqa: E402

logger = logging.getLogger("ingest_corpus")


def discover_files(data_dir: Path, extensions: set[str], only: list[str] | None = None) -> list[Path]:
    """Supported files under ``data_dir``, skipping hidden files and folders.

    ``only`` keeps files whose relative path matches one of the glob patterns.
    """
    files = []
    for path in sorted(data_dir.rglob("*")):
        relative = path.relative_to(data_dir)
        if any(part.startswith(".") for part in relative.parts):
            continue
        if only and not any(fnmatch.fnmatch(relative.as_posix(), pattern) for pattern in only):
            continue
        if path.is_file() and path.suffix.lower() in extensions:
            files.append(path)
    return files


def load_catalog(path: Path) -> dict[str, str]:
    """Titles by source path, or an empty mapping when there is no catalog."""
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must map file paths to titles")
    return {str(source): str(title).strip() for source, title in data.items()}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=settings.data_dir, help="directory to scan recursively")
    parser.add_argument("--reset", action="store_true", help="discard the existing index first")
    parser.add_argument("--skip-existing", action="store_true", help="leave already-indexed files untouched")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="extract and chunk only; do not embed or save (OCR still runs and is cached)",
    )
    parser.add_argument("--no-ocr", action="store_true", help="skip OCR of scanned PDF pages and images")
    parser.add_argument(
        "--catalog",
        type=Path,
        default=None,
        help="JSON file mapping file paths to titles (default: <data-dir>/catalog.json)",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=sorted(SUPPORTED_EXTENSIONS),
        help="file extensions to include (default: all supported)",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        metavar="PATTERN",
        help="index only files whose path (relative to the data directory) matches a glob pattern",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = get_settings()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else settings.log_level.upper(),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    data_dir = args.data_dir.expanduser().resolve()
    if not data_dir.is_dir():
        logger.error("Data directory %s does not exist", data_dir)
        return 2

    extensions = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in args.extensions}
    unknown = extensions - set(SUPPORTED_EXTENSIONS)
    if unknown:
        logger.error("Unsupported extension(s): %s", ", ".join(sorted(unknown)))
        return 2

    files = discover_files(data_dir, extensions, args.only)
    if not files:
        logger.warning("No %s files found in %s", "/".join(sorted(extensions)), data_dir)
        return 0
    logger.info("Found %d file(s) in %s", len(files), data_dir)

    try:
        catalog = load_catalog(args.catalog or data_dir / "catalog.json")
    except (OSError, ValueError) as exc:
        logger.error("Could not read the catalog: %s", exc)
        return 2
    if catalog:
        logger.info("Catalog has titles for %d file(s)", len(catalog))

    segmenter = KhmerSegmenter(settings.khmer_segmenter)
    ocr = None if args.no_ocr else build_ocr(settings)
    if ocr is not None:
        logger.info("OCR: %s %s (mode=%s, cache=%s)", ocr.engine, ocr.model, ocr.mode, ocr.cache_dir)

    if args.dry_run:
        total_chunks = total_formulas = 0
        failures = 0
        for path in files:
            source = path.relative_to(data_dir).as_posix()
            try:
                document = extract_file(path, source=source, ocr=ocr)
                document.title = catalog.get(source, "")
                prepared = prepare_document(
                    document,
                    chunk_size=settings.chunk_size,
                    chunk_overlap=settings.chunk_overlap,
                    segmenter=segmenter,
                )
            except (ExtractionError, LatexIntegrityError, OSError) as exc:
                failures += 1
                logger.error("%s: %s", source, exc)
                continue
            total_chunks += len(prepared.chunks)
            total_formulas += prepared.formulas
            logger.info(
                "%s: %d chunks, %d formulas, %d chars, %d OCR page(s)%s",
                source, len(prepared.chunks), prepared.formulas, prepared.characters,
                len(prepared.ocr_page_numbers),
                f" ({'; '.join(prepared.warnings)})" if prepared.warnings else "",
            )
        logger.info("Dry run: %d chunks, %d formulas, %d failure(s)", total_chunks, total_formulas, failures)
        return 1 if failures else 0

    embedder = build_embedder(settings)
    try:
        store = InMemoryVectorStore.load(
            settings.vector_store_path, embedder.name, allow_model_change=args.reset
        )
    except EmbeddingMismatchError as exc:
        logger.error("%s", exc)
        return 2
    if args.reset:
        store.clear()
        logger.info("Existing index cleared")

    started = time.perf_counter()
    indexed = skipped = 0
    failures: list[str] = []
    try:
        for number, path in enumerate(files, start=1):
            source = path.relative_to(data_dir).as_posix()
            prefix = f"[{number}/{len(files)}] {source}"
            if args.skip_existing and store.has_source(source):
                skipped += 1
                logger.info("%s: already indexed, skipped", prefix)
                continue
            try:
                document = extract_file(path, source=source, ocr=ocr)
                document.title = catalog.get(source, "")
                result = index_document(
                    document,
                    store=store,
                    embedder=embedder,
                    segmenter=segmenter,
                    chunk_size=settings.chunk_size,
                    chunk_overlap=settings.chunk_overlap,
                    replace=not args.skip_existing,
                    persist=False,
                )
            except DuplicateSourceError:
                skipped += 1
                logger.info("%s: already indexed, skipped", prefix)
                continue
            except (ExtractionError, EmptyDocumentError, LatexIntegrityError, OSError) as exc:
                failures.append(source)
                logger.error("%s: %s", prefix, exc)
                continue
            indexed += 1
            logger.info(
                "%s: %d chunks (%d replaced), %d formulas, %d OCR page(s)",
                prefix, result.chunks_added, result.chunks_removed, result.formulas, result.ocr_pages,
            )
            for warning in result.warnings:
                logger.warning("%s: %s", prefix, warning)
    except KeyboardInterrupt:
        logger.warning("Interrupted; saving what has been indexed so far")
    finally:
        saved_to = store.save()

    logger.info(
        "Done in %.1fs: %d indexed, %d skipped, %d failed. Index has %d chunks from %d document(s) -> %s",
        time.perf_counter() - started, indexed, skipped, len(failures),
        len(store), len(store.sources()), saved_to,
    )
    if failures:
        logger.error("Failed: %s", ", ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
