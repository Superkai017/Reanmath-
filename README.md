# Reanmath (រៀនគណិត)

A local retrieval-augmented generation (RAG) tutor for the Cambodian Grade 12
mathematics curriculum (Bac II, science stream). You ingest curriculum
documents (PDF, images, Markdown, plain text). Students ask questions in Khmer or
English, and answers are grounded in the retrieved passages, with LaTeX
rendered in the browser.

The two properties that matter most:

1. **LaTeX survives ingestion unchanged.** Every formula is replaced by a
   UUID token before any text processing and restored byte-for-byte
   afterwards. Chunk boundaries never fall inside a formula.
2. **Khmer is segmented, never mangled.** Khmer has no spaces between words.
   Word boundaries are found with khmer-nltk (or a regex fallback that
   splits into orthographic clusters). The chunker breaks text only at those
   boundaries, and every segmenter is checked to return exactly the input
   code points.

## Architecture

[![System flow: documents are extracted (with Gemini or Kiri OCR for scans), formulas are masked, Khmer is segmented, text is chunked and embedded into the vector store; a question is matched by the retriever, combined with the system prompt, answered by the LLM provider and shown in the bondus UI](system-flow-chart.png)](system-flow-chart.png)

*Click the diagram to open it at full size.*

<details>
<summary>Text version</summary>

```
upload / data/  ──► extract.py ──► latex_guard.mask_latex ──► khmer_segment
                    (PDF pages,      (formulas → ⟦MATH_uuid⟧)    (NFC, ZWSP word
                     images, MD,                                  boundaries)
                     TXT; scans →
                     Gemini or Kiri OCR)
                                                                     │
                     vectorstore (in-memory cosine index,     ◄── chunk.py
                     atomic index.npz on disk)                    (recursive splitter,
                              ▲                                    500 / 50 overlap)
                              │ embedder.py (multilingual-e5 or hashing)
                              │
question ──► retriever.py (top-k, threshold, formula restoration)
                              │
                              ▼
             prompts.py (system prompt + <context> passages)
                              │
                              ▼
             Claude (Anthropic) · Gemini · retrieval-only fallback
                              │
                              ▼
             Frontend/ UI "bondus" (Markdown + KaTeX)
```

</details>

| Path | Role |
|---|---|
| `src/config.py` | `pydantic-settings` configuration from `.env` |
| `src/ingestion/latex_guard.py` | `mask_latex` / `unmask_latex` for `$…$`, `$$…$$`, `\(…\)`, `\[…\]`, math environments |
| `src/ingestion/khmer_segment.py` | normalisation, CRF/regex word segmentation, Khmer/English detection |
| `src/ingestion/extract.py` | PDF (per page), image, Markdown (syntax stripped, formulas kept) and text extraction |
| `src/ingestion/ocr.py` | OCR engine selection, shared page cache, and Gemini vision OCR (Khmer + LaTeX) with retries |
| `src/ingestion/khmer_ocr.py` | Kiri OCR: local, open-source Khmer OCR that keeps Khmer words only |
| `src/ingestion/chunk.py` | `RecursiveCharacterTextSplitter` aware of placeholders and Khmer boundaries |
| `src/ingestion/__init__.py` | `prepare_document` / `index_document` pipeline with a formula-integrity check |
| `src/embeddings/embedder.py` | sentence-transformers embedder (E5 prefixes handled) and offline hashing embedder |
| `src/vectorstore/__init__.py` | in-memory cosine index with persistence and embedding-model checks |
| `src/retrieval/retriever.py` | retrieval, context budgeting and retrieval metrics (hit rate, MRR, P@k, R@k) |
| `src/api.py` | FastAPI app: `/api/query`, `/api/ingest`, `/api/documents`, `/health`, and the web UI |
| `Frontend/` | The web UI (`index.html`, `static/app.js`, `static/style.css`); plain HTML/JS, no build step |
| `schemas.py` / `prompts.py` | API models and the tutor system prompt |
| `scripts/ingest_corpus.py` | bulk indexing of `./data` |
| `testing.py`, `src/tests/` | pytest suites |

## Requirements

- Python 3.11 managed by [uv](https://docs.astral.sh/uv/)
- Linux, macOS or WSL2. PyTorch is installed CPU-only from the PyTorch index.
- Optional: an `ANTHROPIC_API_KEY` or `GEMINI_API_KEY`. Without one, `/api/query`
  returns the best-matching passages instead of a generated answer.

## Setup

```bash
uv sync
cp .env.example .env        # then add an API key
uv run pytest               # offline; uses the hashing embedder
```

The first ingestion or query with the default backend downloads
`intfloat/multilingual-e5-small` (about 470 MB) from Hugging Face. The first
Khmer segmentation loads the khmer-nltk CRF model, which takes a few seconds.

## Usage

Index everything under `./data` (recursively):

```bash
uv run python scripts/ingest_corpus.py            # add or replace documents
uv run python scripts/ingest_corpus.py --dry-run  # extraction and chunking stats only
uv run python scripts/ingest_corpus.py --reset    # rebuild from scratch
```

Start the server and open <http://localhost:8000>:

```bash
uv run uvicorn src.api:app --reload
```

Or with Docker:

```bash
docker compose up reanmath
docker compose run --rm ingest     # index ./data inside the container
```

The UI in `Frontend/` is served by the same process, so there is nothing to
build and no Node.js is needed. It offers:

- a chat with Markdown and KaTeX rendering, collapsible sources (document,
  page and score), copy, regenerate and stop;
- `+` in the composer (or drag and drop) to add a PDF, image, Markdown or text
  file to the knowledge base. Your next question then searches only the files
  attached to it; remove the chip to search everything;
- **Documents** in the sidebar to list and delete indexed documents;
- recent conversations, saved in the browser's `localStorage`.

### Graphs (GeoGebra)

When a picture helps (a function's graph, a circle or other conic, a tangent,
vectors, a 3D surface), or the student asks for one ("គូសក្រាហ្វ…"), the tutor
adds a fenced block of GeoGebra commands:

````markdown
```geogebra
c: (x - 1)^2 + (y + 2)^2 = 9
f(x) = x^2 - 2x
ZoomIn(-4, -6, 6, 5)
```
````

The UI draws each `geogebra` (2D) or `geogebra-3d` block as an interactive
graph with the [GeoGebra Apps API](https://geogebra.github.io/docs/reference/en/GeoGebra_Apps_API/),
which loads from geogebra.org only when a graph first scrolls into view. The
graph can be panned and zoomed, and **PNG** downloads it.

- The final `ZoomIn(...)` sets the view. When the x and y ranges are similar,
  one of them is widened so both axes share a scale and circles stay round.
- Script commands (`Execute`, `SetClickScript`, `RunClickScript`, …) are
  never run. Commands that fail are counted under the graph; hover the count
  to see them.
- If GeoGebra cannot load (for example offline), the commands are shown as
  text instead.
- Graphs need an LLM provider. In passage-only mode there are none.
- GeoGebra is free for non-commercial use; check
  [its license](https://www.geogebra.org/license) before commercial use.

### API

Interactive docs are served at `/docs`.

```bash
# Ask a question
curl -s localhost:8000/api/query -H 'Content-Type: application/json' \
  -d '{"prompt": "គណនា $\\lim_{x \\to 0} \\frac{\\sin x}{x}$", "top_k": 5, "score_threshold": 0.75}'

# Upload a file
curl -s localhost:8000/api/ingest -F file=@data/limits.pdf

# Ingest pasted Markdown
curl -s localhost:8000/api/ingest/text -H 'Content-Type: application/json' \
  -d '{"text": "ដេរីវេនៃ $x^n$ គឺ $nx^{n-1}$ ។", "source_name": "derivatives.md"}'

curl -s localhost:8000/api/documents
curl -s -X DELETE localhost:8000/api/documents/derivatives.md
curl -s localhost:8000/health
```

`/api/query` accepts `prompt`, `top_k`, `score_threshold`, `history` (previous
`user`/`assistant` turns), `sources` (restrict retrieval to specific documents)
and `generate` (`false` returns retrieval results only). The response contains
the answer, the detected language, whether it was grounded, and the passages
used, with formulas restored.

## Configuration

All settings live in `.env`; see `.env.example` for the full list. The most
important ones:

| Variable | Default | Notes |
|---|---|---|
| `LLM_PROVIDER` | `auto` | `anthropic`, `gemini`, `none`, or `auto` (the first provider with a key) |
| `ANTHROPIC_MODEL` | `claude-opus-5` | Sent with adaptive thinking; `ANTHROPIC_EFFORT` sets the effort level |
| `ANTHROPIC_FALLBACKS` | `true` | Server-side fallback if the main model declines a request |
| `GEMINI_MODEL` | `gemini-3.8-flash` | |
| `EMBEDDING_BACKEND` | `sentence-transformers` | `hashing` works offline with lexical matching only |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-small` | Changing it requires `--reset` |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `500` / `50` | Measured in displayed characters |
| `TOP_K` / `SCORE_THRESHOLD` | `5` / `0.75` | Use a threshold around `0.2` with the hashing backend |
| `KHMER_SEGMENTER` | `auto` | `crf` (khmer-nltk), `regex`, or `auto` |
| `OCR_MODE` / `OCR_ENGINE` | `auto` / `auto` | OCR of scanned PDF pages and images; see below |
| `OCR_MODEL` | `gemini-3.5-flash` | Gemini OCR model |
| `KIRI_MIN_CONFIDENCE` / `KIRI_KHMER_ONLY` | `0.2` / `true` | Kiri OCR filtering |

## Scanned PDFs and images (OCR)

Pages with no usable text layer, and uploaded images (`.png`, `.jpg`, `.jpeg`,
`.webp`), are read by OCR. `OCR_ENGINE` picks the engine:

| Engine | What it produces | Needs |
|---|---|---|
| `gemini` | Khmer as Unicode **and** formulas as LaTeX | `GEMINI_API_KEY` |
| `kiri` | **Khmer words only**; no formulas | nothing (runs locally on the CPU) |
| `auto` (default) | `gemini` when `GEMINI_API_KEY` is set, otherwise `kiri` | |

### Kiri OCR (Khmer words only)

[Kiri OCR](https://github.com/mrrtmob/kiri-ocr) is an open-source Khmer/English
OCR model. Set `OCR_ENGINE=kiri` to use it even when a Gemini key is present.
The model (`KIRI_MODEL`) downloads from Hugging Face on first use.

- Each page is split into text lines, and each line is recognised. Lines with
  a confidence below `KIRI_MIN_CONFIDENCE` are dropped. The published model
  scores even clean lines around 0.4–0.45, so the default is 0.2.
- With `KIRI_KHMER_ONLY=true` (default) each line is reduced to its Khmer
  words: Khmer letters, Khmer digits (០–៩) and Khmer punctuation (។ ៕). An
  ASCII `:` inside a word is read back as `ៈ`, which Kiri often misreads. Latin
  letters, Arabic digits and math symbols are removed, because Kiri cannot
  read formulas and would index them as noise. Leftover dependent vowels at
  the start of a word are removed too. The result is then word-segmented and
  chunked like any other Khmer text.
- PDF pages that are a single scanned image are passed to Kiri as-is; other
  pages are rendered with pypdfium2 at `KIRI_RENDER_SCALE`.
- Pages are processed one at a time. Expect a few seconds per page on a CPU
  (`KIRI_DEVICE=cuda` if you have a GPU; `KIRI_DECODE_METHOD=fast` is quicker).

Use Kiri for Khmer prose (lessons, explanations, exercise text). Use Gemini
when the formulas on scanned pages matter.

### Gemini OCR

Gemini vision (`OCR_MODEL`, default `gemini-3.5-flash`) transcribes Khmer to
Unicode and writes formulas as LaTeX.

### Both engines

- Each page transcript is cached in `storage/ocr_cache/<hash>.md`, so
  re-ingesting costs nothing. Review these files to check OCR quality. Kiri
  and Gemini transcripts are cached separately, and changing a Kiri setting
  creates new cache entries.
- `OCR_MODE=always` also OCRs pages that have text, which helps with older PDFs
  whose legacy Khmer fonts extract as garbage.
- Busy Gemini (429/503) responses are retried with backoff. A page that still
  fails is skipped with a warning; the rest of the document is indexed.
- Uploading a scanned PDF in the browser OCRs it during the request, which can
  take several minutes. For large files, prefer
  `uv run python scripts/ingest_corpus.py` (use `--no-ocr` to skip OCR).
- Chunks from OCR'd pages carry `"ocr": true` in their metadata.

## Limitations

- OCR output can contain transcription mistakes, especially in dense formulas.
  Kiri OCR drops formulas entirely by design.
  The tutor prompt tells the model to trust correct mathematics over a passage
  that looks wrong.
- The index is held in memory. It suits a curriculum-sized corpus of tens of
  thousands of chunks, not millions.

## Data and licensing

Source documents live in `data/`, which is git-ignored. Bac II papers are
published by the Ministry of Education, Youth and Sport. Commercial prep books
must not be used as source material.

Code: MIT (see `LICENSE`).
