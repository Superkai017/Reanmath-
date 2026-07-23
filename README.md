# Reanmath-

Khmer-language math practice engine for Cambodian students preparing for
the Bac II examination.

**Status:** early development. Not usable yet.

## What this is

A verified question bank and practice agent for Bac II mathematics
(science stream). The engine powers [ExamKhmer](#), the student-facing app.

Two design principles carry the whole project:

1. **sympy is the source of truth, never the model.** Answers are checked
   symbolically, not judged by an LLM. Grading is arithmetic.
2. **The model is only a language layer.** It explains errors in Khmer
   against solutions that were already verified. It does not pick
   questions, grade, or track progress — that is deterministic code.

## Scope

| | |
|---|---|
| Exam | Bac II, science stream |
| Subject | Mathematics only |
| Years | 2021–2024 past papers |
| Target | ~500 verified items across 16 topics |
| Languages | Khmer stems and explanations, LaTeX math |

Physics, chemistry, and university entrance exams (ITC, CADT, UHS) are
out of scope for v1.

## Requirements

- Linux or WSL2 (the Khmer NLP packages are Linux-first)
- Python 3.11, managed by [uv](https://docs.astral.sh/uv/)
- API keys: [SEA-LION](https://sea-lion.ai) (free), Google AI Studio (free tier)

## Setup

```bash
git clone https://github.com/<you>/reanmath && cd reanmath
uv python install 3.11
uv sync
cp .env.example .env    # add your API keys
uv run pytest           # should pass before you do anything else
```

## Pipeline

```
data/raw/*.pdf
  → make manifest    register sources, sha256, page counts
  → make extract     page images → structured JSON (vision LLM)
  → make verify      sympy checks every answer it can
  → make review      human verifies the rest (streamlit)
  → data/processed/items.jsonl
  → make index       embeddings → chroma
  → make practice    the agent
```

Raw PDFs are gitignored. `data/processed/items.jsonl` and
`data/processed/figures/` are committed — they represent the verification
work and are the only irreplaceable part of the repo.

## Item format

```json
{
  "id": "01HQ...",
  "stem_km": "គណនាដេរីវេនៃអនុគមន៍...",
  "latex": "f(x) = x^3 - 3x^2 + 2",
  "answer_latex": "3x^2 - 6x",
  "answer_type": "exact",
  "solution_steps": ["...", "..."],
  "topic": "calculus.derivatives",
  "sympy_verified": true,
  "difficulty": 2,
  "source": {"file": "bacii_math_2024.pdf", "page": 3}
}
```

Any item with `sympy_verified: false` and `answer_type != "proof"` must be
reviewed by a human before it reaches a student. No exceptions.

## Khmer text handling

- All text is NFC-normalized; ZWSP and ZWNJ are stripped on ingest
- Language detection uses Unicode codepoint ranges, never `langdetect` —
  those libraries confuse Khmer with Thai and Lao
- Generated Khmer is validated for script purity and regenerated on
  failure (models drift into Thai, especially smaller ones)

## Data sources and licensing

Bac II papers are published by the Ministry of Education, Youth and Sport.
Provenance for every source file is recorded in `data/manifest.csv`.
Commercial prep books are not used as source material.

## Layout

```
configs/     taxonomy, settings
data/        raw (ignored), processed (committed)
src/reanmath/
  khmer/     normalization, script detection
  ingest/    manifest, page rendering
  extract/   vision LLM → JSON
  math/      sympy comparators and verification
  llm/       provider clients, Khmer output guard
  bank/      item store, embeddings, sampling
  agent/     the practice loop
  eval/      model comparison on held-out items
tests/
scripts/     manifest builder, review UI
```

## Contributing

Not open to contributions yet. If you are a Khmer-speaking maths teacher
or student and want to help verify items, please open an issue.

## Licence

Code: MIT. Question bank: see `data/LICENSE` — exam papers are public
documents; derived items and written solutions are original work.
