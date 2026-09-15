from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from reanmath.embeddings.embedder import embed_passages
from reanmath.ingestion.extract import extract_and_chunk
from reanmath.llm_client import call_claude
from reanmath.prompts import SYSTEM_PROMPT, build_augmented_prompt
from reanmath.retrieval.retriever import retrieve_context
from reanmath.schemas import ChatRequest, ChatResponse, IngestResponse
from reanmath.vectorstore.qdrant_store import ensure_collection, upsert_chunks

app = FastAPI(title="Reanmath")

STATIC_DIR = Path(__file__).resolve().parents[2] / "static"
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


@app.on_event("startup")
def on_startup() -> None:
    ensure_collection()


@app.get("/")
def root():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    contexts: list[str] = []
    if req.use_rag:
        contexts = retrieve_context(req.message, top_k=req.top_k)

    system = build_augmented_prompt(SYSTEM_PROMPT, contexts)
    reply = call_claude(system=system, message=req.message, history=req.history)
    return ChatResponse(reply=reply, sources=contexts)


@app.post("/api/ingest", response_model=IngestResponse)
async def ingest() -> IngestResponse:
    """Re-index every PDF/Markdown file currently in data/raw/."""
    files = [p for p in DATA_DIR.iterdir() if p.suffix.lower() in {".pdf", ".md", ".markdown", ".txt"}]

    total_chunks = 0
    for path in files:
        chunks = extract_and_chunk(path)
        if not chunks:
            continue
        vectors = embed_passages([c["text"] for c in chunks])
        total_chunks += upsert_chunks(chunks, vectors)

    return IngestResponse(files_processed=len(files), chunks_indexed=total_chunks)


@app.get("/api/health")
def health():
    return {"status": "ok"}
