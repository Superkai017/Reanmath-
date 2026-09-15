"""Loads settings from environment / .env."""
import os

from dotenv import load_dotenv

load_dotenv()

# --- Claude ---
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")

# --- RAG: vector store ---
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "reanmath_grade12")

# --- RAG: embeddings ---
EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-large")
EMBED_DIM = int(os.getenv("EMBED_DIM", "1024"))

# --- RAG: retrieval ---
TOP_K = int(os.getenv("TOP_K", "5"))

# --- Chunking ---
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

if not ANTHROPIC_API_KEY:
    # Don't crash import (useful for tests), but warn loudly.
    print("[reanmath.config] WARNING: ANTHROPIC_API_KEY is not set.")
