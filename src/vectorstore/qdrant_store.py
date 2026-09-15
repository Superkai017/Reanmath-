import uuid
from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from reanmath.config import COLLECTION_NAME, EMBED_DIM, QDRANT_URL


@lru_cache(maxsize=1)
def get_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL)


def ensure_collection(dim: int = EMBED_DIM) -> None:
    client = get_client()
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )


def upsert_chunks(chunks: list[dict], vectors: list[list[float]]) -> int:
    client = get_client()
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={**chunk["metadata"], "text": chunk["text"]},
        )
        for chunk, vector in zip(chunks, vectors)
    ]
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    return len(points)


def search(vector: list[float], top_k: int = 5):
    client = get_client()
    return client.search(
        collection_name=COLLECTION_NAME,
        query_vector=vector,
        limit=top_k,
    )


def collection_stats() -> dict:
    client = get_client()
    if not client.collection_exists(COLLECTION_NAME):
        return {"exists": False, "points": 0}
    info = client.get_collection(COLLECTION_NAME)
    return {"exists": True, "points": info.points_count}
