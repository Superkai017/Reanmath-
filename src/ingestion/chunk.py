from langchain_text_splitters import RecursiveCharacterTextSplitter

from reanmath.config import CHUNK_OVERLAP, CHUNK_SIZE

# "។" and "៕" are Khmer sentence/section-ending marks; prefer breaking
# there before falling back to generic whitespace.
_SEPARATORS = ["\n\n", "\n", "\u17d4", "\u17d5", " ", ""]


def get_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=_SEPARATORS,
    )


def chunk_document(text: str, metadata: dict) -> list[dict]:
    splitter = get_splitter()
    pieces = splitter.split_text(text)
    return [{"text": piece, "metadata": metadata} for piece in pieces]
