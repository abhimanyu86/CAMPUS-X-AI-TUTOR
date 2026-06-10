import os
# Model is cached locally; skip HuggingFace network checks that can hang the run.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import hashlib

client = chromadb.PersistentClient(path="./data/chroma_db")
collection = client.get_or_create_collection(
    name="campus_x_transcripts",
    metadata={"hnsw:space": "cosine"}
)

embedder = SentenceTransformer("all-MiniLM-L6-v2")


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks by word count."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def store_transcript(transcript_data: dict):
    """Chunk and store a single transcript into ChromaDB."""
    text = transcript_data["transcript"]
    title = transcript_data["title"]
    video_id = transcript_data["video_id"]
    url = transcript_data["url"]
    language = transcript_data.get("language", "unknown")

    chunks = chunk_text(text)
    print(f"Storing '{title}' — {len(chunks)} chunks")

    for i, chunk in enumerate(chunks):
        chunk_id = hashlib.md5(f"{video_id}_{i}".encode()).hexdigest()
        embedding = embedder.encode(chunk).tolist()

        collection.add(
            ids=[chunk_id],
            embeddings=[embedding],
            documents=[chunk],
            metadatas=[{
                "title": title,
                "video_id": video_id,
                "url": url,
                "language": language,
                "chunk_index": i
            }]
        )


def get_stored_video_ids() -> set:
    """Return set of video_ids already stored in ChromaDB (for resume/dedup)."""
    try:
        result = collection.get(include=["metadatas"])
        return {m["video_id"] for m in result["metadatas"] if "video_id" in m}
    except Exception as e:
        print(f"Could not read stored video_ids: {e}")
        return set()


def store_all_transcripts(transcripts: list[dict]):
    """Store all transcripts into ChromaDB."""
    for transcript in transcripts:
        store_transcript(transcript)
    print(f"\nDone! Stored {len(transcripts)} transcripts into ChromaDB.")
