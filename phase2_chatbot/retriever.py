import chromadb
from sentence_transformers import SentenceTransformer

client = chromadb.PersistentClient(path="./data/chroma_db")
collection = client.get_or_create_collection(
    name="campus_x_transcripts",
    metadata={"hnsw:space": "cosine"}
)

embedder = SentenceTransformer("all-MiniLM-L6-v2")


def retrieve_relevant_chunks(query: str, top_k: int = 5) -> list[dict]:
    """Embed query and retrieve top-k relevant chunks from ChromaDB."""
    query_embedding = embedder.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    chunks = []
    for i in range(len(results["documents"][0])):
        chunks.append({
            "text": results["documents"][0][i],
            "title": results["metadatas"][0][i]["title"],
            "url": results["metadatas"][0][i]["url"],
            "score": results["distances"][0][i]
        })

    return chunks
