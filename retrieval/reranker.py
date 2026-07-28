import logging
from backend.retrieval.db import vector_store

logger = logging.getLogger("retriever")

class VectorRetriever:
    """
    Pure Vector Search Retriever relying entirely on ChromaDB Cloud.
    Eliminates all in-memory BM25 indexing and document caching.
    """
    def initialize_from_db(self):
        """No-op: Vector store is managed externally in ChromaDB Cloud."""
        pass

    def update_index(self, new_chunks: list[dict]):
        """No-op: Chunks are added directly to ChromaDB."""
        pass

    def retrieve_hybrid(self, query: str, query_embedding: list[float], top_n: int = 5) -> list[dict]:
        """Retrieves top-N matching chunks using vector similarity search."""
        vector_results = vector_store.search(query_embedding, limit=top_n)
        for doc in vector_results:
            doc["bm25_score"] = 0.0
            doc["combined_score"] = doc.get("similarity_score", 0.0)
        return vector_results

    def reset(self):
        """No-op: Reset handled by ChromaDB."""
        pass

# Maintain singleton instance alias for compatibility
hybrid_retriever = VectorRetriever()

