import os
import logging
from pathlib import Path
import chromadb
from backend.config import settings

logger = logging.getLogger("db")

class VectorStore:
    def __init__(self):
        self.db_dir = settings.CHROMA_DB_DIR
        # Ensure the directory exists
        Path(self.db_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initializing ChromaDB persistent client at: {self.db_dir}")
        self.client = chromadb.PersistentClient(path=self.db_dir)
        self.collection_name = "ai_docs"
        
        # Initialize collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, chunks: list[dict], embeddings: list[list[float]]):
        """
        Inserts document chunks and their vector embeddings into ChromaDB.
        Each chunk is: {"id", "url", "title", "parent_header", "content"}
        """
        if not chunks:
            return
            
        ids = [chunk["id"] for chunk in chunks]
        documents = [chunk["content"] for chunk in chunks]
        metadatas = [
            {
                "url": chunk["url"],
                "title": chunk["title"],
                "parent_header": chunk["parent_header"] or ""
            }
            for chunk in chunks
        ]
        
        logger.info(f"Adding {len(chunks)} chunks to ChromaDB collection...")
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        logger.info("ChromaDB update complete.")

    def search(self, query_embedding: list[float], limit: int = 10) -> list[dict]:
        """Queries the vector database for top matching chunks."""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=limit
        )
        
        # Parse query results to standard list of dicts
        parsed_results = []
        if not results or not results["ids"] or len(results["ids"][0]) == 0:
            return parsed_results
            
        ids = results["ids"][0]
        distances = results["distances"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        
        for i in range(len(ids)):
            # Convert cosine distance to a similarity score (cosine distance is usually 1 - cosine_similarity, so similarity = 1 - distance)
            sim_score = 1.0 - float(distances[i])
            parsed_results.append({
                "id": ids[i],
                "content": documents[i],
                "url": metadatas[i]["url"],
                "title": metadatas[i]["title"],
                "parent_header": metadatas[i]["parent_header"],
                "similarity_score": sim_score
            })
            
        return parsed_results

    def reset(self):
        """Clears the collection database completely."""
        logger.info("Resetting ChromaDB collection...")
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def count(self) -> int:
        """Returns the number of elements inside the database."""
        return self.collection.count()

# Global vector store instance
vector_store = VectorStore()
