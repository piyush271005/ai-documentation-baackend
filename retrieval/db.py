import os
import logging
from pathlib import Path
import chromadb
from backend.config import settings

logger = logging.getLogger("db")

class VectorStore:
    def __init__(self):
        self.collection_name = "ai_docs"
        logger.info(f"Connecting to ChromaDB Cloud (tenant={settings.CHROMA_TENANT}, db={settings.CHROMA_DATABASE})")
        self.client = chromadb.CloudClient(
            tenant=settings.CHROMA_TENANT,
            database=settings.CHROMA_DATABASE,
            api_key=settings.CHROMA_API_KEY,
        )
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, chunks: list[dict], embeddings: list[list[float]]):
        if not chunks:
            logger.debug("[DEBUG] add_chunks called with empty chunks list.")
            return
            
        logger.debug(f"[DEBUG] Preparing to add {len(chunks)} chunks and {len(embeddings)} embeddings to ChromaDB Cloud...")
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
        logger.debug(f"[DEBUG] Total collection count after insertion: {self.collection.count()} chunks.")

    def search(self, query_embedding: list[float], limit: int = 10) -> list[dict]:
        """Queries the vector database for top matching chunks."""
        logger.debug(f"[DEBUG] Executing ChromaDB vector search (limit={limit}, query_vector_dim={len(query_embedding)})...")
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=limit
        )
        
        parsed_results = []
        if not results or not results["ids"] or len(results["ids"][0]) == 0:
            logger.debug("[DEBUG] ChromaDB search returned 0 matching results.")
            return parsed_results
            
        ids = results["ids"][0]
        distances = results["distances"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        
        for i in range(len(ids)):
            sim_score = 1.0 - float(distances[i])
            parsed_results.append({
                "id": ids[i],
                "content": documents[i],
                "url": metadatas[i]["url"],
                "title": metadatas[i]["title"],
                "parent_header": metadatas[i]["parent_header"],
                "similarity_score": sim_score
            })
            
        logger.debug(f"[DEBUG] Vector search returned {len(parsed_results)} matching chunks (top score={parsed_results[0]['similarity_score']:.4f})")
        return parsed_results

    def reset(self):
        logger.info("Resetting ChromaDB collection...")
        logger.debug(f"[DEBUG] Deleting collection '{self.collection_name}'...")
        try:
            self.client.delete_collection(self.collection_name)
        except Exception as e:
            logger.debug(f"[DEBUG] Collection delete exception (ignored): {e}")
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.debug("[DEBUG] Collection reset successfully.")

    def count(self) -> int:
        c = self.collection.count()
        logger.debug(f"[DEBUG] Current ChromaDB collection count: {c}")
        return c

# Global vector store instance
vector_store = VectorStore()
