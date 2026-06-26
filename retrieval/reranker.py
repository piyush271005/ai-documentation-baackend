import re
import logging
from rank_bm25 import BM25Okapi
from backend.retrieval.db import vector_store

logger = logging.getLogger("retriever")

def tokenize(text: str) -> list[str]:
    """Tokenize text using lowercase regex matching to avoid nltk dependencies."""
    return re.findall(r"\w+", text.lower())

class HybridRetriever:
    def __init__(self):
        self.bm25 = None
        self.corpus_chunks = []  
        self.is_initialized = False

    def initialize_from_db(self):
        """Initializes the BM25 index using existing chunks in ChromaDB."""
        try:
            count = vector_store.count()
            if count == 0:
                logger.info("ChromaDB is empty. BM25 index not initialized.")
                self.is_initialized = False
                return
                
            logger.info(f"Loading {count} chunks from ChromaDB to initialize BM25 index...")
            
            results = vector_store.collection.get(include=["documents", "metadatas"])
            
            self.corpus_chunks = []
            tokenized_corpus = []
            
            ids = results["ids"]
            documents = results["documents"]
            metadatas = results["metadatas"]
            
            for i in range(len(ids)):
                chunk = {
                    "id": ids[i],
                    "content": documents[i],
                    "url": metadatas[i]["url"],
                    "title": metadatas[i]["title"],
                    "parent_header": metadatas[i].get("parent_header")
                }
                self.corpus_chunks.append(chunk)
                tokenized_corpus.append(tokenize(documents[i]))
                
            if tokenized_corpus:
                self.bm25 = BM25Okapi(tokenized_corpus)
                self.is_initialized = True
                logger.info("BM25 index successfully initialized.")
        except Exception as e:
            logger.error(f"Error initializing BM25 from ChromaDB: {e}")
            self.is_initialized = False

    def update_index(self, new_chunks: list[dict]):
        """Rebuilds the BM25 index with new chunks appended."""
        if not new_chunks:
            return
            
        # If not initialized, try load from DB first to get pre-existing items
        if not self.is_initialized:
            self.initialize_from_db()
            
        # Append new chunks to current corpus
        self.corpus_chunks.extend(new_chunks)
        
        tokenized_corpus = [tokenize(c["content"]) for c in self.corpus_chunks]
        if tokenized_corpus:
            self.bm25 = BM25Okapi(tokenized_corpus)
            self.is_initialized = True
            logger.info(f"BM25 index updated. Total corpus size: {len(self.corpus_chunks)}")

    def retrieve_hybrid(self, query: str, query_embedding: list[float], top_n: int = 5) -> list[dict]:
        """
        Retrieves top chunks using hybrid search (Vector Search + BM25 keyword search)
        and merges lists using Reciprocal Rank Fusion (RRF).
        """
        # If BM25 index is not active, try loading it
        if not self.is_initialized:
            self.initialize_from_db()
            
        # 1. Fetch top 15 results from Vector Search
        vector_results = vector_store.search(query_embedding, limit=15)
        
        # 2. Fetch top 15 results from BM25 Search (if initialized)
        bm25_results = []
        if self.is_initialized and self.bm25:
            tokenized_query = tokenize(query)
            # Get scores for all docs in corpus
            doc_scores = self.bm25.get_scores(tokenized_query)
            
            # Pair each chunk with its score and sort
            scored_corpus = []
            for idx, score in enumerate(doc_scores):
                if score > 0:  # Only consider documents with some match
                    chunk = self.corpus_chunks[idx].copy()
                    chunk["bm25_score"] = float(score)
                    scored_corpus.append(chunk)
                    
            scored_corpus.sort(key=lambda x: x["bm25_score"], reverse=True)
            bm25_results = scored_corpus[:15]

        # 3. Reciprocal Rank Fusion (RRF)
        # RRF score = Sum( 1 / (60 + rank) ) for each rank list
        rrf_scores = {}
        k = 60
        
        # Populate initial tracking dict with documents from both queries
        all_docs = {}
        
        # Process Vector Results
        for rank, doc in enumerate(vector_results):
            doc_id = doc["id"]
            if doc_id not in all_docs:
                all_docs[doc_id] = {
                    "id": doc_id,
                    "content": doc["content"],
                    "url": doc["url"],
                    "title": doc["title"],
                    "parent_header": doc["parent_header"],
                    "similarity_score": doc["similarity_score"],
                    "bm25_score": 0.0
                }
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank + 1))
            
        # Process BM25 Results
        for rank, doc in enumerate(bm25_results):
            doc_id = doc["id"]
            if doc_id not in all_docs:
                all_docs[doc_id] = {
                    "id": doc_id,
                    "content": doc["content"],
                    "url": doc["url"],
                    "title": doc["title"],
                    "parent_header": doc["parent_header"],
                    "similarity_score": 0.0,
                    "bm25_score": doc["bm25_score"]
                }
            else:
                all_docs[doc_id]["bm25_score"] = doc["bm25_score"]
                
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank + 1))
            
        # Sort docs by RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        
        hybrid_results = []
        for doc_id in sorted_ids[:top_n]:
            doc_info = all_docs[doc_id]
            doc_info["combined_score"] = rrf_scores[doc_id]
            hybrid_results.append(doc_info)
            
        # If no results matched, fallback to vector search if available
        if not hybrid_results and vector_results:
            return vector_results[:top_n]
            
        return hybrid_results

    def reset(self):
        """Reset local BM25 corpus cache."""
        self.bm25 = None
        self.corpus_chunks = []
        self.is_initialized = False

# Global hybrid retriever instance
hybrid_retriever = HybridRetriever()
