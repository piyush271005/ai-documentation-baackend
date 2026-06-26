import logging
from sentence_transformers import SentenceTransformer
from backend.config import settings

logger = logging.getLogger("embeddings")

class EmbeddingService:
    def __init__(self):
        self.model_name = settings.EMBEDDING_MODEL_NAME
        self._model = None

    @property
    def model(self):
        """Lazy load the model when first requested."""
        if self._model is None:
            logger.info(f"Loading sentence-transformers model '{self.model_name}'...")
            try:
                # Load locally, it will download and cache automatically
                self._model = SentenceTransformer(self.model_name)
                logger.info("Model loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                raise e
        return self._model

    def embed_query(self, text: str) -> list[float]:
        """Embed a single user query."""
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of document texts."""
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

# Global singleton instance
embedding_service = EmbeddingService()
