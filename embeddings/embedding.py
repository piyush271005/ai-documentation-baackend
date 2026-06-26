import logging
from fastembed import TextEmbedding
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
            logger.info(f"Loading fastembed model '{self.model_name}'...")
            try:
                # Load locally, it will download and cache automatically
                # Map all-MiniLM-L6-v2 to sentence-transformers/all-MiniLM-L6-v2 for fastembed compatibility
                mapped_model_name = self.model_name
                if mapped_model_name == "all-MiniLM-L6-v2":
                    mapped_model_name = "sentence-transformers/all-MiniLM-L6-v2"
                self._model = TextEmbedding(model_name=mapped_model_name)
                logger.info("Model loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                raise e
        return self._model

    def embed_query(self, text: str) -> list[float]:
        """Embed a single user query."""
        embeddings = list(self.model.embed([text]))
        return embeddings[0].tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of document texts."""
        embeddings = list(self.model.embed(texts))
        return [emb.tolist() for emb in embeddings]

# Global singleton instance
embedding_service = EmbeddingService()

