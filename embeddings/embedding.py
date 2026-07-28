import os
import logging
from google import genai
from backend.config import settings

logger = logging.getLogger("embeddings")

class EmbeddingService:
    def __init__(self):
        self.model_name = settings.EMBEDDING_MODEL_NAME or "text-embedding-004"
        self._client = None

    @property
    def client(self) -> genai.Client:
        """Lazy load the Google GenAI Client."""
        if self._client is None:
            api_key = settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY")
            if not api_key:
                logger.warning("GEMINI_API_KEY is not set. Please set GEMINI_API_KEY in settings or environment.")
            logger.info(f"Initializing Google GenAI Client with embedding model '{self.model_name}'...")
            self._client = genai.Client(api_key=api_key or "NO_KEY_PROVIDED")
        return self._client

    def embed_query(self, text: str) -> list[float]:
        """Embed a single user query using Google GenAI API (text-embedding-004)."""
        try:
            api_key = settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY")
            if not api_key:
                logger.error("GEMINI_API_KEY missing for embedding. Returning fallback vector.")
                return [0.0] * 768

            response = self.client.models.embed_content(
                model=self.model_name,
                contents=text
            )
            return list(response.embeddings[0].values)
        except Exception as e:
            logger.error(f"Error embedding query with Google GenAI API: {e}")
            return [0.0] * 768

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of document texts using Google GenAI API (text-embedding-004)."""
        if not texts:
            return []
        try:
            api_key = settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY")
            if not api_key:
                logger.error("GEMINI_API_KEY missing for embedding documents. Returning fallback vectors.")
                return [[0.0] * 768 for _ in texts]

            response = self.client.models.embed_content(
                model=self.model_name,
                contents=texts
            )
            return [list(emb.values) for emb in response.embeddings]
        except Exception as e:
            logger.error(f"Error embedding documents with Google GenAI API: {e}")
            return [[0.0] * 768 for _ in texts]

# Global singleton instance
embedding_service = EmbeddingService()


