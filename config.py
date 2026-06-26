import os
from pathlib import Path
from pydantic_settings import BaseSettings

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    PROJECT_NAME: str = "AI Documentation Search Engine"
    
    # Storage and DB paths
    CHROMA_DB_DIR: str = str(BASE_DIR / "chroma_db")
    
    
    # LLM Settings & Keys
    OPENAI_API_KEY: str = ""
    GEMINI_API_KEY: str = ""
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    DEFAULT_LLM_PROVIDER: str = "mock"  # options: mock, openai, gemini, ollama
    
    # Embedding model name
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
