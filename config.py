import os
from pathlib import Path
from pydantic_settings import BaseSettings

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    PROJECT_NAME: str = "DocSense"
    
    # Storage and DB paths (used when CHROMA_API_KEY is not set)
    CHROMA_DB_DIR: str = str(BASE_DIR / "chroma_db")

    # ChromaDB Cloud settings (set CHROMA_API_KEY to enable cloud mode)
    CHROMA_API_KEY: str = "ck-9Ab3BjrPmhzeMGdehSqBxniGsTw4565oMAVWMBpo8Mpa"
    CHROMA_TENANT: str = "cffdfdb8-3349-4b0c-a112-c05e3c6a571a"
    CHROMA_DATABASE: str = "docsense_db"
    
    # LLM Settings & Keys
    OPENAI_API_KEY: str = ""
    GEMINI_API_KEY: str = ""
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    DEFAULT_LLM_PROVIDER: str = "gemini"  # options: gemini, openai, ollama
    
    # Embedding model name (Google GenAI)
    EMBEDDING_MODEL_NAME: str = "text-embedding-004"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
