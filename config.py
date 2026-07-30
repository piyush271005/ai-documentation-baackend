import os
from pathlib import Path
from pydantic_settings import BaseSettings


BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    PROJECT_NAME: str = "DocSense"
    
    CHROMA_DB_DIR: str = str(BASE_DIR / "chroma_db")

    
    CHROMA_API_KEY: str = "ck-9Ab3BjrPmhzeMGdehSqBxniGsTw4565oMAVWMBpo8Mpa"
    CHROMA_TENANT: str = "cffdfdb8-3349-4b0c-a112-c05e3c6a571a"
    CHROMA_DATABASE: str = "docsense_db"
    
    
    OPENAI_API_KEY: str = ""
    GEMINI_API_KEY: str = ""
    GROQ_API_KEY: str = ""
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    DEFAULT_LLM_PROVIDER: str = "gemini"  
    
    
    EMBEDDING_MODEL_NAME: str = "text-embedding-004"
    

settings = Settings()
