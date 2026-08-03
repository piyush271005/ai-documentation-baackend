from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class CrawlRequest(BaseModel):
    url: str 
    max_pages: int = 50

class CrawlResponse(BaseModel):
    status: str
    message: str
    pages_crawled: int

class SourceChunk(BaseModel):
    id: str
    title: str
    url: str
    content: str
    parent_header: Optional[str] = None
    similarity_score: float
    bm25_score: float
    combined_score: float

class QueryRequest(BaseModel):
    query: str 
    llm_provider: Optional[str] 

class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[SourceChunk]

class StatusResponse(BaseModel):
    is_crawling: bool
    pages_crawled: int
    queue_size: int
    total_chunks: int
    crawled_urls: List[str]

class SettingsRequest(BaseModel):
    llm_provider: str
    openai_key: Optional[str] = ""
    gemini_key: Optional[str] = ""
    groq_key: Optional[str] = ""
    ollama_url: Optional[str] = ""

class SettingsResponse(BaseModel):
    llm_provider: str
    openai_key_configured: bool
    gemini_key_configured: bool
    groq_key_configured: bool
    ollama_url: str
