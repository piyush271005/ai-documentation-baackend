from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional, Dict, Any

class CrawlRequest(BaseModel):
    url: str = Field(..., description="The base URL of the documentation site to crawl")
    max_pages: int = Field(50, ge=1, le=500, description="Maximum number of pages to crawl")
    limit_domain: bool = Field(True, description="Limit crawling to the base domain of the start URL")

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
    query: str = Field(..., min_length=3, description="Search query/question")
    llm_provider: Optional[str] = Field(None, description="Override the default LLM provider")

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
    ollama_url: Optional[str] = ""

class SettingsResponse(BaseModel):
    llm_provider: str
    openai_key_configured: bool
    gemini_key_configured: bool
    ollama_url: str
