import logging
import asyncio
from fastapi import APIRouter, BackgroundTasks, HTTPException
from backend.models.schemas import (
    CrawlRequest, CrawlResponse, QueryRequest, QueryResponse,
    StatusResponse, SettingsRequest, SettingsResponse, SourceChunk
)
from backend.config import settings
from backend.crawler.crawler import crawler_coordinator
from backend.chunking.chunker import Chunker
from backend.embeddings.embedding import embedding_service
from backend.retrieval.db import vector_store
from backend.retrieval.reranker import hybrid_retriever
from backend.llm.llm import llm_service

logger = logging.getLogger("router")
router = APIRouter(prefix="/api")

def get_settings_response() -> SettingsResponse:
    return SettingsResponse(
        llm_provider=settings.DEFAULT_LLM_PROVIDER,
        openai_key_configured=bool(settings.OPENAI_API_KEY),
        gemini_key_configured=bool(settings.GEMINI_API_KEY),
        ollama_url=settings.OLLAMA_BASE_URL
    )

async def run_crawl_pipeline(url: str, max_pages: int, limit_domain: bool):
    """Ingestion pipeline running in background."""
    try:
        # 1. Run the web crawl BFS loop
        await crawler_coordinator.run_crawl(url, max_pages, limit_domain)
        
        crawled_pages = crawler_coordinator.crawled_content
        if not crawled_pages:
            logger.warning("Crawl finished but no pages were extracted.")
            return
            
        # 2. Chunk crawled documents
        chunker = Chunker()
        all_chunks = []
        for page in crawled_pages:
            chunks = chunker.chunk_document(page)
            all_chunks.extend(chunks)
            
        if not all_chunks:
            logger.warning("No chunks generated from crawled documents.")
            return
            
        logger.info(f"Chunked {len(crawled_pages)} pages into {len(all_chunks)} chunks. Generating embeddings...")
        
        # 3. Generate embeddings in batch
        texts_to_embed = [c["content"] for c in all_chunks]
        embeddings = embedding_service.embed_documents(texts_to_embed)
        
        # 4. Insert chunks and embeddings into ChromaDB
        vector_store.add_chunks(all_chunks, embeddings)
        
        # 5. Rebuild / update BM25 search index
        hybrid_retriever.update_index(all_chunks)
        
        logger.info("RAG Ingestion pipeline finished successfully.")
    except Exception as e:
        logger.error(f"Error executing ingestion pipeline: {e}")
        crawler_coordinator.is_crawling = False

@router.post("/crawl", response_model=CrawlResponse)
async def start_crawl(request: CrawlRequest, background_tasks: BackgroundTasks):
    if crawler_coordinator.is_crawling:
        return CrawlResponse(
            status="error",
            message="A crawl task is already running. Please wait.",
            pages_crawled=len(crawler_coordinator.pages_crawled)
        )
        
    background_tasks.add_task(
        run_crawl_pipeline,
        request.url,
        request.max_pages,
        request.limit_domain
    )
    
    return CrawlResponse(
        status="success",
        message="Crawling initiated. Pages are being indexed in the background.",
        pages_crawled=0
    )

@router.post("/query", response_model=QueryResponse)
async def run_query(request: QueryRequest):
    if vector_store.count() == 0:
        raise HTTPException(
            status_code=400,
            detail="The knowledge base is empty. Please crawl a documentation site first!"
        )
        
    try:
        # 1. Embed query
        query_embedding = embedding_service.embed_query(request.query)
        
        # 2. Hybrid retrieve Top-5 chunks (Vector + BM25 RRF)
        top_chunks = hybrid_retriever.retrieve_hybrid(
            request.query,
            query_embedding,
            top_n=5
        )
        
        # 3. Synthesize LLM completion
        answer = await llm_service.generate_answer(
            request.query,
            top_chunks,
            request.llm_provider
        )
        
        # 4. Format sources
        sources = []
        for c in top_chunks:
            sources.append(
                SourceChunk(
                    id=c["id"],
                    title=c["title"],
                    url=c["url"],
                    content=c["content"],
                    parent_header=c.get("parent_header") or "",
                    similarity_score=c.get("similarity_score", 0.0),
                    bm25_score=c.get("bm25_score", 0.0),
                    combined_score=c.get("combined_score", 0.0)
                )
            )
            
        return QueryResponse(
            query=request.query,
            answer=answer,
            sources=sources
        )
    except Exception as e:
        logger.error(f"Error querying RAG pipeline: {e}")
        raise HTTPException(status_code=500, detail=f"Query execution failed: {str(e)}")

@router.get("/status", response_model=StatusResponse)
async def get_status():
    q_size = await crawler_coordinator.get_queue_size()
    total_chunks = vector_store.count()
    return StatusResponse(
        is_crawling=crawler_coordinator.is_crawling,
        pages_crawled=len(crawler_coordinator.pages_crawled),
        queue_size=q_size,
        total_chunks=total_chunks,
        crawled_urls=crawler_coordinator.pages_crawled
    )

@router.post("/reset")
async def reset_system():
    try:
        crawler_coordinator.reset()
        vector_store.reset()
        hybrid_retriever.reset()
        return {"status": "success", "message": "Crawler buffers and ChromaDB collections have been reset."}
    except Exception as e:
        logger.error(f"Reset failed: {e}")
        raise HTTPException(status_code=500, detail=f"Reset operation failed: {str(e)}")

@router.get("/settings", response_model=SettingsResponse)
async def get_settings():
    return get_settings_response()

@router.post("/settings", response_model=SettingsResponse)
async def update_settings(req: SettingsRequest):
    settings.DEFAULT_LLM_PROVIDER = req.llm_provider
    if req.openai_key is not None:
        settings.OPENAI_API_KEY = req.openai_key
    if req.gemini_key is not None:
        settings.GEMINI_API_KEY = req.gemini_key
    if req.ollama_url is not None:
        settings.OLLAMA_BASE_URL = req.ollama_url
    return get_settings_response()
