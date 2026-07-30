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
        groq_key_configured=bool(settings.GROQ_API_KEY),
        ollama_url=settings.OLLAMA_BASE_URL
    )

def _index_pages(pages: list, chunker: Chunker) -> int:
    """Chunk, embed, and store a batch of pages. Returns number of chunks stored."""
    all_chunks = []
    for page in pages:
        chunks = chunker.chunk_document(page)
        all_chunks.extend(chunks)
    if not all_chunks:
        return 0
    texts = [c["content"] for c in all_chunks]
    embeddings = embedding_service.embed_documents(texts)
    vector_store.add_chunks(all_chunks, embeddings)
    hybrid_retriever.update_index(all_chunks)
    return len(all_chunks)


async def run_crawl_pipeline(url: str, max_pages: int):
    """
    Pipelined ingestion: crawling and indexing run concurrently.
    A consumer task processes completed pages in batches while the
    crawler is still running, so post-crawl indexing delay is near-zero.
    """
    chunker = Chunker()
    indexed_count = 0          # number of pages already sent to indexer
    total_chunks_indexed = 0

    async def indexing_consumer():
        """Picks up newly crawled pages every 2s and indexes them in batches."""
        nonlocal indexed_count, total_chunks_indexed
        BATCH_SIZE = 5

        while crawler_coordinator.is_crawling:
            await asyncio.sleep(2)  # poll every 2 seconds
            available = crawler_coordinator.crawled_content
            new_pages = available[indexed_count:]
            if len(new_pages) >= BATCH_SIZE:
                batch = new_pages[:BATCH_SIZE]
                indexed_count += len(batch)
                logger.info(f"[Pipeline] Indexing batch of {len(batch)} pages mid-crawl...")
                n = await asyncio.get_event_loop().run_in_executor(
                    None, _index_pages, batch, chunker
                )
                total_chunks_indexed += n
                logger.info(f"[Pipeline] Batch done — {n} chunks stored ({total_chunks_indexed} total so far)")

    try:
        # Run crawler + indexing consumer concurrently
        crawl_task = asyncio.create_task(
            crawler_coordinator.run_crawl(url, max_pages, limit_domain=True)
        )
        consumer_task = asyncio.create_task(indexing_consumer())

        await crawl_task        # wait for crawl to finish
        consumer_task.cancel()  # stop the consumer loop
        try:
            await consumer_task
        except asyncio.CancelledError:
            pass

        # Final flush: index any pages that arrived after the last batch
        remaining_pages = crawler_coordinator.crawled_content[indexed_count:]
        if remaining_pages:
            logger.info(f"[Pipeline] Final flush: indexing {len(remaining_pages)} remaining pages...")
            n = await asyncio.get_event_loop().run_in_executor(
                None, _index_pages, remaining_pages, chunker
            )
            total_chunks_indexed += n
        elif not crawler_coordinator.crawled_content:
            logger.warning("Crawl finished but no pages were extracted.")
            return

        # Explicit memory cleanup after crawl completes
        import gc
        crawler_coordinator.crawled_content.clear()
        gc.collect()

        logger.info(f"RAG Ingestion pipeline finished. Total chunks indexed: {total_chunks_indexed}")

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
        request.max_pages
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
    if req.groq_key is not None:
        settings.GROQ_API_KEY = req.groq_key
    if req.ollama_url is not None:
        settings.OLLAMA_BASE_URL = req.ollama_url
    return get_settings_response()
