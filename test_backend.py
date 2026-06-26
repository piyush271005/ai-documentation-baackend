import asyncio
import logging
from backend.parser.parser import DocParser
from backend.chunking.chunker import Chunker
from backend.embeddings.embedding import embedding_service
from backend.retrieval.db import vector_store
from backend.retrieval.reranker import hybrid_retriever
from backend.llm.llm import llm_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_backend")

MOCK_HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>FastAPI Tutorial: Middleware</title>
</head>
<body>
    <nav class="navigation">
        <a href="/">Home</a> | <a href="/docs">Docs</a>
    </nav>
    <main>
        <h1>Middleware in FastAPI</h1>
        <p>Middleware allows you to process requests before they reach your route handlers, and responses before they leave.</p>
        
        <h2>Creating Middleware</h2>
        <p>To create a middleware, you use the <code>@app.middleware("http")</code> decorator. This decorator takes the request and a call_next function as arguments.</p>
        
        <pre><code>
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response
        </code></pre>
        
        <h2>Why Use Middleware?</h2>
        <p>Common middleware use cases include logging requests, managing CORS headers, authenticating users, and compressing response bodies.</p>
    </main>
    <footer>
        <p>© 2026 FastAPI Team. All rights reserved.</p>
    </footer>
</body>
</html>
"""

async def run_tests():
    logger.info("Starting Backend Verification Tests...")
    
    # 1. Test HTML Parser
    logger.info("1. Testing HTML Parser...")
    parsed_doc = DocParser.parse_document(MOCK_HTML_PAGE, "https://fastapi.tiangolo.com/tutorial/middleware/")
    assert parsed_doc["title"] == "FastAPI Tutorial: Middleware"
    logger.info(f"Successfully parsed document title: '{parsed_doc['title']}'")
    logger.info(f"Extracted {len(parsed_doc['blocks'])} text blocks.")
    
    # 2. Test Chunker
    logger.info("2. Testing Chunker...")
    chunker = Chunker(target_chunk_size=400)
    chunks = chunker.chunk_document(parsed_doc)
    assert len(chunks) > 0
    logger.info(f"Successfully split document into {len(chunks)} chunks.")
    for idx, c in enumerate(chunks):
        logger.info(f" Chunk {idx+1} (Header: '{c['parent_header']}'): {c['content'][:100]}...")
        
    # 3. Test Embedding Service (MiniLM)
    logger.info("3. Testing Embedding Service...")
    texts = [c["content"] for c in chunks]
    embeddings = embedding_service.embed_documents(texts)
    assert len(embeddings) == len(chunks)
    assert len(embeddings[0]) == 384  # all-MiniLM-L6-v2 dimension
    logger.info(f"Successfully generated {len(embeddings)} embeddings of dimension {len(embeddings[0])}")
    
    # 4. Test ChromaDB and BM25 Reranker
    logger.info("4. Testing Vector Index and Hybrid Retriever...")
    vector_store.reset()
    hybrid_retriever.reset()
    
    vector_store.add_chunks(chunks, embeddings)
    hybrid_retriever.update_index(chunks)
    
    query = "How do I create middleware?"
    query_emb = embedding_service.embed_query(query)
    
    results = hybrid_retriever.retrieve_hybrid(query, query_emb, top_n=3)
    assert len(results) > 0
    logger.info(f"Successfully retrieved {len(results)} hybrid results for query: '{query}'")
    for r in results:
        logger.info(f" Match (Score: {r['combined_score']:.4f}, Vector: {r['similarity_score']:.4f}, BM25: {r['bm25_score']:.4f}): {r['content'][:100]}...")
        
    # 5. Test LLM Answer Synthesis (Mock Provider)
    logger.info("5. Testing LLM Mock Answer Synthesis...")
    answer = await llm_service.generate_answer(query, results, provider_override="mock")
    logger.info(f"Synthesized Answer:\n{answer}")
    
    logger.info("All Backend Verification Tests PASSED successfully!")

if __name__ == "__main__":
    asyncio.run(run_tests())
