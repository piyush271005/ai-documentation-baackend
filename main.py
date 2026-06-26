import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.api.router import router as api_router
from backend.retrieval.reranker import hybrid_retriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("main")

app = FastAPI(
    title="docsense API",
    description="A semantic, keyword-hybrid documentation search engine built with FastAPI and ChromaDB.",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.on_event("startup")
async def on_startup():
    logger.info("Application starting up... Initializing indices.")
   
    hybrid_retriever.initialize_from_db()

@app.get("/")
async def root():
    return {
        "status": "healthy",
        "project": "docsense Backend",
        "docs_url": "/docs"
    }

# Register API Router
app.include_router(api_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
