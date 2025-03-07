import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rerankers import Reranker
from rerankers.models.ranker import BaseRanker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "flashrank/ms-marco-MiniLM-L-12-v2"

# Initialize reranker models
rerankers = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the default model at startup."""
    logger.info("Warming up default model...")
    try:
        reranker = get_reranker(DEFAULT_MODEL)
        logger.info(f"Successfully loaded default model: {DEFAULT_MODEL}")
        _ = await reranker.rank_async(
            query="What is the capital of France?",
            docs=["Paris is the capital of France.", "Paris is the capital of France."],
        )
    except Exception as e:
        logger.error(f"Failed to load default model at startup: {e}")
    yield


app = FastAPI(title="Reranker API", description="A REST API for document reranking", version="1.0.0", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load environment variables
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
CACHE_DIR = "/tmp/.cache/flashrank"


class RerankerNotFoundError(Exception):
    """Exception raised when a reranker model is not found."""

    pass


def get_reranker(model_id: str) -> BaseRanker:
    """Get or initialize a reranker model."""
    model_type, model_name = model_id.split("/", 1)
    if model_id not in rerankers:
        logger.info(f"Loading reranker model: {model_id}")
        try:
            rerankers[model_id] = Reranker(
                model_name,
                model_type=model_type,
                cache_dir=CACHE_DIR,
            )
            if rerankers[model_id] is None:
                raise RerankerNotFoundError("Reranker could not be initialized")
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            raise RuntimeError(f"Model '{model_id}' could not be loaded: {str(e)}")
    return rerankers[model_id]


class RerankRequest(BaseModel):
    model: str = DEFAULT_MODEL
    query: str
    documents: List[str]
    top_n: Optional[int] = None
    return_documents: Optional[bool] = True


class RerankResult(BaseModel):
    index: int
    relevance_score: float
    document: Optional[str]


class RerankResponse(BaseModel):
    results: List[RerankResult]
    meta: Dict[str, Any] = {}


@app.post("/rerank")
async def rerank(request: RerankRequest) -> RerankResponse:
    """Rerank documents based on their relevance to a query."""
    try:
        reranker = get_reranker(request.model)

        # Rerank the documents
        ranking = await reranker.rank_async(query=request.query, docs=request.documents)

        # Format the response according to Cohere's schema
        rerank_results = []
        for result in ranking.results[: request.top_n]:
            score = result.score if ranking.has_scores else 1
            response_result = RerankResult(
                index=result.doc_id,
                relevance_score=score,
                document=None if not request.return_documents else request.documents[result.doc_id],
            )
            rerank_results.append(response_result)

        return RerankResponse(results=rerank_results, meta={"model": request.model})
    except RerankerNotFoundError as e:
        logger.error(f"Model not found: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error during reranking: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/models")
async def list_models():
    """List available models endpoint."""
    try:
        return {
            "models": [
                "flashrank/ms-marco-MiniLM-L-12-v2",
                "t5/unicamp-dl/mt5-base-mmarco-v2"
                "cross-encoder/mixedbread-ai/mxbai-rerank-base-v1"
                "colbert/colbert-ir/colbertv2.0",
            ]
        }
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    logger.info(f"Starting server on {HOST}:{PORT}")
    uvicorn.run("app:app", host=HOST, port=PORT, reload=False)
