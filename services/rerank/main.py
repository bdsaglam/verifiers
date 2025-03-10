import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import bm25s
import Stemmer
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rerankers import Reranker
from rerankers.documents import Document
from rerankers.models.ranker import BaseRanker
from rerankers.results import RankedResults, Result

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
CACHE_DIR = "/tmp/.cache/flashrank"

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
            docs=[
                "Paris is the capital of France.",
                "Paris is the capital of France.",
            ],
        )
    except Exception as e:
        logger.error(f"Failed to load default model at startup: {e}")
    yield


app = FastAPI(
    title="Reranker API",
    description="A REST API for document reranking",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request, call_next):
    """Middleware to log all incoming requests and their responses."""
    logger.info(f"Request: {request.method} {request.url.path}")
    response = await call_next(request)
    logger.info(
        f"Response: {request.method} {request.url.path} - Status: {response.status_code}"
    )
    return response


class RerankerNotFoundError(Exception):
    pass


class BM25Ranker:
    def __init__(
        self,
        stemmer_lang: str | None = "en",
        stopwords: str | list[str] = "english",
        **kwargs,
    ):
        self.stopwords = stopwords
        self.stemmer = Stemmer.Stemmer(stemmer_lang) if stemmer_lang else None

    async def rank_async(self, query: str, docs: List[str], **kwargs):
        """Rank documents using BM25."""
        # Convert docs to format expected by BM25
        doc_dicts = [{"idx": i, "text": doc} for i, doc in enumerate(docs)]

        # Create BM25 instance
        retriever = bm25s.BM25(corpus=doc_dicts)

        # Tokenize corpus and index
        tokenized_corpus = bm25s.tokenize(
            [doc["text"] for doc in doc_dicts],
            stopwords=self.stopwords,
            stemmer=self.stemmer,
        )
        retriever.index(tokenized_corpus)

        # Get results
        documents, scores = retriever.retrieve(
            bm25s.tokenize(query, stemmer=self.stemmer),
            k=len(docs),
        )

        ranked_docs = [
            Result(
                document=Document(doc_id=doc["idx"], text=doc["text"]),
                score=score,
            )
            for doc, score in zip(documents[0].tolist(), scores[0].tolist())
        ]
        return RankedResults(results=ranked_docs, query=query, has_scores=True)


def get_reranker(model_id: str) -> BaseRanker:
    """Get or initialize a reranker model."""
    if model_id == "bm25":
        if model_id not in rerankers:
            logger.info("Loading BM25 reranker")
            try:
                rerankers[model_id] = BM25Ranker()
                if rerankers[model_id] is None:
                    raise RerankerNotFoundError(
                        "BM25 Reranker could not be initialized"
                    )
            except Exception as e:
                logger.error(f"Failed to load BM25 reranker: {e}")
                raise RuntimeError(
                    f"BM25 reranker could not be loaded: {str(e)}"
                )
        return rerankers[model_id]

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
            raise RuntimeError(
                f"Model '{model_id}' could not be loaded: {str(e)}"
            )
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
        ranking = await reranker.rank_async(
            query=request.query,
            docs=request.documents,
        )

        # Format the response according to Cohere's schema
        rerank_results = []
        for result in ranking.results[: request.top_n]:
            score = result.score if ranking.has_scores else 1
            response_result = RerankResult(
                index=result.doc_id,
                relevance_score=score,
                document=None
                if not request.return_documents
                else request.documents[result.doc_id],
            )
            rerank_results.append(response_result)

        return RerankResponse(
            results=rerank_results,
            meta={"model": request.model},
        )
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
                "bm25",
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
    uvicorn.run("main:app", host=HOST, port=PORT, reload=False)
