from typing import Callable


def golden_retriever(docs: list[dict], query: str) -> list[dict]:
    return [doc for doc in docs if doc["is_supporting"]]


def make_bm25_retriever(stemmer_lang: str | None = "en", stopwords: list[str] = [], top_k: int = 3):
    """Create a BM25 retriever function with the given configuration.

    Args:
        stemmer: Language code for stemmer (optional)
        stopwords: List of stopwords to use (optional)
    """
    import bm25s
    import Stemmer

    # Initialize stemmer based on config
    stemmer = None
    if stemmer_lang:
        stemmer = Stemmer.Stemmer(stemmer_lang)

    def retrieve(docs: list[dict], query: str) -> list[dict]:
        """BM25 retriever implementation.

        Args:
            docs: List of documents to search in. Each document should be a dict with
                 'idx' and 'text' fields.
            query: Query string to search for
            top_k: Number of documents to retrieve (default: 3)

        Returns:
            List of documents sorted by relevance score
        """
        k = min(top_k, len(docs))
        retriever = bm25s.BM25(corpus=docs)

        tokenized_corpus = bm25s.tokenize(
            [doc["text"] for doc in docs],
            stopwords=stopwords,
            stemmer=stemmer,
        )
        retriever.index(tokenized_corpus)
        results, scores = retriever.retrieve(bm25s.tokenize(query, stemmer=stemmer), k=k)
        return results[0].tolist()

    return retrieve


def make_reranker_retriever(name: str = "t5", top_k: int = 3) -> Callable:
    from rerankers import Reranker

    reranker = Reranker(name)
    if reranker is None:
        raise ValueError("Ranker is not initialized")

    def retrieve(docs: list[dict], query: str) -> list[dict]:
        texts = [doc["text"] for doc in docs]
        ranking = reranker.rank(query=query, docs=texts, doc_ids=list(range(len(texts))))
        return [docs[result.doc_id] for result in ranking.results[:top_k]]

    return retrieve


def make_retrieve_tool(name: str = "golden", top_k: int = 3) -> Callable:
    if name == "golden":
        retriever = golden_retriever
    elif name == "bm25":
        retriever = make_bm25_retriever(top_k=top_k)
    elif name.startswith("reranker/"):
        retriever = make_reranker_retriever(name=name.split("/")[-1], top_k=top_k)
    else:
        raise ValueError(f"Invalid retriever name: {name}")

    def retrieve(query: str, **kwargs) -> list[str]:
        """Find relevant documents for the query."""
        docs = kwargs['run_context']['input']["docs"]
        retrieved_docs = retriever(docs, query)
        return [x["text"] for x in retrieved_docs]

    return retrieve
