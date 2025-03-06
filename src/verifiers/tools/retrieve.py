from collections import defaultdict
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
            docs: List of documents to search in. Each document should be a dict with 'id' and 'text' fields.
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


def make_rerank_retriever(
    model: str = "t5/unicamp-dl/mt5-base-mmarco-v2",
    top_k: int = 3,
    rerank_client=None,
) -> Callable:
    from verifiers.rerank import RerankClient

    rerank_client = rerank_client or RerankClient()

    def retrieve(docs: list[dict], query: str) -> list[dict]:
        texts = [doc["text"] for doc in docs]
        ranking = rerank_client.rerank(
            query=query,
            documents=texts,
            model=model,
            top_n=top_k,
            return_documents=False,
        )
        return [docs[result.index] for result in ranking.results]

    return retrieve


def combine_retrieval_results(ranked_docs_list: list[list[dict]], docs: list[dict], top_k: int) -> list[dict]:
    """Aggregate retrieval results from multiple retrievers.

    Args:
        ranked_docs_list: List of lists containing ranked documents from each retriever.
        docs: List of documents to search in. Each document should be a dict with 'id' and 'text' fields.
        top_k: Number of documents to retrieve.

    Returns:
        List of documents sorted by combined relevance score.
    """
    combined_ranks = defaultdict(int)

    for ranked_docs in ranked_docs_list:
        seen = set()
        for rank, doc in enumerate(ranked_docs):
            idx = doc["id"]
            combined_ranks[idx] += rank
            seen.add(idx)

        # Add top_k + 1 to documents not retrieved by this retriever
        for doc in docs:
            if doc["id"] not in seen:
                combined_ranks[doc["id"]] += top_k

    # Sort results by the summed rank (ascending)
    sorted_results = sorted(combined_ranks.items(), key=lambda item: item[1])
    docs_mapping = {doc["id"]: doc for doc in docs}
    return [docs_mapping[id] for id, _ in sorted_results[:top_k]]


def make_combined_retriever(*retrievers: list[Callable], top_k: int = 3) -> Callable:
    """Create a combined retriever function with the given retrievers and configuration.

    Args:
        retrievers: List of retriever functions to combine.

    Returns:
        A function that retrieves documents by combining results from all retrievers.
    """

    def retrieve(docs: list[dict], query: str) -> list[dict]:
        """Combined retriever implementation.

        Args:
            docs: List of documents to search in. Each document should be a dict with 'id' and 'text' fields.
            query: Query string to search for.
            top_k: Number of documents to retrieve (default: 3).

        Returns:
            List of documents sorted by combined relevance score.
        """
        ranked_docs_list = [retriever(docs, query, top_k) for retriever in retrievers]
        return combine_retrieval_results(ranked_docs_list, docs, top_k)

    return retrieve


def make_retrieve_tool(name: str = "bm25", top_k: int = 3) -> Callable:
    if name == "golden":
        retriever = golden_retriever
    elif name == "bm25":
        retriever = make_bm25_retriever(top_k=top_k)
    elif name.startswith("rerank"):
        model = name.split("/", 1)[-1].strip()
        kwargs = dict(top_k=top_k)
        if model:
            kwargs["model"] = model
        retriever = make_rerank_retriever(**kwargs)
    elif name == "hybrid":
        rerank_retriever = make_rerank_retriever(top_k=top_k)
        bm25_retriever = make_bm25_retriever(top_k=top_k)
        retriever = make_combined_retriever(rerank_retriever, bm25_retriever, top_k=top_k)
    else:
        raise ValueError(f"Invalid retriever name: {name}")

    def retrieve(query: str, **kwargs) -> str:
        """Search for relevant documents by the query. The results become better if the query is more specific."""
        docs = kwargs["run_context"]["input"]["docs"]
        retrieved_docs = retriever(docs, query)
        return "\n\n".join([x["text"] for x in retrieved_docs])

    return retrieve
