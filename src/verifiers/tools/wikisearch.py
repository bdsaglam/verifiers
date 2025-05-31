from typing import Callable

from verifiers.models import RunContext
from verifiers.wikisearch import SearchDocument, WikiSearchClient, WikiSearchError


class WikiSearchToolError(Exception):
    pass


def format_doc(doc: SearchDocument) -> str:
    return f"Document ID: {doc.id}\n# {doc.title}\n{doc.body}"


def make_wiki_search_tool(
    top_n: int = 3,
    client: WikiSearchClient | None = None,
) -> Callable:
    """Create a wiki search tool that uses the WikiSearchClient.

    Args:
        client: Optional WikiSearchClient instance. If not provided, a new one will be created.

    Returns:
        A function that searches for documents using the wiki search API.
    """
    client = client or WikiSearchClient()

    def wiki_search(query: str, run_context: RunContext | None = None, **kwargs) -> str:
        """
        Search for documents using the wiki search API.

        Args:
            query: The search query.
            run_context: Optional run context (not used in this tool).

        Returns:
            Formatted string with search results.

        Examples:
            {"query": "what is machine learning"}
        """
        try:
            response = client.search(query, top_n=top_n)
            if not response or not response.results:
                return "No documents found"

            return "\n\n".join([format_doc(result.document) for result in response.results])
        except WikiSearchError as e:
            raise WikiSearchToolError(f"Error searching documents: {e}")

    return wiki_search
