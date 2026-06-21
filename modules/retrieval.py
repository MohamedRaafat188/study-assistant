"""
retrieval.py — Query-time retrieval pipeline for the RAG system.

Connects the Embedder and VectorStore into a complete retrieval flow:
query → embed → search → assemble context. Outputs structured results
ready for LLM prompt construction.

This module does NOT handle prompt engineering or LLM generation —
those are separate concerns.

Usage:
    retriever = Retriever(embedder, vector_store, "study_materials")

    result = retriever.retrieve(query="How does multi-head attention work?")

    result.context_text     # formatted context string for LLM
    result.chunks           # list of RetrievedChunk with scores and metadata
"""

from dataclasses import dataclass
from enum import Enum

from embedder import Embedder
from vector_store import VectorStore, SearchResult


# ──────────────────────────────────────────────
# Data Structures
# ──────────────────────────────────────────────

class SearchMode(Enum):
    """Available search strategies."""
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"


class ContextOrder(Enum):
    """How chunks are ordered in the assembled context."""
    RELEVANCE = "relevance"   # by search score (highest first)
    POSITION = "position"     # by document position (chunk_index)


@dataclass
class RetrievedChunk:
    """A single retrieved chunk with all metadata needed for context and citation."""
    chunk_text: str
    document_title: str
    document_id: str
    page_numbers: list[int]
    headings: list[str]
    chunk_index: int
    content_type: str
    score: float
    point_id: str


@dataclass
class RetrievalResult:
    """Complete retrieval output for a single query."""
    query: str
    chunks: list[RetrievedChunk]
    context_text: str
    search_mode: SearchMode
    total_chunks_found: int


# ──────────────────────────────────────────────
# Retriever
# ──────────────────────────────────────────────

DEFAULT_SEARCH_MODE = SearchMode.HYBRID
DEFAULT_CONTEXT_ORDER = ContextOrder.POSITION
DEFAULT_LIMIT = 5
DEFAULT_PREFETCH_LIMIT = 20


class Retriever:
    """
    Query-time retrieval pipeline.

    Embeds a user query, searches the vector store, and assembles
    retrieved chunks into a formatted context for LLM consumption.

    Parameters
    ----------
    embedder : Embedder
        Initialized embedding model (BGE-M3).
    vector_store : VectorStore
        Initialized Qdrant connection.
    collection_name : str
        Name of the collection to search.
    default_search_mode : SearchMode
        Default search strategy. HYBRID recommended. Default: HYBRID.
    default_context_order : ContextOrder
        How chunks are ordered in context assembly.
        POSITION preserves document flow (better for LLM).
        RELEVANCE keeps highest-scored first (better for debugging).
        Default: POSITION.
    """

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        collection_name: str,
        default_search_mode: SearchMode = DEFAULT_SEARCH_MODE,
        default_context_order: ContextOrder = DEFAULT_CONTEXT_ORDER,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.collection_name = collection_name
        self.default_search_mode = default_search_mode
        self.default_context_order = default_context_order

    def retrieve(
        self,
        query: str,
        document_id: str | None = None,
        limit: int = DEFAULT_LIMIT,
        prefetch_limit: int = DEFAULT_PREFETCH_LIMIT,
        search_mode: SearchMode | None = None,
        context_order: ContextOrder | None = None,
    ) -> RetrievalResult:
        """
        Execute the full retrieval pipeline for a user query.

        Parameters
        ----------
        query : str
            The user's question.
        document_id : str, optional
            Restrict search to a specific document.
        limit : int
            Number of chunks to retrieve. Default: 5.
        prefetch_limit : int
            Candidates per index before fusion (hybrid only). Default: 20.
        search_mode : SearchMode, optional
            Override the default search strategy.
        context_order : ContextOrder, optional
            Override the default context ordering.

        Returns
        -------
        RetrievalResult
            Contains retrieved chunks, assembled context text, and metadata.
        """
        mode = search_mode or self.default_search_mode
        order = context_order or self.default_context_order

        # Step 1: Embed the query
        query_embedding = self.embedder.embed_single(query)

        # Step 2: Search Qdrant
        search_results = self._search(
            query_embedding=query_embedding,
            document_id=document_id,
            limit=limit,
            prefetch_limit=prefetch_limit,
            mode=mode,
        )

        # Step 3: Convert search results to RetrievedChunk objects
        chunks = self._to_chunks(search_results)

        # Step 4: Order chunks for context assembly
        ordered_chunks = self._order_chunks(chunks, order)

        # Step 5: Assemble context text
        context_text = self._assemble_context(ordered_chunks)

        return RetrievalResult(
            query=query,
            chunks=ordered_chunks,
            context_text=context_text,
            search_mode=mode,
            total_chunks_found=len(ordered_chunks),
        )

    def _search(
        self,
        query_embedding,
        document_id: str | None,
        limit: int,
        prefetch_limit: int,
        mode: SearchMode,
    ) -> list[SearchResult]:
        """Dispatch to the appropriate search method based on mode."""

        if mode == SearchMode.DENSE:
            return self.vector_store.search_dense(
                collection_name=self.collection_name,
                query_vector=query_embedding.dense_vector,
                document_id=document_id,
                limit=limit,
            )

        elif mode == SearchMode.SPARSE:
            return self.vector_store.search_sparse(
                collection_name=self.collection_name,
                query_sparse=query_embedding.sparse_vector,
                document_id=document_id,
                limit=limit,
            )

        elif mode == SearchMode.HYBRID:
            return self.vector_store.search_hybrid(
                collection_name=self.collection_name,
                query_dense=query_embedding.dense_vector,
                query_sparse=query_embedding.sparse_vector,
                document_id=document_id,
                limit=limit,
                prefetch_limit=prefetch_limit,
            )

        else:
            raise ValueError(f"Unknown search mode: {mode}")

    def _to_chunks(self, search_results: list[SearchResult]) -> list[RetrievedChunk]:
        """Convert raw search results to RetrievedChunk objects with validated fields."""
        chunks = []
        for result in search_results:
            payload = result.payload
            chunks.append(
                RetrievedChunk(
                    chunk_text=payload.get("chunk_text", ""),
                    document_title=payload.get("document_title", "Unknown"),
                    document_id=payload.get("document_id", ""),
                    page_numbers=payload.get("page_numbers", []),
                    headings=payload.get("headings", []),
                    chunk_index=payload.get("chunk_index", 0),
                    content_type=payload.get("content_type", "text"),
                    score=result.score,
                    point_id=result.id,
                )
            )
        return chunks

    def _order_chunks(
        self,
        chunks: list[RetrievedChunk],
        order: ContextOrder,
    ) -> list[RetrievedChunk]:
        """
        Order chunks for context assembly.

        RELEVANCE: highest score first. Useful for debugging and evaluation.
        POSITION: grouped by document, then by chunk_index within each document.
            Preserves the original document flow, which typically produces
            better LLM answers because the context reads naturally.
        """
        if order == ContextOrder.RELEVANCE:
            return sorted(chunks, key=lambda c: c.score, reverse=True)

        elif order == ContextOrder.POSITION:
            return sorted(chunks, key=lambda c: (c.document_id, c.chunk_index))

        else:
            raise ValueError(f"Unknown context order: {order}")

    def _assemble_context(self, chunks: list[RetrievedChunk]) -> str:
        """
        Assemble retrieved chunks into a formatted context string.

        Each chunk is wrapped with source metadata so the LLM can
        generate citations. The format is:

            [Source: "Document Title" | Section: Heading Path | Pages: 1, 2]
            <chunk text>

        This gives the LLM explicit signals for attribution without
        relying on it to infer sources from content alone.
        """
        if not chunks:
            return ""

        context_blocks = []

        for i, chunk in enumerate(chunks, 1):
            # Build the source attribution header
            header_parts = [f'Source: "{chunk.document_title}"']

            if chunk.headings:
                heading_path = " > ".join(chunk.headings)
                header_parts.append(f"Section: {heading_path}")

            if chunk.page_numbers:
                pages = ", ".join(str(p) for p in chunk.page_numbers)
                header_parts.append(f"Pages: {pages}")

            header = " | ".join(header_parts)

            block = f"[{header}]\n{chunk.chunk_text}"
            context_blocks.append(block)

        return "\n\n---\n\n".join(context_blocks)
