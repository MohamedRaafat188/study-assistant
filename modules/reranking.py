"""
reranking.py — Cross-encoder reranking step for the RAG retrieval pipeline.

Takes a RetrievalResult from Retriever.retrieve(), scores each chunk against
the query using BAAI/bge-reranker-v2-m3, trims to top-k, and returns a new
RetrievalResult with updated scores and reassembled context.

Usage:
    reranker = Reranker()
    reranked = reranker.rerank(retrieval_result, top_k=5)

    reranked.chunks          # sorted by reranker score (highest first)
    reranked.context_text    # reassembled in document position order
"""

from dataclasses import replace as dc_replace

from FlagEmbedding import FlagReranker

from .retrieval import RetrievalResult, RetrievedChunk


# ──────────────────────────────────────────────
# Reranker
# ──────────────────────────────────────────────

MODEL_ID = "BAAI/bge-reranker-v2-m3"
DEFAULT_TOP_K = 5


class Reranker:
    """
    Cross-encoder reranker using BAAI/bge-reranker-v2-m3.

    Parameters
    ----------
    model_id : str
        Hugging Face model identifier. Default: BAAI/bge-reranker-v2-m3.
    use_fp16 : bool
        Half-precision inference. Reduces VRAM with negligible quality loss.
        Default: True.
    """

    def __init__(
        self,
        model_id: str = MODEL_ID,
        use_fp16: bool = True,
    ):
        self.model_id = model_id
        self.model = FlagReranker(model_id, use_fp16=use_fp16)

    def rerank(
        self,
        result: RetrievalResult,
        top_k: int = DEFAULT_TOP_K,
    ) -> RetrievalResult:
        """
        Rerank retrieved chunks using cross-encoder scores.

        Parameters
        ----------
        result : RetrievalResult
            Output from Retriever.retrieve().
        top_k : int
            Number of chunks to keep after reranking. Default: 5.

        Returns
        -------
        RetrievalResult
            New result with reranker scores, trimmed to top_k.
            chunks are sorted by reranker score (highest first).
            context_text is assembled in document position order for LLM coherence.
        """
        if not result.chunks:
            return result

        top_k = min(top_k, len(result.chunks))

        # Score all (query, chunk_text) pairs with the cross-encoder
        # normalize=True maps raw logits to [0, 1]
        pairs = [[result.query, chunk.chunk_text] for chunk in result.chunks]
        scores = self.model.compute_score(pairs, normalize=True)

        # Sort by reranker score descending and trim to top_k
        scored = sorted(
            zip(scores, result.chunks),
            key=lambda x: x[0],
            reverse=True,
        )
        top_chunks = [
            dc_replace(chunk, score=float(score))
            for score, chunk in scored[:top_k]
        ]

        # Assemble context in document position order so the LLM reads
        # naturally flowing text rather than score-ranked fragments
        position_ordered = sorted(top_chunks, key=lambda c: (c.document_id, c.chunk_index))
        context_text = _assemble_context(position_ordered)

        return RetrievalResult(
            query=result.query,
            chunks=top_chunks,
            context_text=context_text,
            search_mode=result.search_mode,
            total_chunks_found=top_k,
        )


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _assemble_context(chunks: list[RetrievedChunk]) -> str:
    """Assemble chunks into a formatted context string with source attribution."""
    if not chunks:
        return ""

    context_blocks = []
    for chunk in chunks:
        header_parts = [f'Source: "{chunk.document_title}"']

        # if chunk.headings:
        #     header_parts.append(f"Section: {' > '.join(chunk.headings)}")

        if chunk.page_numbers:
            header_parts.append(f"Pages: {', '.join(str(p) for p in chunk.page_numbers)}")

        header = " | ".join(header_parts)
        context_blocks.append(f"[{header}]\n{chunk.chunk_text}")

    return "\n\n---\n\n".join(context_blocks)
