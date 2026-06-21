"""
point_builder.py — Qdrant point construction for the RAG pipeline.

Maps chunks and embeddings into PointData objects with the correct
payload schema. This module defines the contract between ingestion
and retrieval — the payload fields defined here are what retrieval.py
expects to find.

Usage:
    builder = PointBuilder(document_id="doc_456", document_title="My Paper")
    points = builder.build_points(chunks, embeddings)
"""

from .chunker import ChunkData
from .embedder import EmbeddingResult
from .vector_store import PointData


class PointBuilder:
    """
    Builds Qdrant PointData objects from chunks and embeddings.

    Owns the payload schema — the set of fields stored alongside
    each vector in Qdrant. This schema is the contract between
    the ingestion pipeline and the retrieval pipeline:
    - Ingestion writes these fields via PointBuilder
    - Retrieval reads them in Retriever._to_chunks()

    If you add or change a payload field, update both this class
    and RetrievedChunk in retrieval.py.

    Parameters
    ----------
    document_id : str
        Unique identifier for the document.
    document_title : str
        Human-readable document title for citations.
    """

    def __init__(
        self,
        document_id: str,
        document_title: str,
    ):
        self.document_id = document_id
        self.document_title = document_title

    def build_points(
        self,
        chunks: list[ChunkData],
        embeddings: list[EmbeddingResult],
    ) -> list[PointData]:
        """
        Build PointData objects from paired chunks and embeddings.

        Parameters
        ----------
        chunks : list[ChunkData]
            Chunks from Chunker.chunk(), carrying text and metadata.
        embeddings : list[EmbeddingResult]
            Embeddings from Embedder.embed(), one per chunk.
            Must be same length and order as chunks.

        Returns
        -------
        list[PointData]
            Ready for VectorStore.upsert_points().

        Raises
        ------
        ValueError
            If chunks and embeddings have different lengths.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunk count ({len(chunks)}) != embedding count ({len(embeddings)}). "
                "Each chunk must have exactly one embedding."
            )

        points = []
        for chunk, embedding in zip(chunks, embeddings):
            payload = self._build_payload(chunk)
            points.append(
                PointData.from_embedding_result(
                    embedding=embedding,
                    payload=payload,
                )
            )

        return points

    def _build_payload(self, chunk: ChunkData) -> dict:
        """
        Build the payload dict for a single chunk.

        This is the single source of truth for the payload schema.
        Fields fall into two categories:

        Filtering fields (indexed in Qdrant for fast filtered search):
            - document_id: document-scoped queries and deletion

        Reconstruction fields (stored for retrieval and LLM context):
            - chunk_text: contextualized text for LLM context assembly
            - document_title: for citations
            - page_numbers: for source references
            - headings: section path for navigation
            - chunk_index: position in document for ordering
            - content_type: element type (text, table, formula, etc.)
        """
        return {
            # Filtering fields
            "document_id": self.document_id,
            "content_type": chunk.metadata.content_type,
            # Reconstruction fields
            "chunk_text": chunk.contextualized_text,
            "document_title": self.document_title,
            "page_numbers": chunk.metadata.page_numbers,
            "headings": chunk.metadata.headings,
            "chunk_index": chunk.metadata.chunk_index,
        }
