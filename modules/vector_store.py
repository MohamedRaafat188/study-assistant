"""
vector_store.py — Qdrant integration layer for the RAG pipeline.

Handles collection management, point insertion, filtered search (dense,
sparse, and hybrid), and deletion. Each user has an isolated Qdrant
instance/path — isolation is enforced at the caller level, not here.

Usage:
    store = VectorStore(path="./qdrant_data")       # local file storage
    store = VectorStore(url="http://localhost:6333") # Docker server

    store.create_collection("study_materials", dense_dim=1024)
    store.upsert_points("study_materials", points)
    results = store.search_hybrid("study_materials", query_dense, query_sparse)
"""

import uuid
from dataclasses import dataclass, field

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    Fusion,
    FusionQuery,
    MatchValue,
    NamedSparseVector,
    NamedVector,
    PayloadSchemaType,
    PointStruct,
    Prefetch,
    SparseVector as QdrantSparseVector,
    SparseVectorParams,
    VectorParams,
)

from embedder import EmbeddingResult, SparseVector


# ──────────────────────────────────────────────
# Data Structures
# ──────────────────────────────────────────────

@dataclass
class PointData:
    """
    Data for a single point to insert into Qdrant.

    Built by the ingestion pipeline after chunking and embedding.
    The ID is auto-generated if not provided.
    """
    dense_vector: list[float]
    sparse_vector: SparseVector
    payload: dict
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    @classmethod
    def from_embedding_result(
        cls,
        embedding: EmbeddingResult,
        payload: dict,
        point_id: str | None = None,
    ) -> "PointData":
        """
        Convenience constructor from an EmbeddingResult and payload dict.

        Parameters
        ----------
        embedding : EmbeddingResult
            Output from Embedder.embed() or Embedder.embed_single().
        payload : dict
            Metadata for this point (user_id, document_id, chunk_text, etc.).
        point_id : str, optional
            Custom UUID. Auto-generated if not provided.
        """
        return cls(
            dense_vector=embedding.dense_vector,
            sparse_vector=embedding.sparse_vector,
            payload=payload,
            id=point_id or str(uuid.uuid4()),
        )


@dataclass
class SearchResult:
    """A single search result returned from Qdrant."""
    id: str
    score: float
    payload: dict


# ──────────────────────────────────────────────
# Vector Store
# ──────────────────────────────────────────────

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
DEFAULT_BATCH_SIZE = 100


class VectorStore:
    """
    Qdrant vector store for the RAG pipeline.

    Supports two connection modes:
    - Local file storage (path): No Docker needed, data persists to disk.
      Best for development and testing.
    - Server mode (url): Connects to a running Qdrant instance (Docker).
      Best for production.

    If neither is provided, defaults to local file storage at ./qdrant_data.

    Parameters
    ----------
    url : str, optional
        Qdrant server URL (e.g., "http://localhost:6333").
    path : str, optional
        Local directory for file-based storage.
    """

    def __init__(self, url: str | None = None, path: str | None = None):
        if url and path:
            raise ValueError("Provide either 'url' or 'path', not both.")

        if url:
            self.client = QdrantClient(url=url)
            self.mode = "server"
        else:
            storage_path = path or "./qdrant_data"
            self.client = QdrantClient(path=storage_path)
            self.mode = "local"

    # ──────────────────────────────────────────
    # Collection Management
    # ──────────────────────────────────────────

    def collection_exists(self, collection_name: str) -> bool:
        """Check whether a collection exists."""
        return self.client.collection_exists(collection_name)

    def create_collection(
        self,
        collection_name: str,
        dense_dim: int,
        distance: Distance = Distance.COSINE,
    ) -> None:
        """
        Create a collection configured for dense + sparse hybrid search.

        Sets up:
        - A named dense vector field (dimensionality from embedding model)
        - A named sparse vector field (for BGE-M3 lexical weights)
        - A payload index on document_id for document-scoped queries

        Parameters
        ----------
        collection_name : str
            Name for the collection.
        dense_dim : int
            Dimensionality of dense vectors (1024 for BGE-M3).
        distance : Distance
            Similarity metric. Default: COSINE.

        Raises
        ------
        ValueError
            If the collection already exists.
        """
        if self.collection_exists(collection_name):
            raise ValueError(
                f"Collection '{collection_name}' already exists. "
                "Use delete_collection() first if you want to recreate it."
            )

        # Create collection with named dense and sparse vector configurations
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config={
                DENSE_VECTOR_NAME: VectorParams(
                    size=dense_dim,
                    distance=distance,
                ),
            },
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: SparseVectorParams(),
            },
        )

        # Index document_id for fast document-scoped filtered queries.
        self.client.create_payload_index(
            collection_name=collection_name,
            field_name="document_id",
            field_schema=PayloadSchemaType.KEYWORD,
        )

    def delete_collection(self, collection_name: str) -> None:
        """Delete a collection and all its data."""
        self.client.delete_collection(collection_name)

    def get_collection_info(self, collection_name: str) -> dict:
        """Get collection statistics (point count, config, etc.)."""
        info = self.client.get_collection(collection_name)
        return {
            "name": collection_name,
            "points_count": info.points_count,
            "vectors_count": info.vectors_count,
            "status": info.status.value,
        }

    # ──────────────────────────────────────────
    # Point Operations
    # ──────────────────────────────────────────

    def upsert_points(
        self,
        collection_name: str,
        points: list[PointData],
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        """
        Insert or update points in a collection.

        Automatically batches large insertions to avoid memory issues.
        Uses upsert semantics — if a point with the same ID exists, it
        gets overwritten.

        Parameters
        ----------
        collection_name : str
            Target collection.
        points : list[PointData]
            Points to insert, each containing vectors and payload.
        batch_size : int
            Number of points per upsert call. Default: 100.
        """
        if not points:
            return

        # Convert PointData objects to Qdrant PointStruct format
        qdrant_points = []
        for point in points:
            qdrant_points.append(
                PointStruct(
                    id=point.id,
                    vector={
                        DENSE_VECTOR_NAME: point.dense_vector,
                        SPARSE_VECTOR_NAME: QdrantSparseVector(
                            indices=point.sparse_vector.indices,
                            values=point.sparse_vector.values,
                        ),
                    },
                    payload=point.payload,
                )
            )

        # Batch upsert
        for i in range(0, len(qdrant_points), batch_size):
            batch = qdrant_points[i : i + batch_size]
            self.client.upsert(
                collection_name=collection_name,
                points=batch,
            )

    # ──────────────────────────────────────────
    # Search Operations
    # ──────────────────────────────────────────

    def _build_filter(
        self,
        document_id: str | None = None,
    ) -> Filter | None:
        """
        Build a Qdrant filter for document-scoped queries.

        Returns None when no document_id is given, which tells Qdrant
        to search the entire collection with no filter applied.
        """
        if not document_id:
            return None

        return Filter(
            must=[
                FieldCondition(
                    key="document_id",
                    match=MatchValue(value=document_id),
                )
            ]
        )

    def search_dense(
        self,
        collection_name: str,
        query_vector: list[float],
        document_id: str | None = None,
        limit: int = 5,
    ) -> list[SearchResult]:
        """
        Search using only dense vectors (semantic similarity).

        Parameters
        ----------
        collection_name : str
            Collection to search.
        query_vector : list[float]
            Dense embedding of the query text.
        document_id : str, optional
            Filter to a specific document.
        limit : int
            Number of results to return. Default: 5.

        Returns
        -------
        list[SearchResult]
            Results sorted by relevance (highest score first).
        """
        results = self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            using=DENSE_VECTOR_NAME,
            query_filter=self._build_filter(document_id),
            limit=limit,
        )

        return [
            SearchResult(
                id=str(point.id),
                score=point.score,
                payload=point.payload,
            )
            for point in results.points
        ]

    def search_sparse(
        self,
        collection_name: str,
        query_sparse: SparseVector,
        document_id: str | None = None,
        limit: int = 5,
    ) -> list[SearchResult]:
        """
        Search using only sparse vectors (keyword-style matching).

        Parameters
        ----------
        collection_name : str
            Collection to search.
        query_sparse : SparseVector
            Sparse embedding of the query text.
        document_id : str, optional
            Filter to a specific document.
        limit : int
            Number of results to return. Default: 5.

        Returns
        -------
        list[SearchResult]
            Results sorted by relevance (highest score first).
        """
        results = self.client.query_points(
            collection_name=collection_name,
            query=QdrantSparseVector(
                indices=query_sparse.indices,
                values=query_sparse.values,
            ),
            using=SPARSE_VECTOR_NAME,
            query_filter=self._build_filter(document_id),
            limit=limit,
        )

        return [
            SearchResult(
                id=str(point.id),
                score=point.score,
                payload=point.payload,
            )
            for point in results.points
        ]

    def search_hybrid(
        self,
        collection_name: str,
        query_dense: list[float],
        query_sparse: SparseVector,
        document_id: str | None = None,
        limit: int = 5,
        prefetch_limit: int = 20,
    ) -> list[SearchResult]:
        """
        Hybrid search combining dense and sparse vectors using
        Reciprocal Rank Fusion (RRF).

        This is the primary search method for the RAG pipeline. It
        retrieves candidates from both dense (semantic) and sparse
        (keyword) indexes, then fuses rankings so that a chunk
        appearing high in both lists gets boosted.

        Parameters
        ----------
        collection_name : str
            Collection to search.
        query_dense : list[float]
            Dense embedding of the query text.
        query_sparse : SparseVector
            Sparse embedding of the query text.
        document_id : str, optional
            Filter to a specific document.
        limit : int
            Final number of results after fusion. Default: 5.
        prefetch_limit : int
            Candidates retrieved from each index before fusion.
            Higher values improve recall at the cost of latency.
            Default: 20.

        Returns
        -------
        list[SearchResult]
            Results sorted by fused relevance score.
        """
        query_filter = self._build_filter(document_id)

        results = self.client.query_points(
            collection_name=collection_name,
            prefetch=[
                Prefetch(
                    query=query_dense,
                    using=DENSE_VECTOR_NAME,
                    filter=query_filter,
                    limit=prefetch_limit,
                ),
                Prefetch(
                    query=QdrantSparseVector(
                        indices=query_sparse.indices,
                        values=query_sparse.values,
                    ),
                    using=SPARSE_VECTOR_NAME,
                    filter=query_filter,
                    limit=prefetch_limit,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=limit,
        )

        return [
            SearchResult(
                id=str(point.id),
                score=point.score,
                payload=point.payload,
            )
            for point in results.points
        ]

    # ──────────────────────────────────────────
    # Deletion Operations
    # ──────────────────────────────────────────

    def delete_by_document(
        self,
        collection_name: str,
        document_id: str,
    ) -> None:
        """
        Delete all points belonging to a specific document.

        Parameters
        ----------
        collection_name : str
            Target collection.
        document_id : str
            Document whose chunks should be deleted.
        """
        self.client.delete(
            collection_name=collection_name,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=document_id),
                        ),
                    ]
                )
            ),
        )
