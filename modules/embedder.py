"""
embedder.py — BGE-M3 embedding module for dense and sparse vector generation.

Wraps BAAI/bge-m3 via FlagEmbedding to produce both dense vectors (1024-dim)
and sparse vectors (learned lexical weights) from chunk text. Outputs are
structured for direct insertion into Qdrant.

Usage:
    embedder = Embedder()  # loads model once
    results = embedder.embed(["chunk text 1", "chunk text 2"])
    results[0].dense_vector   # list[float], 1024-dim
    results[0].sparse_vector  # SparseVector(indices=[...], values=[...])
"""

from dataclasses import dataclass
from FlagEmbedding import BGEM3FlagModel
from transformers import AutoTokenizer


# ──────────────────────────────────────────────
# Data Structures
# ──────────────────────────────────────────────

@dataclass
class SparseVector:
    """Qdrant-compatible sparse vector format."""
    indices: list[int]
    values: list[float]


@dataclass
class EmbeddingResult:
    """Embedding output for a single text."""
    dense_vector: list[float]
    sparse_vector: SparseVector


# ──────────────────────────────────────────────
# Embedder
# ──────────────────────────────────────────────

MODEL_ID = "BAAI/bge-m3"
DEFAULT_BATCH_SIZE = 16


class Embedder:
    """
    BGE-M3 embedding wrapper producing dense and sparse vectors.

    Parameters
    ----------
    model_id : str
        Hugging Face model identifier. Default: BAAI/bge-m3.
    batch_size : int
        Max texts per forward pass. Controls GPU memory usage.
        Lower if you hit OOM errors. Default: 16.
    use_fp16 : bool
        Whether to use half-precision. Reduces VRAM usage roughly
        by half with negligible quality loss. Default: True.
    """

    def __init__(
        self,
        model_id: str = MODEL_ID,
        batch_size: int = DEFAULT_BATCH_SIZE,
        use_fp16: bool = True,
    ):
        self.model_id = model_id
        self.batch_size = batch_size

        # Load the embedding model
        # FlagEmbedding handles device placement automatically (GPU if available)
        self.model = BGEM3FlagModel(
            model_id,
            use_fp16=use_fp16,
        )

        # Load tokenizer for sparse vector token-string → token-ID conversion.
        # BGE-M3's lexical_weights returns {token_string: weight}, but Qdrant
        # expects {index: int, value: float}. The tokenizer bridges this gap.
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

    def embed(self, texts: list[str]) -> list[EmbeddingResult]:
        """
        Embed a list of texts, returning dense and sparse vectors for each.

        Parameters
        ----------
        texts : list[str]
            Texts to embed. For RAG chunks, pass the contextualized text
            (with heading prefixes) from HybridChunker.

        Returns
        -------
        list[EmbeddingResult]
            One result per input text, in the same order.

        Raises
        ------
        ValueError
            If texts is empty.
        """
        if not texts:
            raise ValueError("Cannot embed an empty list of texts.")

        # FlagEmbedding handles batching internally, but we pass batch_size
        # to control memory usage per forward pass.
        output = self.model.encode(
            texts,
            batch_size=self.batch_size,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,  # not needed for our retrieval strategy
        )

        dense_vectors = output["dense_vecs"]     # numpy array: (n, 1024)
        sparse_weights = output["lexical_weights"]  # list of dicts: [{token_str: weight}, ...]

        results = []
        for i in range(len(texts)):
            dense = dense_vectors[i].tolist()
            sparse = self._convert_sparse(sparse_weights[i])
            results.append(EmbeddingResult(dense_vector=dense, sparse_vector=sparse))

        return results

    def embed_single(self, text: str) -> EmbeddingResult:
        """Convenience method for embedding a single text."""
        return self.embed([text])[0]

    def _convert_sparse(self, lexical_weights: dict) -> SparseVector:
        """
        Convert BGE-M3 lexical weights to Qdrant sparse vector format.

        BGE-M3 returns: {"token_string": weight, ...}
        Qdrant expects: SparseVector(indices=[int, ...], values=[float, ...])

        The tokenizer's convert_tokens_to_ids handles the mapping.
        Tokens that map to the unknown token ID are dropped — they
        would pollute the sparse index with meaningless entries.
        """
        indices = []
        values = []

        unk_id = self.tokenizer.unk_token_id

        for token_str, weight in lexical_weights.items():
            token_id = self.tokenizer.convert_tokens_to_ids(token_str)

            # Skip unknown tokens — they'd all map to the same ID,
            # collapsing distinct terms into one meaningless entry.
            if token_id == unk_id:
                continue

            indices.append(int(token_id))
            values.append(float(weight))

        return SparseVector(indices=indices, values=values)

    @property
    def dense_dim(self) -> int:
        """Dimensionality of dense vectors. Used for Qdrant collection config."""
        return 1024

    @property
    def model_name(self) -> str:
        """Model identifier. Useful for logging and collection metadata."""
        return self.model_id