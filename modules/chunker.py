"""
chunker.py — Document chunking module for the RAG pipeline.

Wraps Docling's HybridChunker with project-specific configuration
(tokenizer, max_tokens, merge_peers) tied to the embedding model.
Takes a DoclingDocument and returns structured chunks ready for
embedding.

Usage:
    chunker = Chunker()
    chunks = chunker.chunk(docling_document)
    texts = chunker.get_contextualized_texts(chunks)
"""

from dataclasses import dataclass

from docling.chunking import HybridChunker
from docling_core.types.doc.document import DoclingDocument
from transformers import AutoTokenizer


# ──────────────────────────────────────────────
# Data Structures
# ──────────────────────────────────────────────

@dataclass
class ChunkMetadata:
    """Extracted metadata from a single chunk, independent of Docling internals."""
    headings: list[str]
    page_numbers: list[int]
    content_type: str
    chunk_index: int


@dataclass
class ChunkData:
    """A chunk with its text representations and extracted metadata."""
    raw_text: str
    contextualized_text: str
    metadata: ChunkMetadata


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

# Tied to the embedding model — these values are coupled constraints.
# Changing the embedding model may require changing these.
DEFAULT_MODEL_ID = "BAAI/bge-m3"
DEFAULT_MAX_TOKENS = 256
DEFAULT_MERGE_PEERS = True


# ──────────────────────────────────────────────
# Chunker
# ──────────────────────────────────────────────

class Chunker:
    """
    Document chunker wrapping Docling's HybridChunker.

    Configuration is tied to the embedding model:
    - Tokenizer must match the embedding model
    - max_tokens must respect the model's context window
    - merge_peers controls whether adjacent same-level elements merge

    Parameters
    ----------
    model_id : str
        Hugging Face model ID for the tokenizer. Must match the
        embedding model. Default: BAAI/bge-m3.
    max_tokens : int
        Maximum tokens per chunk. Should be well below the embedding
        model's context window to avoid dilution. Default: 512.
    merge_peers : bool
        Whether to merge adjacent same-level elements into one chunk.
        Produces more coherent chunks. Default: True.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        merge_peers: bool = DEFAULT_MERGE_PEERS,
    ):
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.merge_peers = merge_peers

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.hybrid_chunker = HybridChunker(
            tokenizer=self.tokenizer,
            max_tokens=max_tokens,
            merge_peers=merge_peers,
        )

    def chunk(self, document: DoclingDocument) -> list[ChunkData]:
        """
        Chunk a DoclingDocument into retrieval-ready units.

        Returns ChunkData objects containing:
        - raw_text: the chunk's text without heading prefixes
        - contextualized_text: text with heading breadcrumbs prepended
          (used for embedding — gives the model section context)
        - metadata: extracted headings, page numbers, content type, index

        Parameters
        ----------
        document : DoclingDocument
            The processed document (after post-processing and corrections).

        Returns
        -------
        list[ChunkData]
            Ordered list of chunks with metadata.
        """
        raw_chunks = list(self.hybrid_chunker.chunk(document))

        chunk_data_list = []
        for i, raw_chunk in enumerate(raw_chunks):
            metadata = self._extract_metadata(raw_chunk, i)
            chunk_data_list.append(
                ChunkData(
                    raw_text=raw_chunk.text,
                    contextualized_text=self.hybrid_chunker.contextualize(raw_chunk),
                    metadata=metadata,
                )
            )

        return chunk_data_list

    def _extract_metadata(self, raw_chunk, index: int) -> ChunkMetadata:
        """
        Extract structured metadata from a Docling chunk.

        Isolates the Docling-specific metadata access so downstream
        modules don't need to know about Docling's internal types.
        """
        meta = raw_chunk.meta

        # Extract headings
        headings = []
        if hasattr(meta, "headings") and meta.headings:
            headings = list(meta.headings)

        # Extract page numbers from provenance data
        page_numbers = []
        if hasattr(meta, "doc_items"):
            for doc_item in meta.doc_items:
                if hasattr(doc_item, "prov") and doc_item.prov:
                    for prov in doc_item.prov:
                        if hasattr(prov, "page_no") and prov.page_no not in page_numbers:
                            page_numbers.append(prov.page_no)

        # Determine content type from the first doc_item's label
        content_type = "text"
        if hasattr(meta, "doc_items") and meta.doc_items:
            first_item = meta.doc_items[0]
            if hasattr(first_item, "label"):
                content_type = str(first_item.label)

        return ChunkMetadata(
            headings=headings,
            page_numbers=sorted(page_numbers),
            content_type=content_type,
            chunk_index=index,
        )
