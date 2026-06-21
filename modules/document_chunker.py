# Stage 5: Chunking and Vector Database Ingestion

from dataclasses import dataclass
from pathlib import Path
from docling.chunking import HybridChunker
from transformers import AutoTokenizer
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


EMBED_MODEL_ID = "BAAI/bge-m3"
MAX_TOKENS = 256


# ──────────────────────────────────────────────
# Data Structures
# ──────────────────────────────────────────────

@dataclass
class ChunkingResult:
    """Output of the chunking and ingestion stage."""
    success: bool
    chunk_count: int = 0
    vectorstore: object = None  # Chroma instance when successful
    error_message: str = ""
    collection_name: str = ""
    persist_directory: str = ""


# ──────────────────────────────────────────────
# Step 1: Build Chunker
# ──────────────────────────────────────────────

def build_chunker(tokenizer_id=EMBED_MODEL_ID, max_tokens=MAX_TOKENS):
    """
    Build a HybridChunker using the embedding model's own tokenizer.

    Using the same tokenizer for chunking and embedding ensures the
    max_tokens limit maps exactly to the embedding model's context
    window — no silent truncation during embedding.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    return HybridChunker(
        tokenizer=tokenizer,
        max_tokens=max_tokens,
        merge_peers=True,
    )


# ──────────────────────────────────────────────
# Step 2: Chunk the Corrected Document
# ──────────────────────────────────────────────

def chunk_document(docling_result, chunker):
    """
    Run HybridChunker on the DoclingDocument.

    The document must have Stage 3 corrections already applied
    (heading levels, VLM descriptions, formula text) so that
    chunks inherit the cleaned, enriched content.
    """
    doc = docling_result.document
    return list(chunker.chunk(doc))


# ──────────────────────────────────────────────
# Step 3: Convert Chunks to LangChain Documents
# ──────────────────────────────────────────────

def _extract_page_numbers(chunk):
    """
    Extract page numbers from a chunk's doc_items provenance.
    Returns a sorted list of unique page numbers, or [] if unavailable.
    """
    page_nos = set()

    if not hasattr(chunk.meta, "doc_items"):
        return []

    for item in chunk.meta.doc_items:
        if not hasattr(item, "prov") or not item.prov:
            continue
        prov_list = item.prov if isinstance(item.prov, list) else [item.prov]
        for prov in prov_list:
            if hasattr(prov, "page_no"):
                page_nos.add(prov.page_no)

    return sorted(page_nos)


def chunks_to_langchain_docs(chunks, pdf_path):
    """
    Convert DocChunk objects to LangChain Document objects.

    Metadata stored per chunk:
    - source:        original PDF filename
    - chunk_index:   position in the chunk sequence (for ordering)
    - headings:      section breadcrumb as " > " joined string
    - has_captions:  True if the chunk has associated figure/table captions
    - page_numbers:  comma-separated page numbers (empty string if unknown)

    Chroma only accepts str/int/float/bool metadata values, so
    lists are serialized to strings before storing.
    """
    source_name = Path(pdf_path).name

    docs = []
    for i, chunk in enumerate(chunks):
        headings = chunk.meta.headings or []
        captions = chunk.meta.captions or []
        page_nos = _extract_page_numbers(chunk)

        metadata = {
            "source": source_name,
            "chunk_index": i,
            "headings": " > ".join(headings),
            "has_captions": len(captions) > 0,
            "page_numbers": ",".join(str(p) for p in page_nos),
        }

        docs.append(Document(page_content=chunk.text, metadata=metadata))

    return docs


# ──────────────────────────────────────────────
# Step 4: Embed and Store in Chroma
# ──────────────────────────────────────────────

def get_embeddings(model_id=EMBED_MODEL_ID):
    """Load the HuggingFace embedding model."""
    return HuggingFaceEmbeddings(model_name=model_id)


def build_or_load_vectorstore(persist_directory, collection_name, embeddings):
    """
    Load an existing Chroma collection or create a new one.

    Chroma automatically persists to disk. Calling this on an
    existing directory appends to the collection rather than
    overwriting it — safe for multi-document ingestion.
    """
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_directory),
    )


def ingest_to_vectorstore(langchain_docs, persist_directory, collection_name, embeddings):
    """Embed and add LangChain Documents to the Chroma vector store."""
    vectorstore = build_or_load_vectorstore(persist_directory, collection_name, embeddings)
    vectorstore.add_documents(langchain_docs)
    return vectorstore


# ──────────────────────────────────────────────
# Stage 5: Full Chunking and Ingestion Pipeline
# ──────────────────────────────────────────────

def run_chunking_and_ingestion(
    docling_result,
    pdf_path,
    persist_directory,
    collection_name="StudyAssistant",
    embed_model_id=EMBED_MODEL_ID,
    max_tokens=MAX_TOKENS,
    embeddings=None,
):
    """
    Stage 5: Chunk the processed document and store in Chroma.

    Expects Stage 3 corrections to already be applied to the
    DoclingDocument inside docling_result before this is called.

    Args:
        docling_result:    ConversionResult from Docling (post Stage 3)
        pdf_path:          Path to the original PDF (used for source metadata)
        persist_directory: Directory where Chroma stores its data
        collection_name:   Name of the Chroma collection
        embed_model_id:    HuggingFace model ID for tokenizer and embeddings
        max_tokens:        Max tokens per chunk
        embeddings:        Pre-loaded HuggingFaceEmbeddings instance.
                           Pass this when ingesting multiple documents to
                           avoid reloading the model on every call.

    Returns:
        ChunkingResult with vectorstore, chunk count, and status.
    """
    # Step 1: Build chunker and produce chunks
    try:
        chunker = build_chunker(tokenizer_id=embed_model_id, max_tokens=max_tokens)
        chunks = chunk_document(docling_result, chunker)
    except Exception as e:
        return ChunkingResult(
            success=False,
            error_message=f"Chunking failed: {str(e)}",
        )

    if not chunks:
        return ChunkingResult(
            success=False,
            error_message="No chunks produced — document may be empty after post-processing.",
        )

    # Step 2: Convert to LangChain Documents with metadata
    try:
        langchain_docs = chunks_to_langchain_docs(chunks, pdf_path)
    except Exception as e:
        return ChunkingResult(
            success=False,
            chunk_count=len(chunks),
            error_message=f"Chunk metadata conversion failed: {str(e)}",
        )

    # Step 3: Embed and store
    try:
        if embeddings is None:
            embeddings = get_embeddings(embed_model_id)
        vectorstore = ingest_to_vectorstore(
            langchain_docs, persist_directory, collection_name, embeddings
        )
    except Exception as e:
        return ChunkingResult(
            success=False,
            chunk_count=len(chunks),
            error_message=f"Vector store ingestion failed: {str(e)}",
        )

    return ChunkingResult(
        success=True,
        chunk_count=len(chunks),
        vectorstore=vectorstore,
        collection_name=collection_name,
        persist_directory=str(persist_directory),
    )


# ──────────────────────────────────────────────
# Utility: Print Chunking Summary
# ──────────────────────────────────────────────

def print_chunking_summary(result):
    """Print a readable summary of the chunking and ingestion result."""
    print(f"\n{'='*50}")
    print(f"Chunking & Ingestion Summary")
    print(f"{'='*50}")
    print(f"Success: {result.success}")

    if not result.success:
        print(f"Error: {result.error_message}")
        return

    print(f"Chunks produced: {result.chunk_count}")
    print(f"Collection: {result.collection_name}")
    print(f"Persisted to: {result.persist_directory}")
