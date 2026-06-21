"""
test_pipeline.py — End-to-end pipeline test: PDF → classification → parse →
post-processing → quality gate → chunking → retrieval → reranking.

Each stage writes its output to a separate .txt file in test_outputs/.
Set PDF_PATH before running.
"""

import sys
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODULES_DIR = os.path.join(ROOT_DIR, "modules")
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, MODULES_DIR)

from pathlib import Path

# document_parser and point_builder use relative imports so they need the package path.
# Everything else is imported flat to stay consistent with their internal imports.
from modules.document_classifier import classify_document
from modules.document_parser import parse_document
from modules.point_builder import PointBuilder
from modules.post_processor import run_post_processing, run_quality_gate, get_active_items
from modules.chunker import Chunker
from modules.embedder import Embedder
from modules.vector_store import VectorStore
from modules.retrieval import Retriever
from modules.reranking import Reranker


# ── Config ───────────────────────────────────────────────────────────────────

PDF_PATH = r"small_data\attention is all you need.pdf"   # <-- set before running
DOCUMENT_ID = "test_doc_001"
DOCUMENT_TITLE = "Test Document"
COLLECTION = "test_pipeline_collection"
QUERY = "explain the attention mechanism in transformers"
TOP_K_RETRIEVE = 20
TOP_K_RERANK = 5

OUTPUT_DIR = Path("test_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def write_output(filename: str, lines: list[str]) -> None:
    path = OUTPUT_DIR / filename
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  -> {path}")


def section(title: str) -> str:
    bar = "=" * 60
    return f"{bar}\n{title}\n{bar}"


# ── Stage 1: Document Classification ─────────────────────────────────────────

print("\n[Stage 1] Classifying document...")
profile = classify_document(PDF_PATH)

digital = sum(1 for pc in profile.page_classifications if pc.origin == "digital")
scanned = sum(1 for pc in profile.page_classifications if pc.origin == "scanned")

lines = [section("Stage 1: Document Classification"), ""]
lines += [
    f"File:        {profile.filename}",
    f"Total pages: {profile.total_pages}",
    f"Needs OCR:   {profile.needs_ocr}",
    f"Summary:     {digital} digital, {scanned} scanned",
    "",
    "Page Classifications:",
]
for pc in profile.page_classifications:
    lines.append(
        f"  Page {pc.page_number:>3}: {pc.origin:<8}  "
        f"chars={pc.char_count:>6}  density={pc.text_density:.4f}"
    )

write_output("step1_classification.txt", lines)


# ── Stage 2: Parsing ──────────────────────────────────────────────────────────

print("[Stage 2] Parsing document with Docling...")
parse_result = parse_document(PDF_PATH, profile)

if not parse_result.success:
    print(f"  ERROR: {parse_result.error_message}")
    sys.exit(1)

cs = parse_result.profile.content_summary

lines = [section("Stage 2: Document Parsing"), ""]
lines += [
    "Status: SUCCESS",
    f"Markdown length: {len(parse_result.markdown)} chars",
    "",
    "Content Profile:",
    f"  Type:     {cs.content_type}",
    f"  Headings: {cs.total_headings}",
    f"  Tables:   {cs.total_tables}",
    f"  Formulas: {cs.total_formulas}",
    f"  Images:   {cs.total_images}",
    f"  Needs formula processing: {parse_result.profile.needs_formula_processing}",
    f"  Needs image processing:   {parse_result.profile.needs_image_processing}",
    "",
    "-" * 60,
    "Markdown Output:",
    "-" * 60,
    "",
    parse_result.markdown,
]

write_output("step2_parse.txt", lines)


# ── Stage 3: Post-Processing ──────────────────────────────────────────────────

print("[Stage 3] Running post-processing pipeline...")
items, formula_data, image_data, pp_summary = run_post_processing(
    parse_result.docling_result, PDF_PATH
)

active_items = get_active_items(items)

lines = [section("Stage 3: Post-Processing"), ""]
lines += [
    f"Total items:  {pp_summary['total_items']}",
    f"Active items: {pp_summary['active_items']}",
    "",
    "Headings:",
    f"  Demoted:   {pp_summary['headings_demoted']}",
    f"  Hierarchy: {pp_summary['hierarchy']}",
    "",
    "Images:",
    f"  Paired with captions: {pp_summary['images_paired']}",
    f"  Unpaired:             {pp_summary['unpaired_images']}",
    f"  VLM descriptions:     {pp_summary.get('descriptions_generated', 0)}",
    f"  Crop errors:          {pp_summary['image_errors']}",
    "",
    "Tables:",
    f"  Paired with captions: {pp_summary['tables_paired']}",
    f"  Unpaired:             {pp_summary['unpaired_tables']}",
    "",
    "Formulas:",
    f"  Extracted: {pp_summary['formulas_extracted']}",
    f"  Errors:    {pp_summary['formula_errors']}",
    "",
    "Footnotes:",
    f"  Kept (content):     {pp_summary['footnotes_kept']}",
    f"  Removed (metadata): {pp_summary['footnotes_removed']}",
    "",
    "-" * 60,
    f"Active Items ({len(active_items)} total):",
    "-" * 60,
]
for i, item in enumerate(active_items, 1):
    level_tag = f" [H{item.heading_level}]" if item.heading_level else ""
    page_tag = f" p.{item.page_number}" if item.page_number else ""
    preview = item.text[:120].replace("\n", " ")
    lines.append(f"  [{i:>4}] {item.label}{level_tag}{page_tag}: {preview!r}")

write_output("step3_post_processing.txt", lines)


# ── Stage 4: Quality Gate ─────────────────────────────────────────────────────

print("[Stage 4] Running quality gate...")
quality_result = run_quality_gate(PDF_PATH, parse_result.profile, pp_summary, items)

lines = [section("Stage 4: Quality Gate"), ""]
lines += [
    f"Outcome: {quality_result.outcome.upper()}",
    f"Score:   {quality_result.score}",
    f"Passed:  {quality_result.passed}",
    "",
    "Stats:",
]
for k, v in quality_result.stats.items():
    lines.append(f"  {k}: {v}")
lines.append("")
if quality_result.warnings:
    lines.append(f"Warnings ({len(quality_result.warnings)}):")
    for w in quality_result.warnings:
        tag = "[HARD]" if w.severity == "hard" else "[SOFT]"
        lines.append(f"  {tag} {w.check_name}: {w.message}")
        if w.details:
            lines.append(f"        Details: {w.details}")
else:
    lines.append("No warnings — clean extraction.")

write_output("step4_quality_gate.txt", lines)

if quality_result.outcome == "reject":
    print("  Quality gate REJECTED the document. Stopping.")
    sys.exit(1)


# ── Stage 5: Chunking ─────────────────────────────────────────────────────────

print("[Stage 5] Chunking document...")
chunker = Chunker()
chunks = chunker.chunk(parse_result.docling_result.document)

lines = [section("Stage 5: Chunking"), ""]
lines += [
    f"Total chunks:         {len(chunks)}",
    f"Max tokens per chunk: {chunker.max_tokens}",
    "",
]
for chunk in chunks:
    m = chunk.metadata
    lines += [
        f"Chunk {m.chunk_index + 1}:",
        f"  Headings:     {m.headings}",
        f"  Pages:        {m.page_numbers}",
        f"  Content type: {m.content_type}",
        f"  Raw text ({len(chunk.raw_text)} chars):",
        f"    {chunk.raw_text[:300].replace(chr(10), ' ')!r}",
        f"  Contextualized text ({len(chunk.contextualized_text)} chars):",
        f"    {chunk.contextualized_text[:300].replace(chr(10), ' ')!r}",
        "",
    ]

write_output("step5_chunks.txt", lines)


# ── Stage 6: Embedding + Storing + Retrieval ──────────────────────────────────

print("[Stage 6] Embedding and storing in Qdrant...")
embedder = Embedder()
contextualized_texts = [chunk.contextualized_text for chunk in chunks]
embeddings = embedder.embed(contextualized_texts)

store = VectorStore(path="./test_pipeline_qdrant")
if store.collection_exists(COLLECTION):
    store.delete_collection(COLLECTION)
store.create_collection(COLLECTION, dense_dim=embedder.dense_dim)

builder = PointBuilder(document_id=DOCUMENT_ID, document_title=DOCUMENT_TITLE)
points = builder.build_points(chunks, embeddings)
store.upsert_points(COLLECTION, points)

print(f"  Stored {len(points)} points. Retrieving...")
retriever = Retriever(embedder=embedder, vector_store=store, collection_name=COLLECTION)
retrieval_result = retriever.retrieve(
    query=QUERY,
    limit=TOP_K_RETRIEVE,
    prefetch_limit=TOP_K_RETRIEVE * 4,
)

lines = [section("Stage 6: Retrieval"), ""]
lines += [
    f"Query:       {retrieval_result.query!r}",
    f"Search mode: {retrieval_result.search_mode.value}",
    f"Results:     {retrieval_result.total_chunks_found}",
    "",
]
for i, chunk in enumerate(retrieval_result.chunks, 1):
    lines += [
        f"Result {i}:",
        f"  Score:    {chunk.score:.4f}",
        f"  Source:   {chunk.document_title!r}",
        f"  Pages:    {chunk.page_numbers}",
        f"  Headings: {chunk.headings}",
        f"  Text ({len(chunk.chunk_text)} chars):",
        f"    {chunk.chunk_text[:200].replace(chr(10), ' ')!r}",
        "",
    ]
lines += [
    "-" * 60,
    "Assembled Context:",
    "-" * 60,
    "",
    retrieval_result.context_text,
]

write_output("step6_retrieval.txt", lines)


# ── Stage 7: Reranking ────────────────────────────────────────────────────────

print(f"[Stage 7] Reranking {TOP_K_RETRIEVE} candidates → top {TOP_K_RERANK}...")
reranker = Reranker()
reranked_result = reranker.rerank(retrieval_result, top_k=TOP_K_RERANK)

lines = [section("Stage 7: Reranking"), ""]
lines += [
    f"Query:          {reranked_result.query!r}",
    f"Input chunks:   {retrieval_result.total_chunks_found}",
    f"Reranked top-k: {reranked_result.total_chunks_found}",
    "",
]
for i, chunk in enumerate(reranked_result.chunks, 1):
    lines += [
        f"Rank {i}:",
        f"  Reranker score: {chunk.score:.4f}",
        f"  Source:         {chunk.document_title!r}",
        f"  Pages:          {chunk.page_numbers}",
        f"  Headings:       {chunk.headings}",
        f"  Text ({len(chunk.chunk_text)} chars):",
        f"    {chunk.chunk_text[:200].replace(chr(10), ' ')!r}",
        "",
    ]
lines += [
    "-" * 60,
    "Reassembled Context (document position order):",
    "-" * 60,
    "",
    reranked_result.context_text,
]

write_output("step7_reranking.txt", lines)


# ── Cleanup ───────────────────────────────────────────────────────────────────

store.delete_collection(COLLECTION)
store.client.close()

print("\nDone. Output files:")
for f in sorted(OUTPUT_DIR.iterdir()):
    print(f"  {f.name}")
