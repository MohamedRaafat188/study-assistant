"""
app.py — Streamlit UI for the Study Assistant RAG pipeline.

Run with: streamlit run app.py
"""

import sys
import os
import tempfile
import shutil
import uuid
from pathlib import Path

import streamlit as st

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODULES_DIR = os.path.join(ROOT_DIR, "modules")
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, MODULES_DIR)

from modules.document_classifier import classify_document
from modules.document_parser import parse_document
from modules.point_builder import PointBuilder
from modules.post_processor import run_post_processing, run_quality_gate, get_active_items
from modules.chunker import Chunker
from modules.embedder import Embedder
from modules.vector_store import VectorStore
from modules.retrieval import Retriever
from modules.reranking import Reranker

OUTPUT_DIR = Path("test_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def write_output(filename: str, lines: list[str]) -> Path:
    path = OUTPUT_DIR / filename
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def section_header(title: str) -> str:
    bar = "=" * 60
    return f"{bar}\n{title}\n{bar}"


# ── Cached model loaders ──────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading embedding model (BAAI/bge-m3)...")
def load_embedder() -> Embedder:
    return Embedder()


@st.cache_resource(show_spinner="Loading reranker (BAAI/bge-reranker-v2-m3)...")
def load_reranker() -> Reranker:
    return Reranker()


@st.cache_resource(show_spinner="Loading chunker tokenizer...")
def load_chunker() -> Chunker:
    return Chunker()


# ── Page layout ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Study Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Study Assistant — RAG Pipeline")
st.caption(
    "Upload a PDF to watch the full ingestion pipeline run step by step. "
    "Each step's output is saved to a `.txt` file in `test_outputs/`."
)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Configuration")

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        help="The document to process.",
    )

    query = st.text_area(
        "Retrieval Query",
        value="What is the main topic discussed in this document?",
        height=80,
        help="The question used to test retrieval and reranking at the end.",
    )

    st.divider()

    top_k_retrieve = st.slider(
        "Candidates to retrieve",
        min_value=5, max_value=50, value=20, step=5,
        help="Number of chunks fetched before reranking.",
    )
    top_k_rerank = st.slider(
        "Final results after reranking",
        min_value=1, max_value=10, value=5,
        help="Chunks kept after the cross-encoder re-scores.",
    )

    st.divider()

    run_btn = st.button(
        "▶ Run Pipeline",
        type="primary",
        use_container_width=True,
        disabled=not uploaded_file,
    )

    if not uploaded_file:
        st.info("Upload a PDF above to enable the pipeline.")


# ── Pipeline ──────────────────────────────────────────────────────────────────

if run_btn and uploaded_file:

    # Save uploaded bytes to a temp file Docling/fitz can read
    tmp_dir = tempfile.mkdtemp()
    pdf_path = os.path.join(tmp_dir, uploaded_file.name)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    document_id   = str(uuid.uuid4())[:8]
    document_title = Path(uploaded_file.name).stem
    collection_name = f"pipeline_{document_id}"
    qdrant_path   = os.path.join(tmp_dir, "qdrant")

    # ── Step 1: Classification ────────────────────────────────────────────────

    with st.status("**Step 1 — Document Classification**", expanded=True) as s1:
        try:
            profile = classify_document(pdf_path)

            digital = sum(1 for p in profile.page_classifications if p.origin == "digital")
            scanned = sum(1 for p in profile.page_classifications if p.origin == "scanned")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Pages", profile.total_pages)
            c2.metric("Digital", digital)
            c3.metric("Scanned", scanned)
            c4.metric("Needs OCR", "Yes" if profile.needs_ocr else "No")

            with st.expander("Page-by-page breakdown"):
                st.dataframe(
                    [
                        {
                            "Page":    pc.page_number,
                            "Origin":  pc.origin,
                            "Chars":   pc.char_count,
                            "Density": round(pc.text_density, 4),
                        }
                        for pc in profile.page_classifications
                    ],
                    use_container_width=True,
                )

            lines = [section_header("Stage 1: Document Classification"), ""]
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

            s1.update(label="**Step 1 — Document Classification** ✅", state="complete")

        except Exception as e:
            st.error(f"{e}")
            s1.update(label="**Step 1 — Document Classification** ❌", state="error")
            shutil.rmtree(tmp_dir, ignore_errors=True)
            st.stop()

    # ── Step 2: Parsing ───────────────────────────────────────────────────────

    with st.status("**Step 2 — Document Parsing (Docling)**", expanded=True) as s2:
        try:
            st.write("Running Docling layout analysis… this may take a minute.")
            parse_result = parse_document(pdf_path, profile)

            if not parse_result.success:
                st.error(parse_result.error_message)
                s2.update(label="**Step 2 — Document Parsing** ❌", state="error")
                shutil.rmtree(tmp_dir, ignore_errors=True)
                st.stop()

            cs = parse_result.profile.content_summary

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Content Type", cs.content_type)
            c2.metric("Headings",     cs.total_headings)
            c3.metric("Tables",       cs.total_tables)
            c4.metric("Formulas",     cs.total_formulas)
            c5.metric("Images",       cs.total_images)

            with st.expander("Markdown preview (first 3 000 chars)"):
                st.markdown(parse_result.markdown[:3000])

            lines = [section_header("Stage 2: Document Parsing"), ""]
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

            s2.update(label="**Step 2 — Document Parsing** ✅", state="complete")

        except Exception as e:
            st.error(f"{e}")
            s2.update(label="**Step 2 — Document Parsing** ❌", state="error")
            shutil.rmtree(tmp_dir, ignore_errors=True)
            st.stop()

    # ── Step 3: Post-Processing ───────────────────────────────────────────────

    with st.status("**Step 3 — Post-Processing**", expanded=True) as s3:
        try:
            items, formula_data, image_data, pp_summary = run_post_processing(
                parse_result.docling_result, pdf_path
            )
            active_items = get_active_items(items)

            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("Active Items",      pp_summary["active_items"])
            c2.metric("Headings Demoted",  pp_summary["headings_demoted"])
            c3.metric("Images Paired",     pp_summary["images_paired"])
            c4.metric("Tables Paired",     pp_summary["tables_paired"])
            c5.metric("Formulas",          pp_summary["formulas_extracted"])
            c6.metric("Footnotes Removed", pp_summary["footnotes_removed"])

            h = pp_summary.get("hierarchy", {})
            if h:
                col_l, col_r = st.columns(2)
                with col_l:
                    st.write("**Heading Hierarchy**")
                    for level, count in h.items():
                        st.write(f"  {level}: {count}")
                with col_r:
                    st.write("**VLM Image Descriptions**")
                    st.write(f"  Generated: {pp_summary.get('descriptions_generated', 0)}")
                    st.write(f"  Errors: {pp_summary.get('description_errors', 0)}")

            with st.expander(f"Active items ({len(active_items)} total — showing first 60)"):
                for item in active_items[:60]:
                    level_tag = f" [H{item.heading_level}]" if item.heading_level else ""
                    page_tag  = f" p.{item.page_number}"    if item.page_number   else ""
                    preview   = item.text[:120].replace("\n", " ")
                    st.text(f"{item.label}{level_tag}{page_tag}: {preview!r}")
                if len(active_items) > 60:
                    st.caption(f"… and {len(active_items) - 60} more items in the txt file.")

            lines = [section_header("Stage 3: Post-Processing"), ""]
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
                page_tag  = f" p.{item.page_number}"    if item.page_number   else ""
                preview   = item.text[:120].replace("\n", " ")
                lines.append(f"  [{i:>4}] {item.label}{level_tag}{page_tag}: {preview!r}")
            write_output("step3_post_processing.txt", lines)

            s3.update(label="**Step 3 — Post-Processing** ✅", state="complete")

        except Exception as e:
            st.error(f"{e}")
            s3.update(label="**Step 3 — Post-Processing** ❌", state="error")
            shutil.rmtree(tmp_dir, ignore_errors=True)
            st.stop()

    # ── Step 4: Quality Gate ──────────────────────────────────────────────────

    with st.status("**Step 4 — Quality Gate**", expanded=True) as s4:
        try:
            quality_result = run_quality_gate(
                pdf_path, parse_result.profile, pp_summary, items
            )

            outcome_icon = {"pass": "🟢", "warn": "🟡", "reject": "🔴"}.get(
                quality_result.outcome, "⚪"
            )

            c1, c2 = st.columns(2)
            c1.metric("Outcome", f"{outcome_icon} {quality_result.outcome.upper()}")
            c2.metric("Quality Score", f"{quality_result.score:.2f} / 1.00")

            if quality_result.warnings:
                for w in quality_result.warnings:
                    icon = "🔴" if w.severity == "hard" else "🟡"
                    st.warning(f"{icon} **{w.check_name}**: {w.message}")
            else:
                st.success("No warnings — clean extraction.")

            lines = [section_header("Stage 4: Quality Gate"), ""]
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
                st.error("Quality gate **REJECTED** this document. Pipeline stopped.")
                s4.update(label="**Step 4 — Quality Gate** ❌ REJECTED", state="error")
                shutil.rmtree(tmp_dir, ignore_errors=True)
                st.stop()

            s4.update(label="**Step 4 — Quality Gate** ✅", state="complete")

        except Exception as e:
            st.error(f"{e}")
            s4.update(label="**Step 4 — Quality Gate** ❌", state="error")
            shutil.rmtree(tmp_dir, ignore_errors=True)
            st.stop()

    # ── Step 5: Chunking ──────────────────────────────────────────────────────

    with st.status("**Step 5 — Chunking**", expanded=True) as s5:
        try:
            chunker = load_chunker()
            chunks  = chunker.chunk(parse_result.docling_result.document)

            c1, c2 = st.columns(2)
            c1.metric("Total Chunks",   len(chunks))
            c2.metric("Max Tokens/Chunk", chunker.max_tokens)

            with st.expander("Chunk details"):
                for chunk in chunks:
                    m = chunk.metadata
                    st.markdown(
                        f"**Chunk {m.chunk_index + 1}** &nbsp;|&nbsp; "
                        f"Pages `{m.page_numbers}` &nbsp;|&nbsp; "
                        f"Type `{m.content_type}` &nbsp;|&nbsp; "
                        f"Headings `{m.headings}`"
                    )
                    st.caption(chunk.contextualized_text[:250].replace("\n", " "))
                    st.divider()

            lines = [section_header("Stage 5: Chunking"), ""]
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

            s5.update(label="**Step 5 — Chunking** ✅", state="complete")

        except Exception as e:
            st.error(f"{e}")
            s5.update(label="**Step 5 — Chunking** ❌", state="error")
            shutil.rmtree(tmp_dir, ignore_errors=True)
            st.stop()

    # ── Step 6: Embedding + Retrieval ─────────────────────────────────────────

    with st.status("**Step 6 — Embedding & Retrieval**", expanded=True) as s6:
        try:
            embedder = load_embedder()

            st.write(f"Embedding {len(chunks)} chunks with BGE-M3…")
            contextualized_texts = [chunk.contextualized_text for chunk in chunks]
            embeddings = embedder.embed(contextualized_texts)

            st.write("Storing vectors in Qdrant…")
            store = VectorStore(path=qdrant_path)
            if store.collection_exists(collection_name):
                store.delete_collection(collection_name)
            store.create_collection(collection_name, dense_dim=embedder.dense_dim)

            builder = PointBuilder(document_id=document_id, document_title=document_title)
            points  = builder.build_points(chunks, embeddings)
            store.upsert_points(collection_name, points)

            st.write(f"Retrieving top {top_k_retrieve} candidates for query…")
            retriever = Retriever(
                embedder=embedder,
                vector_store=store,
                collection_name=collection_name,
            )
            retrieval_result = retriever.retrieve(
                query=query,
                limit=top_k_retrieve,
                prefetch_limit=top_k_retrieve * 4,
            )

            st.metric("Chunks Retrieved", retrieval_result.total_chunks_found)
            st.caption(f"Search mode: `{retrieval_result.search_mode.value}`")

            with st.expander("Retrieval results"):
                for i, chunk in enumerate(retrieval_result.chunks, 1):
                    st.markdown(
                        f"**#{i}** &nbsp; Score `{chunk.score:.4f}` &nbsp;|&nbsp; "
                        f"Pages `{chunk.page_numbers}` &nbsp;|&nbsp; "
                        f"Headings `{chunk.headings}`"
                    )
                    st.caption(chunk.chunk_text[:250].replace("\n", " "))
                    st.divider()

            lines = [section_header("Stage 6: Retrieval"), ""]
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

            s6.update(label="**Step 6 — Embedding & Retrieval** ✅", state="complete")

        except Exception as e:
            st.error(f"{e}")
            s6.update(label="**Step 6 — Embedding & Retrieval** ❌", state="error")
            shutil.rmtree(tmp_dir, ignore_errors=True)
            st.stop()

    # ── Step 7: Reranking ─────────────────────────────────────────────────────

    with st.status("**Step 7 — Reranking**", expanded=True) as s7:
        try:
            reranker = load_reranker()
            reranked_result = reranker.rerank(retrieval_result, top_k=top_k_rerank)

            c1, c2 = st.columns(2)
            c1.metric("Candidates In",  retrieval_result.total_chunks_found)
            c2.metric("Top-k Kept",     reranked_result.total_chunks_found)

            with st.expander("Reranked results (sorted by cross-encoder score)"):
                for i, chunk in enumerate(reranked_result.chunks, 1):
                    st.markdown(
                        f"**Rank {i}** &nbsp; Score `{chunk.score:.4f}` &nbsp;|&nbsp; "
                        f"Pages `{chunk.page_numbers}` &nbsp;|&nbsp; "
                        f"Headings `{chunk.headings}`"
                    )
                    st.caption(chunk.chunk_text[:300].replace("\n", " "))
                    st.divider()

            st.subheader("Final Context (document position order)")
            st.text_area(
                label="Ready to pass to an LLM",
                value=reranked_result.context_text,
                height=350,
            )

            lines = [section_header("Stage 7: Reranking"), ""]
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

            s7.update(label="**Step 7 — Reranking** ✅", state="complete")

        except Exception as e:
            st.error(f"{e}")
            s7.update(label="**Step 7 — Reranking** ❌", state="error")
            shutil.rmtree(tmp_dir, ignore_errors=True)
            st.stop()

    # ── Cleanup & download links ───────────────────────────────────────────────

    try:
        store.delete_collection(collection_name)
        store.client.close()
    except Exception:
        pass
    shutil.rmtree(tmp_dir, ignore_errors=True)

    st.success("✅ Pipeline complete!")

    st.subheader("📄 Download Output Files")
    output_files = [
        "step1_classification.txt",
        "step2_parse.txt",
        "step3_post_processing.txt",
        "step4_quality_gate.txt",
        "step5_chunks.txt",
        "step6_retrieval.txt",
        "step7_reranking.txt",
    ]
    cols = st.columns(len(output_files))
    for col, fname in zip(cols, output_files):
        fpath = OUTPUT_DIR / fname
        if fpath.exists():
            col.download_button(
                label=fname.replace("step", "Step ").replace("_", " ").replace(".txt", ""),
                data=fpath.read_bytes(),
                file_name=fname,
                mime="text/plain",
            )
