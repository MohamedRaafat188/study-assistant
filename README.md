# 📚 Study Assistant — RAG Pipeline

> A production-ready document ingestion and retrieval pipeline that turns any PDF into a searchable, LLM-ready knowledge base — with a step-by-step Streamlit interface to inspect every stage.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5%2B-EE4C2C?logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.58%2B-FF4B4B?logo=streamlit&logoColor=white)
![Qdrant](https://img.shields.io/badge/Qdrant-1.18%2B-DC244C?logo=qdrant&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Table of Contents

- [Overview](#overview)
- [Pipeline Architecture](#pipeline-architecture)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Pipeline Stages In Detail](#pipeline-stages-in-detail)
- [Configuration](#configuration)
- [Output Files](#output-files)
- [License](#license)

---

## Overview

Study Assistant is a complete **Retrieval-Augmented Generation (RAG)** backend designed for academic documents. You drop in a PDF — a research paper, a lecture slide deck, a textbook chapter — and the pipeline intelligently ingests it into a hybrid vector store, then retrieves and re-ranks the most relevant passages for any question you ask.

The Streamlit UI makes the pipeline transparent: each stage runs live in the browser, showing you exactly what the system extracted, flagged, chunked, and retrieved.

---

## Pipeline Architecture

```
PDF File
   │
   ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 1 — Document Classification                          │
│  Classify each page as digital or scanned (PyMuPDF)        │
│  → Determines whether OCR is needed                         │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 2 — Document Parsing                                 │
│  Layout analysis via Docling (tables, headings, formulas,   │
│  images). Exports to structured DoclingDocument + Markdown. │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 3 — Post-Processing                                  │
│  • Heading validation & hierarchy repair (H1–H4)           │
│  • Image–caption and table–caption pairing                  │
│  • Formula region cropping                                  │
│  • VLM image descriptions (Ollama)                         │
│  • Metadata footnote removal                               │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 4 — Quality Gate                                     │
│  Extraction completeness, structural integrity, error rate. │
│  Scores 0–1 → PASS / WARN / REJECT                         │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 5 — Chunking                                         │
│  Docling HybridChunker with BGE-M3 tokenizer               │
│  Max 512 tokens/chunk, heading breadcrumbs prepended        │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 6 — Embedding & Retrieval                            │
│  BAAI/bge-m3 → dense (1024-dim) + sparse (lexical) vectors │
│  Hybrid search in Qdrant (RRF fusion)                       │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 7 — Reranking                                        │
│  BAAI/bge-reranker-v2-m3 cross-encoder rescores candidates │
│  Trims to top-k, reassembles context in document order      │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
              LLM-ready context string
```

---

## Features

- **Hybrid retrieval** — BGE-M3 produces both dense and sparse vectors in a single forward pass; Qdrant fuses them with Reciprocal Rank Fusion for better recall than either alone.
- **Cross-encoder reranking** — A dedicated reranker model re-scores retrieved candidates for high-precision final context.
- **Profile-driven parsing** — OCR is enabled only when scanned pages are detected, keeping inference fast for digital PDFs.
- **Heading hierarchy repair** — Docling returns flat heading labels; post-processing restores H1–H4 structure from section numbering patterns.
- **VLM image descriptions** — Figures are cropped from the PDF and described by a local vision-language model (Ollama), preserving visual knowledge in the retrieval index.
- **Quality gate** — Automated checks on extraction completeness, structural integrity, and error rates prevent low-quality documents from silently degrading retrieval.
- **Transparent Streamlit UI** — Every pipeline stage runs live with expandable details, metrics, and download buttons for all output files.
- **Step-by-step output files** — Each stage writes a human-readable `.txt` report to `test_outputs/` for debugging and inspection.

---

## Tech Stack

| Component | Library | Model / Version |
|---|---|---|
| PDF classification | PyMuPDF (`fitz`) | — |
| Layout analysis | Docling | `2.92+` |
| Embedding | FlagEmbedding | `BAAI/bge-m3` |
| Reranking | FlagEmbedding | `BAAI/bge-reranker-v2-m3` |
| Chunking | Docling `HybridChunker` | BGE-M3 tokenizer |
| Vector store | Qdrant (local file mode) | `qdrant-client 1.18+` |
| Image descriptions | Ollama | `qwen2.5vl:7b` (configurable) |
| UI | Streamlit | `1.58+` |
| Deep learning | PyTorch | `2.5+ (CPU) / 2.11+cu126 (GPU)` |

---

## Project Structure

```
study-assistant/
│
├── app.py                     # Streamlit UI — runs the full pipeline interactively
├── test_pipeline.py           # Headless end-to-end test script
├── requirements.txt
│
├── modules/
│   ├── document_classifier.py # Stage 1 — page-level digital/scanned classification
│   ├── document_parser.py     # Stage 2 — Docling parsing + content profiling
│   ├── post_processor.py      # Stage 3 — heading repair, captions, VLM, quality gate
│   ├── chunker.py             # Stage 5 — HybridChunker wrapper
│   ├── embedder.py            # Stage 6 — BGE-M3 dense + sparse embeddings
│   ├── vector_store.py        # Stage 6 — Qdrant collection management & search
│   ├── point_builder.py       # Stage 6 — maps chunks+embeddings → Qdrant PointData
│   ├── retrieval.py           # Stage 6 — query embedding → search → context assembly
│   └── reranking.py           # Stage 7 — cross-encoder reranking
│
├── small_data/                # Sample PDFs for testing
├── test_outputs/              # Per-stage .txt output files (generated at runtime)
└── demo.ipynb                 # Exploratory notebook
```

---

## Installation

### Prerequisites

- Python 3.10 or higher
- A CUDA-capable GPU is strongly recommended (the embedding and reranking models are ~3.5 GB combined). CPU inference works but is significantly slower.
- [Ollama](https://ollama.com) installed locally if you want VLM image descriptions.

### 1 — Clone the repository

```bash
git clone https://github.com/MohamedRaafat188/study-assistant.git
cd study-assistant
```

### 2 — Create a virtual environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 3 — Install PyTorch

Install the build that matches your hardware **before** running `pip install -r requirements.txt`.

**CPU only:**
```bash
pip install torch>=2.5.0
```

**CUDA 12.6 (GPU — recommended):**
```bash
pip install torch>=2.11.0 --index-url https://download.pytorch.org/whl/cu126
```

For other CUDA versions, see the [PyTorch install selector](https://pytorch.org/get-started/locally/).

### 4 — Install dependencies

```bash
pip install -r requirements.txt
```

### 5 — (Optional) Set up Ollama for image descriptions

Install Ollama from [ollama.com](https://ollama.com), then pull a vision model:

```bash
ollama pull qwen2.5vl:7b
```

> If Ollama is not running, the pipeline still completes — the image description step falls back to the figure caption text.

### First-run model downloads

On the first run, the following models are downloaded automatically and cached:

| Model | Size | Purpose |
|---|---|---|
| `BAAI/bge-m3` | ~2.3 GB | Dense + sparse embeddings |
| `BAAI/bge-reranker-v2-m3` | ~1.1 GB | Cross-encoder reranking |
| Docling layout models | ~1–2 GB | PDF layout analysis |

---

## Usage

### Streamlit UI (recommended)

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser, then:

1. Upload a PDF from the sidebar.
2. Type a question in the **Retrieval Query** box.
3. Adjust the retrieval and reranking sliders if needed.
4. Click **▶ Run Pipeline**.

Each stage expands as it runs, showing live metrics and detailed results. When the pipeline finishes, download buttons appear for all seven output files.

### Headless script

```bash
python test_pipeline.py
```

Edit the config block at the top of the file to set your PDF path and query before running:

```python
PDF_PATH       = "path/to/your/document.pdf"
DOCUMENT_TITLE = "My Document"
QUERY          = "What is the main topic?"
TOP_K_RETRIEVE = 20
TOP_K_RERANK   = 5
```

Output files are written to `test_outputs/`.

---

## Pipeline Stages In Detail

### Stage 1 — Document Classification

Uses PyMuPDF to extract raw text from each page and measure character density. Pages below a configurable threshold (default: 50 chars) are flagged as **scanned**, enabling OCR only where needed.

### Stage 2 — Document Parsing

Runs [Docling](https://github.com/DS4SD/docling) with configuration driven by Stage 1 results. Detects and structures text blocks, headings, tables, formulas, figures, footnotes, and captions. Exports a `DoclingDocument` and Markdown.

### Stage 3 — Post-Processing

A 7-step correction pipeline on top of Docling's output:
1. **Heading validation** — demotes misclassified headings using numbering pattern analysis and length heuristics.
2. **Hierarchy repair** — assigns H1–H4 levels based on section number dot-depth.
3. **Image–caption pairing** — links figures to their captions by forward-scanning element order.
4. **Table–caption pairing** — same logic for tables.
5. **Formula extraction** — crops formula regions from the original PDF for downstream math OCR.
6. **Image description** — sends cropped figures to a local VLM (Ollama) for text descriptions that enter the retrieval index.
7. **Footnote handling** — removes author/affiliation metadata footnotes while preserving content footnotes.

### Stage 4 — Quality Gate

Runs four automated checks and produces a quality score (0–1):

| Check | Hard / Soft |
|---|---|
| Extraction completeness (Docling chars vs raw PDF chars) | Hard if < 30 %, Soft if < 60 % |
| Structural integrity (headings per page, demotion rate) | Soft |
| Formula / image extraction errors | Soft |
| Digital pages with unexpectedly low text content | Soft |

Outcome thresholds: **≥ 0.7** → pass, **≥ 0.4** → warn, **< 0.4** → reject.

### Stage 5 — Chunking

Wraps Docling's `HybridChunker` with the BGE-M3 tokenizer. Each chunk carries a **contextualized text** variant — the raw text prepended with heading breadcrumbs — which is what gets embedded. This gives the embedding model full section context without losing retrieval granularity.

### Stage 6 — Embedding & Retrieval

`BAAI/bge-m3` is a single model that produces both a dense vector (1024-dim cosine) and a sparse vector (learned lexical weights, BM25-like) in one pass. Both are stored in Qdrant. At query time, **hybrid search** prefetches candidates from each index independently and fuses the ranked lists with RRF.

### Stage 7 — Reranking

`BAAI/bge-reranker-v2-m3` is a cross-encoder: it sees the full `(query, chunk)` pair and scores it holistically rather than comparing independent embeddings. The top-k survivors are sorted by reranker score for the results view, then re-ordered by document position before being assembled into the final context string — preserving natural reading flow for the LLM.

---

## Configuration

All tuneable parameters live in the sidebar of `app.py` (UI) or the config block at the top of `test_pipeline.py` (script).

| Parameter | Default | Description |
|---|---|---|
| `TOP_K_RETRIEVE` | 20 | Candidate chunks fetched from Qdrant before reranking |
| `TOP_K_RERANK` | 5 | Final chunks kept after the cross-encoder |
| `max_tokens` (chunker) | 512 | Max tokens per chunk (coupled to BGE-M3 context window) |
| `use_fp16` (embedder / reranker) | `True` | Half-precision inference — halves VRAM with negligible quality loss |
| VLM model (post-processor) | `qwen2.5vl:7b` | Any Ollama vision model |

---

## Output Files

Running the pipeline writes one `.txt` file per stage to `test_outputs/`:

| File | Contents |
|---|---|
| `step1_classification.txt` | Per-page origin, char count, density |
| `step2_parse.txt` | Content profile + full Markdown export |
| `step3_post_processing.txt` | Item-level post-processing results |
| `step4_quality_gate.txt` | Score, outcome, and all warnings |
| `step5_chunks.txt` | Every chunk with headings, pages, and text |
| `step6_retrieval.txt` | Ranked retrieval results + assembled context |
| `step7_reranking.txt` | Reranker scores + final context |

---

## License

This project is licensed under the MIT License.
