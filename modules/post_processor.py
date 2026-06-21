# Stage 3: Post-Processing Pipeline

from dataclasses import dataclass, field
from typing import Optional
import re
import fitz  # Add this to imports at the top


# ──────────────────────────────────────────────
# Step 1: Heading Validation
# ──────────────────────────────────────────────

# ──────────────────────────────────────────────
# Data Structures
# ──────────────────────────────────────────────

@dataclass
class ProcessedItem:
    """
    Wrapper around a Docling item that holds both original
    data and any corrections made during post-processing.

    The original Docling item is preserved so we can always
    access bounding boxes, confidence scores, and other
    raw data when needed (e.g., for formula/image cropping).
    """
    original_item: object          # Raw Docling item
    label: str                     # Current label (may be corrected)
    original_label: str            # Label as Docling assigned it
    text: str                      # Current text content
    level: int                     # Hierarchy level
    heading_level: Optional[int] = None  # H1, H2, H3, H4 (set during hierarchy repair)
    paired_caption: Optional[str] = None  # For images and tables
    orig_content: Optional[str] = None  # For formulas: the orig attribute
    description: Optional[str] = None  # For images: VLM description placeholder
    latex: Optional[str] = None  # For formulas: LaTeX placeholder
    is_removed: bool = False       # Marked for removal (headers/footers)
    removal_reason: str = ""       # Why it was removed
    page_number: Optional[int] = None


def extract_items(docling_result):
    """
    Convert Docling's document items into our ProcessedItem format.

    This is the bridge between Docling's representation and our
    post-processing pipeline. All downstream steps work with
    ProcessedItem objects, never directly with Docling internals.
    """
    items = []
    for item, level in docling_result.document.iterate_items():
        label = item.label if hasattr(item, "label") else "unknown"
        text = item.text if hasattr(item, "text") else ""

        # Extract orig content for formulas
        orig_content = None
        if hasattr(item, "orig"):
            orig_content = item.orig if item.orig != text else None

        # Extract page number from prov (provenance) if available
        page_number = None
        if hasattr(item, "prov") and item.prov:
            first_prov = item.prov[0] if isinstance(item.prov, list) else item.prov
            if hasattr(first_prov, "page_no"):
                page_number = first_prov.page_no

        processed = ProcessedItem(
            original_item=item,
            label=label,
            original_label=label,
            text=text,
            level=level,
            orig_content=orig_content,
            page_number=page_number,
        )
        items.append(processed)

    return items


# ──────────────────────────────────────────────
# Utility: Get Active Items (not removed)
# ──────────────────────────────────────────────

def get_active_items(items):
    """Return only items that haven't been marked for removal."""
    return [item for item in items if not item.is_removed]


# Pattern to match numbered headings like:
# "1 Introduction", "3.2 Attention", "3.2.1 Scaled Dot-Product Attention"
NUMBERED_HEADING_PATTERN = re.compile(r"^\d+(\.\d+)*\s+")

# Known legitimate unnumbered headings (case-insensitive)
KNOWN_UNNUMBERED_HEADINGS = {
    "abstract",
    "acknowledgements",
    "acknowledgment",
    "acknowledgments",
    "references",
    "bibliography",
    "conclusion",
    "conclusions",
    "appendix",
    "supplementary material",
    "related work",
    "table of contents",
}


def _is_numbered_heading(text):
    """Check if a heading starts with a section number pattern."""
    return bool(NUMBERED_HEADING_PATTERN.match(text.strip()))


def _is_known_unnumbered(text):
    """Check if a heading matches known unnumbered heading conventions."""
    cleaned = text.strip().lower()
    return cleaned in KNOWN_UNNUMBERED_HEADINGS


def _strip_number_prefix(text):
    """
    Remove the number prefix from a heading.
    '3.2.1 Scaled Dot-Product Attention' -> 'Scaled Dot-Product Attention'
    """
    return NUMBERED_HEADING_PATTERN.sub(
        lambda m: "", text.strip(), count=1
    ).strip()


def _is_duplicate_of_numbered(text, numbered_headings):
    """
    Check if an unnumbered heading duplicates the text
    of an existing numbered heading (minus the number).

    'Scaled Dot-Product Attention' would match against
    '3.2.1 Scaled Dot-Product Attention'.
    """
    cleaned = text.strip().lower()
    for numbered in numbered_headings:
        stripped = _strip_number_prefix(numbered).lower()
        if cleaned == stripped:
            return True
    return False


def _document_uses_numbered_headings(headings):
    """
    Determine if the document follows a numbered heading convention.

    If more than 50% of headings are numbered, the document
    uses numbered headings. This threshold handles the fact that
    even numbered documents have some unnumbered headings
    (title, abstract, references).
    """
    if not headings:
        return False

    numbered_count = sum(
        1 for h in headings if _is_numbered_heading(h.text)
    )
    ratio = numbered_count / len(headings)
    return ratio > 0.5


def _clean_heading_text(text):
    """
    Clean heading text that may have garbled content.

    Docling sometimes concatenates heading text with nearby
    figure labels or repeated content. This function:
    1. Takes only the first line if heading spans multiple lines
    2. Strips extra whitespace
    """
    first_line = text.strip().split("\n")[0].strip()
    return first_line


def validate_headings(items):
    """
    Detect and demote misclassified headings.

    Strategy:
    1. If document uses numbered headings, unnumbered headings
       that aren't in the known list are suspicious.
    2. The first heading in the document is treated as the title
       and always kept.
    3. Suspicious headings that duplicate a numbered heading's
       text are demoted to regular text.
    4. Suspicious headings are cleaned (first line only) before
       length evaluation.
    5. Remaining suspicious headings are checked against
       length heuristics — headings significantly longer than
       the document's average heading length are demoted.

    Returns the items list with corrections applied and
    a count of demoted headings.
    """
    active = get_active_items(items)
    headings = [item for item in active if item.label == "section_header"]

    if not headings:
        return items, 0

    uses_numbering = _document_uses_numbered_headings(headings)
    demoted_count = 0

    if not uses_numbering:
        avg_length = sum(len(h.text) for h in headings) / len(headings)
        length_threshold = avg_length * 3

        for item in headings:
            if len(item.text) > length_threshold:
                item.label = "text"
                item.removal_reason = (
                    f"Demoted: heading length ({len(item.text)}) "
                    f"exceeds 3x average ({avg_length:.0f})"
                )
                demoted_count += 1

        return items, demoted_count

    numbered_headings = [
        h.text for h in headings if _is_numbered_heading(h.text)
    ]

    avg_length = sum(len(h.text) for h in headings) / len(headings)
    length_threshold = avg_length * 3

    first_heading_seen = False

    for item in headings:
        if _is_numbered_heading(item.text):
            first_heading_seen = True
            continue

        if _is_known_unnumbered(item.text):
            first_heading_seen = True
            continue

        if not first_heading_seen:
            first_heading_seen = True
            continue

        # Suspicious heading — check for duplication first
        if _is_duplicate_of_numbered(item.text, numbered_headings):
            item.label = "text"
            item.removal_reason = (
                f"Demoted: duplicates numbered heading content"
            )
            demoted_count += 1
            continue

        # Clean heading text before length check
        cleaned_text = _clean_heading_text(item.text)
        if cleaned_text != item.text:
            item.text = cleaned_text

        # Check length heuristic on cleaned text
        if len(item.text) > length_threshold:
            item.label = "text"
            item.removal_reason = (
                f"Demoted: unnumbered, not in known list, "
                f"length ({len(item.text)}) exceeds threshold"
            )
            demoted_count += 1
            continue

        item.removal_reason = "Warning: unnumbered heading, kept but flagged"

    return items, demoted_count


# ──────────────────────────────────────────────
# Step 2: Heading Hierarchy Repair
# ──────────────────────────────────────────────

def _count_dots(text):
    """
    Count dots in the number prefix to determine heading depth.
    '1 Introduction' -> 0 dots -> H2
    '3.1 Encoder' -> 1 dot -> H3
    '3.2.1 Scaled' -> 2 dots -> H4
    """
    match = NUMBERED_HEADING_PATTERN.match(text.strip())
    if not match:
        return None
    number_part = match.group(0).strip()
    return number_part.count(".")


def repair_heading_hierarchy(items):
    """
    Restore proper heading levels (H1-H4) from Docling's
    flat heading output.

    Rules:
    - First heading in the document -> H1 (title)
    - Known unnumbered headings -> H2 (top-level sections)
    - Numbered headings: depth based on dot count
        - '1', '2', '3' (0 dots) -> H2
        - '3.1', '5.2' (1 dot) -> H3
        - '3.2.1' (2 dots) -> H4
    - Unknown unnumbered headings that survived validation
      are assigned H2 as a safe default.

    Returns items with heading_level set and a summary dict.
    """
    active = get_active_items(items)
    headings = [item for item in active if item.label == "section_header"]

    if not headings:
        return items, {}

    first_heading_seen = False
    level_counts = {1: 0, 2: 0, 3: 0, 4: 0}

    for item in headings:
        # First heading is always the document title
        if not first_heading_seen:
            item.heading_level = 1
            first_heading_seen = True
            level_counts[1] += 1
            continue

        # Known unnumbered headings are top-level sections
        if _is_known_unnumbered(item.text):
            item.heading_level = 2
            level_counts[2] += 1
            continue

        # Numbered headings — depth from dot count
        dots = _count_dots(item.text)
        if dots is not None:
            # 0 dots -> H2, 1 dot -> H3, 2 dots -> H4
            # Cap at H4 — deeper nesting is rare and
            # not useful for RAG chunking
            heading_level = min(dots + 2, 4)
            item.heading_level = heading_level
            level_counts[heading_level] += 1
            continue

        # Unknown unnumbered heading that survived validation
        # Default to H2 as safe assumption
        item.heading_level = 2
        level_counts[2] += 1

    summary = {
        "total_headings": sum(level_counts.values()),
        "h1": level_counts[1],
        "h2": level_counts[2],
        "h3": level_counts[3],
        "h4": level_counts[4],
    }

    return items, summary


# ──────────────────────────────────────────────
# Step 3: Pair Images with Captions
# ──────────────────────────────────────────────

def pair_images_with_captions(items):
    """
    Link each image element with its associated caption.

    Handles three patterns:
    - Single image followed by caption: standard pairing
    - Multiple consecutive images followed by one caption:
      shared caption (e.g., side-by-side diagrams in a figure)
    - Image with no following caption: tracked as unpaired

    The general rule: scan forward from each image. If the next
    non-picture element is a caption, pair them. This naturally
    handles any number of consecutive images sharing a caption.
    """
    active = get_active_items(items)
    paired_count = 0
    unpaired_images = []
    already_paired = set()  # Track indices we've already processed

    for i, item in enumerate(active):
        if item.label != "picture":
            continue

        if i in already_paired:
            continue

        # Collect consecutive images starting from this one
        image_group = [i]
        j = i + 1
        while j < len(active) and active[j].label == "picture":
            image_group.append(j)
            j += 1

        # Check if the element after the image group is a caption
        if j < len(active) and active[j].label == "caption":
            caption_text = active[j].text
            for idx in image_group:
                active[idx].paired_caption = caption_text
                already_paired.add(idx)
                paired_count += 1
        else:
            for idx in image_group:
                unpaired_images.append(active[idx])
                already_paired.add(idx)

    return items, paired_count, unpaired_images


# ──────────────────────────────────────────────
# Step 4: Pair Tables with Captions
# ──────────────────────────────────────────────

def pair_tables_with_captions(items):
    """
    Link each table element with its associated caption.

    Same logic as image-caption pairing:
    - Single table followed by caption: standard pairing
    - Multiple consecutive tables followed by one caption:
      shared caption (rare but possible)
    - Table with no following caption: tracked as unpaired

    Table captions typically follow the convention
    'Table N: description'.
    """
    active = get_active_items(items)
    paired_count = 0
    unpaired_tables = []
    already_paired = set()

    for i, item in enumerate(active):
        if item.label != "table":
            continue

        if i in already_paired:
            continue

        # Collect consecutive tables starting from this one
        table_group = [i]
        j = i + 1
        while j < len(active) and active[j].label == "table":
            table_group.append(j)
            j += 1

        # Check if the element after the table group is a caption
        if j < len(active) and active[j].label == "caption":
            caption_text = active[j].text
            for idx in table_group:
                active[idx].paired_caption = caption_text
                already_paired.add(idx)
                paired_count += 1
        else:
            for idx in table_group:
                unpaired_tables.append(active[idx])
                already_paired.add(idx)

    return items, paired_count, unpaired_tables


# ──────────────────────────────────────────────
# Step 5: Formula Extraction Placeholder
# ──────────────────────────────────────────────

def _crop_region_from_pdf(pdf_path, page_number, bbox, padding=5):
    """
    Crop a region from a PDF page and return it as PNG bytes.

    Docling uses BOTTOMLEFT coordinate origin (PDF standard),
    where y=0 is at the bottom of the page. PyMuPDF uses
    TOPLEFT origin where y=0 is at the top. We need to convert.

    Args:
        pdf_path: Path to the original PDF
        page_number: 1-indexed page number
        bbox: Docling BoundingBox with l, t, r, b attributes
        padding: Extra pixels around the crop to avoid cutting edges
    """
    doc = fitz.open(str(pdf_path))
    page = doc[page_number - 1]  # fitz uses 0-indexed pages
    page_height = page.rect.height

    # Convert from BOTTOMLEFT to TOPLEFT origin
    # In BOTTOMLEFT: t is higher y value, b is lower y value
    # In TOPLEFT: we flip y by subtracting from page height
    x0 = bbox.l - padding
    y0 = page_height - bbox.t - padding  # t becomes top in TOPLEFT
    x1 = bbox.r + padding
    y1 = page_height - bbox.b + padding  # b becomes bottom in TOPLEFT

    # Clamp to page boundaries
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(page.rect.width, x1)
    y1 = min(page_height, y1)

    # Crop region
    clip_rect = fitz.Rect(x0, y0, x1, y1)

    # Render at 2x resolution for better OCR quality
    matrix = fitz.Matrix(2, 2)
    pixmap = page.get_pixmap(matrix=matrix, clip=clip_rect)

    image_bytes = pixmap.tobytes("png")

    doc.close()
    return image_bytes


def extract_formulas(items, pdf_path):
    """
    Prepare formula data for downstream math OCR processing.

    For each formula element:
    1. Crop the formula region from the original PDF
    2. Collect the orig text as a hint for the OCR model
    3. Store both on the ProcessedItem for later processing

    Actual model inference (pix2tex) is NOT run here.
    This step prepares the inputs. The model call will be
    plugged in later.

    Returns items with formula data populated, a list of
    FormulaData dicts for inspection, and a count.
    """
    active = get_active_items(items)
    formula_data = []
    extraction_errors = []

    for item in active:
        if item.label != "formula":
            continue

        original = item.original_item

        # Get bounding box and page number
        if not hasattr(original, "prov") or not original.prov:
            extraction_errors.append({
                "page": item.page_number,
                "error": "No provenance data — cannot crop"
            })
            continue

        prov = original.prov[0] if isinstance(original.prov, list) else original.prov

        # Crop the formula region from the PDF
        try:
            image_bytes = _crop_region_from_pdf(
                pdf_path=pdf_path,
                page_number=prov.page_no,
                bbox=prov.bbox,
                padding=5,
            )
        except Exception as e:
            extraction_errors.append({
                "page": prov.page_no,
                "error": f"Crop failed: {str(e)}"
            })
            continue

        # Store on the item for later model processing
        item.latex = item.orig_content  # Use Docling's rough extraction directly, we can replace it with llm output later
        item.orig_content = item.orig_content  # Already set during extraction

        # Build formula data dict for inspection and downstream use
        formula_info = {
            "page": prov.page_no,
            "orig_text": item.orig_content,
            "image_bytes": image_bytes,
            "bbox": {
                "l": prov.bbox.l,
                "t": prov.bbox.t,
                "r": prov.bbox.r,
                "b": prov.bbox.b,
            },
        }
        formula_data.append(formula_info)

    return items, formula_data, extraction_errors


# ──────────────────────────────────────────────
# Step 6: Image Description Placeholder
# ──────────────────────────────────────────────

def extract_images(items, pdf_path):
    """
    Prepare image data for downstream VLM description generation.

    For each image element:
    1. Crop the image region from the original PDF
    2. Collect the paired caption (if available) as context
    3. Store both for later VLM processing

    Actual model inference (VLM) is NOT run here.
    This step prepares the inputs. The model call will be
    plugged in later.

    Returns items with image data prepared, a list of
    ImageData dicts for inspection, and a count of errors.
    """
    active = get_active_items(items)
    image_data = []
    extraction_errors = []

    for item in active:
        if item.label != "picture":
            continue

        original = item.original_item

        # Get bounding box and page number
        if not hasattr(original, "prov") or not original.prov:
            extraction_errors.append({
                "page": item.page_number,
                "error": "No provenance data — cannot crop"
            })
            continue

        prov = original.prov[0] if isinstance(original.prov, list) else original.prov

        # Crop the image region from the PDF
        try:
            image_bytes = _crop_region_from_pdf(
                pdf_path=pdf_path,
                page_number=prov.page_no,
                bbox=prov.bbox,
                padding=10,  # Slightly more padding than formulas
            )
        except Exception as e:
            extraction_errors.append({
                "page": prov.page_no,
                "error": f"Crop failed: {str(e)}"
            })
            continue

        # Store description placeholder on the item
        item.description = None  # Will be filled by VLM later

        # Build image data dict for inspection and downstream use
        image_info = {
            "page": prov.page_no,
            "caption": item.paired_caption,
            "image_bytes": image_bytes,
            "bbox": {
                "l": prov.bbox.l,
                "t": prov.bbox.t,
                "r": prov.bbox.r,
                "b": prov.bbox.b,
            },
        }
        image_data.append(image_info)

    return items, image_data, extraction_errors


# ──────────────────────────────────────────────
# Step 7: Handle Footnotes
# ──────────────────────────────────────────────

# Patterns that indicate non-content footnotes (author attributions,
# affiliations, correspondence info). These are metadata, not
# knowledge base content.
METADATA_FOOTNOTE_PATTERNS = [
    re.compile(r"equal\s+contribution", re.IGNORECASE),
    re.compile(r"work\s+performed\s+while\s+at", re.IGNORECASE),
    re.compile(r"corresponding\s+author", re.IGNORECASE),
    re.compile(r"^\s*[∗†‡§¶\*]+\s*(email|correspondence)", re.IGNORECASE),
    re.compile(r"^\s*[∗†‡§¶\*]+\s*these\s+authors", re.IGNORECASE),
]


def _is_metadata_footnote(text):
    """
    Check if a footnote is author/affiliation metadata
    rather than content.
    """
    for pattern in METADATA_FOOTNOTE_PATTERNS:
        if pattern.search(text):
            return True
    return False


def handle_footnotes(items):
    """
    Classify footnotes as metadata or content.

    Metadata footnotes (author attributions, affiliations)
    are marked for removal — they have no retrieval value.

    Content footnotes (clarifications, technical details)
    are kept. They remain as standalone elements in the
    item list. During chunking, they can be:
    - Attached to the nearest preceding text element
    - Kept as separate searchable chunks
    - Appended to the end of the section they belong to

    The chunking strategy decides how to use them — our job
    here is just to separate signal from noise.

    Returns items with metadata footnotes marked for removal,
    counts of kept vs removed footnotes.
    """
    kept_count = 0
    removed_count = 0

    for item in items:
        if item.label != "footnote":
            continue

        if item.is_removed:
            continue

        if _is_metadata_footnote(item.text):
            item.is_removed = True
            item.removal_reason = "Removed: metadata footnote (no retrieval value)"
            removed_count += 1
        else:
            kept_count += 1

    return items, kept_count, removed_count


# ──────────────────────────────────────────────
# Stage 3: Full Post-Processing Pipeline
# ──────────────────────────────────────────────

def run_post_processing(docling_result, pdf_path, vlm_model="gemma3:12b-cloud"):
    """
    Run the complete Stage 3 post-processing pipeline.

    Steps executed in order:
    1. Heading validation (demote misclassified headings)
    2. Heading hierarchy repair (restore H1/H2/H3/H4)
    3. Pair images with captions
    4. Pair tables with captions
    5. Formula extraction (crop regions, use orig text)
    6. Image extraction (crop regions, prepare for VLM)
    7. Handle footnotes (classify and remove metadata)
    8. Generate image descriptions (VLM)
    9. Apply all corrections to DoclingDocument
    """
    items = extract_items(docling_result)

    # Step 1: Heading validation
    items, demoted_count = validate_headings(items)

    # Step 2: Heading hierarchy repair
    items, hierarchy_summary = repair_heading_hierarchy(items)

    # Step 3: Pair images with captions
    items, images_paired, unpaired_images = pair_images_with_captions(items)

    # Step 4: Pair tables with captions
    items, tables_paired, unpaired_tables = pair_tables_with_captions(items)

    # Step 5: Formula extraction
    items, formula_data, formula_errors = extract_formulas(items, pdf_path)

    # Step 6: Image extraction
    items, image_data, image_errors = extract_images(items, pdf_path)

    # Step 7: Handle footnotes
    items, footnotes_kept, footnotes_removed = handle_footnotes(items)

    # Step 8: Generate image descriptions via VLM
    items, image_data, desc_generated, desc_errors = generate_image_descriptions(
        items, image_data, model_name=vlm_model
    )

    # Step 9: Apply corrections to DoclingDocument
    corrected_doc, correction_counts = apply_corrections_to_document(
        docling_result, items, image_data
    )

    summary = {
        "total_items": len(items),
        "active_items": len(get_active_items(items)),
        "headings_demoted": demoted_count,
        "hierarchy": hierarchy_summary,
        "images_paired": images_paired,
        "unpaired_images": len(unpaired_images),
        "tables_paired": tables_paired,
        "unpaired_tables": len(unpaired_tables),
        "formulas_extracted": len(formula_data),
        "formula_errors": len(formula_errors),
        "images_extracted": len(image_data),
        "image_errors": len(image_errors),
        "descriptions_generated": desc_generated,
        "description_errors": len(desc_errors),
        "footnotes_kept": footnotes_kept,
        "footnotes_removed": footnotes_removed,
        "doc_corrections": correction_counts,
    }

    return items, formula_data, image_data, summary


def print_post_processing_summary(summary):
    """Print a readable summary of all post-processing steps."""
    print(f"\n{'='*50}")
    print(f"Post-Processing Summary")
    print(f"{'='*50}")
    print(f"Total items: {summary['total_items']}")
    print(f"Active items: {summary['active_items']}")
    print(f"\nHeadings:")
    print(f"  Demoted: {summary['headings_demoted']}")
    print(f"  Hierarchy: {summary['hierarchy']}")
    print(f"\nImages:")
    print(f"  Paired with captions: {summary['images_paired']}")
    print(f"  Unpaired: {summary['unpaired_images']}")
    print(f"  Cropped for VLM: {summary['images_extracted']}")
    print(f"  Crop errors: {summary['image_errors']}")
    print(f"\nTables:")
    print(f"  Paired with captions: {summary['tables_paired']}")
    print(f"  Unpaired: {summary['unpaired_tables']}")
    print(f"\nFormulas:")
    print(f"  Cropped for math OCR: {summary['formulas_extracted']}")
    print(f"  Crop errors: {summary['formula_errors']}")
    print(f"\nFootnotes:")
    print(f"  Kept (content): {summary['footnotes_kept']}")
    print(f"  Removed (metadata): {summary['footnotes_removed']}")

    print(f"\nDescriptions:")
    print(f"  Generated by VLM: {summary.get('descriptions_generated', 0)}")
    print(f"  VLM errors: {summary.get('description_errors', 0)}")


# ──────────────────────────────────────────────
# Stage 4: Quality Gate
# ──────────────────────────────────────────────

@dataclass
class QualityWarning:
    """A single quality issue detected during the gate check."""
    severity: str  # "hard" or "soft"
    check_name: str
    message: str
    details: dict = field(default_factory=dict)


@dataclass
class QualityResult:
    """Output of the quality gate."""
    passed: bool          # True if document can enter knowledge base
    score: float          # 0.0 to 1.0
    outcome: str          # "pass", "warn", or "reject"
    warnings: list        # List of QualityWarning objects
    stats: dict = field(default_factory=dict)


def _check_extraction_completeness(pdf_path, post_processing_summary):
    """
    Compare Docling's extraction against raw text baseline.

    A low ratio means Docling missed significant content.
    A very high ratio might mean duplication.
    """
    warnings = []

    doc = fitz.open(str(pdf_path))
    raw_char_count = sum(len(page.get_text().strip()) for page in doc)
    doc.close()

    active_items = post_processing_summary["active_items"]
    docling_char_count = post_processing_summary.get("total_text_chars", 0)

    if raw_char_count == 0:
        warnings.append(QualityWarning(
            severity="hard",
            check_name="extraction_completeness",
            message="No text could be extracted from the PDF. "
                    "The file may be corrupted or entirely image-based.",
            details={"raw_chars": 0, "docling_chars": docling_char_count},
        ))
        return warnings, 0.0

    ratio = docling_char_count / raw_char_count

    if ratio < 0.3:
        warnings.append(QualityWarning(
            severity="hard",
            check_name="extraction_completeness",
            message=f"Extraction ratio is very low ({ratio:.2f}). "
                    f"Most content may be missing.",
            details={"raw_chars": raw_char_count,
                     "docling_chars": docling_char_count, "ratio": ratio},
        ))
    elif ratio < 0.6:
        warnings.append(QualityWarning(
            severity="soft",
            check_name="extraction_completeness",
            message=f"Extraction ratio is below expected ({ratio:.2f}). "
                    f"Some content may be missing.",
            details={"raw_chars": raw_char_count,
                     "docling_chars": docling_char_count, "ratio": ratio},
        ))

    return warnings, ratio


def _check_structural_integrity(post_processing_summary, total_pages):
    """
    Check for suspicious structural patterns.
    """
    warnings = []
    hierarchy = post_processing_summary.get("hierarchy", {})
    total_headings = hierarchy.get("total_headings", 0)

    # Long document with no headings
    if total_pages > 5 and total_headings == 0:
        warnings.append(QualityWarning(
            severity="soft",
            check_name="structural_integrity",
            message=f"Document has {total_pages} pages but no headings detected. "
                    f"Structure may not be correctly parsed.",
            details={"pages": total_pages, "headings": total_headings},
        ))

    # Demoted headings are a large portion of total
    demoted = post_processing_summary.get("headings_demoted", 0)
    if total_headings > 0 and demoted > 0:
        demoted_ratio = demoted / (total_headings + demoted)
        if demoted_ratio > 0.3:
            warnings.append(QualityWarning(
                severity="soft",
                check_name="structural_integrity",
                message=f"High heading demotion rate ({demoted_ratio:.0%}). "
                        f"Layout analysis may be unreliable on this document.",
                details={"demoted": demoted, "total_original": total_headings + demoted},
            ))

    return warnings


def _check_extraction_errors(post_processing_summary):
    """
    Check for errors in formula and image extraction.
    """
    warnings = []

    formula_errors = post_processing_summary.get("formula_errors", 0)
    formulas_total = post_processing_summary.get("formulas_extracted", 0) + formula_errors

    if formula_errors > 0:
        warnings.append(QualityWarning(
            severity="soft",
            check_name="formula_extraction",
            message=f"{formula_errors} of {formulas_total} formulas failed to extract. "
                    f"Some mathematical content may be missing.",
            details={"errors": formula_errors, "total": formulas_total},
        ))

    image_errors = post_processing_summary.get("image_errors", 0)
    images_total = post_processing_summary.get("images_extracted", 0) + image_errors

    if image_errors > 0:
        warnings.append(QualityWarning(
            severity="soft",
            check_name="image_extraction",
            message=f"{image_errors} of {images_total} images failed to extract. "
                    f"Some visual content may be missing.",
            details={"errors": image_errors, "total": images_total},
        ))

    return warnings


def _check_unpaired_elements(post_processing_summary):
    """
    Check for images or tables without captions.
    """
    warnings = []

    unpaired_images = post_processing_summary.get("unpaired_images", 0)
    if unpaired_images > 0:
        warnings.append(QualityWarning(
            severity="soft",
            check_name="unpaired_images",
            message=f"{unpaired_images} image(s) have no caption. "
                    f"These may produce lower quality descriptions.",
            details={"unpaired": unpaired_images},
        ))

    unpaired_tables = post_processing_summary.get("unpaired_tables", 0)
    if unpaired_tables > 0:
        warnings.append(QualityWarning(
            severity="soft",
            check_name="unpaired_tables",
            message=f"{unpaired_tables} table(s) have no caption. "
                    f"Context for these tables may be limited.",
            details={"unpaired": unpaired_tables},
        ))

    return warnings


def _check_empty_pages(profile):
    """
    Flag pages with very low text density that are digital.

    A digital page with very few characters might mean
    the page has content that wasn't extracted — unlike
    scanned pages where low text is expected.
    """
    warnings = []
    suspicious_pages = []

    for pc in profile.page_classifications:
        if pc.origin == "digital" and pc.char_count < 50 and pc.char_count > 0:
            suspicious_pages.append(pc.page_number)

    if suspicious_pages:
        warnings.append(QualityWarning(
            severity="soft",
            check_name="empty_pages",
            message=f"Pages {suspicious_pages} are digital but have very low "
                    f"text content. Some content may not have been extracted.",
            details={"pages": suspicious_pages},
        ))

    return warnings


def _calculate_quality_score(warnings, extraction_ratio):
    """
    Calculate an overall quality score from 0.0 to 1.0.

    Scoring:
    - Start at 1.0
    - Hard failures: -0.5 each
    - Soft failures: -0.1 each
    - Extraction ratio below 0.8: additional penalty

    Score thresholds:
    - >= 0.7: pass
    - >= 0.4: warn (allow with user confirmation)
    - < 0.4: reject
    """
    score = 1.0

    for w in warnings:
        if w.severity == "hard":
            score -= 0.5
        else:
            score -= 0.1

    # Extraction ratio penalty
    if extraction_ratio is not None and extraction_ratio < 0.8:
        score -= (0.8 - extraction_ratio) * 0.5

    return max(0.0, min(1.0, score))


def run_quality_gate(pdf_path, profile, post_processing_summary, items):
    """
    Stage 4: Evaluate parsing quality and determine whether
    the document should enter the knowledge base.

    Runs all quality checks, calculates a score, and returns
    a QualityResult with the outcome and warnings.

    Outcomes:
    - 'pass': document is good quality, proceed to chunking
    - 'warn': document has issues, user decides whether to proceed
    - 'reject': document has critical issues, should not be used
    """
    all_warnings = []

    # Calculate total text chars for extraction check
    active = get_active_items(items)
    total_text_chars = sum(len(item.text) for item in active)
    post_processing_summary["total_text_chars"] = total_text_chars

    # Run all checks
    extraction_warnings, extraction_ratio = _check_extraction_completeness(
        pdf_path, post_processing_summary
    )
    all_warnings.extend(extraction_warnings)
    all_warnings.extend(_check_structural_integrity(
        post_processing_summary, profile.total_pages
    ))
    all_warnings.extend(_check_extraction_errors(post_processing_summary))
    all_warnings.extend(_check_unpaired_elements(post_processing_summary))
    all_warnings.extend(_check_empty_pages(profile))

    # Calculate score
    score = _calculate_quality_score(all_warnings, extraction_ratio)

    # Determine outcome
    if score >= 0.7:
        outcome = "pass"
    elif score >= 0.4:
        outcome = "warn"
    else:
        outcome = "reject"

    has_hard_failure = any(w.severity == "hard" for w in all_warnings)
    if has_hard_failure:
        outcome = "reject"

    return QualityResult(
        passed=(outcome == "pass"),
        score=round(score, 2),
        outcome=outcome,
        warnings=all_warnings,
        stats={
            "extraction_ratio": round(extraction_ratio, 2) if extraction_ratio else None,
            "total_text_chars": total_text_chars,
            "total_active_items": len(active),
        },
    )


def print_quality_result(quality_result):
    """Print a readable quality gate result."""
    print(f"\n{'='*50}")
    print(f"Quality Gate Result")
    print(f"{'='*50}")
    print(f"Outcome: {quality_result.outcome.upper()}")
    print(f"Score: {quality_result.score}")
    print(f"Stats: {quality_result.stats}")

    if quality_result.warnings:
        print(f"\nWarnings ({len(quality_result.warnings)}):")
        for w in quality_result.warnings:
            severity_tag = "[HARD]" if w.severity == "hard" else "[SOFT]"
            print(f"  {severity_tag} {w.check_name}: {w.message}")
    else:
        print(f"\nNo warnings — clean extraction.")


# Add to post_processor.py

# ──────────────────────────────────────────────
# VLM Image Description
# ──────────────────────────────────────────────

def generate_image_descriptions(items, image_data, model_name="gemma3:12b-cloud"):
    """
    Generate descriptions for extracted images using a VLM.

    For each image:
    1. Send the cropped image to the VLM
    2. Include the paired caption as context to improve accuracy
    3. Store the description on the corresponding ProcessedItem

    Args:
        items: List of ProcessedItems
        image_data: List of image dicts from extract_images()
        model_name: Ollama model to use for descriptions
    """
    import ollama

    active = get_active_items(items)
    picture_items = [item for item in active if item.label == "picture"]

    if len(picture_items) != len(image_data):
        print(f"Warning: {len(picture_items)} picture items but "
              f"{len(image_data)} image crops. Matching by index.")

    descriptions_generated = 0
    description_errors = []

    for i, (item, img_info) in enumerate(zip(picture_items, image_data)):
        # Build prompt with caption context if available
        caption_context = ""
        if item.paired_caption:
            caption_context = f" Its caption is: {item.paired_caption}"

        prompt = (
            f"Describe this figure concisely in 3-4 sentences. "
            f"Focus on what components are shown, how they connect, "
            f"and what the figure is demonstrating. "
            f"Do not speculate beyond what is visible."
            f"{caption_context}"
        )

        try:
            response = ollama.chat(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [img_info["image_bytes"]],
                    }
                ],
            )

            description = response["message"]["content"]
            item.description = description
            img_info["description"] = description
            descriptions_generated += 1

        except Exception as e:
            error_msg = f"VLM failed on image {i+1} (page {img_info['page']}): {str(e)}"
            description_errors.append(error_msg)
            # Fallback to caption if VLM fails
            if item.paired_caption:
                item.description = item.paired_caption
                img_info["description"] = item.paired_caption
            print(f"Warning: {error_msg}. Using caption as fallback.")

    return items, image_data, descriptions_generated, description_errors


# ──────────────────────────────────────────────
# Stage 3b: Apply Corrections to DoclingDocument
# ──────────────────────────────────────────────

def apply_corrections_to_document(docling_result, items, image_data):
    """
    Apply all Stage 3 corrections back onto the original
    DoclingDocument so that downstream tools (HybridChunker,
    Markdown export) see the corrected content.

    Modifications:
    1. Fix heading hierarchy (set correct level values)
    2. Update cleaned heading text on document items
    3. Demote misclassified headings (replace with TextItem)
    4. Fix formula text (copy orig to text)
    5. Add VLM descriptions to picture captions
    6. Remove metadata footnotes (clear text)
    """
    from docling_core.types.doc.labels import DocItemLabel
    from docling_core.types.doc.document import TextItem

    doc = docling_result.document
    doc_items = list(doc.iterate_items())

    corrections = {
        "headings_leveled": 0,
        "headings_demoted": 0,
        "formulas_filled": 0,
        "images_described": 0,
        "footnotes_removed": 0,
    }

    # Collect items to demote — we can't modify during iteration
    items_to_demote = []

    for i, (doc_item, doc_level) in enumerate(doc_items):
        if i >= len(items):
            break

        processed = items[i]

        # 1. Fix heading hierarchy
        if (processed.label == "section_header"
                and processed.heading_level is not None):
            new_level = processed.heading_level - 1
            if doc_item.level != new_level:
                doc_item.level = new_level
                corrections["headings_leveled"] += 1
            # Also update cleaned text on the document item
            if hasattr(doc_item, "text") and doc_item.text != processed.text:
                doc_item.text = processed.text
                if hasattr(doc_item, "orig"):
                    doc_item.orig = processed.text

        # 2. Collect demoted headings for replacement
        if (processed.original_label == "section_header"
                and processed.label == "text"):
            items_to_demote.append(doc_item)

        # 3. Fix formula text
        if processed.label == "formula" and processed.latex:
            if hasattr(doc_item, "text"):
                doc_item.text = processed.latex
                corrections["formulas_filled"] += 1

        # 4. Add VLM descriptions to pictures
        if processed.label == "picture" and processed.description:
            if hasattr(doc_item, "captions") and doc_item.captions:
                for cap_ref in doc_item.captions:
                    try:
                        cap_item = cap_ref.resolve(doc)
                        original_caption = cap_item.text
                        cap_item.text = (
                            f"{original_caption}\n"
                            f"[Image Description]: {processed.description}"
                        )
                        corrections["images_described"] += 1
                    except Exception as e:
                        print(f"Warning: Could not modify caption: {e}")
            else:
                print(f"Warning: Image on page {processed.page_number} "
                      f"has no caption to attach description to.")

        # 5. Remove metadata footnotes
        if processed.label == "footnote" and processed.is_removed:
            if hasattr(doc_item, "text"):
                doc_item.text = ""
                corrections["footnotes_removed"] += 1

    # Apply heading demotions via replace_item
    for old_item in items_to_demote:
        new_item = TextItem(
            self_ref=old_item.self_ref,
            parent=old_item.parent,
            children=old_item.children,
            content_layer=old_item.content_layer,
            label=DocItemLabel.TEXT,
            prov=old_item.prov,
            orig=old_item.orig,
            text=old_item.text,
        )
        try:
            doc.replace_item(new_item=new_item, old_item=old_item)
            corrections["headings_demoted"] += 1
        except Exception as e:
            print(f"Warning: Could not demote heading '{old_item.text[:40]}': {e}")

    return doc, corrections