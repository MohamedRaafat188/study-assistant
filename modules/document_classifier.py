from dataclasses import dataclass, field
from pathlib import Path
import fitz  # PyMuPDF


# ──────────────────────────────────────────────
# Data Structures
# ──────────────────────────────────────────────

@dataclass
class PageClassification:
    page_number: int
    origin: str  # "digital" or "scanned"
    char_count: int
    text_density: float  # chars per unit page area


@dataclass
class ContentSummary:
    """Populated after Docling runs (Stage 1b)."""
    total_headings: int = 0
    total_tables: int = 0
    total_formulas: int = 0
    total_images: int = 0
    content_type: str = "unknown"  # text-heavy, math-heavy, table-heavy, image-heavy, mixed


@dataclass
class DocumentProfile:
    filename: str
    total_pages: int
    page_classifications: list
    content_summary: ContentSummary = field(default_factory=ContentSummary)
    needs_ocr: bool = False
    needs_formula_processing: bool = False
    needs_image_processing: bool = False


# ──────────────────────────────────────────────
# Stage 1a: Pre-processing Classification
# ──────────────────────────────────────────────

def classify_page(page, page_number, min_char_threshold=50):
    """
    Classify a single PDF page as digital or scanned.

    Logic:
    - Extract raw text from the page.
    - If character count is above threshold, the page is digital.
    - If below, it's likely scanned or image-based.

    The threshold of 50 is intentionally low to handle pages
    that are mostly visual with short captions — those are still
    digital, just image-heavy.
    """
    text = page.get_text().strip()
    char_count = len(text)

    page_area = page.rect.width * page.rect.height
    text_density = char_count / page_area if page_area > 0 else 0.0

    origin = "digital" if char_count >= min_char_threshold else "scanned"

    return PageClassification(
        page_number=page_number,
        origin=origin,
        char_count=char_count,
        text_density=round(text_density, 4),
    )


def classify_document(pdf_path, min_char_threshold=50):
    """
    Stage 1a: Classify each page and build initial document profile.

    This runs BEFORE Docling. It's lightweight (no AI models)
    and determines:
    - Which pages need OCR
    - Overall document origin mix
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(str(pdf_path))

    page_classifications = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        classification = classify_page(page, page_num + 1, min_char_threshold)
        page_classifications.append(classification)

    doc.close()

    needs_ocr = any(pc.origin == "scanned" for pc in page_classifications)

    scanned_count = sum(1 for pc in page_classifications if pc.origin == "scanned")
    digital_count = sum(1 for pc in page_classifications if pc.origin == "digital")

    profile = DocumentProfile(
        filename=str(pdf_path),
        total_pages=len(page_classifications),
        page_classifications=page_classifications,
        needs_ocr=needs_ocr,
    )

    return profile


# ──────────────────────────────────────────────
# Stage 1b: Post-Parsing Content Profiling
# ──────────────────────────────────────────────

def build_content_profile(profile, docling_result):
    """
    Stage 1b: Analyze Docling's output to determine content types.

    Uses the item.label attribute which Docling assigns to each
    element during layout analysis. This is more reliable than
    checking class names or text content.

    Known label values from Docling:
    text, section_header, table, picture, formula,
    caption, list_item, footnote
    """
    from collections import Counter

    label_counts = Counter()

    for item, level in docling_result.document.iterate_items():
        label = item.label if hasattr(item, "label") else "unknown"
        label_counts[label] += 1

    heading_count = label_counts.get("section_header", 0)
    table_count = label_counts.get("table", 0)
    formula_count = label_counts.get("formula", 0)
    image_count = label_counts.get("picture", 0)
    text_count = label_counts.get("text", 0)

    content_type = _determine_content_type(
        text_count=text_count,
        table_count=table_count,
        formula_count=formula_count,
        image_count=image_count,
    )

    profile.content_summary = ContentSummary(
        total_headings=heading_count,
        total_tables=table_count,
        total_formulas=formula_count,
        total_images=image_count,
        content_type=content_type,
    )

    profile.needs_formula_processing = formula_count > 0
    profile.needs_image_processing = image_count > 0

    return profile


def _determine_content_type(text_count, table_count, formula_count, image_count):
    """
    Classify the dominant content type based on element counts.

    This uses simple ratio-based heuristics. The thresholds
    are starting points — tune them based on real-world testing.
    """
    total = text_count + table_count + formula_count + image_count

    if total == 0:
        return "unknown"

    text_ratio = text_count / total
    table_ratio = table_count / total
    formula_ratio = formula_count / total
    image_ratio = image_count / total

    # If no single type dominates, it's mixed
    dominance_threshold = 0.15  # 15% of elements

    dominant_types = []
    if table_ratio >= dominance_threshold:
        dominant_types.append("table-heavy")
    if formula_ratio >= dominance_threshold:
        dominant_types.append("math-heavy")
    if image_ratio >= dominance_threshold:
        dominant_types.append("image-heavy")

    if len(dominant_types) == 0:
        return "text-heavy"
    elif len(dominant_types) == 1:
        return dominant_types[0]
    else:
        return "mixed"


# ──────────────────────────────────────────────
# Utility: Print Profile Summary
# ──────────────────────────────────────────────

def print_profile(profile):
    """Print a readable summary of the document profile."""
    print(f"\n{'='*50}")
    print(f"Document Profile: {profile.filename}")
    print(f"{'='*50}")
    print(f"Total pages: {profile.total_pages}")
    print(f"Needs OCR: {profile.needs_ocr}")

    print(f"\nPage Classifications:")
    for pc in profile.page_classifications:
        print(f"  Page {pc.page_number}: {pc.origin} "
              f"(chars: {pc.char_count}, density: {pc.text_density})")

    scanned = sum(1 for pc in profile.page_classifications if pc.origin == "scanned")
    digital = sum(1 for pc in profile.page_classifications if pc.origin == "digital")
    print(f"\nSummary: {digital} digital, {scanned} scanned")

    if profile.content_summary.content_type != "unknown":
        cs = profile.content_summary
        print(f"\nContent Profile:")
        print(f"  Content type: {cs.content_type}")
        print(f"  Headings: {cs.total_headings}")
        print(f"  Tables: {cs.total_tables}")
        print(f"  Formulas: {cs.total_formulas}")
        print(f"  Images: {cs.total_images}")
        print(f"  Needs formula processing: {profile.needs_formula_processing}")
        print(f"  Needs image processing: {profile.needs_image_processing}")