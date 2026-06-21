# Stage 2: Docling Parsing with Profile-Driven Configuration

from dataclasses import dataclass
from pathlib import Path
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from .document_classifier import DocumentProfile, build_content_profile, print_profile


# ──────────────────────────────────────────────
# Data Structures
# ──────────────────────────────────────────────

@dataclass
class ParseResult:
    """Wraps Docling's output with status and error tracking."""
    success: bool
    docling_result: object = None  # ConversionResult when successful
    profile: DocumentProfile = None  # Updated profile after Stage 1b
    error_message: str = ""
    markdown: str = ""
    document_dict: dict = None


# ──────────────────────────────────────────────
# Docling Configuration
# ──────────────────────────────────────────────

def build_pipeline_options(profile):
    """
    Configure Docling's pipeline based on the document profile.

    This is where Stage 1a's classification drives parsing behavior:
    - OCR is enabled only if the profile says scanned pages exist.
    - Table mode defaults to ACCURATE since students upload
      diverse content and we prioritize quality over speed.
    """
    pipeline_options = PdfPipelineOptions()

    # OCR: driven by Stage 1a classification
    pipeline_options.do_ocr = profile.needs_ocr

    # Table structure: always enabled since we don't know
    # if tables exist until after parsing.
    # Use ACCURATE mode — for a student platform, quality
    # matters more than processing speed.
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE

    return pipeline_options


def create_converter(profile):
    """
    Create a configured DocumentConverter based on the document profile.
    """
    pipeline_options = build_pipeline_options(profile)

    converter = DocumentConverter(
        format_options={
            "pdf": PdfFormatOption(
                pipeline_cls=StandardPdfPipeline,
                pipeline_options=pipeline_options,
                backend=PyPdfiumDocumentBackend,
            )
        }
    )

    return converter


# ──────────────────────────────────────────────
# Stage 2: Parse Document
# ──────────────────────────────────────────────

def parse_document(pdf_path, profile):
    """
    Run Docling on the document with profile-driven configuration.

    Steps:
    1. Build a converter configured by the profile.
    2. Run conversion.
    3. Check conversion status.
    4. If successful, run Stage 1b to complete the content profile.
    5. Extract markdown and structured dict for downstream use.
    6. Return everything wrapped in a ParseResult.

    If conversion fails, return a ParseResult with success=False
    and a descriptive error message for the user.
    """
    pdf_path = Path(pdf_path)

    # Step 1: Create configured converter
    try:
        converter = create_converter(profile)
    except Exception as e:
        return ParseResult(
            success=False,
            profile=profile,
            error_message=f"Failed to initialize parser: {str(e)}",
        )

    # Step 2: Run conversion
    try:
        result = converter.convert(str(pdf_path))
    except Exception as e:
        return ParseResult(
            success=False,
            profile=profile,
            error_message=f"Document conversion failed: {str(e)}",
        )

    # Step 3: Check conversion status
    status = result.status.name if hasattr(result.status, "name") else str(result.status)
    if status != "SUCCESS":
        return ParseResult(
            success=False,
            profile=profile,
            error_message=f"Conversion completed with status: {status}. "
                          f"The document may be corrupted or in an unsupported format.",
        )

    # Step 4: Run Stage 1b — complete the content profile
    try:
        updated_profile = build_content_profile(profile, result)
    except Exception as e:
        # If profiling fails, we still have a valid conversion.
        # Log the error but don't fail the entire parse.
        updated_profile = profile
        print(f"Warning: Content profiling failed: {str(e)}")

    # Step 5: Extract outputs
    try:
        markdown = result.document.export_to_markdown()
    except Exception as e:
        markdown = ""
        print(f"Warning: Markdown export failed: {str(e)}")

    try:
        document_dict = result.document.export_to_dict()
    except Exception as e:
        document_dict = None
        print(f"Warning: Dict export failed: {str(e)}")

    # Step 6: Return wrapped result
    return ParseResult(
        success=True,
        docling_result=result,
        profile=updated_profile,
        markdown=markdown,
        document_dict=document_dict,
    )


# ──────────────────────────────────────────────
# Utility: Print Parse Summary
# ──────────────────────────────────────────────

def print_parse_summary(parse_result):
    """Print a readable summary of the parsing result."""
    print(f"\n{'='*50}")
    print(f"Parse Result")
    print(f"{'='*50}")
    print(f"Success: {parse_result.success}")

    if not parse_result.success:
        print(f"Error: {parse_result.error_message}")
        return

    print(f"Markdown length: {len(parse_result.markdown)} chars")
    print(f"Dict exported: {'Yes' if parse_result.document_dict else 'No'}")

    # Print the updated profile (now includes content summary)
    if parse_result.profile:
        print_profile(parse_result.profile)