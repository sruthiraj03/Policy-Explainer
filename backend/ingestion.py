"""
Ingestion Pipeline: PDF to Vector Search.

This module implements the PolicyExplainer ingestion "ETL" workflow:
- Extract: Parse PDF text while preserving page boundaries (for citation accuracy).
- Transform: Clean page text (remove headers/footers/noise) and chunk content into
  LLM-friendly segments with overlap to preserve context across boundaries.
- Load: Persist extracted pages/chunks and add chunks to the vector store for retrieval.

Core design goal:
- Keep citations accurate by ensuring every chunk is associated with a specific page number.
  This enables downstream summarization and QA to cite evidence reliably.

Operational overview:
- `run_ingest(pdf_content)` is the primary entry point used by the API.
- It saves the raw PDF, extracts pages, validates that the document appears to be a health
  insurance policy (keyword-based), chunks each page, persists artifacts, and indexes chunks.

Notes:
- Chunking is performed page-by-page to avoid citations spanning multiple pages.
- Header/footer removal is both heuristic (page numbers) and statistical (repeated lines).
"""

import re
import shutil
from collections import Counter
from pathlib import Path

import fitz  # PyMuPDF: High-performance PDF parsing

from backend.schemas import Chunk, ExtractedPage
from backend import storage

# --- Chunking Constants ---
# These values are tuned for insurance documents, which are typically dense and structured.
# Chunk size is calibrated to be large enough for coherent reasoning while staying within
# practical LLM context and embedding constraints.
TARGET_TOKENS = 600  # Ideal size for the LLM to process in one 'thought'
MIN_TOKENS = 500
MAX_TOKENS = 800  # Upper limit to stay within the LLM's context window effectively
OVERLAP_TOKENS = 80  # Sentences shared between chunks so context isn't lost at the 'cut'
MIN_CHUNK_CHARS = 50  # Ignore tiny fragments that lack meaningful info


# --- Policy Keywords ---

# Expanded keywords based on standard health insurance document schemas.
# These are used by `is_likely_policy` to quickly reject non-policy uploads.
POLICY_KEYWORDS = [
    # General Identifiers
    "summary of benefits", "evidence of coverage", "policy number", "group number",
    "health care",

    # Financial Terms (Cost Sharing)
    "deductible", "coinsurance", "copayment", "out-of-pocket", "annual limit",
    "maximum out of pocket", "premium", "cost-sharing", "insurance", "medical",

    # Service Categories
    "primary care", "specialist visit", "emergency room", "urgent care",
    "inpatient hospital", "outpatient surgery", "preventive care",

    # Pharmacy & Meds
    "prescription drug", "formulary", "generic drug", "preferred brand", "mail order",

    # Managed Care & Admin
    "prior authorization", "pre-authorization", "referral", "network provider",
    "non-preferred provider", "medically necessary", "exclusions", "limitations",

    #Plan Types
    "PPO", "HMO", "HSA", "EPO", "POS", "HDHP"

    #
]


# --- Policy Checking Code ---
def is_likely_policy(text: str) -> bool:
    """
    Heuristically determine whether extracted text appears to be a health insurance policy/SBC.

    Current approach:
    - Lowercase the text
    - Count unique keyword matches from POLICY_KEYWORDS
    - Accept only if the unique keyword match count meets threshold

    This function is intended to be a fast guardrail to prevent ingestion of irrelevant PDFs
    (e.g., resumes, invoices, random reports), which would degrade retrieval quality.

    Args:
        text: Extracted text sample (typically from first pages of the PDF).

    Returns:
        bool: True if document appears to be a policy/SBC, otherwise False.
    """
    text_lower = text.lower()

    # 1. Check for mandatory "Anchor Phrases" first
    # NOTE: Placeholder comment retained from current design; no anchor logic implemented here.

    # 2. Count standard keywords (unique keyword hits)
    matches = {word for word in POLICY_KEYWORDS if word in text_lower}

    # Only pass if it has an anchor AND meets the keyword count
    # NOTE: "anchor" is not currently implemented; decision is solely based on keyword count.
    is_valid = len(matches) >= 10

    # Debug logging: helps tune thresholds during development.
    # (Consider routing through a logger later; intentionally left as print to preserve behavior.)
    print(f" DEBUG: Keywords: {len(matches)}")
    return is_valid


def _approx_tokens(text: str) -> int:
    """
    Rough token estimator used for chunk sizing.

    The approximation assumes ~4 characters per token, which is a common quick heuristic.
    This avoids dependency on an actual tokenizer at ingestion time.

    Args:
        text: Input text.

    Returns:
        int: Approximate token count (non-negative).
    """
    return max(0, len(text.strip()) // 4)


def _split_into_sentences(text: str) -> list[str]:
    """
    Split text into sentence-like units while keeping sentence-ending punctuation.

    Uses a regex that splits on whitespace following '.', '!', or '?'.
    This supports chunking by sentence boundaries (reduces mid-sentence splits).

    Args:
        text: Page text to split.

    Returns:
        list[str]: List of sentence-like strings (trimmed, non-empty).
    """
    if not text.strip():
        return []

    # Looks for punctuation followed by space; keeps punctuation attached to preceding sentence.
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def _chunk_page_text(page_number: int, doc_id: str, text: str) -> list[Chunk]:
    """
    Break a single page into manageable chunks while preserving citation integrity.

    Key invariant:
    - Every produced Chunk includes:
        * page_number: the source page (1-indexed)
        * doc_id: the document identifier
        * chunk_id: stable identifier encoding page and within-page chunk index

    Chunking strategy:
    - Split page text into sentences
    - Accumulate sentences until adding another would exceed MAX_TOKENS
    - Emit a chunk, then carry over a sliding-window overlap of sentences to the next chunk

    Args:
        page_number: 1-indexed PDF page number.
        doc_id: Document identifier used across storage and retrieval.
        text: Cleaned (or raw) page text.

    Returns:
        list[Chunk]: A list of chunks for the page. If no text, a single empty chunk is returned.
    """
    # If the page is empty/whitespace, return a single empty chunk so the page boundary exists.
    if not text.strip():
        return [Chunk(chunk_id=f"c_{page_number}_0", page_number=page_number, doc_id=doc_id, chunk_text="")]

    # Split into sentences; if sentence splitting fails, fall back to a single chunk.
    sentences = _split_into_sentences(text)
    if not sentences:
        return [Chunk(chunk_id=f"c_{page_number}_0", page_number=page_number, doc_id=doc_id, chunk_text=text.strip())]

    chunks: list[Chunk] = []

    # `current` accumulates the sentences for the current chunk being built.
    current: list[str] = []
    current_tokens = 0

    # Chunk index is within-page; combined with page_number to form a globally unique chunk_id.
    chunk_index = 0

    for sent in sentences:
        sent_tokens = _approx_tokens(sent)

        # If adding this sentence would exceed the maximum chunk size,
        # finalize the current chunk before adding the new sentence.
        if current and current_tokens + sent_tokens > MAX_TOKENS:
            chunk_text = " ".join(current)

            # Emit the completed chunk with stable metadata.
            chunks.append(Chunk(
                chunk_id=f"c_{page_number}_{chunk_index}",
                page_number=page_number,
                doc_id=doc_id,
                chunk_text=chunk_text
            ))
            chunk_index += 1

            # --- Sliding Window Overlap ---
            # Carry a trailing slice of the previous chunk into the next chunk so that:
            # - important definitions are not separated from their context
            # - retrieval has continuity when chunk boundaries occur mid-topic
            overlap_sentences = []
            overlap_tokens = 0

            # Walk backward through the current chunk, accumulating sentences until overlap target met.
            for s in reversed(current):
                overlap_tokens += _approx_tokens(s)
                overlap_sentences.insert(0, s)
                if overlap_tokens >= OVERLAP_TOKENS:
                    break

            # Start the new chunk with the overlap context.
            current = overlap_sentences.copy()
            current_tokens = overlap_tokens

        # Add the sentence to the current chunk accumulator.
        current.append(sent)
        current_tokens += sent_tokens

    # Capture any remaining sentences as the final chunk.
    if current:
        chunk_text = " ".join(current)
        chunks.append(Chunk(
            chunk_id=f"c_{page_number}_{chunk_index}",
            page_number=page_number,
            doc_id=doc_id,
            chunk_text=chunk_text
        ))

    return chunks


def chunk_pages(pages: list[ExtractedPage], doc_id: str) -> list[Chunk]:
    """
    Convert a list of extracted pages into a flat list of chunks.

    This is a thin helper that:
    - iterates pages in order
    - chunks each page independently (preserving page boundaries)
    - returns a single flattened chunk list suitable for storage/vector indexing

    Args:
        pages: Extracted pages (page_number + text).
        doc_id: Document identifier used for all chunk metadata.

    Returns:
        list[Chunk]: Flattened list of all chunks across all pages.
    """
    all_chunks: list[Chunk] = []
    for page in pages:
        all_chunks.extend(_chunk_page_text(page.page_number, doc_id, page.text))
    return all_chunks


# --- PDF Parsing & Cleaning ---

def _normalize_line(s: str) -> str:
    """
    Normalize a line by collapsing whitespace and trimming.

    This is used when comparing lines across pages to identify repeated headers/footers.
    """
    return " ".join(s.split()).strip()


def _looks_like_page_number(line: str) -> bool:
    """
    Detect whether a line looks like a page marker/footer.

    Examples detected:
    - "1"
    - "Page 1 of 10"
    - "5/12"

    These lines are commonly repeated boilerplate and should not be indexed.

    Args:
        line: Raw line text.

    Returns:
        bool: True if the line should be treated as page-number-like noise.
    """
    line = _normalize_line(line)
    if not line:
        return True
    if re.match(r"^\d+$", line):
        return True
    if re.match(r"^page\s+\d+\s+of\s+\d+$", line, re.I):
        return True
    if re.match(r"^\d+\s*/\s*\d+$", line):
        return True
    return False


def _clean_page_text(raw: str, drop_first_last_lines: bool = True) -> str:
    """
    Clean raw page text by removing likely headers/footers and collapsing excessive whitespace.

    Cleaning steps:
    - Split into lines
    - Optionally remove short/page-number-like lines from the start and end of the page
    - Collapse excessive blank lines

    Args:
        raw: Raw text extracted from a PDF page.
        drop_first_last_lines: Whether to heuristically strip top/bottom noise lines.

    Returns:
        str: Cleaned page text.
    """
    # Split into individual lines; keep order for better readability.
    lines = [ln.strip() for ln in raw.splitlines()]
    if not lines:
        return ""

    if drop_first_last_lines:
        # Pop lines from the top if they are short or look like page numbers.
        # This removes common headers like "Page 2 of 10" or empty lines.
        while lines and (_looks_like_page_number(lines[0]) or len(lines[0]) < 3):
            lines.pop(0)

        # Pop lines from the bottom using the same heuristics.
        while lines and (_looks_like_page_number(lines[-1]) or len(lines[-1]) < 3):
            lines.pop()

    # Rejoin lines with newlines to preserve some structure (tables/sections).
    text = "\n".join(lines)

    # Collapse runs of 3+ newlines into 2 newlines to reduce whitespace without flattening structure.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _remove_repeated_header_footer(lines_by_page: list[list[str]]) -> list[list[str]]:
    """
    Remove repeated header/footer lines using a statistical approach.

    Idea:
    - Count normalized line frequency across pages.
    - If a normalized line appears on more than ~50% of pages, it is likely boilerplate
      (header/footer) rather than meaningful policy content.
    - Remove those repeated lines from each page.

    This improves retrieval quality by reducing noise that would otherwise dominate embeddings.

    Args:
        lines_by_page: A list of pages, each containing a list of raw text lines.

    Returns:
        list[list[str]]: Same shape as input but with repeated lines removed.
    """
    # If only one page (or empty), there's no meaningful "repetition" signal.
    if len(lines_by_page) < 2:
        return lines_by_page

    # Count occurrences of normalized lines across all pages.
    all_lines: Counter[str] = Counter()
    for page_lines in lines_by_page:
        for ln in page_lines:
            n = _normalize_line(ln)
            # Ignore empty/very short lines that are unlikely to be meaningful.
            if n and len(n) > 2:
                all_lines[n] += 1

    # Define repetition threshold:
    # - At least 2 occurrences (avoid removing content from very short docs)
    # - More than half the number of pages indicates likely header/footer boilerplate
    threshold = max(2, len(lines_by_page) // 2)

    # Build the set of normalized lines considered boilerplate.
    repeated = {ln for ln, count in all_lines.items() if count > threshold}

    # Remove repeated lines from each page (comparison is done using normalized form).
    return [[ln for ln in page_lines if _normalize_line(ln) not in repeated] for page_lines in lines_by_page]


def extract_pages(pdf_path: Path | str, clean_headers_footers: bool = True) -> list[ExtractedPage]:
    """
    Extract text from a PDF while maintaining page boundaries.

    This uses PyMuPDF (fitz) for extraction and returns a list of ExtractedPage objects:
    - page_number is 1-indexed (matching typical human page numbering)
    - text contains the extracted text for that page

    If clean_headers_footers is enabled, the function additionally:
    - removes statistically repeated lines across pages (common headers/footers)
    - runs per-page cleanup to drop page-number-like lines at top/bottom

    Args:
        pdf_path: Path (or string path) to the PDF file on disk.
        clean_headers_footers: Whether to perform header/footer cleanup.

    Returns:
        list[ExtractedPage]: Extracted pages with text.
    """
    path = Path(pdf_path)

    # Basic file validation before attempting to open.
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Not a PDF file: {path}")

    # Open the PDF document with PyMuPDF.
    doc = fitz.open(path)

    # Guard against empty documents.
    if len(doc) == 0:
        doc.close()
        raise ValueError(f"PDF has no pages: {path}")

    pages: list[ExtractedPage] = []
    raw_lines_by_page: list[list[str]] = []

    try:
        # Iterate through each page in order.
        for i in range(len(doc)):
            # Convert 0-indexed i into 1-indexed page number for citations.
            page_num = i + 1
            page = doc[i]

            # 'sort=True' encourages top-to-bottom reading order; helpful for tables/structured docs.
            text = page.get_text("text", sort=True) or ""

            # Keep per-line raw text for statistical header/footer detection.
            raw_lines_by_page.append([ln.strip() for ln in text.splitlines()])

            # Store raw page text initially; later cleaned if requested.
            pages.append(ExtractedPage(page_number=page_num, text=text))
    finally:
        # Always close the document handle even if extraction fails.
        doc.close()

    # If cleanup is disabled (or no pages), return as-is.
    if not clean_headers_footers or not pages:
        return pages

    # Run statistical removal of repeated header/footer lines.
    cleaned_lines = _remove_repeated_header_footer(raw_lines_by_page)

    # Reconstruct pages using cleaned lines + per-page heuristic cleanup.
    # `zip(..., strict=True)` ensures page alignment stays correct (no silent truncation).
    return [
        ExtractedPage(
            page_number=p.page_number,
            text=_clean_page_text("\n".join(lines))
        )
        for p, lines in zip(pages, cleaned_lines, strict=True)
    ]


# --- Execution Pipeline ---

def run_ingest(pdf_content: bytes, base_path: Path | None = None) -> str:
    """
        Orchestrates the end-to-end ingestion pipeline for health insurance documents.

        This function manages the lifecycle of a document upload by performing the following steps:
        1. Generates a unique UUID for the session to ensure document isolation.
        2. Saves the raw binary PDF to local persistent storage.
        3. Extracts text and performs 'is_likely_policy' validation to prevent non-insurance uploads.
        4. Segments the validated text into searchable chunks and generates vector embeddings.
        5. Synchronizes the chunks with ChromaDB using strict doc_id metadata filtering.

        Args:
            pdf_content (bytes): The raw file data uploaded by the user.
            base_path (Path, optional): Override for the default document storage directory.

        Returns:
            str: The generated doc_id, used for subsequent summary and Q&A requests.

        Raises:
            ValueError: If the PDF is empty, lacks extractable text, or fails policy validation keywords.
        """
    # Generate a new document id to isolate this upload in storage and the vector DB.
    doc_id = storage.generate_document_id()

    # Persist the raw PDF bytes so the ingestion pipeline can operate on a stable file path.
    pdf_path = storage.save_raw_pdf(pdf_content, doc_id, base_path)

    try:
        # Extract pages with boundary preservation and header/footer cleanup enabled.
        pages = extract_pages(pdf_path, clean_headers_footers=True)

        # If there are no pages or all pages have no text, reject early (cannot proceed).
        if not pages or sum(len(p.text or "") for p in pages) == 0:
            raise ValueError("PDF has no extractable text")

        # --- NEW: Policy Validation Step ---
        # Combine text from the first few pages (where SBC summaries and plan details often appear)
        # and ensure the document contains sufficient insurance-related terminology.
        sample_text = " ".join([p.text or "" for p in pages[:5]])
        if not is_likely_policy(sample_text):
            raise ValueError(
                "Validation Failed: This document does not appear to be a health insurance "
                "policy or Summary of Benefits. Please upload a valid SBC PDF."
            )
        # -----------------------------------

        # Persist extracted pages for traceability/debugging and potential UI display.
        storage.save_extracted_pages(pages, doc_id, base_path)

        # Chunk page text into retrieval units (each chunk retains page_number + doc_id).
        chunks = chunk_pages(pages, doc_id)

        # If chunking produced nothing, abort (retrieval cannot function without chunks).
        if not chunks:
            raise ValueError("PDF produced no chunks")

        # Persist chunks to disk so they can be reused without re-parsing.
        storage.save_chunks(chunks, doc_id, base_path)

        # Add chunks to the vector database for embedding-based retrieval.
        # This assumes `storage.add_chunks` handles embeddings/indexing internally.
        storage.add_chunks(doc_id, chunks)

    except Exception as e:
        # Cleanup on failure:
        # If ingestion fails midway, remove the document directory to avoid leaving partial artifacts
        # (e.g., raw PDF saved but chunks missing).
        doc_dir = storage.get_document_dir(doc_id, base_path)
        if doc_dir.exists():
            import shutil  # local import to preserve runtime behavior and avoid unused-import warnings

            shutil.rmtree(doc_dir)

        # Re-raise the original exception so the API layer can translate it to an HTTP error.
        raise e  # Let the API handle the error message

    # On success, return the doc_id for subsequent summary/Q&A requests.
    return doc_id