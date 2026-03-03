"""
Evaluation Module: The Judge.

This module centralizes validation, confidence scoring, and quantitative evaluation
metrics for PolicyExplainer outputs. It is designed to be used by the backend to:

1) Validate structural correctness of section summaries and QA responses
2) Assign heuristic "confidence" labels (high/medium/low)
3) Compute quantitative metrics:
   - Faithfulness (Groundedness): Are summary bullets supported by cited chunks?
   - Completeness (Coverage): Are required policy sections addressed?

High-level approach:

- Validation functions focus on schema/consistency checks (e.g., citations exist,
  page numbers look valid, chunk IDs match expected format).
- Confidence functions are simple heuristics layered on top of validation results
  and presence/coverage signals.
- Metric computations load stored artifacts (summaries/chunks) via `backend.storage`
  and compute scores using lightweight lexical/numeric overlap checks.

Notes:
- This module intentionally uses deterministic heuristics (regex/token overlap)
  to avoid adding additional LLM calls to evaluation.
- No functionality is performed here beyond reading stored artifacts and returning
  JSON-serializable metric payloads.
"""

import json
import re
from pathlib import Path
from typing import Any

from backend import storage
from backend.schemas import SectionSummaryWithConfidence, BulletWithCitations

# --- Constants & Patterns ---

# Standard UI format for showing citations to end users.
# (Currently unused in this file, but kept as a central definition for consistency.)
USER_FACING_CITATION_FORMAT = "(p. {page})"

# Regex used to split text into sentence-like units for simple verbosity checks.
# This is intentionally naive (punctuation-based) but fast and deterministic.
SENTENCE_PATTERN = re.compile(r"[.!?]+")

# Weighting for Completeness Score based on Business/Policy Importance.
# These weights allow the coverage score to prioritize critical sections
# such as costs and covered services over less critical sections.
SECTION_WEIGHTS: dict[str, float] = {
    "Plan Snapshot": 0.05,
    "Cost Summary": 0.35,
    "Summary of Covered Services": 0.30,
    "Administrative Conditions": 0.15,
    "Exclusions & Limitations": 0.10,
    "Claims, Appeals & Member Rights": 0.05,
}


# --- Internal Helpers ---

def _count_sentences(text: str) -> int:
    """
    Count sentence-like segments in text.

    This is used as a simple verbosity/readability proxy. The logic is:
    - Split on punctuation (., !, ?)
    - Count the number of non-empty segments

    Args:
        text: Input text to analyze.

    Returns:
        int: Number of detected sentence segments.
    """
    # Guard against None/empty/whitespace-only input.
    if not text or not text.strip():
        return 0

    # Split by sentence punctuation and count non-empty segments.
    return len([p for p in SENTENCE_PATTERN.split(text.strip()) if p.strip()])


def _section_addressed(sec: SectionSummaryWithConfidence) -> bool:
    """
    Determine whether a policy section is considered "addressed" for completeness scoring.

    A section is considered addressed if:
    - The section is NOT present in the document (present=False), and the model correctly
      recognized that fact; OR
    - The section is present and contains at least one bullet with citations.

    Rationale:
    - Completeness is about coverage/handling, not necessarily the number of bullets.
    - If a section doesn't exist in the source document, it's still "addressed" if the
      summary explicitly acknowledges it is missing.

    Args:
        sec: Section summary output (with presence flag and bullets).

    Returns:
        bool: True if addressed, else False.
    """
    # If the section truly isn't in the document and the system marks it as not present,
    # we treat that as addressed (it was handled correctly).
    if not sec.present:
        return True

    # If the section is present but no bullets were produced, it's not addressed.
    if not sec.bullets:
        return False

    # If any bullet has at least one citation, we consider the section grounded/covered.
    return any(b.citations for b in sec.bullets)


def _normalize_tokens(text: str) -> set[str]:
    """
    Normalize text into a set of tokens for overlap analysis.

    Tokenization here is intentionally simple:
    - Lowercase
    - Extract alphanumeric tokens via regex
    - Return unique tokens (set)

    This supports fast, deterministic similarity checks between bullet text and chunk text.

    Args:
        text: Raw text.

    Returns:
        set[str]: A set of normalized tokens.
    """
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _extract_numbers(text: str) -> set[str]:
    """
    Extract numeric strings from text.

    This is used for crude "hard fact" checks: if a bullet mentions specific numbers
    (copays, deductibles, coinsurance rates, etc.), ensure the cited chunk contains
    those numbers too.

    Args:
        text: Raw text.

    Returns:
        set[str]: Set of numeric substrings (e.g., {"500", "20", "0.2"}).
    """
    return set(re.findall(r"\d+\.?\d*", text))


def _chunk_supports_bullet(bullet_text: str, chunk: Any, min_overlap: float = 0.15) -> bool:
    """
    Verify whether a chunk plausibly supports a summary bullet.

    This function is a heuristic "support" test. It does NOT attempt true semantic
    entailment, but it is designed to catch obvious mismatches and hallucinations.

    It uses two checks (either can pass):
      1) Token overlap check:
         If the overlap ratio between bullet tokens and chunk tokens is above a
         small threshold, treat as supported.
      2) Numeric containment check:
         If the bullet contains numbers and ALL bullet numbers are contained in the
         chunk numbers, treat as supported (strong signal for cost facts).

    Args:
        bullet_text: The bullet text produced by the model.
        chunk: The cited chunk object (backend storage schema varies; accessed via getattr).
        min_overlap: Minimum overlap ratio needed for the token overlap check.

    Returns:
        bool: True if the chunk supports the bullet under these heuristics.
    """
    # Normalize bullet text into token set.
    bullet_tokens = _normalize_tokens(bullet_text)

    # Pull chunk text in a backend-agnostic way.
    # Using getattr keeps compatibility if chunk schema changes or multiple backends exist.
    chunk_text = getattr(chunk, "chunk_text", "") or ""
    chunk_tokens = _normalize_tokens(chunk_text)

    # If bullet has no tokens (e.g., empty or only punctuation), treat as trivially supported.
    # This prevents division by zero and avoids penalizing malformed bullets here.
    if not bullet_tokens:
        return True

    # --- Check 1: Keyword overlap (lightweight grounding proxy) ---
    # Ratio = (# tokens shared) / (# tokens in bullet).
    # Note: This is asymmetric (relative to bullet length) by design:
    # we want to ensure the bullet's content is present in the chunk.
    if len(bullet_tokens & chunk_tokens) / len(bullet_tokens) >= min_overlap:
        return True

    # --- Check 2: Numeric fact checking (stronger signal for cost-related bullets) ---
    bullet_nums = _extract_numbers(bullet_text)
    chunk_nums = _extract_numbers(chunk_text)

    # If bullet includes numeric values and all of them appear in the chunk, consider supported.
    # This helps catch hallucinated copays/deductibles that are not in the cited chunk.
    if bullet_nums and bullet_nums <= chunk_nums:
        return True

    # If neither heuristic passes, treat as unsupported.
    return False


# --- Validation Logic ---
def validate_section_summary(
        section_out: SectionSummaryWithConfidence,
        detail_level: str = "standard",
) -> tuple[bool, list[str]]:
    """
    Validate a section summary output for citation integrity and bullet count rules.

    This validation checks:
    - If section is marked not present => automatically valid (nothing to cite).
    - Bullet count constraints:
        * standard: 3–6 bullets
        * detailed: 6–12 bullets
      (Note: fewer than min bullets triggers an issue only if there is at least 1 bullet;
       this avoids penalizing truly empty output the same way as "too short" output.)
    - Each bullet must have at least one citation.
    - Each citation must have:
        * valid page number (> 0)
        * chunk_id that starts with "c_" (expected internal format)

    Args:
        section_out: The section summary output (with bullets/citations/presence).
        detail_level: "standard" or a more verbose level (treated as "detailed" here).

    Returns:
        tuple:
          - is_valid (bool): True if no issues found
          - issues (list[str]): list of issue codes for downstream use/debug/UI
    """
    issues: list[str] = []

    # If the section isn't present in the document, there is nothing to validate.
    # The system explicitly acknowledging absence is treated as correct behavior.
    if not section_out.present:
        return (True, [])

    # Normalize bullets to an empty list so downstream loops can assume list semantics.
    bullets = section_out.bullets or []

    # Choose bullet count expectations based on detail level.
    # Standard summaries should be concise; detailed should be more comprehensive.
    min_b, max_b = (3, 6) if detail_level == "standard" else (6, 12)

    # Too many bullets can indicate verbosity/rambling.
    if len(bullets) > max_b:
        issues.append(f"bullet_count_high: {len(bullets)} bullets (max {max_b})")

    # Too few bullets can indicate under-coverage; only flag if there is at least one bullet.
    # (If there are zero bullets, other logic may treat the section as unaddressed.)
    if len(bullets) < min_b and len(bullets) > 0:
        issues.append(f"bullet_count_low: {len(bullets)} bullets (min {min_b})")

    # Validate each bullet and its citations.
    for i, b in enumerate(bullets):
        # Every summary point MUST have at least one citation to be considered faithful.
        if not b.citations:
            issues.append(f"bullet_{i + 1}_missing_citations")

        # Validate each citation entry for basic sanity.
        for c in b.citations:
            # Page numbers should be positive; 0/negative is treated as hallucinated/invalid.
            if c.page <= 0:
                issues.append(f"bullet_{i + 1}_invalid_page_number: {c.page}")

            # Chunk IDs should match our expected internal format.
            # The summarization pipeline/storage layer typically uses "c_<...>" IDs.
            if not c.chunk_id or not str(c.chunk_id).startswith("c_"):
                issues.append(f"bullet_{i + 1}_invalid_chunk_id: {c.chunk_id}")

    # A section is valid only if there are no issues.
    return (len(issues) == 0, issues)


# --- Confidence Scoring ---

def confidence_for_section(
        section_out: SectionSummaryWithConfidence
) -> str:
    """
    Determine heuristic confidence label for a section summary.

    This confidence is NOT a probabilistic score; it is an interpretable label
    based on validation signals and citation coverage.

    Rules:
    - If section not present OR no bullets => "low"
    - If validation issues exist:
        * If any are "critical" (invalid/missing) => "low"
        * Else => "medium"
    - If every bullet has at least one citation => "high"
    - Otherwise => "medium"

    Args:
        section_out: The section summary output with validation metadata.

    Returns:
        str: "high", "medium", or "low"
    """
    # We now pull the issues directly from the object we built in summarization.py
    issues = section_out.validation_issues or []

    # If the section isn't there or has no bullets, it's a low-confidence state.
    # (Either the content is missing or we don't have grounded statements to trust.)
    if not section_out.present or not section_out.bullets:
        return "low"

    total_bullets = len(section_out.bullets)
    bullets_with_citations = sum(1 for b in section_out.bullets if b.citations)

    # If there are issues, classify them and decide confidence.
    if issues:
        # "Critical" issues are those indicating missing grounding or invalid references.
        # These are strong reasons to distrust the section output.
        critical = [i for i in issues if any(x in i.lower() for x in ["invalid", "missing"])]
        if critical:
            return "low"
        return "medium"

    # If every bullet has at least one citation, treat as high confidence.
    if bullets_with_citations >= total_bullets:
        return "high"

    # Otherwise, some bullets may be weakly grounded; treat as medium.
    return "medium"


# --- Main Metrics Computation ---

def compute_faithfulness(doc_id: str, base_path: Path | None = None) -> dict[str, Any]:
    """
    Compute faithfulness (groundedness) score for a document summary.

    Faithfulness is measured by verifying each bullet is supported by at least one of
    its cited chunks using `_chunk_supports_bullet(...)`.

    Process:
    - Load stored summary and chunk list for the given doc_id.
    - Build a chunk lookup dictionary by chunk_id.
    - For each bullet in each present section:
        * Count it as one evaluation unit
        * Mark it "supported" if ANY cited chunk passes support heuristics
    - Score = supported_units / total_units

    Args:
        doc_id: Document identifier.
        base_path: Optional storage override (e.g., for tests or alternate storage roots).

    Returns:
        dict: Includes doc_id, faithfulness_score (0–1), and total_units.
              If data is missing, returns a payload with error information.
    """
    try:
        # Load stored artifacts needed for grounding checks.
        summary = storage.load_policy_summary(doc_id, base_path)
        chunks_list = storage.load_chunks(doc_id, base_path)
    except FileNotFoundError:
        # If either summary or chunks are missing, evaluation cannot proceed reliably.
        return {"error": "data_missing", "faithfulness_score": 0.0}

    # Index chunks by ID for fast citation lookup.
    chunks_by_id = {c.chunk_id: c for c in chunks_list}

    # Each bullet is treated as a single evaluation unit.
    total_units = supported_units = 0

    # Iterate through each section and each bullet to evaluate support.
    for sec in summary.sections:
        # Skip sections not present or without bullets; they do not contribute to faithfulness units.
        if not sec.present or not sec.bullets:
            continue

        for b in sec.bullets:
            total_units += 1
            is_supported = False

            # A bullet is supported if at least one cited chunk supports it.
            for cit in b.citations:
                # Look up the cited chunk by ID.
                ch = chunks_by_id.get(cit.chunk_id)

                # Apply support heuristics: token overlap and/or numeric containment.
                if ch and _chunk_supports_bullet(b.text, ch):
                    is_supported = True
                    break

            if is_supported:
                supported_units += 1

    # Avoid division by zero by using (total_units or 1) in the denominator.
    return {
        "doc_id": doc_id,
        "faithfulness_score": round(supported_units / (total_units or 1), 4),
        "total_units": total_units
    }


def compute_completeness(doc_id: str, base_path: Path | None = None) -> dict[str, Any]:
    """
    Compute completeness (coverage) score for a document summary.

    Completeness is a weighted coverage score across policy sections.
    Each section is marked as:
      - 1.0 if addressed (see `_section_addressed`)
      - 0.0 otherwise

    The final score is the weighted average across SECTION_WEIGHTS.

    Args:
        doc_id: Document identifier.
        base_path: Optional storage override.

    Returns:
        dict: Includes doc_id, completeness_score (0–1), and per-section scores.
              If the summary is missing, returns an error payload.
    """
    try:
        # Completeness depends only on the summary structure/output.
        summary = storage.load_policy_summary(doc_id, base_path)
    except FileNotFoundError:
        return {"doc_id": doc_id, "error": "summary_missing", "completeness_score": 0.0}

    section_scores = {}
    weighted_sum = 0.0
    total_weight = sum(SECTION_WEIGHTS.values())

    # Evaluate each section in the produced summary.
    for sec in summary.sections:
        name = sec.section_name

        # Weight defaults to 0.0 for unknown section names (i.e., not in SECTION_WEIGHTS).
        weight = SECTION_WEIGHTS.get(name, 0.0)

        # Addressed = either correctly marked missing, or has at least one cited bullet.
        addressed = _section_addressed(sec)

        # Store a binary score for explainability in the output payload.
        section_scores[name] = 1.0 if addressed else 0.0

        # Accumulate weighted score.
        weighted_sum += weight * section_scores[name]

    # Compute weighted average; guard against divide-by-zero if weights somehow sum to 0.
    return {
        "doc_id": doc_id,
        "completeness_score": round(weighted_sum / (total_weight or 1), 4),
        "section_scores": section_scores,
    }


def run_all_evaluations(doc_id: str, base_path: Path | None = None) -> dict[str, Any]:
    """
    Run the full evaluation suite and return a compact metrics payload.

    This is the top-level entry point typically called by the API layer.
    It orchestrates:
    - Faithfulness computation
    - Completeness computation

    Args:
        doc_id: Document identifier.
        base_path: Optional storage override.

    Returns:
        dict: {"doc_id": ..., "faithfulness": <float>, "completeness": <float>}
    """
    # Compute each metric independently so partial failures can still return defaults.
    f_rep = compute_faithfulness(doc_id, base_path)
    c_rep = compute_completeness(doc_id, base_path)

    # Normalize output to a stable API contract.
    return {
        "doc_id": doc_id,
        "faithfulness": f_rep.get("faithfulness_score", 0.0),
        "completeness": c_rep.get("completeness_score", 0.0)
    }


# -- qa validation set --
def validate_qa_response(
        response_json: dict[str, Any],
        *,
        valid_page_numbers: set[int] | None = None,
) -> tuple[bool, list[str], str]:
    """
    Validate a QA response payload for required fields and citation sanity.

    Current validation rules:
    - A disclaimer must be present (response_json["disclaimer"] truthy).
    - If `valid_page_numbers` is provided, each citation page must be an int and
      must exist in the allowed set.

    Args:
        response_json: The QA response dict (typically already model_dump()'d).
        valid_page_numbers: Optional set of page numbers considered valid for this doc.

    Returns:
        tuple:
          - is_valid (bool): True if no issues
          - issues (list[str]): issue codes for downstream consumption
          - display_text (str): the answer text for UI display
    """
    issues = []

    # Pull core fields from the response with safe defaults.
    answer = response_json.get("answer", "")
    citations = response_json.get("citations", [])

    # Disclaimers are mandatory for UI/legal/policy reasons.
    if not response_json.get("disclaimer"):
        issues.append("disclaimer_required")

    # If the caller supplies a valid page set, verify cited pages belong to it.
    if valid_page_numbers:
        for c in citations:
            p = c.get("page")
            if isinstance(p, int) and p not in valid_page_numbers:
                issues.append(f"invalid_page_citation:{p}")

    # Return (is_valid, issues, display_text)
    return len(issues) == 0, issues, answer


def confidence_for_qa(
        answer_type: str,
        citation_count: int,
        *,
        validation_issues: list[str] | None = None,
        retrieval_chunk_count: int = 0,
) -> str:
    """
    Determine heuristic confidence label for QA responses.

    Rules:
    - If answer type is "not_found" OR retrieval returned no chunks => "low"
    - If any validation issue contains "invalid" => "low"
    - If there are at least 2 citations AND at least 3 retrieved chunks => "high"
    - Otherwise => "medium"

    Rationale:
    - No retrieval / not-found indicates we have little grounding.
    - Invalid citations are critical integrity failures.
    - Multiple citations + multiple retrieved chunks suggests stronger grounding context.

    Args:
        answer_type: Classification from QA pipeline (e.g., "standard", "scenario", "not_found").
        citation_count: Number of citations included in the answer.
        validation_issues: Optional list of validation issue codes.
        retrieval_chunk_count: Number of chunks considered during retrieval.

    Returns:
        str: "high", "medium", or "low"
    """
    issues = validation_issues or []

    # If we couldn't retrieve evidence or explicitly didn't find an answer, confidence is low.
    if answer_type == "not_found" or retrieval_chunk_count == 0:
        return "low"

    # Any invalid citation-related issue is treated as a hard failure for trust.
    if any("invalid" in i for i in issues):
        return "low"

    # Strong evidence signals: multiple citations and broader retrieval context.
    if citation_count >= 2 and retrieval_chunk_count >= 3:
        return "high"

    # Default case: some evidence exists but isn't strong enough to be labeled "high".
    return "medium"