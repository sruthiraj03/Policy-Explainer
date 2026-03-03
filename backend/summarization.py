"""
Summarization Module: The LLM Orchestrator.

This module transforms retrieved policy chunks into structured, schema-validated summaries.

Primary responsibilities:
- Build an LLM-readable "context" block from retrieved chunks while preserving:
    * chunk_id (for traceability)
    * page_number (for user-facing citations)
- Call the LLM in JSON mode to produce a structured section summary.
- Enforce grounding by filtering citations to only the retrieved chunk IDs.
- Normalize terminology in the generated text for clarity and consistent wording.
- Validate the resulting Pydantic model (citation integrity, bullet counts, etc.).
- Assign a heuristic confidence label (high/medium/low) based on validation outcomes.
- Orchestrate full-document summarization across all canonical sections and persist results.

Key safety/quality controls:
- JSON-only output requested via `response_format={"type": "json_object"}` plus defensive parsing.
- Strict citation filtering: discard any citations not referring to retrieved chunks.
- Conservative temperature to reduce hallucinations.
- Bullet caps to avoid overly verbose outputs and "lost in the middle" effects.
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from openai import OpenAI

from backend import storage
from backend.config import get_settings
from backend.evaluation import confidence_for_section, validate_section_summary
from backend.retrieval import CORE_SECTIONS, retrieve_for_section
from backend.schemas import (
    BulletWithCitations,
    Citation,
    DEFAULT_DISCLAIMER,
    DocMetadata,
    NOT_FOUND_MESSAGE,
    PolicySummaryOutput,
    SectionSummaryWithConfidence,  # Use the consolidated model
)
from backend.utils import load_terminology_map, normalize_text

# Define DetailLevel locally as it is a specific logic toggle for the pipeline.
# - "standard": concise summaries (fewer bullets)
# - "detailed": deeper coverage (more bullets)
DetailLevel = Literal["standard", "detailed"]

# Consistency constants for bullet counts.
# These caps protect downstream LLM reasoning and keep UI output readable.
STANDARD_MAX_BULLETS = 6
DETAILED_MAX_BULLETS = 12


def _build_context(chunks: list[dict[str, Any]]) -> str:
    """
    Convert retrieved chunk dicts into a structured text block for the LLM.

    Each chunk is labeled with:
    - chunk_id (used later as an allowed citation key)
    - page number (used for user-facing citations)

    This labeling is critical: it gives the LLM explicit identifiers to reference in JSON output.

    Args:
        chunks: Retrieval hits from the vector store (storage.query / retrieve_for_section).

    Returns:
        str: Formatted context string or "" if chunks is empty.
    """
    if not chunks:
        return ""

    parts = []
    for c in chunks:
        # Default to 0/"" if missing to keep formatting stable.
        page = c.get("page_number", 0)
        cid = c.get("chunk_id", "")
        text = (c.get("chunk_text") or "").strip()

        # Separate chunks with a delimiter to make boundaries visually obvious.
        parts.append(f"---\nChunk {cid} (page {page}):\n{text}\n")

    return "\n".join(parts).strip()


def _parse_llm_json(raw: str) -> dict[str, Any] | None:
    """
    Robust JSON extraction from an LLM response.

    Even when using JSON mode, it is common to encounter responses that include:
    - markdown fences ```json ... ```
    - leading conversational text
    - extra whitespace

    Strategy:
    1) Attempt to extract a fenced JSON object if present.
    2) Otherwise, slice from the first '{' to the last '}'.
    3) Attempt json.loads; return None on decode failure.

    Args:
        raw: Raw response content string from the LLM.

    Returns:
        dict[str, Any] | None: Parsed JSON dict, or None if parsing fails.
    """
    s = raw.strip()

    # Prefer extracting JSON inside ```json fences if present.
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", s, re.DOTALL)
    if m:
        s = m.group(1)
    else:
        # Fallback: attempt to carve out the first full JSON object.
        start, end = s.find("{"), s.rfind("}")
        if start != -1 and end != -1 and end > start:
            s = s[start: end + 1]

    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return None


def summarize_section(
        section_name: str,
        chunks: list[dict[str, Any]],
        detail_level: DetailLevel = "standard",
) -> SectionSummaryWithConfidence:
    """
    Generate a structured summary for a single policy section.

    This function maps directly to the consolidated `SectionSummaryWithConfidence` model
    (rather than producing an intermediate output), which enables:
    - downstream validation using evaluation.py
    - confidence scoring based on validation and citation coverage

    Grounding controls:
    - Only citations referencing retrieved chunk IDs are kept (allowed_ids whitelist).
    - Bullets without at least one valid citation are discarded.

    Args:
        section_name: Canonical section name (must align with schemas.py literals).
        chunks: Retrieval hits for this section.
        detail_level: "standard" or "detailed" (controls bullet cap).

    Returns:
        SectionSummaryWithConfidence: Validated, citation-filtered summary output.
    """
    # Build a whitelist of allowed chunk IDs. This prevents hallucinated citations.
    allowed_ids = {str(c.get("chunk_id")) for c in chunks if c.get("chunk_id")}

    # Pre-define the "not found" state. This is returned if:
    # - no chunks were retrieved
    # - the model indicates the section isn't present
    # - parsing or API calls fail
    empty_res = SectionSummaryWithConfidence(
        section_name=section_name,
        present=False,
        bullets=[],
        not_found_message=NOT_FOUND_MESSAGE,
        confidence="low",
        validation_issues=["No relevant document chunks found."]
    )

    # If retrieval returned nothing, skip the LLM call.
    if not chunks:
        return empty_res

    # Build formatted evidence context for the model.
    context = _build_context(chunks)

    # Load runtime settings and create OpenAI client.
    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)

    # System prompt: enforce strict JSON shape and strict grounding.
    # Note: This is intentionally minimal and structural; deeper stylistic guidance can be
    # added without changing logic, but behavior is preserved as-is here.
    system_prompt = f"""You are a policy document summarizer. Use ONLY provided chunks.
    You MUST output your response as a valid JSON object with exactly these keys:
    - "present": boolean (true if info is found, false if not)
    - "bullets": a list of dicts, each with "text" (the summary point) and "citations" (a list of dicts with "chunk_id" and "page").
    """

    # Call the model and parse JSON output.
    try:
        response = client.chat.completions.create(
            model=settings.llm_model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Summarize {section_name} using: {context}"},
            ],
            temperature=0.1,  # Low temperature to reduce hallucination
        )

        raw = (response.choices[0].message.content or "").strip()

        # Debug snippet to help diagnose formatting/parsing issues in development logs.
        print(f"🤖 DEBUG SUMMARY LLM OUTPUT for {section_name}: {raw[:100]}...")

        parsed = _parse_llm_json(raw)

    except Exception as e:
        # Preserve current behavior: log and return the empty result.
        print(f"❌ DEBUG: OpenAI API Call Failed for {section_name}: {e}")
        return empty_res

    # If parsing fails or model indicates the section isn't present, return not-found state.
    if not parsed or not parsed.get("present"):
        return empty_res

    # 1) Normalize terminology and validate citations.
    # - Normalize bullet text for readability (plan-specific term map).
    # - Filter citations to allowed chunk IDs only.
    # - Force page to int to satisfy Pydantic typing and avoid runtime validation errors.
    term_map = load_terminology_map()
    valid_bullets = []

    for b in parsed.get("bullets", []):
        # Normalize text output for consistent phrasing and expanded terminology.
        text = normalize_text(b.get("text", ""), term_map)

        cites = []
        for c in b.get("citations", []):
            # Only keep citations that refer to chunks we actually retrieved.
            if str(c.get("chunk_id")) in allowed_ids:
                # Proactive fix: coerce page to int to avoid Pydantic type errors.
                try:
                    page_num = int(c.get('page', 0))
                except (ValueError, TypeError):
                    page_num = 0

                cites.append(Citation(page=page_num, chunk_id=str(c.get("chunk_id"))))

        # Only keep bullets that have at least one valid source.
        if cites:
            valid_bullets.append(BulletWithCitations(text=text, citations=cites))

    # 2) Apply bullet cap based on detail level to keep output bounded and consistent.
    final_bullets = valid_bullets[:DETAILED_MAX_BULLETS if detail_level == "detailed" else STANDARD_MAX_BULLETS]

    # Construct the preliminary summary object first.
    # This ensures the downstream validator/confidence logic can safely read `.present`, `.bullets`, etc.
    preliminary_summary = SectionSummaryWithConfidence(
        section_name=section_name,
        present=True if final_bullets else False,
        bullets=final_bullets,
        not_found_message=None if final_bullets else NOT_FOUND_MESSAGE,
        confidence="low",  # Temporary placeholder; updated after validation/scoring
        validation_issues=[]
    )

    # 3) Validate the complete object (citation integrity, bullet count rules, etc.).
    # validate_section_summary returns (is_valid, issues); we store issues for transparency.
    _, issues = validate_section_summary(preliminary_summary)
    preliminary_summary.validation_issues = issues

    # 4) Compute and set final confidence label based on validation outcomes and citation coverage.
    conf = confidence_for_section(preliminary_summary)
    preliminary_summary.confidence = conf

    return preliminary_summary


def run_full_summary_pipeline(
    doc_id: str,
    detail_level: DetailLevel = "standard",
    base_path: Path | None = None,
) -> PolicySummaryOutput:
    """
    Run full-document summarization across all canonical policy sections.

    Steps:
    1) Load extracted pages to compute total_pages for metadata.
    2) For each section in CORE_SECTIONS:
        - retrieve relevant chunks via retrieve_for_section(...)
        - summarize the section via summarize_section(...)
    3) Construct PolicySummaryOutput payload with metadata and disclaimer.
    4) Persist the summary to storage for later use by the frontend and FAQ generation.

    Args:
        doc_id: Document identifier.
        detail_level: "standard" or "detailed" summaries.
        base_path: Optional override for storage paths (useful for tests).

    Returns:
        PolicySummaryOutput: Complete summary payload for the document.
    """
    # Load metadata for audit trail and UI display.
    total_pages = len(storage.load_extracted_pages(doc_id, base_path))
    metadata = DocMetadata(
        doc_id=doc_id,
        generated_at=datetime.now(tz=timezone.utc).isoformat(),
        total_pages=total_pages
    )

    # Summarize each canonical section.
    final_sections = []
    for section_name in CORE_SECTIONS:
        # Fetch relevant chunks from the vector store.
        chunks = retrieve_for_section(doc_id, section_name)

        # Convert chunks into a structured summary for this section.
        summary = summarize_section(section_name, chunks, detail_level)

        final_sections.append(summary)

    # Build final output payload.
    full_output = PolicySummaryOutput(
        metadata=metadata,
        disclaimer=DEFAULT_DISCLAIMER,
        sections=final_sections
    )

    # Persist the summary so the frontend and other modules (e.g., FAQ generation) can reuse it.
    storage.save_policy_summary(full_output, doc_id, base_path)
    return full_output