"""
Evaluation Module: The Judge.

This module centralizes validation, confidence scoring, and quantitative evaluation
metrics for PolicyExplainer outputs. It is designed to be used by the backend to:

1) Validate structural correctness of section summaries and QA responses
2) Assign heuristic "confidence" labels (high/medium/low)
3) Compute quantitative metrics:
   - Faithfulness: Are summary bullets supported by cited chunks without introducing
     unsupported or contradictory claims?
   - Completeness: Are required policy sections addressed?
   - Simplicity: Is the generated summary easier to read than the original policy text?

High-level approach:

- Validation functions focus on schema/consistency checks (e.g., citations exist,
  page numbers look valid, chunk IDs match expected format).
- Confidence functions are simple heuristics layered on top of validation results
  and presence/coverage signals.
- Metric computations load stored artifacts (summaries/chunks) via `backend.storage`
  and compute scores using deterministic lexical/numeric checks.

Faithfulness logic:
- A bullet is evaluated against its cited chunks.
- A bullet is considered supported if at least one cited chunk supports it via:
    1) lexical token overlap, or
    2) numeric match with compatible policy context
- A bullet is considered contradictory only if a cited chunk signals:
    1) numeric mismatch, or
    2) wrong-context numeric use
- Faithfulness penalizes both hallucinated and contradictory bullets.

Simplicity logic:
- Uses Flesch Reading Ease.
- Compares original document text vs generated summary text.
- Higher summary Flesch score means easier readability.

Notes:
- This module intentionally uses deterministic heuristics (regex/token overlap,
  numeric checks, keyword-pair checks) to avoid adding additional LLM calls.
- No functionality is performed here beyond reading stored artifacts and returning
  JSON-serializable metric payloads.
"""

import re
from pathlib import Path
from typing import Any

from backend import storage
from backend.schemas import SectionSummaryWithConfidence

# --- Constants & Patterns ---

USER_FACING_CITATION_FORMAT = "(p. {page})"

SENTENCE_PATTERN = re.compile(r"[.!?]+")

CONTEXT_KEYWORDS: set[str] = {
    "copay", "copayment", "coinsurance", "deductible", "premium", "maximum", "max",
    "oop", "out", "pocket", "limit", "visit", "visits", "day", "days", "year", "years",
    "month", "months", "network", "authorization", "referral", "emergency", "urgent",
    "preventive", "diagnostic", "primary", "care", "specialist", "generic", "brand",
    "drug", "prescription", "hospital", "inpatient", "outpatient", "therapy",
    "rehabilitation", "mental", "health", "maternity", "dental", "vision",
}

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
    """
    if not text or not text.strip():
        return 0
    return len([p for p in SENTENCE_PATTERN.split(text.strip()) if p.strip()])


def _section_addressed(sec: SectionSummaryWithConfidence) -> bool:
    """
    Determine whether a policy section is considered addressed for completeness scoring.
    """
    if not sec.present:
        return True

    if not sec.bullets:
        return False

    return any(b.citations for b in sec.bullets)


def _normalize_tokens(text: str) -> set[str]:
    """
    Normalize text into a set of alphanumeric lowercase tokens.
    """
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def _extract_numbers(text: str) -> set[str]:
    """
    Extract numeric substrings from text.
    """
    return set(re.findall(r"\d+\.?\d*", text or ""))


def _extract_context_keywords(text: str) -> set[str]:
    """
    Extract domain-specific context keywords from text.
    """
    tokens = _normalize_tokens(text)
    return {t for t in tokens if t in CONTEXT_KEYWORDS}


def _number_has_matching_context(
    bullet_text: str,
    chunk_text: str,
    *,
    min_shared_context: int = 1,
) -> bool:
    """
    Verify that numeric overlap also occurs in a compatible policy context.
    """
    bullet_nums = _extract_numbers(bullet_text)
    chunk_nums = _extract_numbers(chunk_text)

    if not bullet_nums:
        return False

    if not bullet_nums.issubset(chunk_nums):
        return False

    bullet_context = _extract_context_keywords(bullet_text)
    chunk_context = _extract_context_keywords(chunk_text)

    return len(bullet_context & chunk_context) >= min_shared_context


def _chunk_supports_bullet(bullet_text: str, chunk: Any, min_overlap: float = 0.15) -> bool:
    """
    Verify whether a chunk plausibly supports a summary bullet.
    """
    bullet_tokens = _normalize_tokens(bullet_text)
    chunk_text = getattr(chunk, "chunk_text", "") or ""
    chunk_tokens = _normalize_tokens(chunk_text)

    if not bullet_tokens:
        return True

    if len(bullet_tokens & chunk_tokens) / len(bullet_tokens) >= min_overlap:
        return True

    if _number_has_matching_context(bullet_text, chunk_text):
        return True

    return False


def _chunk_contradicts_bullet(bullet_text: str, chunk: Any) -> bool:
    """
    Verify whether a chunk contradicts a summary bullet using numeric mismatch rules.
    """
    chunk_text = getattr(chunk, "chunk_text", "") or ""
    bullet_nums = _extract_numbers(bullet_text)

    if bullet_nums and not _number_has_matching_context(bullet_text, chunk_text):
        return True

    return False


def _count_words(text: str) -> int:
    """
    Count words using regex.
    """
    return len(re.findall(r"\b\w+\b", text or ""))


def _estimate_syllables(word: str) -> int:
    """
    Estimate syllables in a word using a simple heuristic.
    """
    word = (word or "").lower()
    vowels = "aeiouy"

    count = 0
    prev_char_was_vowel = False

    for char in word:
        if char in vowels:
            if not prev_char_was_vowel:
                count += 1
            prev_char_was_vowel = True
        else:
            prev_char_was_vowel = False

    if word.endswith("e") and count > 1:
        count -= 1

    return max(count, 1)


def _count_syllables(text: str) -> int:
    """
    Estimate total syllables in text.
    """
    words = re.findall(r"\b\w+\b", text or "")
    return sum(_estimate_syllables(w) for w in words)


def _flesch_reading_ease(text: str) -> float:
    """
    Compute the Flesch Reading Ease score.
    """
    sentences = _count_sentences(text)
    words = _count_words(text)
    syllables = _count_syllables(text)

    if sentences == 0 or words == 0:
        return 0.0

    return (
        206.835
        - 1.015 * (words / sentences)
        - 84.6 * (syllables / words)
    )


# --- Validation Logic ---

def validate_section_summary(
    section_out: SectionSummaryWithConfidence,
    detail_level: str = "standard",
) -> tuple[bool, list[str]]:
    """
    Validate a section summary output for citation integrity and bullet count rules.
    """
    issues: list[str] = []

    if not section_out.present:
        return True, []

    bullets = section_out.bullets or []
    min_b, max_b = (3, 6) if detail_level == "standard" else (6, 12)

    if len(bullets) > max_b:
        issues.append(f"bullet_count_high: {len(bullets)} bullets (max {max_b})")

    if 0 < len(bullets) < min_b:
        issues.append(f"bullet_count_low: {len(bullets)} bullets (min {min_b})")

    for i, b in enumerate(bullets):
        if not b.citations:
            issues.append(f"bullet_{i + 1}_missing_citations")

        for c in b.citations:
            if c.page <= 0:
                issues.append(f"bullet_{i + 1}_invalid_page_number: {c.page}")

            if not c.chunk_id or not str(c.chunk_id).startswith("c_"):
                issues.append(f"bullet_{i + 1}_invalid_chunk_id: {c.chunk_id}")

    return len(issues) == 0, issues


# --- Confidence Scoring ---

def confidence_for_section(section_out: SectionSummaryWithConfidence) -> str:
    """
    Determine heuristic confidence label for a section summary.
    """
    issues = section_out.validation_issues or []

    if not section_out.present or not section_out.bullets:
        return "low"

    total_bullets = len(section_out.bullets)
    bullets_with_citations = sum(1 for b in section_out.bullets if b.citations)

    if issues:
        critical = [i for i in issues if any(x in i.lower() for x in ["invalid", "missing"])]
        if critical:
            return "low"
        return "medium"

    if bullets_with_citations >= total_bullets:
        return "high"

    return "medium"


# --- Main Metrics Computation ---

def compute_faithfulness(doc_id: str, base_path: Path | None = None) -> dict[str, Any]:
    """
    Compute faithfulness score for a document summary.
    """
    try:
        summary = storage.load_policy_summary(doc_id, base_path)
        chunks_list = storage.load_chunks(doc_id, base_path)
    except FileNotFoundError:
        return {
            "doc_id": doc_id,
            "error": "data_missing",
            "faithfulness_score": 0.0,
        }

    chunks_by_id = {c.chunk_id: c for c in chunks_list}

    total_units = 0
    hallucinated_units = 0
    contradictory_units = 0
    debug_details: list[dict[str, Any]] = []

    for sec in summary.sections:
        if not sec.present or not sec.bullets:
            continue

        for b in sec.bullets:
            total_units += 1
            is_supported = False
            is_contradictory = False

            bullet_debug = {
                "section": sec.section_name,
                "bullet_text": b.text,
                "citations": [],
                "result": None,
            }

            for cit in b.citations:
                ch = chunks_by_id.get(cit.chunk_id)
                chunk_text = getattr(ch, "chunk_text", "") if ch else ""

                citation_debug = {
                    "chunk_id": cit.chunk_id,
                    "page": cit.page,
                    "support_match": False,
                    "contradiction_match": False,
                    "reason": [],
                    "chunk_preview": chunk_text[:300],
                }

                if not ch:
                    citation_debug["reason"].append("chunk_not_found")
                    bullet_debug["citations"].append(citation_debug)
                    continue

                if _chunk_contradicts_bullet(b.text, ch):
                    is_contradictory = True
                    citation_debug["contradiction_match"] = True

                    bullet_nums = _extract_numbers(b.text)
                    if bullet_nums and not _number_has_matching_context(b.text, chunk_text):
                        citation_debug["reason"].append("number_mismatch_or_wrong_context")

                if _chunk_supports_bullet(b.text, ch):
                    is_supported = True
                    citation_debug["support_match"] = True
                    citation_debug["reason"].append("supported_by_chunk")

                bullet_debug["citations"].append(citation_debug)

            if is_contradictory:
                contradictory_units += 1
                bullet_debug["result"] = "contradictory"
            elif not is_supported:
                hallucinated_units += 1
                bullet_debug["result"] = "hallucinated"
            else:
                bullet_debug["result"] = "supported"

            debug_details.append(bullet_debug)

    faithfulness_score = 1 - ((hallucinated_units + contradictory_units) / (total_units or 1))

    print("\n=== FAITHFULNESS DEBUG REPORT ===")
    print(f"Document ID: {doc_id}")
    print(f"Faithfulness Score: {round(faithfulness_score, 4)}")
    print(f"Total Units: {total_units}")
    print(f"Hallucinated Units: {hallucinated_units}")
    print(f"Contradictory Units: {contradictory_units}")

    for i, detail in enumerate(debug_details, start=1):
        if detail["result"] in {"contradictory", "hallucinated"}:
            print(f"\n--- Bullet {i} ---")
            print(f"Section: {detail['section']}")
            print(f"Result: {detail['result']}")
            print(f"Bullet: {detail['bullet_text']}")

            for c in detail["citations"]:
                print(f"  Citation -> chunk_id={c['chunk_id']}, page={c['page']}")
                print(f"    support_match={c['support_match']}")
                print(f"    contradiction_match={c['contradiction_match']}")
                print(f"    reason={c['reason']}")
                print(f"    chunk_preview={c['chunk_preview']}")

    return {
        "doc_id": doc_id,
        "faithfulness_score": round(faithfulness_score, 4),
        "total_units": total_units,
        "hallucinated_units": hallucinated_units,
        "contradictory_units": contradictory_units,
        "debug_details": debug_details,
    }


def compute_completeness(doc_id: str, base_path: Path | None = None) -> dict[str, Any]:
    """
    Compute completeness (coverage) score for a document summary.
    """
    try:
        summary = storage.load_policy_summary(doc_id, base_path)
    except FileNotFoundError:
        return {
            "doc_id": doc_id,
            "error": "summary_missing",
            "completeness_score": 0.0,
        }

    section_scores = {}
    weighted_sum = 0.0
    total_weight = sum(SECTION_WEIGHTS.values())

    for sec in summary.sections:
        name = sec.section_name
        weight = SECTION_WEIGHTS.get(name, 0.0)
        addressed = _section_addressed(sec)

        section_scores[name] = 1.0 if addressed else 0.0
        weighted_sum += weight * section_scores[name]

    return {
        "doc_id": doc_id,
        "completeness_score": round(weighted_sum / (total_weight or 1), 4),
        "section_scores": section_scores,
    }


def compute_simplicity(doc_id: str, base_path: Path | None = None) -> dict[str, Any]:
    """
    Compute the Simplicity Score using Flesch Reading Ease.

    Simplicity measures how much easier the generated summary is
    to read compared to the original policy document.

    Returns:
        dict containing:
        - simplicity_score (0–1)
        - original_flesch
        - summary_flesch
        - improvement
    """
    try:
        pages = storage.load_extracted_pages(doc_id, base_path)
        summary = storage.load_policy_summary(doc_id, base_path)
    except FileNotFoundError:
        return {
            "doc_id": doc_id,
            "error": "data_missing",
            "simplicity_score": 0.0,
        }

    original_text = " ".join((p.text or "") for p in pages).strip()

    summary_text = " ".join(
        (b.text or "")
        for sec in summary.sections
        for b in sec.bullets
    ).strip()

    if not original_text or not summary_text:
        return {
            "doc_id": doc_id,
            "error": "empty_text",
            "simplicity_score": 0.0,
        }

    original_flesch = _flesch_reading_ease(original_text)
    summary_flesch = _flesch_reading_ease(summary_text)

    improvement = summary_flesch - original_flesch
    simplicity_score = max(0.0, min(improvement / 100.0, 1.0))

    print("[Simplicity Evaluation]")
    print(f"Original Flesch Score: {original_flesch:.2f}")
    print(f"Summary Flesch Score: {summary_flesch:.2f}")
    print(f"Improvement: {improvement:.2f}")
    print(f"Simplicity Score: {simplicity_score:.3f}")

    return {
        "doc_id": doc_id,
        "original_flesch": round(original_flesch, 2),
        "summary_flesch": round(summary_flesch, 2),
        "improvement": round(improvement, 2),
        "simplicity_score": round(simplicity_score, 4),
    }


def run_all_evaluations(doc_id: str, base_path: Path | None = None) -> dict[str, Any]:
    """
    Run the full evaluation suite and return a compact metrics payload.
    """
    f_rep = compute_faithfulness(doc_id, base_path)
    c_rep = compute_completeness(doc_id, base_path)
    s_rep = compute_simplicity(doc_id, base_path)

    return {
        "doc_id": doc_id,
        "faithfulness": f_rep.get("faithfulness_score", 0.0),
        "completeness": c_rep.get("completeness_score", 0.0),

        # Main simplicity value shown in UI
        "simplicity": s_rep.get("summary_flesch", 0.0),

        # Extra details for dropdown / expandable section
        "summary_flesch": s_rep.get("summary_flesch"),
        "original_flesch": s_rep.get("original_flesch"),
        "improvement": s_rep.get("improvement"),
        "simplicity_score": s_rep.get("simplicity_score", 0.0),
    }


# -- QA validation set --

def validate_qa_response(
    response_json: dict[str, Any],
    *,
    valid_page_numbers: set[int] | None = None,
) -> tuple[bool, list[str], str]:
    """
    Validate a QA response payload for required fields and citation sanity.
    """
    issues = []

    answer = response_json.get("answer", "")
    citations = response_json.get("citations", [])

    if not response_json.get("disclaimer"):
        issues.append("disclaimer_required")

    if valid_page_numbers:
        for c in citations:
            p = c.get("page")
            if isinstance(p, int) and p not in valid_page_numbers:
                issues.append(f"invalid_page_citation:{p}")

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
    """
    issues = validation_issues or []

    if answer_type == "not_found" or retrieval_chunk_count == 0:
        return "low"

    if any("invalid" in i for i in issues):
        return "low"

    if citation_count >= 2 and retrieval_chunk_count >= 3:
        return "high"

    return "medium"