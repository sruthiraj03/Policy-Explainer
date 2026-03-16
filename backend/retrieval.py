"""
Retrieval Module: The "Search Engine" of PolicyExplainer.

This module is responsible for retrieving the most relevant document chunks for
each canonical Policy Summary section. It implements a lightweight multi-query
retrieval strategy to improve recall on heterogeneous insurance language.

Why multi-query retrieval?
- Insurance documents often use inconsistent wording for the same concept.
  For example, "Cost Summary" content may appear under "cost sharing", "deductible",
  "out-of-pocket maximum", etc. Searching with several targeted sub-queries increases
  the chance of retrieving the right evidence.

Core responsibilities:
- Define the canonical section names (CORE_SECTIONS) aligned to schemas.py literals.
- Define section-specific query expansions (SECTION_QUERIES).
- Provide `retrieve_for_section(...)` to:
    * run multiple sub-queries against the vector store
    * deduplicate overlapping hits across queries
    * keep the best match per chunk based on vector distance
    * return chunks sorted by page order for coherent LLM context

Output contract:
- Retrieval returns a list of dicts (as provided by `storage.query`) with an added
  "section" field indicating which section the chunk was retrieved for.
"""

from typing import Any, Final
from backend import storage

# --- Constants ---

# Canonical policy sections: These align with your SectionName Literal in schemas.py.
# Using Final ensures these cannot be changed accidentally during runtime.
# Updated retrieval.py keys to match schemas.py Literals.
CORE_SECTIONS: Final[tuple[str, ...]] = (
    "Plan Snapshot",
    "Cost Summary",
    "Summary of Covered Services",  # Match Schema
    "Administrative Conditions",
    "Exclusions & Limitations",
    "Claims, Appeals & Member Rights"  # Match Schema
)

# Multi-Query Mapping:
# NLP Strategy: Instead of searching only for section titles, search for multiple
# synonymous queries. This improves recall because policies use varied jargon.
SECTION_QUERIES: Final[dict[str, tuple[str, ...]]] = {
    "Plan Snapshot": (
        "plan name and type",
        "summary of benefits overview",
        "plan overview and key features",
    ),
    "Cost Summary": (
        "deductible amount and when it applies",
        "copay and coinsurance",
        "out of pocket maximum OOP",
        "annual deductible",
        "cost sharing requirements",
    ),
    "Summary of Covered Services": (
        "what is covered",
        "covered benefits and services",
        "coverage details",
        "covered medical services",
        "benefits included in plan",
    ),
    "Administrative Conditions": (
        "prior authorization",
        "referrals required",
        "administrative requirements",
    ),
    "Exclusions & Limitations": (
        "exclusions not covered",
        "limitations and restrictions",
        "what is not covered",
    ),
    "Claims, Appeals & Member Rights": (
        "how to file a claim",
        "appeals and grievances",
        "member rights and responsibilities",
    ),
}

# Retrieval tuning knobs:
# - TOP_K_PER_QUERY: How many hits to request for each sub-query.
# - MAX_CHUNKS_SECTION: Global cap to avoid passing too much context to the LLM.
TOP_K_PER_QUERY = 3
MAX_CHUNKS_SECTION = 10


def retrieve_for_section(
    doc_id: str,
    section_name: str,
    top_k_per_query: int = TOP_K_PER_QUERY,
    max_chunks: int = MAX_CHUNKS_SECTION,
) -> list[dict[str, Any]]:
    """
    Retrieve relevant chunks for a specific policy section using multi-query expansion.

    Flow:
    1) Look up the section's list of sub-queries from SECTION_QUERIES.
    2) For each sub-query, call `storage.query(...)` against the vector store.
    3) Deduplicate hits across queries by chunk_id:
        - The same chunk may be retrieved by multiple sub-queries.
        - Keep the hit with the smallest `distance` (closest vector match).
    4) Sort results by page_number (and chunk_id for deterministic ordering) so downstream
       LLM summarization reads the document in its natural order.
    5) Return at most `max_chunks` to reduce "lost in the middle" effects in long contexts.

    Args:
        doc_id: Identifier for the ingested document (used to scope vector retrieval).
        section_name: Canonical section name (must exist in SECTION_QUERIES to retrieve).
        top_k_per_query: Number of vector hits requested per sub-query.
        max_chunks: Maximum number of chunks returned for this section after deduping/sorting.

    Returns:
        list[dict[str, Any]]: Retrieved chunk dicts enriched with `section` metadata.
                              If section_name is unknown, returns [].
    """
    # Fetch sub-queries for this section. If none exist, retrieval cannot proceed.
    queries = SECTION_QUERIES.get(section_name)
    if not queries:
        return []

    # `seen` maps chunk_id -> best hit dict so far for that chunk.
    # Best is defined as the smallest vector distance (highest similarity).
    seen: dict[str, dict[str, Any]] = {}

    for q in queries:
        # Skip accidental empty strings in query list (defensive).
        if not q.strip():
            continue

        # storage.query connects to the underlying vector database (e.g., ChromaDB).
        # It should return a list of dicts containing chunk metadata and similarity info.
        hits = storage.query(doc_id, q.strip(), top_k=top_k_per_query)

        for h in hits:
            # chunk_id is the stable identifier used across the pipeline (citations, storage, etc.).
            cid = h.get("chunk_id") or ""
            if not cid:
                # If a hit lacks a chunk_id, we cannot use it reliably downstream.
                continue

            # --- Deduping & Distance Logic ---
            # If the same chunk appears for multiple queries, keep the "best" match.
            # We define best as the smallest distance (closest in vector space).
            if cid not in seen:
                # First time we see this chunk_id: store it as the current best.
                out = dict(h)
                out["section"] = section_name  # annotate for downstream explainability
                seen[cid] = out
            else:
                # Compare vector distance if present.
                d = h.get("distance")

                # Lower distance = higher similarity.
                # Only update if both distances are present and the new one is better.
                if d is not None and seen[cid].get("distance") is not None and d < seen[cid]["distance"]:
                    out = dict(h)
                    out["section"] = section_name
                    seen[cid] = out

    # --- Sorting for Context ---
    # Sort by page_number so the LLM sees content in document order.
    # Secondary sort by chunk_id provides deterministic ordering within the same page.
    ordered = sorted(
        seen.values(),
        key=lambda x: (x.get("page_number", 0), x.get("chunk_id", ""))
    )

    # Cap the results to prevent overly long contexts that harm LLM performance.
    return ordered[:max_chunks]