"""
Q&A Module: Grounded RAG, Section Deep-Dive, and Scenario Generation.

This module is the PolicyExplainer "routing engine" for user questions. It provides:
- Standard retrieval-augmented Q&A (RAG) grounded strictly in retrieved chunks.
- Section-level "deep dive" summaries for core policy sections.
- Hypothetical scenario generation (e.g., cost walk-throughs) constrained to policy terms.
- Lightweight conversational handling (greetings/small talk) to avoid unnecessary retrieval/LLM calls.
- Dynamic FAQ generation based on an already-produced document summary.

Core guarantees / design goals:
- Data fidelity: Answers should be based ONLY on retrieved policy chunks or known summary artifacts.
- Citation integrity: Citations returned by the LLM are filtered to only include chunk IDs that were
  actually retrieved for the request (strict citation whitelisting).
- Schema adherence: All public functions return objects conforming to the defined Pydantic schemas,
  or JSON payloads that map cleanly into them.

Important implementation details:
- OpenAI responses are requested in JSON mode via `response_format={"type": "json_object"}`.
  However, the `_parse_llm_json` helper still defensively strips markdown fences and tries to
  recover JSON from imperfect responses.
- Confidence labeling is heuristic and derived from evidence availability and citation coverage.
"""

import json
import re
from typing import Any

from openai import OpenAI

from backend import storage
from backend.config import get_settings
from backend.retrieval import CORE_SECTIONS, retrieve_for_section
from backend.schemas import (
    Citation,
    ScenarioStepOutput,
    NOT_FOUND_MESSAGE,
    ScenarioQAResponseOutput,
    QAResponseOutput,
    SectionSummaryWithConfidence
)
from backend.summarization import summarize_section
from backend.utils import load_terminology_map, normalize_text

# Standard "not found" answer used for user-facing responses when evidence is missing.
NOT_FOUND_ANSWER = "I couldn't find an answer to that specific question in this policy document."

# Disclaimer included with all QA outputs for compliance / expectations management.
QA_RESPONSE_DISCLAIMER = "This explanation is for informational purposes only. Refer to official policy documents."

# Regex patterns used to detect user intent for deep-dive / detailed responses.
DETAIL_INTENT_PATTERNS = [
    r"more\s+detail\s+about", r"in\s+more\s+detail", r"deeper\s+summary\s+of",
    r"detailed\s+summary\s+of", r"deep\s+dive\s+(?:into|on)",
]

# Substring triggers used to classify scenario-style questions.
SCENARIO_TRIGGER_PHRASES = ["what would happen if", "example scenario", "how much would i pay if"]

# Catch simple greetings before hitting the database/vector store.
# This reduces latency and avoids expensive calls for conversational inputs.
GREETING_PATTERNS = [
    r"^(hi|hello|hey|greetings|good morning|good afternoon|good evening)\b",
    r"^how are you", r"^who are you", r"^what can you do"
]


def _qa_build_context(chunks: list[dict[str, Any]]) -> str:
    """
    Build an LLM-ready context string from retrieved chunks.

    The context format includes:
    - chunk_id (for traceability)
    - page_number (for user-facing citations)
    - chunk_text (the actual evidence)

    Args:
        chunks: List of chunk dicts returned from `storage.query(...)`.

    Returns:
        str: A compact, formatted context string.
    """
    parts = []
    for c in chunks:
        # Each chunk is formatted as a small block; separators help the model "see" boundaries.
        parts.append(
            f"---\nChunk {c.get('chunk_id', '')} (page {c.get('page_number', 0)}):\n"
            f"{(c.get('chunk_text') or '').strip()}\n"
        )
    return "\n".join(parts).strip()


def _parse_llm_json(raw: str) -> dict:
    """
    Parse JSON output from the LLM, stripping common markdown wrappers.

    Even with JSON response_format, some models/tools may include:
    - ```json fences
    - ``` fences
    - extra leading/trailing whitespace

    This function:
    1) strips fences if present
    2) attempts direct json.loads
    3) falls back to extracting the first {...} block via regex

    Args:
        raw: Raw text returned by the LLM.

    Returns:
        dict: Parsed JSON object (empty dict if parsing fails).
    """
    raw = raw.strip()

    # Remove common markdown code fence wrappers that break JSON parsing.
    if raw.startswith("```json"):
        raw = raw[7:]
    if raw.startswith("```"):
        raw = raw[3:]
    if raw.endswith("```"):
        raw = raw[:-3]

    try:
        # First attempt: parse the full cleaned string.
        return json.loads(raw.strip())
    except Exception:
        # Fallback: attempt to recover an object-shaped substring.
        match = re.search(r"(\{.*\})", raw, re.DOTALL)
        return json.loads(match.group(1)) if match else {}


def handle_greeting(doc_id: str, question: str) -> QAResponseOutput:
    """
    Handle greetings/small-talk without retrieval or LLM calls.

    This keeps the system responsive and prevents unnecessary vector DB queries.

    Args:
        doc_id: Document identifier (included for schema consistency).
        question: Original user text.

    Returns:
        QAResponseOutput: A friendly response with no citations required.
    """
    return QAResponseOutput(
        doc_id=doc_id,
        question=question,
        answer="Hello! I am your Policy Explainer assistant. How can I help you understand your insurance document today?",
        answer_type="answerable",
        citations=[],
        confidence="high",
        disclaimer=QA_RESPONSE_DISCLAIMER
    )


def ask(doc_id: str, question: str, top_k: int = 6) -> QAResponseOutput:
    """
    Standard grounded RAG Q&A for a policy document.

    Steps:
    1) Normalize the question.
    2) Retrieve top_k relevant chunks from vector store.
    3) Build an evidence context string.
    4) Ask the LLM to answer using ONLY that context, requesting JSON output.
    5) Post-process answer text (terminology normalization).
    6) Strictly filter citations to only allow retrieved chunk IDs (whitelist).
    7) Assign heuristic confidence based on citation presence and retrieval breadth.

    Args:
        doc_id: Document identifier.
        question: User question.
        top_k: Number of chunks to retrieve for grounding.

    Returns:
        QAResponseOutput: Structured answer with citations and confidence label.
    """
    # Normalize question to avoid empty strings and accidental whitespace-only queries.
    question = (question or "").strip()

    # Retrieve relevant chunks from storage/vector DB.
    # Expected chunk schema: dict with at least chunk_id, page_number, chunk_text.
    chunks = storage.query(doc_id, question, top_k=top_k)

    # Build a whitelist of allowed chunk IDs for strict citation filtering later.
    # This prevents hallucinated citations that do not correspond to retrieved evidence.
    allowed_ids = {str(c.get("chunk_id")) for c in chunks if c.get("chunk_id")}

    # If retrieval returns nothing, immediately return a not_found response.
    if not chunks:
        return QAResponseOutput(
            doc_id=doc_id,
            question=question,
            answer=NOT_FOUND_ANSWER,
            answer_type="not_found",
            citations=[],
            confidence="low",
            disclaimer=QA_RESPONSE_DISCLAIMER
        )

    # Convert chunks to an LLM-readable context.
    context = _qa_build_context(chunks)

    # Load runtime settings (API key, model name, etc.) from environment.
    settings = get_settings()

    # Initialize OpenAI client with API key (never log or expose).
    client = OpenAI(api_key=settings.openai_api_key)

    # System prompt enforces strict grounding, conversational handling, and JSON-only output.
    system_prompt = """You are a strictly factual Policy Q&A system. 
    1. Answer using ONLY the provided chunks.
    2. If the user's input is conversational (e.g., "thanks", "ok"), respond politely and leave "citations" empty [].
    3. If the answer is NOT in the chunks, say exactly "Not found in this document." and leave "citations" empty [].
    4. Otherwise, provide the answer and cite your sources.
    
    Output ONLY valid JSON containing: 
    {"answer": "your text", "answer_type": "answerable", "citations": [{"chunk_id": "c_1_0", "page": 1}]}
    """

    # Call the model in JSON mode.
    response = client.chat.completions.create(
        model=settings.llm_model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            # Provide both evidence context and the user question in one message for clarity.
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ],
        temperature=0.1  # Low temperature to reduce creativity/hallucination.
    )

    # Parse the JSON response; defensive parsing handles occasional fence wrapping.
    parsed = _parse_llm_json(response.choices[0].message.content or "")

    # Load terminology map for rewriting plan-specific terms into clearer language.
    term_map = load_terminology_map()

    # Extract answer; if missing, fall back to a generic not-found.
    raw_answer = parsed.get("answer") or NOT_FOUND_ANSWER

    # Catch LLM attempting to answer unanswerable questions.
    # The system prompt instructs the exact phrase "Not found in this document."
    # We interpret that as a not-found state and return the standard user-facing message.
    if "not found in this document" in raw_answer.lower():
        answer_text = NOT_FOUND_ANSWER
        answer_type = "not_found"
    else:
        # Normalize answer text for readability/consistency using the term map.
        answer_text = normalize_text(raw_answer, term_map)

        # The model may return answer_type; default to "answerable" if absent.
        answer_type = parsed.get("answer_type", "answerable")

    # STRICT CITATION FILTERING:
    # Only accept citations whose chunk_id exists in `allowed_ids` (retrieved evidence).
    citations = []
    for c in parsed.get("citations", []):
        c_id = c.get("chunk_id")

        # Whitelist check: reject hallucinated chunk references.
        if c_id in allowed_ids:
            try:
                # Normalize and validate page number.
                page_num = int(c.get('page', 0))
                if page_num > 0:
                    citations.append(Citation(page=page_num, chunk_id=str(c_id)))
            except (ValueError, TypeError):
                # If the LLM returns invalid page types (None/str/etc.), skip citation.
                continue

    # Heuristic confidence:
    # - high: citations present AND enough retrieved context
    # - medium: citations present but retrieval context is thinner
    # - low: no citations
    confidence = "high" if citations and len(chunks) >= 3 else "medium" if citations else "low"

    # Return schema-compliant QAResponseOutput for API serialization.
    return QAResponseOutput(
        doc_id=doc_id,
        question=question,
        answer=answer_text,
        answer_type=answer_type,
        citations=citations,
        confidence=confidence,
        disclaimer=QA_RESPONSE_DISCLAIMER
    )


def ask_scenario(doc_id: str, question: str, scenario_type: str = "General") -> ScenarioQAResponseOutput:
    """
    Generate a hypothetical cost scenario grounded in policy terms.

    This pathway is triggered for questions that look like scenario prompts (e.g., "What would
    happen if I go to the ER?"). Instead of retrieving by the exact user question (which can be
    underspecified), we query using a cost-terms-focused prompt to retrieve relevant benefits
    and cost-sharing rules.

    Steps:
    1) Construct a cost-focused retrieval query.
    2) Retrieve chunks and build allowed_ids whitelist.
    3) Ask the LLM to output scenario steps in JSON.
    4) If model reports not_found, return not-found scenario response.
    5) Normalize step text and filter citations to allowed_ids.
    6) Return ScenarioQAResponseOutput.

    Args:
        doc_id: Document identifier.
        question: Original scenario question from the user.
        scenario_type: Scenario label (e.g., "ER", "General") used to guide retrieval.

    Returns:
        ScenarioQAResponseOutput: Stepwise scenario with citations and confidence.
    """
    # Retrieval query aims at cost-sharing language often needed for scenario calculation.
    query = f"{scenario_type} deductible copay coinsurance out of pocket"

    # Retrieve a slightly larger context for scenarios, since they often require multiple facts.
    chunks = storage.query(doc_id, query, top_k=8)

    # Whitelist chunk IDs that the model is allowed to cite.
    allowed_ids = {str(c.get("chunk_id")) for c in chunks}

    # If nothing is retrieved, return not-found scenario response immediately.
    if not chunks:
        return ScenarioQAResponseOutput(
            doc_id=doc_id,
            question=question,
            scenario_type=scenario_type,
            not_found_message=NOT_FOUND_MESSAGE,
            confidence="low",
            disclaimer=QA_RESPONSE_DISCLAIMER
        )

    # Build formatted context for the LLM.
    context = _qa_build_context(chunks)

    # Load settings and initialize OpenAI client.
    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)

    # System prompt: enforce strict grounding, step-wise output, and JSON-only format.
    system_prompt = """You generate hypothetical cost scenarios based strictly on policy terms. 
    Use ONLY provided chunks. If you cannot calculate a scenario based on the chunks, set "not_found" to true.
    Output ONLY valid JSON containing: 
    {"steps": [{"step_number": 1, "text": "description", "citations": [{"chunk_id": "c_1_0", "page": 1}]}], "not_found": false}
    """

    response = client.chat.completions.create(
        model=settings.llm_model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            # Provide evidence context and the scenario prompt.
            {"role": "user", "content": f"Context: {context}\nScenario: {question}"}
        ],
        temperature=0.1  # Low temperature to keep outputs consistent and grounded.
    )

    parsed = _parse_llm_json(response.choices[0].message.content or "")

    # If the model cannot produce a scenario grounded in chunks, return not-found.
    if not parsed or parsed.get("not_found"):
        return ScenarioQAResponseOutput(
            doc_id=doc_id,
            question=question,
            scenario_type=scenario_type,
            not_found_message=NOT_FOUND_MESSAGE,
            confidence="low",
            disclaimer=QA_RESPONSE_DISCLAIMER
        )

    # Normalize terminology for readability and consistency.
    term_map = load_terminology_map()
    final_steps = []

    # Transform model steps into Pydantic ScenarioStepOutput objects.
    for i, s in enumerate(parsed.get("steps", [])):
        text = normalize_text(s.get("text", ""), term_map)

        # Filter citations to only those chunk IDs that were retrieved.
        cites = []
        for c in s.get("citations", []):
            if c.get("chunk_id") in allowed_ids:
                try:
                    p_num = int(c.get("page", 0))
                    if p_num > 0:
                        cites.append(Citation(page=p_num, chunk_id=c.get("chunk_id")))
                except (ValueError, TypeError):
                    continue

        # Preserve step_number if provided; otherwise fall back to sequential numbering.
        final_steps.append(
            ScenarioStepOutput(
                step_number=s.get("step_number", i + 1),
                text=text,
                citations=cites
            )
        )

    # Confidence heuristic for scenarios:
    # More steps generally implies richer grounded coverage (though still heuristic).
    return ScenarioQAResponseOutput(
        doc_id=doc_id,
        question=question,
        scenario_type=scenario_type,
        steps=final_steps,
        confidence="high" if len(final_steps) >= 3 else "medium",
        disclaimer=QA_RESPONSE_DISCLAIMER
    )


def _handle_section_detail(doc_id: str, question: str, section_name: str) -> QAResponseOutput:
    """
    Handle a deep-dive request for a specific core policy section.

    This path:
    - retrieves chunks specifically for the requested section
    - runs the summarization pipeline in "detailed" mode
    - formats the result as a bulleted textual answer
    - aggregates unique citations across bullets

    Args:
        doc_id: Document identifier.
        question: Original user question (used for response schema).
        section_name: Exact section name from CORE_SECTIONS.

    Returns:
        QAResponseOutput: Section detail response with merged citations.
    """
    # Retrieve evidence targeted to the section requested.
    chunks = retrieve_for_section(doc_id, section_name)

    # Generate a detailed summary for this section.
    summary: SectionSummaryWithConfidence = summarize_section(section_name, chunks, detail_level="detailed")

    # Render bullets into a readable answer string (still grounded in summary output).
    answer_text = f"Detailed overview of {section_name}:\n" + "\n".join([f"- {b.text}" for b in summary.bullets])

    # Merge citations across bullets while de-duplicating by chunk_id.
    all_citations = []
    seen_chunks = set()
    for b in summary.bullets:
        for c in b.citations:
            if c.chunk_id not in seen_chunks:
                all_citations.append(c)
                seen_chunks.add(c.chunk_id)

    return QAResponseOutput(
        doc_id=doc_id,
        question=question,
        answer=answer_text,
        answer_type="section_detail",
        citations=all_citations,
        confidence=summary.confidence,
        disclaimer=QA_RESPONSE_DISCLAIMER,
        validation_issues=summary.validation_issues
    )


def route_question(doc_id: str, question: str) -> Any:
    """
    Route a user question to the appropriate handler based on intent.

    Routing order matters (fast/cheap checks first):
    1) Greetings/small talk -> handle_greeting (no retrieval/LLM)
    2) Scenario triggers   -> ask_scenario
    3) Deep-dive requests  -> section-specific summarization in detailed mode
    4) Default             -> standard grounded RAG via ask()

    Args:
        doc_id: Document identifier.
        question: Raw user question.

    Returns:
        Any: Typically QAResponseOutput or ScenarioQAResponseOutput depending on route.
    """
    q_lower = (question or "").strip().lower()

    # 1. Catch Greetings & Small Talk First
    # Use regex search to match greetings at the start or common small-talk patterns.
    if any(re.search(p, q_lower) for p in GREETING_PATTERNS):
        return handle_greeting(doc_id, question)

    # 2. Catch Scenario Triggers
    # Use simple substring checks for common scenario formulations.
    if any(phrase in q_lower for phrase in SCENARIO_TRIGGER_PHRASES):
        # Lightweight scenario type classification (currently only special-casing ER/emergency).
        scenario_type = "ER" if "er" in q_lower or "emergency" in q_lower else "General"
        return ask_scenario(doc_id, question, scenario_type=scenario_type)

    # 3. Catch Deep Dive Requests
    # Look for language indicating the user wants a detailed overview of a section.
    if any(re.search(p, q_lower) for p in DETAIL_INTENT_PATTERNS):
        # If the question contains a core section name, treat it as a section deep dive.
        for section in CORE_SECTIONS:
            if section.lower() in q_lower:
                return _handle_section_detail(doc_id, question, section)

    # 4. Standard RAG Q&A
    return ask(doc_id, question)


# Add this to the bottom of backend/qa.py

def generate_document_faqs(doc_id: str) -> dict:
    """
    Generate dynamic FAQs (4–5) based on the document's existing summary.

    This endpoint leverages already-generated summary content (if available) to:
    - produce plan-specific questions a user is likely to ask
    - provide concise answers
    - avoid re-reading the entire source document

    Fallback behavior:
    - If the summary cannot be loaded (missing or error), generate general insurance FAQs.

    Args:
        doc_id: Document identifier.

    Returns:
        dict: JSON object shaped like:
              {"faqs": [{"question": "...", "answer": "..."}]}
              (parsed from model output via `_parse_llm_json`).
    """
    try:
        # Load the summary we already generated during ingestion.
        summary = storage.load_policy_summary(doc_id)

        # Build a compact context string from the summary bullets to keep token usage low.
        context = ""

        for sec in summary.sections:
            if sec.present:
                context += f"{sec.section_name}:\n"
                # Include only the top few bullets per section to fit within prompt budgets.
                for b in sec.bullets[:3]:  # Top 3 bullets per section to save tokens
                    context += f"- {b.text}\n"
    except Exception:
        # If anything goes wrong (missing summary, parse errors), fall back to a generic prompt.
        context = "No summary available. Generate general health insurance FAQs."

    # Load runtime settings and initialize OpenAI client.
    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)

    system_prompt = """You are a helpful insurance assistant. Based on the provided policy summary, generate 4 to 5 Frequently Asked Questions (FAQs) that a user might have specifically about this plan.
    Keep answers clear, accurate, and concise.
    Output ONLY valid JSON containing a list of faqs: {"faqs": [{"question": "...", "answer": "..."}]}
    """

    response = client.chat.completions.create(
        model=settings.llm_model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Policy Summary:\n{context}"}
        ],
        temperature=0.3  # Slightly higher temperature to diversify FAQ phrasing while staying grounded.
    )

    # Parse and return the JSON payload from the model response.
    return _parse_llm_json(response.choices[0].message.content or "")