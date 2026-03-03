"""
API and Domain Models (Pydantic).

This module defines the typed "contracts" between:
- The LLM (structured outputs we ask the model to produce)
- The backend (parsing/validation, storage, routing, evaluation)
- The frontend/UI (stable response shapes for rendering)

Why this matters:
- Pydantic enforces strict schemas at runtime.
- If the LLM hallucinates a field name, wrong type, or missing required field,
  the code fails early with a validation error rather than propagating bad data.
- These models also serve as a single source of truth for the API response formats.

Design goals:
- Grounding: Important statements are represented as bullets with citations.
- Traceability: Citations include a chunk_id pointing to the exact retrieved chunk.
- Safety: Missing information should be marked explicitly via `present=False` or
  `answer_type="not_found"` rather than hallucinating.
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field

# --- Constants & Literals ---

# Strict section-name vocabulary.
# This should stay aligned with:
# - retrieval.py CORE_SECTIONS
# - summarization schema expectations
# - UI section rendering logic
SectionName = Literal[
    "Plan Snapshot",
    "Cost Summary",
    "Summary of Covered Services",
    "Administrative Conditions",
    "Exclusions & Limitations",
    "Claims, Appeals & Member Rights"
]

# Standardized confidence labels (kept categorical rather than numeric for UI simplicity).
ConfidenceLevel = Literal["high", "medium", "low"]

# Categorizes the type of Q&A response for different UI rendering logic.
# The UI can use this field to decide which component/template to show.
QA_ANSWER_TYPE = Literal[
    "answerable",     # Direct answer found in doc
    "not_found",      # Information missing; refer to website
    "ambiguous",      # Doc is unclear
    "conflict",       # Two parts of the doc say different things
    "section_detail", # Deep dive into one policy section
    "scenario"        # Example: 'If I break my leg, what do I pay?'
]

# Standard message used when evidence is not found in the document context.
NOT_FOUND_MESSAGE = "Not found in this document."

# Legal/UX disclaimer used for summaries and QA outputs (can be overridden per response).
DEFAULT_DISCLAIMER = (
    "This summary is for informational purposes only. It does not replace the full policy document."
)

# --- Base Components ---

class Citation(BaseModel):
    """
    A single evidence reference supporting a claim.

    Each citation ties a statement back to:
    - `page`: human-readable location in the PDF (1-based)
    - `chunk_id`: exact retrieval unit ID in the vector store / persisted chunks

    This enables:
    - UI display of page references
    - Auditing and debugging by mapping chunk_id back to stored chunk text
    """
    page: int = Field(..., ge=1, description="1-based page number from the PDF")
    chunk_id: str = Field(..., description="The unique ID of the text chunk (e.g., c_1_15)")


class BulletWithCitations(BaseModel):
    """
    A single atomic policy statement with supporting citations.

    The pipeline produces bullet lists rather than a single paragraph to:
    - force smaller, verifiable claims
    - encourage coverage of multiple aspects of a section
    - simplify UI rendering and citation display

    Every bullet should ideally have at least one citation for grounding.
    """
    text: str = Field(..., description="The policy detail written in plain English")
    citations: list[Citation] = Field(default_factory=list, description="Source links for this bullet")

# --- Section Summaries ---

class SectionSummaryBase(BaseModel):
    """
    Base structure for a policy section summary.

    Key fields:
    - section_name: which canonical section this summary refers to
    - present: "truth flag" indicating whether the information exists in the document
    - bullets: list of extracted grounded statements (may be empty if present=False)
    - not_found_message: optional explanation when present=False

    The 'present' flag is critical: if the system cannot find evidence for a section,
    it should mark present=False instead of hallucinating content.
    """
    section_name: SectionName
    present: bool
    bullets: list[BulletWithCitations] = Field(default_factory=list)
    not_found_message: Optional[str] = Field(default=None)


class SectionSummaryWithConfidence(SectionSummaryBase):
    """
    Section summary extended with evaluation metadata.

    Added fields:
    - confidence: categorical trust label (high/medium/low)
    - validation_issues: list of issue codes/notes describing missing or questionable data

    This structure allows:
    - the API/UI to show a confidence indicator to the user
    - internal evaluation modules to track and debug extraction failures
    """
    confidence: ConfidenceLevel
    validation_issues: list[str] = Field(
        default_factory=list,
        description="Notes on missing data or extraction errors"
    )

# --- Full Policy Summary ---

class DocMetadata(BaseModel):
    """
    Metadata describing the processed document.

    This is included in summary outputs to support:
    - versioning
    - traceability (doc_id)
    - basic document stats (total_pages)
    - UI display of source filename (if available)
    """
    doc_id: str = Field(..., description="Unique hash or ID for the policy")
    generated_at: str = Field(..., description="Timestamp for versioning")
    total_pages: int = Field(..., ge=0)
    source_file: Optional[str] = None


class PolicySummaryOutput(BaseModel):
    """
    Full policy summary payload returned to the frontend.

    Components:
    - metadata: identity + versioning info
    - disclaimer: default legal/UX disclaimer
    - sections: list of all section summaries with confidence metadata

    The UI typically renders this as a multi-section report with citations.
    """
    metadata: DocMetadata
    disclaimer: str = Field(default=DEFAULT_DISCLAIMER)
    sections: list[SectionSummaryWithConfidence]

# --- Q&A Components ---

class QAResponseOutput(BaseModel):
    """
    Structured output for the Q&A feature.

    Guarantees:
    - answer_type indicates how the UI should interpret the response
    - citations link the answer back to chunk evidence (may be empty if not_found/greeting)
    - confidence communicates trust level to the user
    - disclaimer is included by default for safe communication

    This model is used directly by the API layer for serialization.
    """
    doc_id: str
    question: str
    answer: str
    answer_type: QA_ANSWER_TYPE
    citations: list[Citation] = Field(default_factory=list)
    confidence: ConfidenceLevel
    disclaimer: str = Field(default=DEFAULT_DISCLAIMER)
    validation_issues: list[str] = Field(default_factory=list)

# -- Chunking and Extracting --

class ExtractedPage(BaseModel):
    """
    Represents the extracted text for a single PDF page.

    This model is used during ingestion prior to chunking, and may also be stored
    for debugging, auditability, or UI preview.
    """
    page_number: int = Field(..., ge=1, description="1-based page number")
    text: str = Field(default="", description="The full cleaned text of the page")


class Chunk(BaseModel):
    """
    Retrieval unit stored in the vector database.

    A chunk is the fundamental "unit of evidence" used across:
    - retrieval (vector search)
    - summarization (grounding and citations)
    - Q&A answers
    - evaluation (faithfulness checks)

    chunk_id convention:
    - typically "c_{page}_{index}" where:
        * page is 1-based PDF page number
        * index is within-page chunk index
    """
    chunk_id: str = Field(..., description="Unique ID, usually c_{page}_{index}")
    page_number: int = Field(..., ge=1)
    doc_id: str = Field(..., description="ID of the parent document")
    chunk_text: str = Field(..., description="The actual text content used for embeddings")

# -- Scenario Logic --

class ScenarioStepOutput(BaseModel):
    """
    One step in a hypothetical scenario walkthrough.

    Scenarios break down cost/coverage reasoning into sequential steps.
    Any numeric value or policy rule should be supported by citations.
    """
    step_number: int = Field(..., ge=1)
    text: str = Field(..., description="Plain English description of the cost step")
    citations: list[Citation] = Field(default_factory=list)


class ScenarioQAResponseOutput(BaseModel):
    """
    Response model for scenario generation.

    This response type is intended for UI display as a step-by-step explanation.
    Fields include:
    - scenario_type: label guiding what kind of scenario this is (e.g., ER vs General)
    - header: UI-friendly title string
    - steps: ordered scenario steps (typically 3–6)
    - not_found_message: populated if the system can't construct a grounded scenario
    - confidence: trust label
    """
    doc_id: str = Field(...)
    question: str = Field(...)
    answer_type: Literal["scenario"] = Field(default="scenario")
    scenario_type: str = Field(...)
    header: str = Field(default="Example Scenario (Hypothetical – Based on Policy Terms)")
    steps: list[ScenarioStepOutput] = Field(default_factory=list)
    not_found_message: Optional[str] = Field(default=None)
    confidence: ConfidenceLevel = Field(...)
    disclaimer: str = Field(default=DEFAULT_DISCLAIMER)