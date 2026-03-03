"""
API Routes Orchestrator.

This module defines the FastAPI routers (route groups) for the PolicyExplainer backend.
Each router acts as a gateway to a specific stage of the PolicyExplainer pipeline:

- Ingestion: Accept a PDF upload, validate it, and kick off parsing/chunking/indexing.
- Summarization: Run the full document summary pipeline or a section-level "deep dive".
- Q&A: Accept user questions and route them through the unified question router.
- Evaluation: Trigger automated evaluation metrics (e.g., faithfulness/completeness).

Design notes:
- These handlers are intentionally thin: they validate inputs, delegate to backend modules,
  and translate errors into HTTP status codes appropriate for the frontend.
- Successful responses are returned as plain JSON-serializable dicts. When backend modules
  return Pydantic models, we convert them via `.model_dump()`.

Key inputs/outputs:
- Ingestion expects an uploaded PDF (UploadFile) and returns a document identifier `doc_id`.
- Summary and Q&A endpoints accept a `doc_id` path parameter (string) referencing stored/indexed content.
- Summary endpoints return structured Pydantic outputs serialized to dict.
- Evaluation returns computed metrics (implementation-defined by `run_all_evaluations`).

Error handling conventions:
- Validation / user input issues: HTTP 400 (e.g., non-PDF upload, invalid section id, empty question).
- Known ingestion validation failures: HTTP 400 (propagated from `ValueError` with a safe message).
- Unexpected server errors: HTTP 500 with a descriptive message.
"""

from fastapi import APIRouter, HTTPException, UploadFile
from pydantic import BaseModel

from backend import storage
from backend.evaluation import run_all_evaluations
from backend.ingestion import run_ingest
from backend.qa import route_question  # We only need the unified router now!
from backend.retrieval import CORE_SECTIONS, retrieve_for_section
from backend.summarization import run_full_summary_pipeline, summarize_section

# --- Ingest: PDF Upload & Processing ---
# This router groups endpoints related to accepting documents and initiating ingestion.
router_ingest = APIRouter()


@router_ingest.post("/ingest")
async def ingest(file: UploadFile) -> dict:
    """
    Receive a PDF upload, validate it, and trigger the ingestion pipeline.

    Validation performed here is intentionally lightweight and fast:
    - Check filename extension indicates PDF.
    - Check file content begins with the PDF magic header (%PDF).

    On success, delegates to `run_ingest(content)` which is responsible for:
    parsing, cleaning, chunking, embedding/indexing, and persisting any artifacts,
    then returns a unique document identifier (`doc_id`) used across the API.

    Returns:
        dict: {"doc_id": <str>, "filename": <original filename>}
    """
    # Basic filename-based validation to provide immediate feedback to the UI.
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        # 400 indicates a client-side input problem (not a server failure).
        raise HTTPException(status_code=400, detail="File must be a PDF")

    try:
        # UploadFile.read() is async; we read the full content into memory here.
        # (In the future, streaming could be used if extremely large files are expected.)
        content = await file.read()

        # Minimal header validation: a real PDF begins with the bytes "%PDF".
        # This avoids wasting time on obviously wrong uploads with a .pdf extension.
        if len(content) < 4 or content[:4] != b"%PDF":
            raise HTTPException(status_code=400, detail="Invalid PDF format")

        # Delegate core ingestion work to the ingestion module.
        # `run_ingest` should raise ValueError for "expected" validation failures.
        doc_id = run_ingest(content)

        # Return the doc_id so the frontend can reference the newly ingested document.
        return {"doc_id": doc_id, "filename": file.filename}

    except ValueError as ve:
        # This catches ingestion-level validation errors (e.g., keyword checks, schema checks,
        # or any intentional rejection where the document is considered "invalid" for the app).
        # We return HTTP 400 so the frontend treats it as a user/document issue rather than
        # a server crash, and can display a helpful message to the user.
        raise HTTPException(status_code=400, detail=str(ve))

    except Exception as e:
        # Catch-all for unexpected system failures (e.g., storage/indexing failures,
        # internal exceptions, dependency outages).
        # We respond with HTTP 500 so the frontend knows the server encountered an error.
        raise HTTPException(status_code=500, detail=f"Ingest failed: {e}")


# --- Summary: LLM Extraction ---
# This router groups endpoints that produce summaries from an ingested document.
router_summary = APIRouter()


@router_summary.post("/{doc_id}")
async def post_summary(doc_id: str) -> dict:
    """
    Run the full document summarization pipeline for the given document.

    This endpoint is intended to produce the full "schema" summary (e.g., all core sections).
    The heavy lifting is performed by `run_full_summary_pipeline(doc_id)`.

    Args:
        doc_id: The document identifier returned by /ingest.

    Returns:
        dict: A JSON-serializable dict representation of the summary Pydantic model.
    """
    try:
        # Delegate to the summarization pipeline which should:
        # - retrieve relevant chunks per section
        # - call the LLM to summarize/extract structured data
        # - assemble the final multi-section summary object
        summary = run_full_summary_pipeline(doc_id)

        # Convert the Pydantic model output into a standard dict for JSON response.
        return summary.model_dump()
    except Exception as e:
        # Debug printing: in development, forcing a traceback helps reveal the true error
        # in terminal logs rather than only returning the exception string to the client.
        import traceback  # <-- intentionally local import to avoid overhead unless error occurs

        traceback.print_exc()  # <-- prints a full stack trace to server logs
        raise HTTPException(status_code=500, detail=str(e))


@router_summary.post("/{doc_id}/section/{section_id}")
async def post_section_summary(doc_id: str, section_id: str) -> dict:
    """
    Generate a detailed summary ("Deep Dive") for a single section of the document.

    This endpoint supports the UI workflow where the user requests a more detailed
    explanation for one specific core section (rather than regenerating the entire document).

    Args:
        doc_id: The document identifier returned by /ingest.
        section_id: One of the allowed core section ids (validated against CORE_SECTIONS).

    Returns:
        dict: A JSON-serializable dict representation of the section summary Pydantic model.
    """
    # Reject unknown section ids early with a client-friendly message.
    if section_id not in CORE_SECTIONS:
        raise HTTPException(status_code=400, detail="Invalid section ID")

    try:
        # Retrieve the most relevant text chunks for the requested section.
        # Retrieval strategy and filtering are implemented inside `retrieve_for_section`.
        chunks = retrieve_for_section(doc_id, section_id)

        # Generate the section-level summary using a "detailed" preset.
        # The summarizer should be grounded in the retrieved chunks.
        out = summarize_section(section_id, chunks, detail_level="detailed")

        # Convert the Pydantic model output into a dict for JSON response.
        return out.model_dump()
    except Exception as e:
        # Unexpected failure while retrieving/summarizing for this section.
        raise HTTPException(status_code=500, detail=str(e))


# --- Q&A: Grounded Retrieval ---
# This router groups endpoints for question answering against an ingested policy.
router_qa = APIRouter()


class QABody(BaseModel):
    """
    Request body schema for the Q&A endpoint.

    Fields:
        question: The user's natural-language question about the policy document.
    """

    question: str


@router_qa.post("/{doc_id}")
async def ask_endpoint(doc_id: str, body: QABody) -> dict:
    """
    Answer a user question for the given document using the unified Q&A router.

    The `route_question` function is responsible for classifying/routing the question
    (e.g., standard QA vs scenario vs deep-dive) and returning a structured response.

    Args:
        doc_id: The document identifier returned by /ingest.
        body: JSON request body containing the question.

    Returns:
        dict: A JSON-serializable dict representation of the Q&A response Pydantic model.
    """
    # Defensive normalization: strip whitespace and guard against empty strings.
    question = (body.question or "").strip()
    if not question:
        # 400 indicates the client did not provide a required input field.
        raise HTTPException(status_code=400, detail="Question is required")

    try:
        # Delegate to the unified question router, which encapsulates:
        # - intent detection / classification
        # - retrieval / grounding
        # - answer generation
        # - structured response assembly
        response_obj = route_question(doc_id, question)

        # Convert the Pydantic model output into a dict for JSON response.
        return response_obj.model_dump()
    except Exception as e:
        # Catch-all server error: return 500 so the UI can show a generic failure state.
        raise HTTPException(status_code=500, detail=str(e))


@router_qa.get("/{doc_id}/faqs")
async def get_faqs(doc_id: str) -> dict:
    """
    Generate and return dynamic FAQs for the given document.

    This endpoint delegates FAQ generation to `backend.qa.generate_document_faqs`.
    The output shape is implementation-defined by that function and is returned directly.

    Args:
        doc_id: The document identifier returned by /ingest.

    Returns:
        dict: A JSON-serializable dict containing generated FAQs.
    """
    try:
        # Local import keeps module load lighter and avoids importing FAQ generation
        # unless this endpoint is called.
        from backend.qa import generate_document_faqs

        return generate_document_faqs(doc_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Evaluate: Analytics & Metrics ---
# This router groups endpoints that compute evaluation metrics for a document.
router_evaluate = APIRouter()


@router_evaluate.post("/{doc_id}")
async def evaluate(doc_id: str) -> dict:
    """
    Trigger evaluation for the given document and return computed metrics.

    The evaluation module is expected to compute metrics such as:
    - faithfulness / groundedness
    - completeness / coverage
    - other quality signals as implemented in `run_all_evaluations`

    Args:
        doc_id: The document identifier returned by /ingest.

    Returns:
        dict: A JSON-serializable dict of evaluation results.
    """
    try:
        # Delegate to the evaluation runner; expected to return a dict.
        return run_all_evaluations(doc_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))