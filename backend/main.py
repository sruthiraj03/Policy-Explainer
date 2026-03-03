"""
FastAPI Application Entrypoint.

This module initializes the PolicyExplainer FastAPI server, configures middleware,
and registers the API routers that expose the core pipeline functionality.

High-level responsibilities:
- Create the FastAPI `app` instance.
- Enable CORS so the Streamlit frontend (or any other client) can call the API.
- Register feature routers for:
    * Ingest: PDF upload and processing
    * Summary: Full-document and section-level summarization
    * Q&A: Retrieval-grounded question answering
    * Evaluate: Automated evaluation metrics (faithfulness/completeness)

Notes:
- Routers are defined in `backend.api` and imported here to keep this file minimal.
- CORS is currently configured as wide-open (allow_origins=["*"]) for development
  convenience. In production, this should typically be restricted to known domains.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Updated imports to match the simplified api.py.
# Each router groups endpoints by feature area and is mounted below.
from backend.api import (
    router_ingest,
    router_summary,
    router_qa,
    router_evaluate
)

# Create the FastAPI app instance. The title appears in the OpenAPI/Swagger UI.
app = FastAPI(title="PolicyExplainer API")

# Enable CORS so Streamlit can talk to FastAPI.
# This is required when the frontend is served from a different origin (host/port).
# Current config is permissive for development/testing.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Allow requests from any origin (development-friendly)
    allow_methods=["*"],   # Allow all HTTP methods (GET/POST/etc.)
    allow_headers=["*"],   # Allow all headers (Authorization, Content-Type, etc.)
)

# Include the routers.
# - Ingest is mounted at its internal route (e.g., POST /ingest).
# - Summary, QA, and Evaluate are namespaced with prefixes for clarity.
app.include_router(router_ingest, tags=["Ingest"])
app.include_router(router_summary, prefix="/summary", tags=["Summary"])
app.include_router(router_qa, prefix="/qa", tags=["Q&A"])
app.include_router(router_evaluate, prefix="/evaluate", tags=["Evaluate"])


@app.get("/")
async def root():
    """
    Health-check / root endpoint.

    This is a lightweight endpoint used to confirm the API process is running and reachable.
    It can also be used by deployment health checks / uptime monitoring.

    Returns:
        dict: Simple status message.
    """
    return {"message": "PolicyExplainer API is running"}