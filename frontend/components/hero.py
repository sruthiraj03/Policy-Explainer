"""
Hero View Component for Landing Page and File Upload.

This module renders the "hero" landing view shown before a document is uploaded.
It provides:
- A branded landing header and short value proposition.
- A PDF file uploader for the user's policy / Summary of Benefits document.
- A 3-step progress flow that orchestrates backend processing:
    1) Ingest (upload + validation + chunking + vector indexing)
    2) Summarize (generate structured policy highlights)
    3) Evaluate (compute quality/grounding metrics)

It also handles:
- Document validation failures returned by the backend (HTTP 400 with a "detail" message).
- Network/server exceptions (connection errors, timeouts, HTTP errors).
- Session state transitions so the app can move from hero view -> dashboard view.

Expected session_state usage:
- "upload_error": str
    Set when the backend rejects the uploaded document as invalid.
- "processing": bool
    Used as a guard to prevent duplicate processing when Streamlit re-runs.
- "doc_id": str
    Document identifier returned by /ingest, used in later requests.
- "summary": dict
    JSON summary returned by /summary/{doc_id}.
- "eval_data": dict
    JSON evaluation payload returned by /evaluate/{doc_id} (if successful).
- "uploader_key": int
    Used to force re-mounting/refreshing the uploader widget (if managed elsewhere).

Configuration:
- API_BASE is read from the environment variable "API_BASE".
  NOTE: The default value below appears formatted like a Markdown link; it is preserved
  exactly as-is to avoid changing existing behavior.
"""

import os
import requests
import streamlit as st
from dotenv import load_dotenv

# Load environment variables for the API URL.
load_dotenv()

# Backend base URL for FastAPI.
# NOTE: Default value is a Markdown-formatted link string; preserved as provided.
API_BASE = os.getenv("API_BASE", "[http://127.0.0.1:8000](http://127.0.0.1:8000)")


def render_hero_view():
    """
    Render the landing page hero view with upload flow.

    UI layout:
    - Two-column layout:
        * Left: title, subtitle, upload controls and progress flow
        * Right: header image (if present)

    Processing flow:
    - When a PDF is uploaded, the app performs:
        1) POST /ingest (multipart file upload)
        2) POST /summary/{doc_id}
        3) POST /evaluate/{doc_id}
    - Results are stored in session_state and the app reruns to transition views.

    Returns:
        None
    """
    col_text, col_img = st.columns([1, 1], gap="large")

    # --- Left Column: Text + Upload Controls ---
    with col_text:
        # Hero title and subtitle are rendered as HTML to allow custom CSS classes.
        st.markdown(
            '<h1 class="hero-title">Policy Explainer</h1><div class="hero-subtitle">Understand your health insurance.</div>',
            unsafe_allow_html=True
        )

        # If a previous upload attempt failed validation, show the error and provide a reset button.
        if "upload_error" in st.session_state:
            st.error(f"🚫 **Validation Failed:** {st.session_state['upload_error']}")

            # Reset clears session state entirely so user can try a new upload.
            if st.button("❌ Reset & Try Again", key="exit_reset"):
                st.session_state.clear()
                st.rerun()

        else:
            # Value proposition text displayed when not in an error state.
            st.markdown(
                '<div class="hero-text">Decode your health insurance instantly. Upload your <b>Summary of Benefits</b> (PDF) to reveal hidden costs.</div>',
                unsafe_allow_html=True
            )

            # File uploader:
            # - "key" is dynamic to allow forcing a remount when uploader_key changes.
            # - label is collapsed for a cleaner UI.
            uploaded_file = st.file_uploader(
                "Drop your policy PDF here",
                type=["pdf"],
                key=f"uploader_{st.session_state.get('uploader_key', 0)}",
                label_visibility="collapsed",
            )

            # Guard against duplicate submissions:
            # Streamlit reruns frequently; we use a "processing" flag to ensure only one
            # ingest/summarize/evaluate sequence runs per upload event.
            if uploaded_file and "processing" not in st.session_state:
                st.session_state["processing"] = True

                # Placeholder for step-by-step status messages.
                status_help = st.empty()

                try:
                    # Use spinner to indicate work-in-progress while we call the backend.
                    with st.spinner("Processing..."):
                        # --- Step 1: Ingest ---
                        # Upload PDF and trigger backend ingestion + validation.
                        status_help.markdown("🔍 **Step 1/3:** Validating policy...")

                        # requests expects multipart "files" mapping: {field: (filename, bytes, content_type)}
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                        ingest_r = requests.post(f"{API_BASE}/ingest", files=files, timeout=120)

                        # Validation failures are surfaced by the backend as HTTP 400 with a "detail" message.
                        # This branch converts backend messaging into a cleaner UI error state.
                        if ingest_r.status_code == 400:
                            status_help.empty()
                            error_detail = ingest_r.json().get("detail", "Invalid document.")

                            # Remove the generic "Validation Failed:" prefix if present for a cleaner message.
                            st.session_state["upload_error"] = error_detail.replace("Validation Failed:", "").strip()

                            # Clear processing flag before rerun so the user can try again.
                            del st.session_state["processing"]
                            st.rerun()

                        # Any other non-2xx status is treated as a hard error.
                        ingest_r.raise_for_status()

                        # Document ID used to reference the uploaded file in subsequent endpoints.
                        doc_id = ingest_r.json().get("doc_id")

                        # --- Step 2: Summarize ---
                        status_help.markdown("📑 **Step 2/3:** Generating summary...")
                        summary_r = requests.post(f"{API_BASE}/summary/{doc_id}", timeout=120)
                        summary_r.raise_for_status()

                        # --- Step 3: Evaluate ---
                        status_help.markdown("⚖️ **Step 3/3:** Verifying accuracy...")
                        eval_r = requests.post(f"{API_BASE}/evaluate/{doc_id}", timeout=120)

                        # --- Update Session State ---
                        # Store results needed by the dashboard view.
                        st.session_state["doc_id"] = doc_id
                        st.session_state["summary"] = summary_r.json()
                        st.session_state["eval_data"] = eval_r.json() if eval_r.status_code == 200 else {}

                        # Clear processing flag and rerun to transition UI.
                        del st.session_state["processing"]
                        st.rerun()

                # --- Error Handling ---
                except requests.exceptions.ConnectionError:
                    st.error(
                        "🚨 **Connection Error:** Cannot connect to the backend API. "
                        "Please make sure your FastAPI server is running (`uvicorn backend.main:app --reload`)."
                    )
                    # Ensure processing flag is cleared so user can retry.
                    if "processing" in st.session_state:
                        del st.session_state["processing"]

                except requests.exceptions.Timeout:
                    st.error("⏳ **Timeout:** The document processing took too long and timed out. Try a smaller PDF.")
                    if "processing" in st.session_state:
                        del st.session_state["processing"]

                except requests.exceptions.HTTPError as e:
                    st.error(
                        f"⚠️ **Server Error:** The backend failed with status code {e.response.status_code}. "
                        "Check the FastAPI console for details."
                    )
                    if "processing" in st.session_state:
                        del st.session_state["processing"]

                except Exception as e:
                    # Catch-all for unexpected errors (parsing, missing keys, etc.).
                    st.error(f"🚨 **System Error:** {str(e)}")
                    if "processing" in st.session_state:
                        del st.session_state["processing"]

    # --- Right Column: Hero Image ---
    with col_img:
        # Resolve the path relative to this file's location inside the components folder.
        # base_dir is set to the parent directory of the components folder.
        base_dir = os.path.dirname(os.path.dirname(__file__))
        image_path = os.path.join(base_dir, "assets", "header_image.jpg")

        # Prefer the assets/ directory location, fall back to base directory if needed.
        if os.path.exists(image_path):
            st.image(image_path, use_container_width=True)
        else:
            fallback_path = os.path.join(base_dir, "header_image.jpg")
            if os.path.exists(fallback_path):
                st.image(fallback_path, use_container_width=True)