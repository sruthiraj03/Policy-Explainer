"""
Dashboard Component for the main post-upload interface.

This module renders the primary Streamlit "dashboard" view shown after a user uploads
a policy document and the backend has produced a summary.

It is responsible for:
- Displaying the policy summary in expandable, section-based UI components.
- Displaying a dynamically generated FAQ tab (generated once per document/session).
- Providing navigation controls:
    * Switch between Summary and FAQs
    * Reset session state to upload a new policy
    * Download a generated PDF version of the summary
- Rendering the chat assistant panel alongside the main content area.

Assumptions / session_state dependencies:
- st.session_state["doc_id"]: str
    Active document identifier used for backend requests and download naming.
- st.session_state["summary"]: dict
    The summary payload returned by the FastAPI summary endpoint.
- st.session_state["active_tab"]: str
    Tracks which tab is active: "Summary" or "FAQs".
- st.session_state.chat_history: list[dict] (used by the chat component)
    Chat transcript rendered in the right-hand panel.

External dependencies:
- backend API base URL is read from environment variable API_BASE.
- utils.pdf_generator.generate_policy_pdf(...) converts a summary dict into a PDF byte payload.
- components.chat.render_chat_panel(...) renders the Streamlit chat UI.
"""

import os
import requests
import streamlit as st
from dotenv import load_dotenv

from utils.pdf_generator import generate_policy_pdf
from components.chat import render_chat_panel

# Load environment variables for the API URL.
# This allows different deployment environments to configure the backend location cleanly.
load_dotenv()
API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")


def render_summary_content():
    """
    Render the "Policy Highlights" section, displaying a structured summary.

    The summary is expected to follow the backend schema shape (PolicySummaryOutput),
    but is handled here as a dict (already JSON-serialized/deserialized).

    UI behavior:
    - Each section is rendered as an expander.
    - Confidence is displayed with a colored label (green/orange/red).
    - Each bullet is shown with inline citations formatted as page numbers.

    Returns:
        None
    """
    st.subheader("Policy Highlights")
    summary_data = st.session_state.get("summary", {})

    # The summary payload contains a list of sections.
    for section in summary_data.get("sections", []):
        # Only render sections marked present=True (avoid showing empty placeholders).
        if section.get("present"):
            # Expanders are opened by default to make the summary immediately visible.
            with st.expander(f"{section['section_name']}", expanded=True):
                # Confidence is a string like "high"/"medium"/"low".
                # We display it uppercased and color-coded for quick scanning.
                conf = section.get("confidence", "low").upper()
                st.markdown(
                    f"**Confidence:** :{'green' if conf == 'HIGH' else 'orange' if conf == 'MEDIUM' else 'red'}[{conf}]"
                )

                # Render bullets with citations.
                for bullet in section.get("bullets", []):
                    # Format citations as a comma-separated list of page references.
                    cites = ", ".join([f"p. {c['page']}" for c in bullet.get("citations", [])])

                    # Escape $ so Streamlit Markdown doesn't interpret it as LaTeX math.
                    safe_text = bullet["text"].replace("$", "\\$")

                    # Use a bullet symbol with italicized citations for a clean look.
                    st.markdown(f"• {safe_text} *({cites})*")


def render_faq_content():
    """
    Render the "Frequently Asked Questions" section.

    FAQs are generated dynamically the first time the user opens this view for a given
    session/document and then stored in `st.session_state["faqs"]` to avoid repeated
    backend calls and LLM usage.

    Behavior:
    - If FAQs are not cached yet, call GET /qa/{doc_id}/faqs and store the result.
    - If the request fails, show a friendly error and store an empty list.
    - Render each FAQ as an expander with the answer inside.

    Returns:
        None
    """
    st.subheader("Frequently Asked Questions")

    # If FAQs aren't in memory yet, generate them once per session.
    if "faqs" not in st.session_state:
        with st.spinner("🧠 Generating FAQs tailored to your policy..."):
            try:
                doc_id = st.session_state["doc_id"]

                # Backend endpoint returns a JSON object like: {"faqs": [{"question": "...", "answer": "..."}]}
                r = requests.get(f"{API_BASE}/qa/{doc_id}/faqs", timeout=120)
                r.raise_for_status()

                # Store FAQs in session state so future renders are instant.
                st.session_state["faqs"] = r.json().get("faqs", [])
            except Exception as e:
                # Preserve current behavior: do not display exception details to end-user by default.
                st.error("⚠️ Could not generate FAQs at this time. Please try again later.")
                st.session_state["faqs"] = []

    faqs = st.session_state.get("faqs", [])

    # If no FAQs are available (either empty response or failure), show an info message.
    if not faqs:
        st.info("No FAQs available for this document.")
    else:
        # Render each FAQ as an expander for clean scan-and-open interaction.
        for faq in faqs:
            with st.expander(faq.get("question", "Question")):
                st.write(faq.get("answer", ""))


def render_dashboard_view():
    """
    Orchestrate the main dashboard layout with navigation, content area, and chat panel.

    Layout:
    - Title + divider
    - Two main columns:
        * Left (main): Summary/FAQs content and navigation controls
        * Right (chat): Sticky chat assistant panel

    Navigation:
    - "📑 Summary": sets active_tab to "Summary"
    - "❓ FAQs": sets active_tab to "FAQs"
    - "🔄 New Policy": clears session state entirely and reruns (fresh start)
    - "📥 Save Insurance Summary": generates a PDF and exposes it via download_button

    Returns:
        None
    """
    # Custom HTML title allows custom CSS styling via the "dash-title" class.
    st.markdown('<h1 class="dash-title">Policy Explainer</h1>', unsafe_allow_html=True)
    st.divider()

    # Main layout: content on the left, chat on the right.
    col_main, col_chat = st.columns([0.74, 0.26], gap="large")

    with col_main:
        # Navigation Bar: 4 columns with actions.
        col_nav1, col_nav2, col_nav3, col_nav4 = st.columns([1, 1, 1.5, 1])

        # 1) Summary tab button.
        # Use a visual "primary" style when active.
        with col_nav1:
            if st.button("📑 Summary", type="primary" if st.session_state["active_tab"] == "Summary" else "secondary"):
                st.session_state["active_tab"] = "Summary"
                st.rerun()

        # 2) New Policy button: reset everything and rerun.
        # This clears summary/chat/doc_id so the user can start fresh.
        with col_nav2:
            if st.button("🔄 New Policy", key="dash_new"):
                st.session_state.clear()
                st.rerun()

        # 3) Download Summary as PDF.
        # We generate the PDF bytes from the summary currently in memory.
        with col_nav3:
            summary_data = st.session_state.get("summary", {})
            pdf_bytes = generate_policy_pdf(summary_data)

            st.download_button(
                label="📥 Save Insurance Summary",
                data=pdf_bytes,
                file_name=f"Policy Summary-{st.session_state.get('doc_id', 'Analysis')}.pdf",
                mime="application/pdf",
            )

        # 4) FAQs tab button.
        with col_nav4:
            if st.button("❓ FAQs", type="primary" if st.session_state["active_tab"] == "FAQs" else "secondary"):
                st.session_state["active_tab"] = "FAQs"
                st.rerun()

        # Spacer for visual separation.
        st.write("")

        # Display the active tab's content.
        # Default behavior: if active_tab is "Summary", render summary; else render FAQs.
        if st.session_state["active_tab"] == "Summary":
            render_summary_content()
        else:
            render_faq_content()

    with col_chat:
        # This marker div is likely used by custom CSS to make the chat column sticky.
        st.markdown('<div class="chat-sticky-marker"></div>', unsafe_allow_html=True)

        # Render the chat panel inside its own container for consistent spacing.
        with st.container():
            render_chat_panel()