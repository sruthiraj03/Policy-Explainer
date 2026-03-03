"""
PolicyExplainer Streamlit App (Frontend Main Entry Point)

This file is the main entry point for the Streamlit frontend application.

Responsibilities:
- Configure global Streamlit settings (page title, layout, favicon).
- Initialize shared styling (custom CSS).
- Initialize session state variables.
- Render persistent UI elements (sidebar).
- Route between:
    * Hero View (file upload / landing page)
    * Dashboard View (summary, FAQs, chat assistant)

Architecture:
- This file contains no business logic.
- All feature-specific UI and logic are delegated to modular components.
- Acts purely as an orchestrator/router for frontend state.

Execution:
- When run directly (`streamlit run app.py`), main() is invoked.
"""

import streamlit as st

from components.hero import render_hero_view
from components.dashboard import render_dashboard_view
from components.sidebar import render_sidebar
from utils.state import init_session_state
from utils.style import load_css


# --- Streamlit App Configuration ---
# IMPORTANT: Must be the first Streamlit command in the file.
# Controls browser tab title, layout width, and favicon.
st.set_page_config(
    page_title="Policy Explainer",
    layout="wide",
    page_icon="📄"
)


def main():
    """
    Main application router and orchestrator.

    Execution Order:
    1) Inject custom CSS for consistent styling.
    2) Initialize required session_state variables.
    3) Render the sidebar (controls + evaluation metrics).
    4) Route to either:
        - Hero view (if no document uploaded yet)
        - Dashboard view (if doc_id exists)

    Error Handling:
    - Wraps routing logic in a try/except block.
    - Displays a user-friendly error message.
    - Provides expandable technical details for debugging.
    """

    # 1. Load Custom CSS
    # Injects global styling overrides before rendering UI components.
    load_css()

    # 2. Initialize Session State Variables
    # Ensures all required keys exist to prevent KeyError during reruns.
    init_session_state()

    # 3. Render the Sidebar (Controls & Metrics)
    # Sidebar is always visible regardless of main view.
    render_sidebar()

    # 4. Main Page Routing Logic
    try:
        # Routing Condition:
        # If no document has been uploaded (doc_id is None),
        # render the Hero view (landing + upload).
        if st.session_state.get("doc_id") is None:
            render_hero_view()

        # Otherwise, render the Dashboard view:
        # - Policy summary
        # - FAQs
        # - Chat assistant
        else:
            render_dashboard_view()

    except Exception as e:
        # Global UI-level error guard.
        # Prevents the entire app from crashing due to unexpected frontend issues.
        st.error("🚨 Oops! Something went wrong while processing your request.")

        # Provide expandable technical details for debugging.
        with st.expander("Technical Error Details"):
            st.write(f"Error: {type(e).__name__} - {str(e)}")


# Standard Python entry point guard.
# Ensures main() only runs when this file is executed directly.
if __name__ == "__main__":
    main()