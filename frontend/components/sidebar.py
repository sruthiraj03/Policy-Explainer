"""
Sidebar Component for App Controls and Metrics.

This module renders the Streamlit sidebar used across the PolicyExplainer app.

Responsibilities:
- Provide global application controls (e.g., full reset).
- Display AI-generated evaluation metrics when available.
- Surface high-level quality signals (Faithfulness and Completeness)
  computed by the backend Evaluation Engine.

Expected session_state keys:
- "eval_data": dict
    Returned from POST /evaluate/{doc_id}, typically shaped like:
    {
        "doc_id": "...",
        "faithfulness": float (0.0–1.0),
        "completeness": float (0.0–1.0)
    }

Behavior:
- Metrics are shown only if eval_data exists and is non-empty.
- Scores are converted from 0–1 scale to percentage format for readability.
"""

import streamlit as st


def render_sidebar():
    """
    Render the application sidebar with controls and performance metrics.

    Sections:
    1) App Controls
       - "Reset Application" clears all session state and reruns the app.
    2) AI Performance Metrics (conditional)
       - Displays Faithfulness and Completeness scores if available.

    Returns:
        None
    """
    # All sidebar content must be rendered within the st.sidebar context.
    with st.sidebar:
        # --- App Controls Section ---
        st.header("App Controls")

        # Full reset button:
        # Clears all session state keys (doc_id, summary, chat history, etc.)
        # and forces a rerun to return to the hero view.
        if st.button("Reset Application", use_container_width=True):
            st.session_state.clear()
            st.rerun()

        # --- Evaluation Metrics Section ---
        # Only display metrics if they exist and are non-empty.
        if "eval_data" in st.session_state and st.session_state["eval_data"]:
            st.divider()
            st.header("AI Performance Metrics")
            st.caption("Automatically calculated by the Evaluation Engine.")

            # Faithfulness and completeness are returned as floats between 0 and 1.
            # Convert to percentage format for display.
            f_score = st.session_state["eval_data"].get("faithfulness", 0) * 100
            c_score = st.session_state["eval_data"].get("completeness", 0) * 100

            # Streamlit metric components provide a clean numeric display.
            st.metric("Faithfulness (Accuracy)", f"{f_score:.1f}%")
            st.metric("Completeness (Coverage)", f"{c_score:.1f}%")