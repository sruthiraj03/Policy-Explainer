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
    """
    with st.sidebar:
        st.header("App Controls")

        if st.button("Reset Application", use_container_width=True):
            st.session_state.clear()
            st.rerun()

        if "eval_data" in st.session_state and st.session_state["eval_data"]:
            st.divider()
            st.header("AI Performance Metrics")
            st.caption("Automatically calculated by the Evaluation Engine.")

            eval_data = st.session_state["eval_data"]

            # Percentage-style metrics
            f_score = eval_data.get("faithfulness", 0) * 100
            c_score = eval_data.get("completeness", 0) * 100

            # Simplicity should show Summary Flesch directly
            s_score = eval_data.get("simplicity")
            improvement = eval_data.get("improvement")

            st.metric("Faithfulness (Accuracy)", f"{f_score:.1f}%")
            st.metric("Completeness (Coverage)", f"{c_score:.1f}%")

            if s_score is not None:
                st.metric("Simplicity (Readability)", f"{s_score:.2f}")

            if improvement is not None:
                with st.expander("Readability Details"):
                    st.metric("Improvement Score", f"{improvement:.2f}")