"""
Session State Manager for the Streamlit Frontend.

This module centralizes initialization of all required Streamlit `st.session_state`
keys used throughout the PolicyExplainer frontend.

Why this exists:
- Streamlit reruns the entire script on most interactions.
- Accessing a missing session_state key raises a KeyError.
- Centralizing initialization ensures predictable, stable UI behavior
  across hero view, dashboard view, chat panel, and sidebar.

Design principle:
- Only initialize keys if they do NOT already exist.
- Never overwrite existing values (to preserve user progress).
"""

import streamlit as st


def init_session_state():
    """
    Initialize all required session state variables if they don't already exist.

    This function should be called once near the top of the main Streamlit app
    script before rendering UI components.

    Session State Groups:

    Core App State:
    - doc_id: str | None
        The backend document identifier returned from /ingest.
        Controls whether the app shows hero view or dashboard view.
    - active_tab: str
        Current dashboard tab ("Summary" or "FAQs").
    - uploader_key: int
        Used to force-refresh the file uploader component when needed.

    Chat State:
    - chat_history: list[dict]
        Conversation transcript with entries like:
        {"role": "user"|"assistant", "content": "..."}
    - chat_open: bool
        Reserved toggle for showing/hiding chat panel (if implemented in layout logic).
    - chat_input_text: str
        Placeholder for manual input state (not required when using st.chat_input,
        but kept for compatibility).
    - pending_question: str | None
        Temporary storage for two-phase chat submission logic.
    """

    # --- Core App States ---

    # Unique document identifier returned by backend after ingestion.
    if "doc_id" not in st.session_state:
        st.session_state["doc_id"] = None

    # Tracks which dashboard tab is active.
    # Defaults to "Summary" so the main highlights are shown first.
    if "active_tab" not in st.session_state:
        st.session_state["active_tab"] = "Summary"

    # Used to force re-mounting of the file uploader (e.g., after reset).
    if "uploader_key" not in st.session_state:
        st.session_state["uploader_key"] = 0

    # --- Chat States ---

    # Chat transcript history.
    # Initialize with a friendly assistant greeting to guide the user.
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = [{
            "role": "assistant",
            "content": (
                "Hi, I am your Policy Explainer AI Chatbot here to assist you. "
                "Feel free to ask me any questions about your policy."
            )
        }]

    # Flag for whether chat panel is considered open.
    # (May be used by layout logic or future UI toggles.)
    if "chat_open" not in st.session_state:
        st.session_state["chat_open"] = True

    # Stores raw chat input text if needed.
    # Retained for compatibility even if st.chat_input is used elsewhere.
    if "chat_input_text" not in st.session_state:
        st.session_state["chat_input_text"] = ""

    # Holds a user question temporarily between reruns
    # so that UI can show "Thinking..." before API call completes.
    if "pending_question" not in st.session_state:
        st.session_state["pending_question"] = None