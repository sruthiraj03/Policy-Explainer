"""
Chat Component for the interactive Q&A assistant.

This module renders the Streamlit chat UI used for PolicyExplainer's interactive Q&A experience.
It is responsible for:

- Displaying a persistent chat transcript stored in `st.session_state.chat_history`.
- Providing a native Streamlit chat input (`st.chat_input`) for user questions.
- Calling the FastAPI backend Q&A endpoint to generate grounded answers.
- Rendering different answer formats, including:
    * Standard Q&A answers (with optional citations)
    * Scenario-style step breakdowns (with step-level citations)
- Handling common networking and server errors gracefully (connection, timeout, HTTP errors).

Expected session_state prerequisites (set elsewhere in the app):
- st.session_state["doc_id"]: str
    The active document identifier returned by the ingest endpoint.
- st.session_state.chat_history: list[dict]
    Chat transcript containing dicts like {"role": "user"|"assistant", "content": "..."}.
- st.session_state.pending_question: Optional[str]
    Used as a two-phase UI pattern to:
      1) immediately display the user's message + "Thinking..."
      2) perform the API call after the UI updates
"""

import os
import requests
import streamlit as st

# Base URL for the FastAPI backend.
# NOTE: The default value below appears to be formatted like a Markdown link; we preserve it
# exactly as-is to avoid changing behavior. The rest of the app will use it as provided.
API_BASE = st.secrets["API_BASE"].rstrip("/")
API_PREFIX = ""

def api_url(path: str) -> str:
    return f"{API_BASE}{API_PREFIX}{path}"

def render_chat_panel():
    """
    Render the PolicyExplainer chat panel.

    UI/interaction pattern:
    - Phase 1: Render transcript + chat input. If user submits a question, store it in
      session state and rerun immediately so the UI reflects the new message.
    - Phase 2: On rerun, detect the pending question and perform the API request, then
      append the assistant response to chat history and rerun again to display it.

    This two-phase approach makes the interface feel responsive by showing:
    - the user's message instantly
    - a "Thinking..." indicator while the backend request is in-flight (from the UI's POV)

    Returns:
        None
    """

    # 1. Clean Header
    # Use Streamlit's built-in typography for a consistent look with the rest of the app.
    st.subheader("💬 Policy Assistant")
    st.caption("Ask questions about your coverage, limits, and costs. Answers are grounded in your document.")

    # 2. Message History Area
    # A bounded container provides a consistent "chat window" with scrolling.
    msg_container = st.container(height=480, border=True)

    with msg_container:
        # Render the existing transcript from session state.
        # Each entry is expected to have:
        # - msg["role"]   -> "user" or "assistant"
        # - msg["content"]-> markdown-safe text
        for msg in st.session_state.chat_history:
            # st.chat_message provides native avatars and formatting per role.
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # If a question was just submitted, show a typing indicator until the API response arrives.
        if st.session_state.get("pending_question"):
            with st.chat_message("assistant"):
                st.markdown("💭 *Thinking...*")

    # 3. Native Chat Input
    # st.chat_input is pinned to the bottom and submits on Enter.
    if prompt := st.chat_input("Ask a question about your policy..."):
        # Append the user's message to the transcript.
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        # Store the message as "pending" so the next rerun can execute the API call.
        st.session_state.pending_question = prompt

        # Rerun immediately so the new message appears and the typing indicator shows.
        st.rerun()

    # 4. Handle the API call AFTER the UI updates
    # This block runs only on rerun after a user has submitted a prompt.
    if st.session_state.get("pending_question"):
        q = st.session_state.pending_question

        # Clear pending flag so the next rerun doesn't duplicate the same API call.
        st.session_state.pending_question = None

        try:
            # Call the backend Q&A endpoint.
            # Endpoint format: POST {API_BASE}/qa/{doc_id} with JSON {"question": "..."}
            qa_r = requests.post(
                api_url(f"/qa/{st.session_state['doc_id']}"),
                json={"question": q},
                timeout=120  # generous timeout to allow retrieval + LLM response time
            )

            # Raise HTTPError for non-2xx responses so we handle them in a dedicated except block.
            qa_r.raise_for_status()

            # Parse JSON payload produced by the backend.
            out = qa_r.json()

            # Scenario format support:
            # Scenario responses are structured as a header + step list, each step with citations.
            if out.get("answer_type") == "scenario":
                answer = f"**{out.get('header', 'Scenario Breakdown')}**\n\n"

                for step in out.get("steps", []):
                    # Format citations as "p. X" page references for user readability.
                    cites = ", ".join([f"p. {c['page']}" for c in step.get("citations", [])])

                    # Escape "$" so Streamlit Markdown doesn't interpret it as LaTeX math.
                    safe_text = (step.get("text", "") or "").replace("$", "\\$")

                    # Render each step as a numbered line with optional citation list.
                    answer += f"**{step.get('step_number')}.** {safe_text} *({cites})*\n"

            # Standard Q&A / Section detail format:
            else:
                raw_answer = (out.get("answer", "") or "").strip()

                # Guardrail: If the backend returns the same text as the user question,
                # treat it as an unhelpful echo and request a rephrase.
                if raw_answer.lower() == q.strip().lower():
                    answer = "I don’t see that information in your policy—could you rephrase?"
                else:
                    answer = raw_answer

                    # Only show citations when the answer type is one the UI considers valid/grounded.
                    is_valid_answer = out.get("answer_type") in ["answerable", "section_detail"]

                    # If citations exist and the answer type indicates the response is grounded,
                    # append a compact "Sources" line listing page numbers.
                    if out.get("citations") and is_valid_answer:
                        cites = ", ".join([f"p. {c['page']}" for c in out.get("citations", [])])
                        answer += f"\n\n*Sources: {cites}*"

            # Append assistant message to transcript for rendering on the next rerun.
            st.session_state.chat_history.append({"role": "assistant", "content": answer})

        # --- Error Handling ---
        # These cases provide user-friendly messages while preserving app stability.
        except requests.exceptions.ConnectionError:
        except requests.exceptions.ConnectionError:
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": "🚨 **Connection Error:** I couldn't reach the backend server. Please ensure the FastAPI server is running."
            })
        except requests.exceptions.Timeout:
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": "⏳ **Timeout:** The server took too long to respond. It might be processing a heavy load."
            })
        except requests.exceptions.HTTPError as e:
            # HTTPError contains response details like status_code.
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": f"⚠️ **Server Error:** The backend returned an error ({e.response.status_code}). Please try again."
            })
        except Exception as e:
            # Catch-all for unexpected runtime issues (parsing errors, missing keys, etc.).
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": f"❌ **Unexpected Error:** Something went wrong ({str(e)})."
            })

        # Final rerun to display the assistant response appended above.
        st.rerun()