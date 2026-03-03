"""
CSS Injection for the Streamlit Frontend.

This module centralizes all custom CSS overrides used by the PolicyExplainer
Streamlit interface.

Purpose:
- Improve visual hierarchy (hero title, subtitles, dashboard title).
- Reduce default Streamlit padding for tighter layout control.
- Customize file uploader messaging.
- Implement a sticky chat column for improved UX.
- Standardize button styling across the app.

Design philosophy:
- Use targeted selectors (e.g., data-testid attributes) to override
  specific Streamlit components.
- Keep all CSS in one place for easier maintenance and iteration.
- Inject CSS once at app startup via `load_css()`.
"""

import streamlit as st


def load_css():
    """
    Inject custom CSS into the Streamlit app.

    This function should be called once at the top of the main app script
    (after `st.set_page_config()` if used).

    Notes:
    - Uses `unsafe_allow_html=True` because raw <style> tags are required.
    - Relies on Streamlit's internal `data-testid` attributes and layout structure.
      These may change across Streamlit versions.
    """
    st.markdown("""
    <style>

      /* --- Global Layout Adjustments --- */

      /* Keep the top app header white and layered above content */
      .stAppHeader {
        background-color: white;
        z-index: 99;
      }

      /* Reduce default vertical padding for tighter layout */
      .block-container {
        padding-top: 1rem;
      }

      /* --- HERO SECTION STYLES --- */

      /* Main landing page title */
      .hero-title {
        font-size: 60px !important;
        font-weight: 800;
        color: #0F172A;
        margin-bottom: 0px;
        line-height: 1.1;
      }

      /* Subtitle below hero title */
      .hero-subtitle {
        font-size: 28px !important;
        font-weight: 500;
        color: #334155;
        margin-bottom: 20px;
        margin-top: 10px;
      }

      /* Supporting hero text paragraph */
      .hero-text {
        font-size: 18px !important;
        line-height: 1.6;
        color: #475569;
        margin-bottom: 30px;
      }

      /* Dashboard title styling */
      .dash-title {
        font-size: 32px !important;
        font-weight: 800;
        color: #0F172A;
        margin: 0;
        line-height: 1.2;
      }

      /* --- File Uploader Customization --- */

      /* Replace default drag-and-drop text */
      [data-testid="stFileUploaderDropzone"] div div::before {
        content: "Select your insurance document ";
        visibility: visible;
      }

      /* Hide the original helper text */
      [data-testid="stFileUploaderDropzone"] div div span {
        display: none;
      }

      /* --- Sticky Chat Column --- */

      /*
        Trick:
        - Insert a marker div in layout (.chat-sticky-marker).
        - Hide its container.
        - Make the following container sticky.
      */

      /* Hide the placeholder marker container */
      div[data-testid="element-container"]:has(.chat-sticky-marker) {
        display: none !important;
      }

      /* Make the actual chat container sticky */
      div[data-testid="element-container"]:has(.chat-sticky-marker) + div {
        position: sticky !important;
        top: 110px !important;
        align-self: flex-start !important;
      }

      /* --- Button Styling --- */

      /* Standardize primary and download button appearance */
      div.stButton > button,
      div.stDownloadButton > button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        font-weight: 600;
      }

    </style>
    """, unsafe_allow_html=True)