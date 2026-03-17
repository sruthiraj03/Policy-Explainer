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
    st.markdown("""
    <style>

      /* --- Global Layout Adjustments --- */

      .stAppHeader {
        background-color: white;
        z-index: 99;
      }

      .block-container {
        padding-top: 1rem;
      }

      /* --- HERO SECTION STYLES --- */

      .hero-title {
        font-size: 38px !important; 
        font-weight: 600;
        color: #0F172A;
        margin-bottom: 0px;
        line-height: 1.1;
      }

      .hero-subtitle {
        font-size: 25px !important;  
        font-weight: 500;
        color: #334155;
        margin-bottom: 20px;
        margin-top: 10px;
      }

      .hero-text {
        font-size: 15px !important;  
        line-height: 1.6;
        color: #475569;
        margin-bottom: 20px;
      }

      .dash-title {
        font-size: 28px !important;
        font-weight: 800;
        color: #0F172A;
        margin: 0;
        line-height: 1.2;
      }

      /* --- File Uploader Customization --- */

      [data-testid="stFileUploaderDropzone"] {
        position: relative;
        padding: 10px 10px 10px 10px !important;  
        min-height: 10px;
        border-radius: 10px;
      }

      [data-testid="stFileUploaderDropzone"]::before {
        content: "Select your insurance document";
        position: absolute;
        top: 24px;
        left: 24px;
        font-size: 16px;  
        font-weight: 600;
        color: #0F172A;
        line-height: 1.1;
        pointer-events: none;
        z-index: 2;
      }

      [data-testid="stFileUploaderDropzone"] button {
        position: absolute;
        right: 16px;
        top: 50%;
        transform: translateY(-50%);

        font-size: 16px !important; 
        font-weight: 400;
        padding: 8px 8px !important;
        border-radius: 8px;
      }

      [data-testid="stFileUploaderDropzoneInstructions"] {
        display: none !important;
      }

      [data-testid="stFileUploaderDropzone"] small {
        display: none !important;
      }

      /* --- Sticky Chat Column --- */

      div[data-testid="element-container"]:has(.chat-sticky-marker) {
        display: none !important;
      }

      div[data-testid="element-container"]:has(.chat-sticky-marker) + div {
        position: sticky !important;
        top: 110px !important;
        align-self: flex-start !important;
      }

      /* --- Button Styling --- */

      div.stButton > button,
      div.stDownloadButton > button {
        width: 100%;
        border-radius: 8px;
        height: 2.6em;   /* slightly reduced */
        font-weight: 600;
        font-size: 14px;
      }

    </style>
    """, unsafe_allow_html=True)