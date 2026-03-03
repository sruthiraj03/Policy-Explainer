"""
Utility for generating PDF summaries.

This module provides a lightweight utility for converting a structured
policy summary dictionary (as returned by the backend) into a downloadable
PDF document using `fpdf2`.

Design goals:
- Produce a clean, readable summary document suitable for saving/sharing.
- Handle common Unicode characters that FPDF (Latin-1 based) cannot encode.
- Preserve section structure and bullet formatting.
- Return raw PDF bytes for use with Streamlit's `st.download_button`.

Important limitation:
- FPDF (default core fonts like Arial) uses Latin-1 encoding.
  Therefore, we proactively sanitize text to prevent encoding errors.
"""

from fpdf import FPDF  # pip install fpdf2


def generate_policy_pdf(summary_data: dict) -> bytes:
    """
    Generate a PDF file from a policy summary dictionary.

    Expected summary_data shape (simplified):
    {
        "sections": [
            {
                "section_name": "...",
                "present": True,
                "bullets": [
                    {"text": "...", "citations": [...]},
                    ...
                ]
            },
            ...
        ]
    }

    Args:
        summary_data (dict): Summary payload (typically from PolicySummaryOutput JSON).

    Returns:
        bytes: Encoded PDF binary suitable for file download.
    """

    def clean_text(text: str) -> str:
        """
        Normalize and sanitize text for FPDF (Latin-1 encoding).

        FPDF's default fonts do not support full Unicode. To prevent encoding
        exceptions, we:
        - Replace common Unicode punctuation with ASCII equivalents.
        - Encode to Latin-1 with "replace" to gracefully handle unsupported chars.

        Args:
            text (str): Raw text to sanitize.

        Returns:
            str: Latin-1-safe string.
        """
        replacements = {
            "\u2022": "-",   # Bullet
            "\u2018": "'",   # Left single quote
            "\u2019": "'",   # Right single quote
            "\u201c": '"',   # Left double quote
            "\u201d": '"',   # Right double quote
            "\u2013": "-",   # En dash
            "\u2014": "-",   # Em dash
        }

        # Replace known problematic Unicode characters.
        for k, v in replacements.items():
            text = text.replace(k, v)

        # Force Latin-1 compatibility (replace unsupported chars).
        return text.encode("latin-1", "replace").decode("latin-1")

    # Initialize PDF document.
    pdf = FPDF()
    pdf.add_page()

    # --- Title ---
    pdf.set_font("Arial", "B", 20)
    pdf.cell(0, 15, "Policy Explainer Summary", ln=True, align="C")

    # Reset to normal font for body.
    pdf.set_font("Arial", "", 10)

    # --- Sections ---
    for section in summary_data.get("sections", []):
        # Only include sections that are marked present.
        if section.get("present"):
            # Section header background fill (light gray).
            pdf.set_fill_color(240, 240, 240)
            pdf.set_font("Arial", "B", 14)

            # Section title cell with fill.
            pdf.cell(
                0,
                10,
                clean_text(f" {section['section_name']}"),
                ln=True,
                fill=True
            )
            pdf.ln(2)

            # Reset font for bullet text.
            pdf.set_font("Arial", "", 11)

            for bullet in section.get("bullets", []):
                # Multi-cell allows automatic line wrapping.
                pdf.multi_cell(
                    0,
                    7,
                    clean_text(f"- {bullet['text']}")
                )
                pdf.ln(1)

            # Extra spacing between sections.
            pdf.ln(5)

    # Generate output.
    # dest="S" returns the document as a string (Latin-1) in fpdf2.
    out = pdf.output(dest="S")

    # Ensure we return raw bytes (Streamlit download_button expects bytes).
    return out.encode("latin-1") if isinstance(out, str) else bytes(out)