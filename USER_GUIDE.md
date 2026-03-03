# User Guide

This guide explains how to use the deployed PolicyExplainer application.

PolicyExplainer helps users understand complex health insurance policy documents by generating structured summaries, grounded answers, FAQs, and evaluation-backed reliability metrics.

No technical knowledge is required.

---

# Application Overview

PolicyExplainer allows you to:

- Upload a health insurance policy PDF
- Generate structured policy summaries
- Ask grounded questions through a RAG-powered assistant
- Automatically generate FAQs specific to your policy
- Download a structured PDF summary
- View confidence indicators for generated content
- Evaluate summary quality using measurable metrics

All outputs are strictly grounded in your uploaded document.

If information is not found in the document, the system will explicitly state:

```
Not found in this document.
```

The system does not use external knowledge.

---

# First Screen: Upload Your Policy

When you first open the application, you will see:

- A header explaining the purpose of PolicyExplainer
- A file upload section labeled:
  "Select your insurance document"
- A "Browse files" button

To begin:

1. Click "Browse files"
2. Select your insurance policy PDF
3. Wait for processing to complete

Once ingestion finishes, the application transitions to the main summary dashboard.

---

# Main Dashboard

After uploading a policy, you will see:

- Navigation buttons at the top:
  - Summary
  - New Policy
  - Save Insurance Summary
  - FAQs
- A Policy Highlights section
- A Policy Assistant chat interface on the right

---

# Policy Highlights (Summary View)

The Summary tab displays structured highlights of your policy.

Sections may include:

- Plan Snapshot
- Cost Summary
- Covered Services
- Administrative Conditions
- Exclusions & Limitations
- Claims & Appeals

Each section:

- Is collapsible
- Displays structured bullet points
- Includes page citations
- Displays a Confidence level (e.g., HIGH)

Example:

```
Confidence: HIGH
• Bullet point summary (p. 3)
```

Confidence reflects citation validation strength and structural consistency.

---

# Save Insurance Summary (Download PDF)

You may download a structured summary of your policy.

To download:

1. Click "Save Insurance Summary"
2. A formatted PDF summary will be generated
3. Save the file locally

The exported PDF includes:

- Structured sections
- Bullet-point summaries
- Page references

This is useful for sharing or record-keeping.

---

# FAQs (Policy-Specific Frequently Asked Questions)

The FAQs button generates policy-specific frequently asked questions based on your uploaded document.

FAQs are:

- Automatically generated from the policy content
- Grounded in document text
- Structured for clarity
- Citation-backed

This provides quick insight into commonly important coverage details.

---

# Policy Assistant (RAG-Based Chat)

On the right side of the dashboard is the Policy Assistant.

You can ask natural language questions such as:

- What is my deductible?
- Is emergency care covered?
- Do I need prior authorization?
- What is excluded?

To ask a question:

1. Type your question in the input field
2. Click the arrow button to submit
3. Review the response

The assistant will:

- Retrieve relevant document chunks
- Generate a structured answer
- Validate citations
- Display a confidence indicator

If the policy does not contain the requested information, the assistant responds:

```
Not found in this document.
```

---

# New Policy

To analyze a different policy:

1. Click "New Policy"
2. Upload a new PDF
3. The previous document context is cleared
4. A new document ID is created

Each policy is processed independently.

---

# Confidence Levels

Each generated section and response includes a confidence level.

Confidence is based on:

- Citation density
- Citation validity
- Retrieval strength
- Structural validation checks

Confidence indicates reliability relative to the document, not legal correctness.

---

# Evaluation Metrics

The system can evaluate summary quality using three metrics.

## 1. Faithfulness

Measures whether summary bullets are supported by cited document text.

Higher score = stronger grounding.

---

## 2. Completeness

Measures coverage across important policy sections using weighted scoring.

Higher score = more comprehensive summary.

---

## 3. Simplicity

Measures how much easier the summary is to understand compared to the original policy text.

Simplicity considers:

- Reduction in sentence complexity
- Lower jargon density
- Improved readability
- Structural clarity through bullet formatting

Higher score = clearer explanation.

---

# Important Limitations

PolicyExplainer:

- Does not provide medical advice
- Does not provide legal advice
- Does not interpret policy intent
- Cannot extract text from scanned PDFs without OCR

If a PDF contains only images, ingestion may fail.

---

# Best Practices

For best results:

- Ask specific, focused questions
- Review citations for critical decisions
- Use section summaries before detailed questions
- Generate FAQs for quick insights
- Download summaries for offline review

---

# Data Handling

- Uploaded documents are stored locally on the server.
- Documents are not used for model training.
- API keys are stored in environment variables.
- Documents may be deleted depending on deployment configuration.

---

# Summary

PolicyExplainer is designed to make health insurance policies:

- Transparent
- Grounded
- Measurable
- Easier to understand

The system prioritizes reliability and clarity over speculation.

---

End of User Guide.
