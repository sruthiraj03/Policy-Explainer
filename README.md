# PolicyExplainer

PolicyExplainer is a modular Retrieval-Augmented Generation (RAG) system that transforms complex health insurance policy PDFs into structured, grounded, and evaluable outputs.

The system is engineered for:

- Deterministic preprocessing
- Strict citation enforcement
- Retrieval precision
- Multi-metric evaluation
- Reproducibility and traceability

PolicyExplainer is not a general chatbot. It is a controlled document intelligence pipeline built to reduce hallucination risk and improve interpretability in applied LLM systems.

---

# Problem Statement

Insurance policies are legally dense and difficult for consumers to interpret. Traditional LLM-based systems often:

- Hallucinate unsupported details
- Provide incomplete summaries
- Overuse jargon
- Fail to quantify output reliability

PolicyExplainer addresses these issues by:

- Restricting answers strictly to document content
- Enforcing citation validation
- Dropping unsupported claims
- Separating deterministic processing from probabilistic generation
- Measuring output quality across three evaluation dimensions:
  - Faithfulness
  - Completeness
  - Simplicity

---

# System Capabilities

## 1. Deterministic PDF Ingestion

- Page-level text extraction (PyMuPDF)
- Header/footer cleaning
- Heuristic validation of likely policy structure
- Token-based chunking (500–800 tokens)
- Sliding overlap (~80 tokens)
- Deterministic chunk IDs:

```text
c_{page_number}_{chunk_index}
```

All artifacts are persisted for reproducibility.

---

## 2. Section-Aware Retrieval

Rather than a single embedding query, each canonical policy section triggers multiple semantic sub-queries.

Example (Cost Summary):

- deductible
- copay
- coinsurance
- out-of-pocket maximum
- premium

Retrieval pipeline:

1. Run vector search per sub-query
2. Deduplicate by chunk_id
3. Retain lowest-distance match
4. Sort by (page_number, chunk_id)
5. Cap context window to avoid overload

This improves recall and mitigates the "Lost-in-the-Middle" problem.

---

## 3. Structured Summarization

Each section summary enforces a strict JSON contract:

- present (boolean)
- bullets[]
  - text
  - citations[] (chunk_id + page)

Post-generation enforcement:

- Filter citations to allowed chunk_ids
- Drop bullets without valid citations
- Record validation issues
- Compute confidence score

Unsupported statements never appear in final output.

---

## 4. Grounded Q&A

Users can ask natural language questions about the uploaded policy.

Q&A pipeline:

1. Retrieve top-k relevant chunks
2. Sort in document order
3. Force structured JSON response
4. Filter invalid citations
5. Remove unsupported claims
6. Compute confidence score

If no supporting chunks exist, the system returns exactly:

```text
Not found in this document.
```

No external knowledge is used.

---

# Evaluation Framework

PolicyExplainer includes deterministic post-generation evaluation via:

```text
POST /evaluate/{doc_id}
```

The system measures output quality across three independent axes.

---

## 1. Faithfulness (0.0 – 1.0)

Measures whether each summary bullet is supported by its cited chunk.

Support logic:

- Token overlap threshold
- Numeric consistency checks
- Citation validation against retrieved chunk_ids

High faithfulness means claims are grounded in the document.

---

## 2. Completeness (0.0 – 1.0)

Measures coverage across canonical policy sections.

Weighted scoring:

- Cost Summary (35%)
- Covered Services (30%)
- Administrative Conditions (15%)
- Exclusions & Limitations (10%)
- Plan Snapshot (5%)
- Claims & Appeals (5%)

Encourages balanced and comprehensive summaries.

---

## 3. Simplicity (0.0 – 1.0)

Measures how much more understandable the generated summary is compared to the original policy text.

The Simplicity Score evaluates:

1. Readability Improvement
   - Reduction in average sentence length
   - Flesch Reading Ease delta between source and summary

2. Jargon Reduction
   - Decrease in domain-specific terms using a predefined jargon dictionary
   - Percentage of simplified terminology substitutions

3. Structural Clarity
   - Bullet formatting vs dense paragraph text
   - Reduced clause complexity

Example logic:

```text
Simplicity Score =
0.4 * readability_improvement
+ 0.4 * jargon_reduction
+ 0.2 * structural_simplification
```

Interpretation:

- ≥ 0.75 → Strong simplification
- 0.50–0.75 → Moderate simplification
- < 0.50 → Limited simplification

This metric ensures summaries are not only grounded and complete, but genuinely easier to understand.

---

# Architecture Overview

The system is divided into:

- Frontend (Streamlit) — UI layer
- Backend (FastAPI) — document processing & LLM orchestration

Core backend modules:

```text
backend/
├─ api.py
├─ ingestion.py
├─ retrieval.py
├─ summarization.py
├─ qa.py
├─ evaluation.py
├─ storage.py
├─ schemas.py
└─ utils.py
```

Artifacts are stored per document:

```text
data/documents/{doc_id}/
├─ raw.pdf
├─ pages.json
├─ chunks.jsonl
└─ policy_summary.json
```

Vector embeddings are stored in:

```text
./chroma_data
```

---

# Deterministic vs Probabilistic Layers

Deterministic:

- Chunking
- Retrieval ordering
- Deduplication
- Citation filtering
- Faithfulness scoring
- Completeness scoring
- Simplicity scoring
- Confidence scoring

Probabilistic:

- LLM generation

By isolating deterministic validation layers, the system reduces hallucination exposure and increases auditability.

---

# Hallucination Mitigation Strategy

PolicyExplainer reduces hallucination risk through:

- Context-limited retrieval
- Multi-query recall
- Strict JSON contract enforcement
- Citation filtering
- Automatic removal of unsupported bullets
- Explicit "Not found" requirement
- Deterministic faithfulness scoring

Reliability is prioritized over verbosity.

---

# Technology Stack

Backend:
- Python
- FastAPI
- Pydantic
- PyMuPDF
- Chroma Vector Database
- OpenAI API

Frontend:
- Streamlit

Persistence:
- Local JSON artifact storage
- Persistent vector index

---

# Key Engineering Decisions

- Section-aware multi-query retrieval
- Deterministic chunking for reproducibility
- Post-generation citation validation
- Automatic removal of unsupported claims
- Weighted completeness scoring
- Readability-based simplicity scoring
- Separation of evaluation from generation

These decisions prioritize interpretability, reliability, and measurable output quality.

---

# Why This Project Matters

PolicyExplainer demonstrates applied RAG system design with:

- Retrieval optimization
- Guardrail engineering
- Structured output enforcement
- Multi-axis evaluation metrics
- Reproducibility and traceability

It reflects deliberate engineering around LLM reliability rather than simple API integration.

---

End of README.
