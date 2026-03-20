# PolicyExplainer

PolicyExplainer is a full-stack Retrieval-Augmented Generation (RAG) system that transforms complex health insurance policy PDFs into structured, grounded, and evaluable outputs.

It is designed as a controlled document intelligence pipeline, not a generic chatbot, with strong emphasis on:

- Grounded generation (no external knowledge)
- Strict citation enforcement
- Deterministic preprocessing
- Multi-metric evaluation
- Reproducibility and traceability

---

# Problem Statement

Health insurance policies are long, technical, and difficult for users to interpret.

Traditional LLM-based solutions often:
- Hallucinate unsupported information  
- Miss critical coverage details  
- Use complex terminology  
- Provide no measure of answer reliability  

PolicyExplainer addresses this by:

- Restricting outputs strictly to document content  
- Enforcing citation-backed answers  
- Removing unsupported claims  
- Separating deterministic and probabilistic layers  
- Quantifying output quality using:
  - Faithfulness  
  - Completeness  
  - Simplicity  

---

# Key Features

## 1. End-to-End Document Intelligence Pipeline

Upload → Process → Summarize → Evaluate → Ask Questions

- Upload policy PDF (Streamlit UI)
- Backend ingestion and validation
- Structured summary generation
- Evaluation scoring
- Interactive grounded Q&A

---

## 2. Deterministic PDF Ingestion

- Page-level extraction using PyMuPDF
- Header and footer cleaning
- Policy validation using keyword heuristics
- Token-based chunking with overlap
- Page-aware chunking for accurate citations

Artifacts stored per document:

```
data/documents/{doc_id}/
├─ raw.pdf
├─ pages.json
├─ chunks.jsonl
```

---

## 3. Retrieval-Augmented Generation (RAG)

- Vector embeddings using OpenAI
- Stored in ChromaDB
- Section-aware multi-query retrieval
- Deduplication by chunk_id
- Context ordering by document structure

Improves:
- Recall
- Relevance
- Citation grounding

---

## 4. Structured Summarization

Summaries follow a strict schema:

```json
{
  "section_name": "...",
  "present": true,
  "bullets": [
    {
      "text": "...",
      "citations": [{ "page": 1, "chunk_id": "c_1_0" }]
    }
  ]
}
```

---

## 5. Grounded Question Answering

* Retrieval of top-k relevant chunks
* Context sorted in document order
* Strict JSON response enforcement
* Citation validation and filtering
* Removal of unsupported claims
* Deterministic fallback:

```
Not found in this document.
```

---

# System Capabilities

## 1. Deterministic PDF Ingestion

* Page-level text extraction using PyMuPDF
* Header and footer cleaning
* Heuristic validation of policy structure
* Token-based chunking (500 to 800 tokens)
* Sliding overlap (~80 tokens)
* Deterministic chunk IDs:

```
c_{page_number}_{chunk_index}
```

All artifacts are persisted for reproducibility.

---

## 2. Section-Aware Retrieval

Instead of a single embedding query, each canonical policy section triggers multiple semantic sub-queries.

Example (Cost Summary):

* deductible
* copay
* coinsurance
* out-of-pocket maximum
* premium

Retrieval pipeline:

1. Run vector search per sub-query
2. Deduplicate by chunk_id
3. Retain lowest-distance match
4. Sort by (page_number, chunk_id)
5. Cap context window

This improves recall and mitigates the "lost in the middle" problem.

---

## 3. Structured Summarization

Each section summary enforces a strict JSON contract:

* present (boolean)
* bullets[]

  * text
  * citations[] (chunk_id and page)

Post-generation enforcement:

* Filter citations to allowed chunk_ids
* Drop bullets without valid citations
* Record validation issues
* Compute confidence score

Unsupported statements never appear in final output.

---

## 4. Grounded Q&A

Q&A pipeline:

1. Retrieve top-k relevant chunks
2. Sort in document order
3. Generate structured response
4. Validate citations
5. Remove unsupported claims
6. Compute confidence score

If no supporting chunks exist:

```
Not found in this document.
```

---

# Evaluation Framework

PolicyExplainer includes deterministic post-generation evaluation:

```
POST /evaluate/{doc_id}
```

---

## 1. Faithfulness (0.0 to 1.0)

Measures whether each summary bullet is supported by its cited chunk.

Support logic:

* Token overlap threshold
* Numeric consistency checks
* Citation validation

---

## 2. Completeness (0.0 to 1.0)

Measures coverage across canonical policy sections.

Weighted scoring:

* Cost Summary (35 percent)
* Covered Services (30 percent)
* Administrative Conditions (15 percent)
* Exclusions and Limitations (10 percent)
* Plan Snapshot (5 percent)
* Claims and Appeals (5 percent)

---

## 3. Simplicity (0.0 to 1.0)

Measures how much more understandable the generated summary is compared to the original policy.

Evaluates:

* Readability improvement (sentence length, Flesch score)
* Jargon reduction using predefined dictionary
* Structural clarity (bullet formatting, reduced complexity)

Example:

```
Simplicity Score =
0.4 * readability_improvement
+ 0.4 * jargon_reduction
+ 0.2 * structural_simplification
```

---

# Architecture Overview

The system consists of:

* Frontend: Streamlit
* Backend: FastAPI

Core backend modules:

```
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

Artifacts:

```
data/documents/{doc_id}/
├─ raw.pdf
├─ pages.json
├─ chunks.jsonl
└─ policy_summary.json
```

Vector storage:

```
./chroma_data
```

---

# Deterministic vs Probabilistic Layers

Deterministic:

* Chunking
* Retrieval ordering
* Deduplication
* Citation filtering
* Faithfulness scoring
* Completeness scoring
* Simplicity scoring
* Confidence scoring

Probabilistic:

* LLM generation

This separation improves auditability and reduces hallucination risk.

---

# Hallucination Mitigation Strategy

* Context-limited retrieval
* Multi-query recall
* Strict JSON contract enforcement
* Citation filtering
* Removal of unsupported bullets
* Explicit fallback response
* Deterministic evaluation

Reliability is prioritized over verbosity.

---

# Technology Stack

Backend:

* Python
* FastAPI
* Pydantic
* PyMuPDF
* ChromaDB
* OpenAI API

Frontend:

* Streamlit

Storage:

* Local JSON artifacts
* Persistent vector index

---

# Key Engineering Decisions

* Section-aware multi-query retrieval
* Deterministic chunking
* Post-generation citation validation
* Removal of unsupported claims
* Weighted completeness scoring
* Readability-based simplicity scoring
* Separation of evaluation from generation

---

# Why This Project Matters

PolicyExplainer demonstrates real-world RAG system design with:

* Retrieval optimization
* Guardrail engineering
* Structured output enforcement
* Multi-axis evaluation
* Reproducibility and traceability

It focuses on building reliable, auditable AI systems rather than simple LLM integrations.

---

*End of README.*
