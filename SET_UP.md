# Setup Guide

This document explains how to install, configure, and run PolicyExplainer locally.

PolicyExplainer consists of:

- A FastAPI backend (document processing + RAG pipeline)
- A Streamlit frontend (UI layer)
- A persistent Chroma vector database
- Local artifact storage per document

---

# System Requirements

- Python 3.10+
- pip
- Virtual environment support
- OpenAI API key

Optional (recommended):

- Git
- VS Code or similar IDE

---

# 1. Clone the Repository

```bash
git clone <your-repo-url>
cd PolicyExplainer
```

---

# 2. Create Virtual Environment

Create a virtual environment:

```bash
python -m venv .venv
```

Activate:

Windows:

```bash
.venv\Scripts\activate
```

Mac/Linux:

```bash
source .venv/bin/activate
```

---

# 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If you are adding the Simplicity metric using readability scoring, you may later include:

```bash
pip install textstat
```

---

# 4. Configure Environment Variables

Copy the template:

Windows:

```bash
copy .env.example .env
```

Mac/Linux:

```bash
cp .env.example .env
```

Edit `.env` and set:

```text
OPENAI_API_KEY=your_api_key_here
```

Optional configuration:

```text
LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
VECTOR_DB_PATH=./chroma_data
API_BASE_URL=http://localhost:8000
```

Notes:

- `OPENAI_API_KEY` is required.
- Do not hardcode secrets.
- `.env` should never be committed to version control.

---

# 5. Run Backend (FastAPI)

From the project root:

```bash
uvicorn backend.main:app --reload
```

Backend will run at:

```text
http://127.0.0.1:8000
```

API documentation (Swagger UI):

```text
http://127.0.0.1:8000/docs
```

---

# 6. Run Frontend (Streamlit)

In a separate terminal (with virtual environment activated):

```bash
streamlit run frontend/app.py
```

Frontend will run at:

```text
http://localhost:8501
```

---

# 7. Verify Installation

1. Upload a sample insurance policy PDF
2. Confirm a `doc_id` is generated
3. Generate a section summary
4. Ask a grounded question
5. Run evaluation via:

```text
POST /evaluate/{doc_id}
```

Successful evaluation returns:

- Faithfulness score
- Completeness score
- Simplicity score
- Validation results

---

# Project Storage Layout

After ingesting a document, artifacts are saved to:

```text
data/documents/{doc_id}/
├─ raw.pdf
├─ pages.json
├─ chunks.jsonl
├─ policy_summary.json
└─ evaluation_report.json
```

Vector embeddings are stored in:

```text
./chroma_data
```

These artifacts enable reproducibility and evaluation.

---

# Running Evaluation

You may evaluate a processed document using:

```bash
curl -X POST http://127.0.0.1:8000/evaluate/{doc_id}
```

Evaluation includes:

- Faithfulness scoring
- Completeness scoring
- Simplicity scoring
- Structural validation checks

Evaluation is deterministic.

---

# Optional: Enable Simplicity Metric

If implementing readability scoring (recommended):

Add dependency:

```bash
pip install textstat
```

The Simplicity Score may use:

- Flesch Reading Ease delta
- Average sentence length reduction
- Jargon frequency reduction

Ensure simplicity logic remains deterministic.

---

# Common Issues

## 1. OPENAI_API_KEY not set

Error:
```text
Missing API key
```

Solution:
- Confirm `.env` exists
- Ensure key is valid
- Restart backend after editing `.env`

---

## 2. PDF has no extractable text

Cause:
- Scanned PDF (image-only)

Solution:
- Run OCR before ingestion

---

## 3. Port Already in Use

Change backend port:

```bash
uvicorn backend.main:app --reload --port 8001
```

Change frontend port:

```bash
streamlit run frontend/app.py --server.port 8502
```

---

# Production Considerations

For deployment:

- Restrict CORS origins
- Add authentication
- Remove debug logging
- Use fixed model versions
- Containerize with Docker
- Add CI pipeline

---

# Summary

PolicyExplainer runs locally using:

- FastAPI backend
- Streamlit frontend
- Chroma vector persistence
- Deterministic evaluation framework

All components are modular and reproducible.

---

End of Setup Guide.
