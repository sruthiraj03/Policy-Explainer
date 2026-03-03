"""
Document storage (doc_store) and vector store (Chroma).

This module provides two complementary storage layers used by PolicyExplainer:

1) Local file-system document store ("doc_store")
   - Persists raw PDFs and derived artifacts (extracted pages, chunks, summaries)
   - Organizes each document under a UUID-named directory for natural isolation
   - Enables reproducibility and debugging (you can inspect raw inputs/outputs on disk)

2) Vector database store (Chroma)
   - Stores chunk embeddings for semantic retrieval
   - Queried by the QA and summarization pipelines to retrieve evidence
   - Updated to operate in a strict "stateless" mode:
       * The Chroma collection is wiped prior to ingesting any new document
       * This prevents cross-document retrieval leakage and reduces hallucination risk

Design notes:
- File artifacts are cached in-memory (via backend.utils cache helpers) to reduce repeated disk I/O.
- The Chroma client is cached (lru_cache) to reuse a persistent connection to the same directory.
- The Chroma collection is NOT cached on purpose, because the collection may be deleted/recreated
  during ingestion (wipe_database), and caching would return a stale reference.
"""

import json
import uuid
from pathlib import Path
from typing import Any
from functools import lru_cache

import chromadb
from chromadb.utils import embedding_functions

from backend.config import get_settings
from backend.schemas import Chunk, ExtractedPage, PolicySummaryOutput
from backend.utils import cache_get, cache_invalidate, cache_set

# --- File Paths & Constants ---
# Base directory where document artifacts are stored.
# Each document uses a subdirectory named by document_id.
DEFAULT_DOC_STORAGE_PATH = Path(__file__).resolve().parent.parent / "data" / "documents"

# Filenames used within each document directory.
RAW_PDF_FILENAME = "raw.pdf"
PAGES_JSON_FILENAME = "pages.json"
CHUNKS_JSONL_FILENAME = "chunks.jsonl"
POLICY_SUMMARY_FILENAME = "Policy_summary.json"

# Chroma collection name used for storing embedded chunks.
COLLECTION_NAME = "policy_chunks"


# --- 1. Local File System Storage ---
# These functions manage the physical PDF and JSON files on your hard drive.
# Because they are saved in UUID-named folders, they naturally isolate themselves.

def generate_document_id() -> str:
    """
    Generate a unique identifier for a document ingestion session.

    Returns:
        str: A UUID4 string used as doc_id throughout the pipeline.
    """
    return str(uuid.uuid4())


def _doc_dir(document_id: str, base_path: Path | None) -> Path:
    """
    Safely create (if needed) and return the directory path for a document.

    Security note:
    - This function rejects document_ids containing path separators to prevent path traversal.

    Args:
        document_id: UUID-like document identifier.
        base_path: Optional override for the root document storage path.

    Returns:
        Path: Directory for this document (created if missing).
    """
    # Resolve base path (default if not provided).
    base = base_path if base_path is not None else DEFAULT_DOC_STORAGE_PATH
    base = base.resolve()

    # Simple path-traversal prevention: disallow separators and "." / "..".
    if "/" in document_id or "\\" in document_id or document_id in (".", ".."):
        raise ValueError("Invalid document_id: must not contain path separators")

    # Ensure the document directory exists.
    doc_dir = base / document_id
    doc_dir.mkdir(parents=True, exist_ok=True)
    return doc_dir


def get_document_dir(document_id: str, base_path: Path | None = None) -> Path:
    """
    Return the storage directory path without creating it.

    This is useful for checking existence or constructing file paths without side effects.

    Args:
        document_id: UUID-like document identifier.
        base_path: Optional override for the root document storage path.

    Returns:
        Path: Directory path (may or may not exist on disk).
    """
    base = base_path if base_path is not None else DEFAULT_DOC_STORAGE_PATH
    base = base.resolve()
    return base / document_id


def save_raw_pdf(content: bytes, document_id: str, base_path: Path | None = None) -> Path:
    """
    Persist the raw uploaded PDF bytes to disk.

    Args:
        content: Raw PDF file bytes.
        document_id: UUID-like document identifier.
        base_path: Optional override for the root document storage path.

    Returns:
        Path: Path to the saved PDF file.
    """
    doc_dir = _doc_dir(document_id, base_path)
    path = doc_dir / RAW_PDF_FILENAME
    path.write_bytes(content)
    return path


def save_extracted_pages(pages: list[ExtractedPage], document_id: str, base_path: Path | None = None) -> Path:
    """
    Persist extracted page text to disk as JSON.

    This captures page boundaries explicitly, which is important for citations and debugging.

    Args:
        pages: List of ExtractedPage models.
        document_id: UUID-like document identifier.
        base_path: Optional override for the root document storage path.

    Returns:
        Path: Path to the saved pages.json file.
    """
    doc_dir = _doc_dir(document_id, base_path)
    path = doc_dir / PAGES_JSON_FILENAME
    with path.open("w", encoding="utf-8") as f:
        json.dump([p.model_dump() for p in pages], f, ensure_ascii=False, indent=2)
    return path


def save_chunks(chunks: list[Chunk], document_id: str, base_path: Path | None = None) -> Path:
    """
    Persist chunk objects to disk as JSONL.

    JSONL is used because:
    - It streams well for large numbers of chunks
    - Each line is independently parseable
    - It avoids large monolithic JSON arrays for big documents

    Cache behavior:
    - Invalidate the in-memory chunk cache for this document so future reads reflect new data.

    Args:
        chunks: List of Chunk models.
        document_id: UUID-like document identifier.
        base_path: Optional override for the root document storage path.

    Returns:
        Path: Path to the saved chunks.jsonl file.
    """
    doc_dir = _doc_dir(document_id, base_path)
    path = doc_dir / CHUNKS_JSONL_FILENAME
    with path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.model_dump(), ensure_ascii=False) + "\n")

    # Ensure any cached chunk list for this doc is cleared after write.
    cache_invalidate(f"chunks:{document_id}")
    return path


def load_chunks(document_id: str, base_path: Path | None = None) -> list[Chunk]:
    """
    Load chunk objects from disk (chunks.jsonl), with memoization via cache helpers.

    Cache behavior:
    - First checks in-memory cache
    - If missing, reads from disk, validates via Pydantic, then caches the result

    Args:
        document_id: UUID-like document identifier.
        base_path: Optional override for the root document storage path.

    Returns:
        list[Chunk]: Parsed chunk models.

    Raises:
        FileNotFoundError: If the chunks file does not exist for this document.
    """
    cache_key = f"chunks:{document_id}"
    cached = cache_get(cache_key)
    if cached is not None:
        return cached

    path = get_document_dir(document_id, base_path) / CHUNKS_JSONL_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"No chunks found for document {document_id}: {path}")

    chunks = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Validate each line into a Chunk model to enforce schema integrity.
            chunks.append(Chunk.model_validate(json.loads(line)))

    # Cache the parsed chunk list for subsequent calls.
    cache_set(cache_key, chunks)
    return chunks


def load_extracted_pages(document_id: str, base_path: Path | None = None) -> list[ExtractedPage]:
    """
    Load extracted pages (pages.json) from disk.

    Args:
        document_id: UUID-like document identifier.
        base_path: Optional override for the root document storage path.

    Returns:
        list[ExtractedPage]: Parsed extracted page models.

    Raises:
        FileNotFoundError: If pages.json does not exist for this document.
    """
    path = get_document_dir(document_id, base_path) / PAGES_JSON_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"No extracted pages found for document {document_id}: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Validate list items into ExtractedPage models.
    return [ExtractedPage.model_validate(item) for item in data]


def save_policy_summary(summary: PolicySummaryOutput, document_id: str, base_path: Path | None = None) -> Path:
    """
    Persist the final policy summary output to disk.

    Cache behavior:
    - Invalidate the in-memory summary cache for this document after saving.

    Args:
        summary: PolicySummaryOutput model.
        document_id: UUID-like document identifier.
        base_path: Optional override for the root document storage path.

    Returns:
        Path: Path to the saved Policy_summary.json file.
    """
    doc_dir = _doc_dir(document_id, base_path)
    path = doc_dir / POLICY_SUMMARY_FILENAME
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary.model_dump(), f, ensure_ascii=False, indent=2)

    cache_invalidate(f"summary:{document_id}")
    return path


def load_policy_summary(document_id: str, base_path: Path | None = None) -> PolicySummaryOutput:
    """
    Load a stored policy summary from disk (Policy_summary.json), with caching.

    Args:
        document_id: UUID-like document identifier.
        base_path: Optional override for the root document storage path.

    Returns:
        PolicySummaryOutput: Parsed and validated summary model.

    Raises:
        FileNotFoundError: If the summary file does not exist for this document.
    """
    cache_key = f"summary:{document_id}"
    cached = cache_get(cache_key)
    if cached is not None:
        return cached

    path = get_policy_summary_path(document_id, base_path)
    if not path.exists():
        raise FileNotFoundError(f"No policy summary for document {document_id}: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Validate JSON into the Pydantic summary model to enforce schema correctness.
    summary = PolicySummaryOutput.model_validate(data)

    # Cache validated object for future calls.
    cache_set(cache_key, summary)
    return summary


def get_policy_summary_path(document_id: str, base_path: Path | None = None) -> Path:
    """
    Return the full path to the stored policy summary JSON file.

    Args:
        document_id: UUID-like document identifier.
        base_path: Optional override for the root document storage path.

    Returns:
        Path: Path to Policy_summary.json under the document directory.
    """
    return get_document_dir(document_id, base_path) / POLICY_SUMMARY_FILENAME


# --- 2. Vector Database Storage (Chroma) ---
# These functions manage the AI's searchable memory.

@lru_cache(maxsize=1)
def _get_client():
    """
    Establish a persistent ChromaDB client connection.

    The client points to a local persistence directory derived from config settings.
    `@lru_cache(maxsize=1)` ensures a single client instance is reused across calls,
    reducing overhead and preventing repeated initialization.

    Returns:
        chromadb.PersistentClient: Connected client instance.
    """
    settings = get_settings()
    path = settings.get_vector_db_path_resolved()

    # Ensure persistence directory exists before initializing client.
    path.mkdir(parents=True, exist_ok=True)

    return chromadb.PersistentClient(path=str(path))


def _get_embedding_function():
    """
    Create the embedding function used by Chroma to embed documents and queries.

    This uses OpenAI embeddings with:
    - API key from environment
    - embedding model name from settings

    Returns:
        embedding_functions.OpenAIEmbeddingFunction: Embedding callable for Chroma.
    """
    settings = get_settings()
    return embedding_functions.OpenAIEmbeddingFunction(
        api_key=settings.openai_api_key,
        model_name=settings.embedding_model,
    )


def _get_collection():
    """
    Retrieve (or create) the active Chroma collection.

    Important:
    - We do NOT cache the collection reference here.
      The ingestion pipeline may delete the collection (wipe_database),
      and caching would return a stale object.

    Returns:
        chromadb.api.models.Collection.Collection: Active collection handle.
    """
    client = _get_client()
    ef = _get_embedding_function()

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"description": "Policy document chunks"},
    )


def wipe_database() -> None:
    """
    Delete the entire Chroma collection used for policy chunks.

    This is intentionally destructive and supports a "stateless" ingestion model:
    each new document ingest starts with an empty vector store, preventing
    any cross-document retrieval leakage.

    Side effects:
    - Deletes the collection if it exists.
    - Prints debug messages for visibility during development.
    """
    client = _get_client()
    try:
        client.delete_collection(COLLECTION_NAME)
        print("🧹 DEBUG: Database completely wiped. Slate is clean.")
    except Exception:
        # If the collection doesn't exist yet, deletion can raise an exception.
        # It is safe to ignore in that case.
        pass


def add_chunks(doc_id: str, chunks: list[Chunk]) -> None:
    """
    Wipe old vector data, then add the new document's chunks to the vector store.

    Steps:
    1) If no chunks, do nothing.
    2) Wipe the existing Chroma collection (stateless mode).
    3) Recreate/get an empty collection.
    4) Format chunk payloads (ids, documents, metadatas).
    5) Add to Chroma and force a heartbeat to ensure persistence.

    Args:
        doc_id: Document identifier (used for metadata scoping).
        chunks: Chunk models to embed/store.
    """
    if not chunks:
        return

    # 1. Annihilate all previous vector data before doing anything else.
    wipe_database()

    # 2. Get a brand new, empty collection.
    collection = _get_collection()
    doc_id_str = str(doc_id)

    # 3. Format the new data in the shape Chroma expects.
    # - ids: unique identifiers for each stored item
    # - documents: raw text strings
    # - metadatas: additional searchable fields (used for filtering and citation support)
    ids = [str(c.chunk_id) for c in chunks]
    documents = [c.chunk_text for c in chunks]
    metadatas = [
        {
            "chunk_id": str(c.chunk_id),
            "page_number": int(c.page_number),
            "doc_id": doc_id_str  # Still tagging it for good measure
        } for c in chunks
    ]

    # 4. Save the new document into the fresh database.
    collection.add(ids=ids, documents=documents, metadatas=metadatas)

    # 5. Force the client to "touch" the DB.
    # This is used here as a practical way to encourage flush/sync on certain platforms.
    _get_client().heartbeat()
    print(f"✅ DEBUG: New document ingested. Total DB Count: {collection.count()}")


def query(doc_id: str, query_text: str, top_k: int = 5) -> list[dict[str, Any]]:
    """
    Query the active vector store for chunks relevant to `query_text`.

    Safety/quality measures:
    - Empty queries return [] immediately.
    - Uses a `where` filter for doc_id as an extra guard, even though the collection is wiped
      per ingest. This reduces the chance of future changes accidentally mixing documents.

    Args:
        doc_id: Document identifier to scope retrieval.
        query_text: Natural language query string.
        top_k: Number of top results requested from Chroma.

    Returns:
        list[dict[str, Any]]: List of chunk-like dicts including:
          - chunk_id
          - page_number
          - doc_id
          - chunk_text
          - distance (vector distance / similarity measure)
        If query fails, returns [].
    """
    # Reject whitespace-only queries to avoid unnecessary DB calls.
    if not query_text.strip():
        return []

    collection = _get_collection()
    doc_id_str = str(doc_id)

    # Extra safety: filter results to this document_id.
    where_filter = {"doc_id": doc_id_str}

    # Heartbeat + debug log help diagnose persistence and query behavior.
    _get_client().heartbeat()
    print(f"🔎 DEBUG: Searching Vector DB for: '{query_text}'")

    try:
        # Query the vector DB for nearest-neighbor matches.
        results = collection.query(
            query_texts=[query_text.strip()],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        # Chroma returns results as lists-of-lists (one list per query text).
        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        out = []
        for i, (cid, doc_text) in enumerate(zip(ids, docs, strict=False)):
            # Defensive indexing: metadata/distances might be shorter than ids in edge cases.
            meta = metas[i] if i < len(metas) else {}
            distance = dists[i] if i < len(dists) else None

            # Normalize output into a stable dict shape used throughout the app.
            out.append({
                "chunk_id": cid,
                "page_number": meta.get("page_number", 0),
                "doc_id": meta.get("doc_id", doc_id),
                "chunk_text": doc_text or "",
                "distance": distance
            })
        return out

    except Exception as e:
        # If Chroma throws (persistence issues, embedding errors, etc.), return empty results.
        # Debug print preserves existing behavior and helps during development.
        print(f"❌ DEBUG: ChromaDB Query Failed: {e}")
        return []