"""
Shared utilities: terminology normalization and document cache.

This module contains small, reusable helpers used across the PolicyExplainer backend:

1) Terminology normalization
   - Loads a canonical-terms map from JSON (canonical -> list of synonyms).
   - Rewrites text to replace synonyms with canonical terms for consistency.
   - Preserves quoted text (single-quoted and double-quoted) so that:
       * exact policy phrases, user quotes, or literal strings are not altered
       * downstream validation/citation alignment is less likely to break

2) Lightweight in-memory TTL cache
   - Stores Python objects (e.g., parsed chunks, summaries) for a short time window.
   - Avoids repeated disk reads and JSON parsing for frequently accessed artifacts.
   - Intended as a per-process cache (not shared across machines/processes).
"""

import json
import re
import time
from pathlib import Path
from typing import Any

# --- Terminology normalization ---

# Default location for the terminology map used by normalize_text().
# The file is expected to be JSON shaped like:
# {
#   "canonical term": ["synonym 1", "synonym 2", ...],
#   ...
# }
DEFAULT_TERMINOLOGY_PATH = Path(__file__).resolve().parent.parent / "schema" / "terminology_map.json"


def load_terminology_map(path: Path | str | None = None) -> dict[str, list[str]]:
    """
    Load a canonical->synonyms terminology map from JSON.

    Behavior is intentionally defensive:
    - If the file does not exist, returns {}
    - If JSON parsing fails, returns {}
    - If the JSON root is not a dict, returns {}
    - Ensures keys are strings, and values are lists of strings (or [] if invalid)

    Args:
        path: Optional path override. If None, DEFAULT_TERMINOLOGY_PATH is used.

    Returns:
        dict[str, list[str]]: Canonical term -> list of synonym strings.
    """
    p = path if path is not None else DEFAULT_TERMINOLOGY_PATH
    p = Path(p)

    # Missing file: no normalization possible.
    if not p.exists():
        return {}

    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        # Any read/parse error results in an empty map.
        return {}

    # Ensure correct top-level structure.
    if not isinstance(data, dict):
        return {}

    # Normalize structure to dict[str, list[str]] regardless of input types.
    return {str(k): [str(s) for s in v] if isinstance(v, list) else [] for k, v in data.items()}


def _extract_quoted_placeholders(text: str) -> tuple[str, list[str]]:
    """
    Replace quoted substrings with placeholders, returning:
    - transformed text (with placeholders)
    - list of original quoted substrings (in capture order)

    This prevents normalization from rewriting content inside quotes.

    Example:
        Input:  She said "out of network" costs differ.
        Output: She said <placeholder0> costs differ.
        Quoted: ['"out of network"']

    Args:
        text: Input string.

    Returns:
        tuple[str, list[str]]: (text_with_placeholders, quoted_substrings)
    """
    quoted: list[str] = []
    placeholder = "\x00QUOTE\x00"  # low-likelihood sentinel token to avoid accidental collisions

    def repl(m: re.Match) -> str:
        # Store the entire matched quoted substring (including quote characters).
        quoted.append(m.group(0))
        # Replace it with a unique placeholder token containing its index.
        return f"{placeholder}{len(quoted) - 1}{placeholder}"

    # Replace double-quoted substrings: "..."
    t = re.sub(r'"[^"]*"', repl, text)

    # Replace single-quoted substrings: '...'
    t = re.sub(r"'[^']*'", repl, t)

    return t, quoted


def _restore_quoted(text: str, quoted: list[str]) -> str:
    """
    Restore previously extracted quoted substrings back into the text.

    Args:
        text: Text containing placeholders.
        quoted: Quoted substrings captured by _extract_quoted_placeholders.

    Returns:
        str: Text with placeholders replaced by original quoted content.
    """
    placeholder = "\x00QUOTE\x00"
    for i, q in enumerate(quoted):
        text = text.replace(f"{placeholder}{i}{placeholder}", q)
    return text


def normalize_text(text: str, terminology_map: dict[str, list[str]] | None = None) -> str:
    """
    Replace synonym phrases with canonical terms.

    Rules:
    - Whole-phrase only: matches must be bounded by non-word characters.
      (Prevents partial replacements inside longer words.)
    - Case-insensitive matching.
    - Quoted text is left unchanged (both 'single' and "double" quotes).
    - Synonym replacement order is longest-first to prevent shorter phrases
      from pre-empting longer, more specific matches.

    Args:
        text: Input string to normalize.
        terminology_map: Optional canonical->synonyms map. If None, it is loaded from disk.

    Returns:
        str: Normalized text.
    """
    # Fast-path: empty/whitespace-only text has nothing to normalize.
    if not text.strip():
        return text

    # Load default terminology map if one wasn't supplied.
    if terminology_map is None:
        terminology_map = load_terminology_map()

    # If no mappings exist, return input unchanged.
    if not terminology_map:
        return text

    # Temporarily replace quoted substrings with placeholders to avoid rewriting them.
    work, quoted = _extract_quoted_placeholders(text)

    # Build (synonym -> canonical) replacement pairs.
    pairs: list[tuple[str, str]] = []
    for canonical, synonyms in terminology_map.items():
        for syn in synonyms:
            if syn.strip():
                pairs.append((syn.strip(), canonical))

    # Replace longer synonyms first to reduce conflicts/overlaps.
    pairs.sort(key=lambda x: -len(x[0]))

    # Apply phrase-level replacements.
    for synonym, canonical in pairs:
        # Word-boundary style protection:
        # - (?<!\w) ensures the synonym is not preceded by a word character
        # - (?!\w) ensures the synonym is not followed by a word character
        # This approximates "whole phrase" matching even for multi-word synonyms.
        pattern = r"(?<!\w)" + re.escape(synonym) + r"(?!\w)"
        work = re.sub(pattern, canonical, work, flags=re.IGNORECASE)

    # Restore quoted substrings exactly as originally captured.
    return _restore_quoted(work, quoted)


# --- Document cache (TTL) ---

# Time-to-live for cached values (seconds).
# This is intentionally short to reduce staleness while still preventing repeated I/O.
CACHE_TTL_SECONDS = 300

# In-memory cache dict storing: key -> (expiry_timestamp, value)
_cache: dict[str, tuple[float, Any]] = {}


def cache_get(key: str) -> Any | None:
    """
    Retrieve a cached value if present and not expired.

    Args:
        key: Cache key.

    Returns:
        Any | None: Cached value if valid; otherwise None.
    """
    if key not in _cache:
        return None

    expiry, value = _cache[key]

    # Expire entries lazily on access.
    if time.time() > expiry:
        del _cache[key]
        return None

    return value


def cache_set(key: str, value: Any) -> None:
    """
    Store a value in the cache with TTL.

    Args:
        key: Cache key.
        value: Arbitrary Python object to cache.
    """
    _cache[key] = (time.time() + CACHE_TTL_SECONDS, value)


def cache_invalidate(key: str) -> None:
    """
    Remove a specific key from the cache if it exists.

    Args:
        key: Cache key to remove.
    """
    _cache.pop(key, None)


def cache_clear() -> None:
    """
    Clear the entire cache.

    This is useful for tests or when you want to force fresh reads across the app.
    """
    _cache.clear()