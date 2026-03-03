"""
Centralized configuration from environment variables.

This module defines a single source of truth for all runtime configuration
used by the PolicyExplainer backend. Configuration values are loaded from:

- Environment variables
- A local `.env` file (if present)
- Safe, non-secret defaults (where applicable)

Usage:

    from backend.config import get_settings

    settings = get_settings()
    settings.openai_api_key  # never log or expose
    settings.embedding_model
    settings.llm_model
    settings.vector_db_path

Security principles:
- No secrets are hardcoded.
- OPENAI_API_KEY is required and must be non-empty.
- The settings object is cached to ensure consistent configuration
  across the entire application lifecycle.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    This class extends `BaseSettings`, meaning:
    - Fields are automatically populated from environment variables.
    - Values can also be loaded from a `.env` file.
    - Type validation and constraints are enforced automatically.

    Required:
        OPENAI_API_KEY (must be set and non-empty)

    Optional (with safe defaults):
        EMBEDDING_MODEL
        LLM_MODEL
        VECTOR_DB_PATH

    Environment variable names are case-insensitive due to configuration.
    """

    # Configuration for how Pydantic loads environment variables.
    model_config = SettingsConfigDict(
        env_file=".env",                # Automatically load variables from .env file
        env_file_encoding="utf-8",      # Ensure consistent encoding
        extra="ignore",                 # Ignore unexpected environment variables
        case_sensitive=False,           # Allow flexible casing (OPENAI_API_KEY vs openai_api_key)
    )

    # Required field: no default value provided.
    # The ellipsis (...) tells Pydantic this is mandatory.
    openai_api_key: str = Field(
        ...,
        min_length=1,
        description="OpenAI API key; set via OPENAI_API_KEY. Never committed.",
    )

    # Safe default embedding model.
    # This does NOT contain sensitive information.
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="Model used for embedding policy chunks (EMBEDDING_MODEL).",
    )

    # Safe default LLM model for summarization and Q&A.
    llm_model: str = Field(
        default="gpt-4o-mini",
        description="Model used for summarization and Q&A (LLM_MODEL).",
    )

    # Default local persistence directory for Chroma vector database.
    # If empty string is provided via environment, validator will normalize it.
    vector_db_path: str = Field(
        default="./chroma_data",
        description="Directory for Chroma persistence (VECTOR_DB_PATH). Use empty for in-memory.",
    )

    @field_validator("openai_api_key", mode="before")
    @classmethod
    def strip_api_key(cls, v: str | None) -> str:
        """
        Validate and normalize the OpenAI API key before model instantiation.

        - Ensures the key exists.
        - Ensures it is non-empty after stripping whitespace.
        - Strips leading/trailing whitespace to prevent subtle configuration bugs.

        Raises:
            ValueError if the key is missing or blank.
        """
        if v is None or (isinstance(v, str) and not v.strip()):
            raise ValueError("OPENAI_API_KEY must be set and non-empty")

        # Normalize whitespace to avoid authentication issues due to trailing spaces.
        return v.strip() if isinstance(v, str) else v

    @field_validator("vector_db_path", mode="before")
    @classmethod
    def normalize_vector_db_path(cls, v: str | None) -> str:
        """
        Normalize vector database path.

        Behavior:
        - If unset or blank, fall back to default "./chroma_data".
        - Strip whitespace to prevent accidental path misconfiguration.

        This ensures downstream Chroma initialization receives a clean path string.
        """
        if v is None or (isinstance(v, str) and not v.strip()):
            return "./chroma_data"

        return v.strip()

    def get_vector_db_path_resolved(self) -> Path:
        """
        Return `vector_db_path` as a fully resolved Path object.

        This is useful when:
        - Initializing Chroma persistence
        - Performing filesystem operations
        - Avoiding ambiguity with relative paths

        Returns:
            pathlib.Path: Absolute, resolved filesystem path.
        """
        return Path(self.vector_db_path).resolve()


@lru_cache
def get_settings() -> Settings:
    """
    Return a cached Settings instance.

    Why caching?
    - Ensures configuration is loaded only once.
    - Prevents repeated environment parsing.
    - Guarantees consistent settings across modules.

    Because of `@lru_cache`, repeated calls to `get_settings()`
    will return the same in-memory Settings instance.

    Usage:

        from backend.config import get_settings
        settings = get_settings()
    """
    return Settings()