"""
Settings and Configuration for FoodieHelp AI
Centralizes application settings, file paths, model configurations, and environment variables.
"""

from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

# Base directory of the repository (d:\Foodie)
BASE_DIR: Path = Path(__file__).resolve().parent.parent

# Load environment variables from .env file
load_dotenv(dotenv_path=BASE_DIR / ".env", override=False)


class Settings:
    """Application-wide settings with fallback defaults."""

    # Paths
    BASE_DIR: Path = BASE_DIR
    DATA_DIR: Path = BASE_DIR / "data"
    KNOWLEDGE_BASE_DIR: Path = BASE_DIR / os.getenv("KNOWLEDGE_BASE_DIR", "data/knowledge_base")
    CHROMA_PERSIST_DIR: Path = BASE_DIR / os.getenv("CHROMA_PERSIST_DIR", "data/chroma_db")
    DATABASE_PATH: Path = BASE_DIR / os.getenv("DATABASE_PATH", "data/foodiehelp.db")

    SAMPLE_ORDERS_FILE: Path = DATA_DIR / "sample_orders.json"
    SAMPLE_CUSTOMERS_FILE: Path = DATA_DIR / "sample_customers.json"

    # Vector store & Embeddings
    CHROMA_COLLECTION_NAME: str = "foodiehelp_policies"
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.35"))
    TOP_K_CHUNKS: int = int(os.getenv("TOP_K_CHUNKS", "4"))
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 80

    # LLM Settings (Configurable via GEMINI_MODEL env var)
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "").strip()
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3.5-flash").strip()
    GEMINI_TEMPERATURE: float = 0.2

    @classmethod
    def reload_env(cls) -> None:
        """Reload configuration from disk (.env)."""
        load_dotenv(dotenv_path=cls.BASE_DIR / ".env", override=True)
        cls.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
        cls.GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash").strip()
        cls.SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.35"))
        cls.TOP_K_CHUNKS = int(os.getenv("TOP_K_CHUNKS", "4"))

    @classmethod
    def is_gemini_configured(cls) -> bool:
        """Check if a non-placeholder Gemini API key is provided."""
        key = cls.GEMINI_API_KEY
        if not key:
            return False
        # Catch common placeholder text
        if key in ("your_gemini_api_key_here", "TODO", "YOUR_API_KEY"):
            return False
        return len(key) > 15


settings = Settings()

