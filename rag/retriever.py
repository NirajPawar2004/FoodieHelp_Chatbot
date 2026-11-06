"""
Retriever Module for FoodieHelp AI
Fetches relevant policy passages from persistent ChromaDB and formats context for LLM generation.
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional
from config.settings import settings
from ingestion.vector_store import VectorStore


class PolicyRetriever:
    """Retrieves and formats policy knowledge for grounded RAG generation."""

    def __init__(self, vector_store: Optional[VectorStore] = None):
        self.vector_store = vector_store or VectorStore()

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Query vector store for passages semantically close to customer's inquiry.
        Filters by similarity threshold.
        """
        k = top_k if top_k is not None else settings.TOP_K_CHUNKS
        threshold = similarity_threshold if similarity_threshold is not None else settings.SIMILARITY_THRESHOLD

        return self.vector_store.search(
            query=query,
            top_k=k,
            similarity_threshold=threshold
        )

    def format_context(self, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """Format retrieved chunks into a clean context block with source document tags."""
        if not retrieved_chunks:
            return ""

        context_blocks = []
        for idx, chunk in enumerate(retrieved_chunks, start=1):
            source = chunk["metadata"].get("source", "Unknown Document")
            page = chunk["metadata"].get("page", 1)
            similarity = chunk.get("similarity", 0.0)

            header = f"[Source {idx}: {source} (Page {page}) - Match Score: {similarity:.2f}]"
            content = chunk["text"]
            context_blocks.append(f"{header}\n{content}")

        return "\n\n---\n\n".join(context_blocks)

