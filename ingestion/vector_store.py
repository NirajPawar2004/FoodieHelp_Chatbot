"""
Vector Store for FoodieHelp AI
Manages sentence-transformer embeddings and persistent ChromaDB storage.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

from config.settings import settings
from ingestion.document_loader import DocumentLoader, LoadedDocument
from ingestion.text_chunker import TextChunker, DocumentChunk


class VectorStore:
    """Persistent ChromaDB vector database with HuggingFace SentenceTransformer embeddings."""

    _instance: Optional["VectorStore"] = None
    _embedder: Optional[SentenceTransformer] = None

    def __new__(cls, *args, **kwargs):
        """Singleton pattern so the embedding model is loaded only once in memory."""
        if cls._instance is None:
            cls._instance = super(VectorStore, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        persist_dir: Optional[Path] = None,
        collection_name: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        if getattr(self, "_initialized", False):
            return

        self.persist_dir = Path(persist_dir) if persist_dir else settings.CHROMA_PERSIST_DIR
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name or settings.CHROMA_COLLECTION_NAME
        self.model_name = model_name or settings.EMBEDDING_MODEL_NAME

        # Load embedding model
        print(f"[VectorStore] Loading embedding model: {self.model_name}...")
        self.embedder = SentenceTransformer(self.model_name)

        # Initialize persistent Chroma client
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self._initialized = True
        print(f"[VectorStore] ChromaDB initialized at {self.persist_dir} (Collection: {self.collection_name})")

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Compute normalized vector embeddings for a list of strings."""
        if not texts:
            return []
        embeddings = self.embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return embeddings.tolist()

    def add_chunks(self, chunks: List[DocumentChunk]) -> int:
        """Add text chunks into ChromaDB with deduplication checks."""
        if not chunks:
            return 0

        # Filter out chunks that might already exist
        existing_ids = set(self.collection.get()["ids"])
        new_chunks = [c for c in chunks if c.chunk_id not in existing_ids]

        if not new_chunks:
            return 0

        ids = [c.chunk_id for c in new_chunks]
        documents = [c.text for c in new_chunks]
        metadatas = [c.metadata for c in new_chunks]

        # Generate embeddings
        embeddings = self.embed_texts(documents)

        # Add to Chroma collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        return len(new_chunks)

    def is_document_indexed(self, source_name: str) -> bool:
        """Check if any chunks for this document already exist in ChromaDB."""
        results = self.collection.get(
            where={"source": source_name},
            limit=1
        )
        return len(results["ids"]) > 0

    def get_indexed_sources(self) -> List[str]:
        """Return list of distinct source document filenames indexed."""
        data = self.collection.get(include=["metadatas"])
        sources = set()
        for meta in data.get("metadatas", []):
            if meta and "source" in meta:
                sources.add(meta["source"])
        return sorted(list(sources))

    def get_stats(self) -> Dict[str, Any]:
        """Return summary statistics of the vector store."""
        count = self.collection.count()
        sources = self.get_indexed_sources()
        return {
            "total_chunks": count,
            "total_documents": len(sources),
            "sources": sources,
            "collection_name": self.collection_name,
            "embedding_model": self.model_name
        }

    def clear_collection(self) -> None:
        """Clear the collection for safe re-indexing."""
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"[VectorStore] Cleared collection {self.collection_name}.")

    def search(
        self,
        query: str,
        top_k: int = 4,
        similarity_threshold: float = 0.35
    ) -> List[Dict[str, Any]]:
        """
        Search for top-k chunks most semantically similar to query.
        Applies similarity threshold filter.
        """
        if not query.strip():
            return []

        if self.collection.count() == 0:
            return []

        query_embedding = self.embed_texts([query])

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"]
        )

        matched_chunks: List[Dict[str, Any]] = []

        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for chunk_id, text, metadata, distance in zip(ids, docs, metas, distances):
            # For cosine distance in Chroma: similarity = 1.0 - distance
            similarity = round(1.0 - float(distance), 4)

            # Apply similarity threshold
            if similarity >= similarity_threshold:
                matched_chunks.append({
                    "chunk_id": chunk_id,
                    "text": text,
                    "metadata": metadata,
                    "similarity": similarity,
                    "distance": round(float(distance), 4)
                })

        # Sort by similarity descending
        matched_chunks.sort(key=lambda x: x["similarity"], reverse=True)
        return matched_chunks


def index_directory(
    directory_path: Optional[Path] = None,
    force_reindex: bool = False
) -> Dict[str, Any]:
    """Helper to load, chunk, and index all documents from knowledge_base directory."""
    dir_path = Path(directory_path) if directory_path else settings.KNOWLEDGE_BASE_DIR
    vstore = VectorStore()

    if force_reindex:
        vstore.clear_collection()

    loader = DocumentLoader()
    chunker = TextChunker(chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP)

    docs = loader.load_directory(dir_path)
    chunks = chunker.chunk_documents(docs)
    added_count = vstore.add_chunks(chunks)

    stats = vstore.get_stats()
    stats["newly_added_chunks"] = added_count
    return stats

