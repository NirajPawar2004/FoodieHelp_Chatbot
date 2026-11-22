"""
Vector Store for FoodieHelp AI
Manages sentence-transformer embeddings, persistent ChromaDB storage,
and ultra-lightweight precomputed semantic search for serverless (Vercel/Netlify).
"""

from __future__ import annotations

import os
import json
import math
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
except ImportError:
    chromadb = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

from google import genai
from config.settings import settings
from ingestion.document_loader import DocumentLoader, LoadedDocument
from ingestion.text_chunker import TextChunker, DocumentChunk


class VectorStore:
    """Hybrid Vector Store supporting ChromaDB and serverless precomputed vector indices."""

    _instance: Optional["VectorStore"] = None
    _embedder: Optional[Any] = None

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
        self.collection_name = collection_name or settings.CHROMA_COLLECTION_NAME
        self.model_name = model_name or settings.EMBEDDING_MODEL_NAME
        self._cached_index: Optional[List[Dict[str, Any]]] = None

        self.embedder = None
        self.client = None
        self.collection = None

        # Only initialize heavy local SentenceTransformer + ChromaDB if available and not in serverless mode
        if chromadb is not None and SentenceTransformer is not None and not getattr(settings, "IS_SERVERLESS", False):
            try:
                self.persist_dir.mkdir(parents=True, exist_ok=True)
                print(f"[VectorStore] Loading embedding model: {self.model_name}...")
                self.embedder = SentenceTransformer(self.model_name)
                self.client = chromadb.PersistentClient(path=str(self.persist_dir))
                self.collection = self.client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
                print(f"[VectorStore] ChromaDB initialized at {self.persist_dir} (Collection: {self.collection_name})")
            except Exception as e:
                print(f"[VectorStore] Note: Local ChromaDB init skipped ({e}). Using serverless precomputed index.")
        else:
            print("[VectorStore] Serverless/Lightweight mode active: Using precomputed vector index.")

        self._initialized = True

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Compute normalized vector embeddings for a list of strings."""
        if not texts:
            return []
        if self.embedder is not None:
            embeddings = self.embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            return embeddings.tolist()
        return []

    def _load_precomputed_index(self) -> List[Dict[str, Any]]:
        """Load precomputed policy chunks and embeddings from JSON file."""
        if self._cached_index is not None:
            return self._cached_index
        index_file = settings.PRECOMPUTED_INDEX_FILE
        if index_file.exists():
            try:
                with open(index_file, "r", encoding="utf-8") as f:
                    self._cached_index = json.load(f)
                    return self._cached_index
            except Exception as e:
                print(f"[VectorStore] Failed loading precomputed index: {e}")
        return []

    def _search_precomputed(
        self,
        query: str,
        top_k: int = 4,
        similarity_threshold: float = 0.35
    ) -> List[Dict[str, Any]]:
        """Fast cosine similarity search over precomputed embeddings."""
        chunks = self._load_precomputed_index()
        if not chunks:
            return []

        # Attempt semantic search using Gemini embeddings
        if settings.is_gemini_configured():
            try:
                client = genai.Client(api_key=settings.GEMINI_API_KEY)
                res = client.models.embed_content(model="gemini-embedding-001", contents=query)
                q_emb = res.embeddings[0].values
                norm_q = math.sqrt(sum(x * x for x in q_emb))

                scores = []
                for item in chunks:
                    c_emb = item.get("embedding", [])
                    if not c_emb:
                        continue
                    dot = sum(x * y for x, y in zip(q_emb, c_emb))
                    norm_c = math.sqrt(sum(y * y for y in c_emb))
                    sim = dot / (norm_q * norm_c) if norm_q and norm_c else 0.0
                    if sim >= similarity_threshold:
                        scores.append({
                            "chunk_id": item["chunk_id"],
                            "text": item["text"],
                            "metadata": item["metadata"],
                            "similarity": round(float(sim), 4),
                            "distance": round(float(1.0 - sim), 4)
                        })

                if scores:
                    scores.sort(key=lambda x: x["similarity"], reverse=True)
                    return scores[:top_k]
            except Exception as e:
                print(f"[VectorStore] Gemini embedding search fallback: {e}")

        # Offline / Lexical Token Overlap fallback
        q_words = set(re.findall(r"\w+", query.lower()))
        scores = []
        for item in chunks:
            text = item.get("text", "")
            c_words = set(re.findall(r"\w+", text.lower()))
            common = q_words.intersection(c_words)
            sim = len(common) / (math.sqrt(len(q_words)) * math.sqrt(len(c_words))) if q_words and c_words else 0.0
            if sim >= similarity_threshold:
                scores.append({
                    "chunk_id": item["chunk_id"],
                    "text": text,
                    "metadata": item.get("metadata", {}),
                    "similarity": round(float(sim), 4),
                    "distance": round(float(1.0 - sim), 4)
                })

        scores.sort(key=lambda x: x["similarity"], reverse=True)
        return scores[:top_k]

    def add_chunks(self, chunks: List[DocumentChunk]) -> int:
        """Add text chunks into ChromaDB if available."""
        if not chunks or self.collection is None:
            return 0

        existing_ids = set(self.collection.get()["ids"])
        new_chunks = [c for c in chunks if c.chunk_id not in existing_ids]

        if not new_chunks:
            return 0

        ids = [c.chunk_id for c in new_chunks]
        documents = [c.text for c in new_chunks]
        metadatas = [c.metadata for c in new_chunks]
        embeddings = self.embed_texts(documents)

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        return len(new_chunks)

    def is_document_indexed(self, source_name: str) -> bool:
        """Check if any chunks for this document already exist."""
        if self.collection is not None:
            try:
                results = self.collection.get(where={"source": source_name}, limit=1)
                return len(results["ids"]) > 0
            except Exception:
                pass

        chunks = self._load_precomputed_index()
        return any(c.get("metadata", {}).get("source") == source_name for c in chunks)

    def get_indexed_sources(self) -> List[str]:
        """Return list of distinct source document filenames indexed."""
        if self.collection is not None:
            try:
                data = self.collection.get(include=["metadatas"])
                sources = set()
                for meta in data.get("metadatas", []):
                    if meta and "source" in meta:
                        sources.add(meta["source"])
                if sources:
                    return sorted(list(sources))
            except Exception:
                pass

        chunks = self._load_precomputed_index()
        return sorted(list(set(c["metadata"]["source"] for c in chunks if "source" in c.get("metadata", {}))))

    def get_stats(self) -> Dict[str, Any]:
        """Return summary statistics of the vector store."""
        if self.collection is not None:
            try:
                count = self.collection.count()
                sources = self.get_indexed_sources()
                return {
                    "total_chunks": count,
                    "total_documents": len(sources),
                    "sources": sources,
                    "collection_name": self.collection_name,
                    "embedding_model": self.model_name
                }
            except Exception:
                pass

        chunks = self._load_precomputed_index()
        sources = self.get_indexed_sources()
        return {
            "total_chunks": len(chunks),
            "total_documents": len(sources),
            "sources": sources,
            "collection_name": "precomputed_vector_index",
            "embedding_model": "gemini-embedding-001"
        }

    def clear_collection(self) -> None:
        """Clear the collection for safe re-indexing."""
        if self.client is not None:
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

        # Use ChromaDB if available and populated
        if self.collection is not None and self.embedder is not None:
            try:
                if self.collection.count() > 0:
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
                        similarity = round(1.0 - float(distance), 4)
                        if similarity >= similarity_threshold:
                            matched_chunks.append({
                                "chunk_id": chunk_id,
                                "text": text,
                                "metadata": metadata,
                                "similarity": similarity,
                                "distance": round(float(distance), 4)
                            })

                    matched_chunks.sort(key=lambda x: x["similarity"], reverse=True)
                    if matched_chunks:
                        return matched_chunks
            except Exception as e:
                print(f"[VectorStore] Chroma query exception, using precomputed fallback: {e}")

        # Fallback to precomputed semantic index (Vercel/Netlify/Offline)
        return self._search_precomputed(query, top_k, similarity_threshold)


def index_directory(
    directory_path: Optional[Path] = None,
    force_reindex: bool = False
) -> Dict[str, Any]:
    """Helper to load, chunk, and index all documents from knowledge_base directory."""
    dir_path = Path(directory_path) if directory_path else settings.KNOWLEDGE_BASE_DIR
    vstore = VectorStore()

    if force_reindex and vstore.collection is not None:
        vstore.clear_collection()

    loader = DocumentLoader()
    chunker = TextChunker(chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP)

    docs = loader.load_directory(dir_path)
    chunks = chunker.chunk_documents(docs)
    added_count = vstore.add_chunks(chunks)

    stats = vstore.get_stats()
    stats["newly_added_chunks"] = added_count
    return stats
