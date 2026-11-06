"""
Text Chunker for FoodieHelp AI
Splits long documents into overlapping passages while preserving source metadata.
"""

from __future__ import annotations

import re
from typing import List, Dict, Any
from ingestion.document_loader import LoadedDocument


class DocumentChunk:
    """Represents an individual text chunk ready for vector embedding."""
    def __init__(self, chunk_id: str, text: str, metadata: Dict[str, Any]):
        self.chunk_id = chunk_id
        self.text = text.strip()
        self.metadata = metadata

    def __repr__(self) -> str:
        return f"<DocumentChunk id='{self.chunk_id}' source='{self.metadata.get('source')}' len={len(self.text)}>"


class TextChunker:
    """Splits loaded documents into semantically coherent overlapping chunks."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 80):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str) -> List[str]:
        """Split a string into overlapping slices respecting paragraph & sentence boundaries."""
        text = text.strip()
        if not text:
            return []

        if len(text) <= self.chunk_size:
            return [text]

        chunks: List[str] = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + self.chunk_size, text_len)

            # If not at the end of the text, look for a clean break (newline or period)
            if end < text_len:
                # Look backwards for a double newline, newline, or sentence end
                break_point = -1
                for sep in ["\n\n", "\n", ". ", "? ", "! "]:
                    idx = text.rfind(sep, start + self.chunk_size // 2, end)
                    if idx != -1:
                        break_point = idx + len(sep)
                        break

                if break_point != -1:
                    end = break_point

            chunk_content = text[start:end].strip()
            if chunk_content:
                chunks.append(chunk_content)

            if end >= text_len:
                break

            # Advance start by (chunk length - overlap)
            step = max(end - start - self.chunk_overlap, 1)
            start += step

        return chunks

    def chunk_documents(self, documents: List[LoadedDocument]) -> List[DocumentChunk]:
        """Convert a list of LoadedDocuments into DocumentChunks with enriched metadata."""
        all_chunks: List[DocumentChunk] = []

        for doc in documents:
            raw_chunks = self.chunk_text(doc.text)
            source_name = doc.metadata.get("source", "document")
            page_num = doc.metadata.get("page", 1)

            # Sanitize source name for ID construction
            clean_source = re.sub(r"[^\w\-.]", "_", source_name)

            for idx, chunk_text in enumerate(raw_chunks):
                chunk_id = f"{clean_source}_p{page_num}_c{idx:03d}"
                metadata = {
                    **doc.metadata,
                    "chunk_id": chunk_id,
                    "chunk_index": idx,
                    "total_chunks_in_doc": len(raw_chunks),
                    "character_count": len(chunk_text)
                }
                all_chunks.append(DocumentChunk(chunk_id=chunk_id, text=chunk_text, metadata=metadata))

        return all_chunks

