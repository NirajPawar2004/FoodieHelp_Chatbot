"""
Document Loader for FoodieHelp AI
Extracts plain text and metadata from TXT, PDF, and DOCX files.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Dict, Any
import pypdf
import docx


class LoadedDocument:
    """Holds extracted text and associated metadata for a document segment."""
    def __init__(self, text: str, metadata: Dict[str, Any]):
        self.text = text.strip()
        self.metadata = metadata

    def __repr__(self) -> str:
        source = self.metadata.get("source", "Unknown")
        page = self.metadata.get("page", 1)
        return f"<LoadedDocument source='{source}' page={page} length={len(self.text)}>"


class DocumentLoader:
    """Extracts text and metadata across TXT, PDF, and DOCX formats."""

    @staticmethod
    def load_txt(file_path: Path) -> List[LoadedDocument]:
        """Load text from a .txt file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(file_path, "r", encoding="latin-1") as f:
                content = f.read()

        if not content.strip():
            return []

        metadata = {
            "source": file_path.name,
            "file_type": "txt",
            "file_path": str(file_path),
            "page": 1,
            "total_pages": 1,
        }
        return [LoadedDocument(text=content, metadata=metadata)]

    @staticmethod
    def load_pdf(file_path: Path) -> List[LoadedDocument]:
        """Load text from a .pdf file, capturing per-page metadata."""
        docs: List[LoadedDocument] = []
        reader = pypdf.PdfReader(str(file_path))
        total_pages = len(reader.pages)

        for page_idx, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            if page_text.strip():
                metadata = {
                    "source": file_path.name,
                    "file_type": "pdf",
                    "file_path": str(file_path),
                    "page": page_idx + 1,
                    "total_pages": total_pages,
                }
                docs.append(LoadedDocument(text=page_text, metadata=metadata))
        return docs

    @staticmethod
    def load_docx(file_path: Path) -> List[LoadedDocument]:
        """Load text from a .docx file."""
        doc = docx.Document(str(file_path))
        full_text: List[str] = []
        for para in doc.paragraphs:
            if para.text.strip():
                full_text.append(para.text.strip())

        joined_text = "\n\n".join(full_text)
        if not joined_text.strip():
            return []

        metadata = {
            "source": file_path.name,
            "file_type": "docx",
            "file_path": str(file_path),
            "page": 1,
            "total_pages": 1,
        }
        return [LoadedDocument(text=joined_text, metadata=metadata)]

    @classmethod
    def load_file(cls, file_path: Path) -> List[LoadedDocument]:
        """Dispatch document loading based on file extension."""
        path = Path(file_path)
        ext = path.suffix.lower()

        if ext == ".txt":
            return cls.load_txt(path)
        elif ext == ".pdf":
            return cls.load_pdf(path)
        elif ext == ".docx":
            return cls.load_docx(path)
        else:
            raise ValueError(f"Unsupported file format: {ext}. FoodieHelp supports .txt, .pdf, and .docx.")

    @classmethod
    def load_directory(cls, directory_path: Path) -> List[LoadedDocument]:
        """Load all supported documents from a directory."""
        dir_path = Path(directory_path)
        all_docs: List[LoadedDocument] = []

        if not dir_path.exists():
            return all_docs

        for item in sorted(dir_path.iterdir()):
            if item.is_file() and item.suffix.lower() in [".txt", ".pdf", ".docx"]:
                try:
                    all_docs.extend(cls.load_file(item))
                except Exception as e:
                    print(f"Warning: Failed to load {item.name}: {e}")

        return all_docs

