"""
Unit Tests for FoodieHelp AI Document Ingestion and RAG Pipeline (Modules B & C)
Tests document loaders (TXT, PDF, DOCX), text chunking, ChromaDB retrieval, and grounded RAG.
Does not require a paid Gemini API call (tests offline mode and mocked generation).
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock

from ingestion.document_loader import DocumentLoader, LoadedDocument
from ingestion.text_chunker import TextChunker, DocumentChunk
from ingestion.vector_store import VectorStore
from rag.retriever import PolicyRetriever
from rag.rag_pipeline import RAGPipeline
from services.gemini_service import GeminiService
from config.settings import settings


def test_document_loader_txt():
    """Verify loading and metadata extraction from a text document."""
    txt_file = settings.KNOWLEDGE_BASE_DIR / "cancellation_policy.txt"
    docs = DocumentLoader.load_file(txt_file)
    assert len(docs) == 1
    assert "FOODIEHELP ORDER CANCELLATION POLICY" in docs[0].text
    assert docs[0].metadata["source"] == "cancellation_policy.txt"
    assert docs[0].metadata["file_type"] == "txt"


def test_document_loader_docx():
    """Verify loading DOCX policy document."""
    docx_file = settings.KNOWLEDGE_BASE_DIR / "packaging_and_hygiene_standards.docx"
    if docx_file.exists():
        docs = DocumentLoader.load_file(docx_file)
        assert len(docs) >= 1
        assert "Tamper-Evident" in docs[0].text
        assert docs[0].metadata["file_type"] == "docx"


def test_document_loader_pdf():
    """Verify loading PDF policy document."""
    pdf_file = settings.KNOWLEDGE_BASE_DIR / "rider_safety_and_delivery_terms.pdf"
    if pdf_file.exists():
        docs = DocumentLoader.load_file(pdf_file)
        assert len(docs) >= 1
        assert docs[0].metadata["file_type"] == "pdf"
        assert docs[0].metadata["page"] == 1


def test_text_chunker():
    """Verify text splitting, chunk overlap, and metadata propagation."""
    chunker = TextChunker(chunk_size=120, chunk_overlap=30)
    sample_text = (
        "FoodieHelp ensures high hygiene. All packages must have tamper seals. "
        "If a seal is broken, do not accept the package. Report to customer support "
        "within 24 hours for a full refund or immediate replacement."
    )
    doc = LoadedDocument(text=sample_text, metadata={"source": "test_policy.txt", "page": 1})
    chunks = chunker.chunk_documents([doc])

    assert len(chunks) >= 2
    for c in chunks:
        assert len(c.text) <= 150
        assert c.metadata["source"] == "test_policy.txt"
        assert c.chunk_id.startswith("test_policy")


def test_vector_store_search():
    """Verify semantic search returns relevant policy chunks with similarity scores."""
    vstore = VectorStore()
    results = vstore.search(
        query="What happens if the food package tamper seal is broken?",
        top_k=3,
        similarity_threshold=0.25
    )

    assert len(results) > 0
    top_result = results[0]
    assert "similarity" in top_result
    assert top_result["similarity"] >= 0.25
    assert "metadata" in top_result
    assert "source" in top_result["metadata"]


def test_rag_pipeline_offline_fallback():
    """Verify that RAG pipeline synthesizes grounded passages cleanly when Gemini is offline."""
    mock_llm = MagicMock(spec=GeminiService)
    mock_llm.is_available.return_value = False

    retriever = PolicyRetriever()
    pipeline = RAGPipeline(retriever=retriever, llm_service=mock_llm)

    res = pipeline.answer_question("Can I cancel an order while it is Preparing?")
    assert res["grounded"] is True
    assert res["retrieved_count"] > 0
    assert len(res["sources"]) > 0
    assert "cancellation_policy.txt" in [s["source"] for s in res["sources"]]
    assert "cancel" in res["answer"].lower()


def test_rag_pipeline_missing_context_no_hallucination():
    """Verify that questions completely outside the policy domain return polite missing-context notice."""
    retriever = PolicyRetriever()
    pipeline = RAGPipeline(retriever=retriever)

    # Completely alien query that should not exceed similarity threshold
    res = pipeline.answer_question(
        question="What is the quantum mechanics equation for dark matter subatomic particles?",
        similarity_threshold=0.75
    )

    assert res["grounded"] is False
    assert res["retrieved_count"] == 0
    assert "don't see specific details" in res["answer"].lower() or "could not find" in res["answer"].lower()

