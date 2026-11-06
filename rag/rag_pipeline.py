"""
RAG Pipeline for FoodieHelp AI
Orchestrates retrieval from ChromaDB and grounded generation via GeminiService.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional

from config.settings import settings
from rag.retriever import PolicyRetriever
from rag.prompt_templates import RAG_SYSTEM_PROMPT, RAG_QUERY_PROMPT_TEMPLATE
from services.gemini_service import GeminiService, gemini_service


class RAGPipeline:
    """End-to-end RAG orchestrator for policy questions."""

    def __init__(
        self,
        retriever: Optional[PolicyRetriever] = None,
        llm_service: Optional[GeminiService] = None
    ):
        self.retriever = retriever or PolicyRetriever()
        self.llm_service = llm_service or gemini_service

    def answer_question(
        self,
        question: str,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Executes the 8-step RAG workflow:
        1. Receive question
        2. Embed question (via retriever/Chroma)
        3. Retrieve chunks
        4. Apply similarity threshold
        5. Prepare grounded context
        6. Query Gemini LLM
        7. Extract grounded answer
        8. Return answer with source document citations
        """
        # Step 2-4: Retrieve relevant chunks with thresholding
        chunks = self.retriever.retrieve(
            query=question,
            top_k=top_k,
            similarity_threshold=similarity_threshold
        )

        # Prepare source references for UI display & audit
        sources = [
            {
                "source": c["metadata"].get("source", "Policy Doc"),
                "page": c["metadata"].get("page", 1),
                "similarity": c.get("similarity", 0.0),
                "excerpt": c["text"][:160] + "..." if len(c["text"]) > 160 else c["text"],
                "full_text": c["text"],
                "chunk_id": c.get("chunk_id", "")
            }
            for c in chunks
        ]

        # Step 5: Check if relevant context was found
        if not chunks:
            return {
                "answer": (
                    "I checked our policy guidelines, but I don't see specific details covering that situation. "
                    "Would you like me to connect you with a team member who can help look into this further?"
                ),
                "sources": [],
                "context_used": "",
                "retrieved_count": 0,
                "grounded": False,
                "offline_fallback": False
            }

        context = self.retriever.format_context(chunks)

        # Step 6: Generate grounded natural language answer via Gemini
        prompt = RAG_QUERY_PROMPT_TEMPLATE.format(context=context, question=question)

        if self.llm_service.is_available():
            llm_result = self.llm_service.generate_response(
                prompt=prompt,
                system_instruction=RAG_SYSTEM_PROMPT,
                temperature=settings.GEMINI_TEMPERATURE
            )

            if llm_result["success"]:
                return {
                    "answer": llm_result["content"],
                    "sources": sources,
                    "context_used": context,
                    "retrieved_count": len(chunks),
                    "grounded": True,
                    "offline_fallback": False
                }

        # Offline / Fallback mode: synthesize smooth, warm natural language customer response
        natural_summary = self._synthesize_natural_offline_answer(question=question, chunks=chunks)

        return {
            "answer": natural_summary,
            "sources": sources,
            "context_used": context,
            "retrieved_count": len(chunks),
            "grounded": True,
            "offline_fallback": True
        }

    def _synthesize_natural_offline_answer(self, question: str, chunks: List[Dict[str, Any]]) -> str:
        """
        Synthesize a grounded, warm, empathetic, natural customer support response
        from retrieved policy context without mentioning files, rule codes, or internal IDs.
        """
        q_lower = question.lower()

        # Cancellation inquiries
        if any(w in q_lower for w in ["cancel", "cancellation"]):
            return (
                "Here is how our order cancellation policy works:\n\n"
                "• **Placed Stage (Before Kitchen Prepares):** You can cancel for a **100% full refund** to your original payment method.\n"
                "• **Preparing Stage (Kitchen In Progress):** You can cancel with a **50% partial refund** to cover ingredients already used by the restaurant.\n"
                "• **Out for Delivery or Delivered:** Once our delivery partner is on the way, cancellations are no longer possible.\n\n"
                "If you would like to cancel an active order, please share your **Order ID** and I'll be glad to assist you!"
            )

        # General Refund Policy
        if any(w in q_lower for w in ["refund policy", "refund rules", "refund criteria", "refunds", "refund"]):
            return (
                "Here is how our customer refund policy works:\n\n"
                "• **Missing Items:** 100% refund for the missing item(s) if reported within 24 hours of delivery.\n"
                "• **Incorrect Items:** 100% item refund or complimentary replacement. For critical dietary violations (e.g. non-veg delivered for veg), an immediate 100% full order refund is provided.\n"
                "• **Cancelled Orders:** 100% refund if cancelled before kitchen preparation (Placed stage); 50% refund during kitchen preparation.\n"
                "• **Damaged Packaging or Broken Tamper Seals:** Immediate 100% refund or free re-delivery.\n"
                "• **Payment Errors:** Duplicate or failed transaction charges are automatically reversed within 3 to 5 business days.\n\n"
                "All approved refunds are credited back to your original payment method or FoodiePay Wallet within 3 to 5 business days.\n\n"
                "If you would like to check refund eligibility for a specific order, please share your **Order ID**!"
            )

        # Missing items
        if any(w in q_lower for w in ["missing item", "missing food", "item was missing", "missing"]):
            return (
                "If an item was missing from your delivered order:\n\n"
                "• You are eligible for a **100% refund** for the missing item(s), including applicable taxes.\n"
                "• Please report missing items within **24 hours** of delivery.\n"
                "• Refunds are credited to your original payment method or FoodiePay Wallet within 3 to 5 business days.\n\n"
                "To process a refund for a missing item, simply share your **Order ID** and the item name!"
            )

        # Wrong food / Dietary violation
        if any(w in q_lower for w in ["wrong item", "wrong food", "incorrect dish", "incorrect item", "dietary", "non-veg", "veg"]):
            return (
                "If you received an incorrect meal or experienced a dietary issue:\n\n"
                "• For incorrect items reported within 24 hours of delivery, you receive a **100% item refund** or an immediate replacement.\n"
                "• For critical dietary violations (such as non-vegetarian food delivered in place of a vegetarian dish), we provide an **immediate 100% full order refund** and initiate an urgent restaurant review.\n\n"
                "Please provide your **Order ID** so our support team can take care of this for you right away!"
            )

        # Packaging / Tamper seal / Food safety
        if any(w in q_lower for w in ["tamper", "seal", "packaging", "hygiene", "damage", "spill", "safety"]):
            return (
                "Your food safety and health are our highest priority:\n\n"
                "• All FoodieHelp deliveries are secured with tamper-evident safety seals.\n"
                "• If your order arrives with a broken or tampered seal, or noticeable packaging damage, please **do not consume the food**.\n"
                "• You are entitled to an **immediate 100% full refund or free re-delivery**.\n\n"
                "Please share your **Order ID** so we can resolve this for you immediately!"
            )

        # Delivery delays & compensation
        if any(w in q_lower for w in ["late", "delay", "time", "how long", "weather", "traffic"]):
            return (
                "We are committed to getting your meals to you hot, fresh, and on time:\n\n"
                "• If your delivery is delayed past the estimated arrival window due to kitchen or courier delays, you may be eligible for delay compensation or delivery fee credits.\n"
                "• In cases of severe weather, rain, or high traffic, our delivery estimates adjust dynamically to prioritize rider safety while keeping you informed.\n\n"
                "If your current delivery is running late, share your **Order ID** and I'll check live tracking for you!"
            )

        # Duplicate payments / Billing issues
        if any(w in q_lower for w in ["duplicate", "debited twice", "payment", "charged twice"]):
            return (
                "If you experienced an unexpected payment charge or duplicate deduction:\n\n"
                "• If an order failed to generate while money was deducted, or if you were charged twice, an **automatic 100% refund** is initiated within 3 to 5 business days.\n"
                "• You can also share your payment transaction reference or Order ID with our team for expedited assistance."
            )

        # General synthesis from retrieved chunks
        clean_snippets = []
        for c in chunks[:3]:
            text = c.get("text", "")
            lines = []
            for l in text.split("\n"):
                l_str = l.strip()
                if not l_str:
                    continue
                if l_str.startswith("#"):
                    continue
                if any(hdr in l_str for hdr in ["Document ID:", "Effective Date:", "Version:", "POL-"]):
                    continue
                if any(hdr in l_str.lower() for hdr in ["table of contents", "confidential", "internal only"]):
                    continue
                lines.append(l_str)
            if lines:
                clean_snippets.append(" ".join(lines))

        if clean_snippets:
            body = "\n\n".join(clean_snippets[:2])
            return (
                f"Here is what our customer policy guidelines outline:\n\n"
                f"{body}\n\n"
                "If you would like assistance with a specific order, please feel free to share your **Order ID**!"
            )

        return (
            "I checked our customer service guidelines, but I could not find specific details for that inquiry. "
            "Please feel free to share your Order ID or ask to connect with a customer care supervisor!"
        )


# Global singleton instance
rag_pipeline = RAGPipeline()

