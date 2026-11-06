"""
Agent Orchestrator for FoodieHelp AI
Directs incoming user queries to specialized tools, RAG, business logic, or human handoff.
Validates all inputs, maintains conversation state, and enforces confirmation safety gates.
"""

from __future__ import annotations

import re
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

from agent.guardrails import Guardrails
from agent.tools import (
    tool_get_order_status,
    tool_get_customer_orders,
    tool_get_order_details,
    tool_check_refund_eligibility,
    tool_cancel_order,
    tool_create_ticket,
    tool_escalate_to_human,
    tool_search_policies,
)
from services.gemini_service import gemini_service
from services.order_service import order_service
from config.settings import settings


@dataclass
class AgentResponse:
    """Structured response returned by the orchestrator for the Streamlit UI."""
    content: str
    action_taken: str
    tool_used: Optional[str] = None
    tool_data: Optional[Dict[str, Any]] = None
    sources: List[Dict[str, Any]] = field(default_factory=list)
    pending_confirmation: Optional[Dict[str, Any]] = None
    suggested_replies: List[str] = field(default_factory=list)


class AgentOrchestrator:
    """Central decision-maker and tool caller for the FoodieHelp AI agent."""

    def __init__(self):
        self.guardrails = Guardrails()

    def process_message(
        self,
        user_message: str,
        customer_id: str,
        session_state: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """
        Main entry point for processing customer input.
        Coordinates intent classification, entity extraction, security checks, and tool execution.
        """
        clean_msg = self.guardrails.sanitize_text(user_message)
        if not clean_msg:
            return AgentResponse(
                content="Please type a message or choose one of the suggested prompts below.",
                action_taken="None",
                suggested_replies=["Track my order", "What is your refund policy?", "View my recent orders"]
            )

        # 1. Guardrail Security Check
        is_violation, warning_msg = self.guardrails.check_for_injection(clean_msg)
        if is_violation:
            return AgentResponse(
                content=warning_msg or "Input flagged by security guardrails.",
                action_taken="Security Alert",
                suggested_replies=["Track my order", "Refund policy"]
            )

        # 2. Check for Pending Confirmation (e.g., Order Cancellation confirmation)
        pending_action = session_state.get("pending_confirmation") if session_state else None
        if pending_action:
            resp = self._handle_pending_confirmation(clean_msg, customer_id, pending_action)
            if resp:
                return resp

        # 3. Entity Extraction
        extracted_order_id = self.guardrails.extract_order_id(clean_msg)
        lower_msg = clean_msg.lower()

        # 4. Intent Classification and Tool Routing

        # Intent A: Human Escalation
        if any(kw in lower_msg for kw in ["speak to human", "speak to a human", "talk to human", "talk to a human", "human agent", "real person", "escalate", "agent support", "customer care executive", "representative", "supervisor", "support agent", "human"]):
            return self._handle_human_escalation(clean_msg, customer_id, extracted_order_id)

        # Specific Order ID provided in message: handle action directly on this order
        if extracted_order_id:
            if any(kw in lower_msg for kw in ["cancel", "cancellation"]):
                return self._handle_cancellation_request(clean_msg, customer_id, extracted_order_id)
            elif any(kw in lower_msg for kw in ["refund", "missing", "wrong", "food quality", "stale", "spill", "cold", "duplicate", "debited"]):
                return self._handle_refund_request(clean_msg, customer_id, extracted_order_id)
            else:
                return self._handle_order_status_request(clean_msg, customer_id, extracted_order_id)

        # Intent B: Greetings & General Welcome
        greeting_words = ["hi", "hello", "hey", "help", "who are you", "what can you do", "good morning", "good evening", "good afternoon"]
        if lower_msg in greeting_words or any(lower_msg.startswith(g + " ") for g in ["hi", "hello", "hey"]):
            return self._handle_greeting(customer_id)

        # Intent C: Policy / FAQ Inquiries (Questions, Guidelines, Rules, Conditions)
        # When user asks a question (starts with question words, ends with '?', or asks about policies)
        is_policy_inquiry = (
            clean_msg.strip().endswith("?")
            or any(kw in lower_msg for kw in [
                "policy", "policies", "faq", "rules", "rule", "terms", "condition",
                "can i", "do you", "how do", "how does", "how can", "how long",
                "what is", "what if", "what happens", "is it possible",
                "tamper", "seal", "hygiene", "packaging", "weather", "compensation",
                "missing item refund", "order cancellation policy", "wrong food / dietary violation", "food safety & tamper seal"
            ])
            or (any(kw in lower_msg for kw in ["refund", "cancel"]) and any(kw in lower_msg for kw in ["when", "how", "why", "what", "can i", "allowed", "stage"]))
        )

        if is_policy_inquiry:
            return self._handle_policy_inquiry(clean_msg)

        # Intent D: Order Cancellation (without explicit order_id)
        if any(kw in lower_msg for kw in ["cancel order", "cancel my order", "want to cancel", "cancellation of", "cancel"]):
            return self._handle_cancellation_request(clean_msg, customer_id, None)

        # Intent E: Refund Request (without explicit order_id)
        if any(kw in lower_msg for kw in ["refund", "money back", "missing item", "wrong item", "missing food", "wrong food", "duplicate payment", "debited twice", "food quality"]):
            return self._handle_refund_request(clean_msg, customer_id, None)

        # Intent F: Order Status Lookup or Tracking (without explicit order_id)
        if any(kw in lower_msg for kw in ["status", "track", "where is", "when will", "delivery time", "my order", "orders"]):
            return self._handle_order_status_request(clean_msg, customer_id, None)

        # Intent G: Complaints or Tickets
        if any(kw in lower_msg for kw in ["complaint", "raise ticket", "create ticket", "lodge a complaint", "bad service", "terrible"]):
            return self._handle_complaint_request(clean_msg, customer_id, extracted_order_id)

        # Fallback: Attempt RAG search first; if no chunks match threshold, provide conversational help
        rag_attempt = tool_search_policies(clean_msg)
        if rag_attempt.get("grounded", False):
            return AgentResponse(
                content=rag_attempt["answer"],
                action_taken="Support Inquiry",
                tool_used="tool_search_policies",
                sources=[],
                suggested_replies=["Track my order", "What is your refund policy?", "Speak to a human"]
            )

        # General conversational response
        return self._handle_general_query(clean_msg)

    # --- Intent Handlers ---

    def _handle_pending_confirmation(
        self,
        clean_msg: str,
        customer_id: str,
        pending_action: Dict[str, Any]
    ) -> Optional[AgentResponse]:
        """Handle Yes/No answers to pending confirmation requests."""
        lower = clean_msg.lower().strip()
        action_type = pending_action.get("type")

        if action_type == "cancel_order":
            order_id = pending_action.get("order_id")

            # Check if user confirmed
            if any(confirm_word in lower for confirm_word in ["yes", "confirm", "proceed", "cancel it", "cancel order", "ok", "sure", "yep"]):
                result = tool_cancel_order(order_id=order_id, customer_id=customer_id, confirmed=True)
                return AgentResponse(
                    content=result["message"],
                    action_taken="Order Cancelled",
                    tool_used="tool_cancel_order",
                    tool_data=result,
                    pending_confirmation=None,
                    suggested_replies=["Check my recent orders", "What is your refund timeline?", "Help with another issue"]
                )
            elif any(decline_word in lower for decline_word in ["no", "keep", "abort", "don't cancel", "stop", "nevermind", "wait"]):
                return AgentResponse(
                    content=f"Understood! Order **{order_id}** has NOT been cancelled and remains active. Is there anything else I can assist you with?",
                    action_taken="Cancellation Aborted by Customer",
                    pending_confirmation=None,
                    suggested_replies=["Track my order", "View my recent orders", "Help"]
                )

        return None

    def _handle_order_status_request(
        self,
        clean_msg: str,
        customer_id: str,
        order_id: Optional[str]
    ) -> AgentResponse:
        """Handle order status inquiries, order listing, or tracking."""
        # Case 1: Specific order ID mentioned
        if order_id:
            status_data = tool_get_order_status(order_id=order_id, customer_id=customer_id)
            if not status_data["success"]:
                return AgentResponse(
                    content=status_data["message"],
                    action_taken="Order Lookup Failed",
                    tool_used="tool_get_order_status",
                    tool_data=status_data,
                    suggested_replies=["View my recent orders", "What is your delivery policy?"]
                )

            msg = (
                f"📦 **Order Status for #{status_data['order_id']}**\n\n"
                f"- **Restaurant:** {status_data['restaurant_name']}\n"
                f"- **Status:** `{status_data['order_status']}`\n"
                f"- **Total Amount:** ₹{status_data['order_amount']:.2f}\n"
                f"- **Estimated Delivery:** {status_data['estimated_delivery_time']}\n"
                f"- **Order Placed At:** {status_data['order_date_time']}\n\n"
            )

            if status_data["order_status"] in ["Placed", "Preparing"]:
                msg += "💡 *Tip: This order is currently in the kitchen and can still be modified or cancelled if needed.*"

            return AgentResponse(
                content=msg,
                action_taken="Order Status Lookup",
                tool_used="tool_get_order_status",
                tool_data=status_data,
                suggested_replies=[f"Cancel order {order_id}", "Check refund policy", "View my recent orders"]
            )

        # Case 2: User wants to see all their orders
        cust_orders = tool_get_customer_orders(customer_id)
        orders = cust_orders.get("orders", [])
        if not orders:
            return AgentResponse(
                content="You currently have no recorded food orders under this account.",
                action_taken="Customer Orders Lookup",
                tool_used="tool_get_customer_orders",
                suggested_replies=["What can you do?", "What is your delivery policy?"]
            )

        latest_orders = orders[:4]
        lines = [f"Here are your latest orders ({len(orders)} total on file):\n"]
        for o in latest_orders:
            lines.append(
                f"- **{o['order_id']}**: {o['restaurant_name']} — "
                f"**₹{o['order_amount']:.2f}** (`{o['order_status']}`) [{o['order_date_time'][:16]}]"
            )
        lines.append("\nYou can ask me to track or check details for any of these Order IDs!")

        return AgentResponse(
            content="\n".join(lines),
            action_taken="Customer Orders Listing",
            tool_used="tool_get_customer_orders",
            tool_data={"count": len(orders)},
            suggested_replies=[f"Track {latest_orders[0]['order_id']}", f"Cancel {latest_orders[0]['order_id']}", "Refund policy"]
        )

    def _handle_cancellation_request(
        self,
        clean_msg: str,
        customer_id: str,
        order_id: Optional[str]
    ) -> AgentResponse:
        """Handle order cancellation requests with policy validation and confirmation."""
        # If no order ID was provided, ask or pick the most recent eligible order
        if not order_id:
            cust_orders = order_service.get_customer_orders(customer_id)
            cancellable = [
                o for o in cust_orders
                if (o.get("order_status") if isinstance(o, dict) else o.order_status) in ["Placed", "Preparing"]
            ]
            if cancellable:
                target = cancellable[0]
                order_id = target["order_id"] if isinstance(target, dict) else target.order_id
            elif cust_orders:
                first = cust_orders[0]
                first_id = first["order_id"] if isinstance(first, dict) else first.order_id
                first_status = first["order_status"] if isinstance(first, dict) else first.order_status
                return AgentResponse(
                    content=(
                        f"Which order would you like to cancel? Your most recent order is **{first_id}** "
                        f"(Status: `{first_status}`). Please specify the Order ID."
                    ),
                    action_taken="Order Identification Needed",
                    suggested_replies=[f"Cancel {first_id}"]
                )
            else:
                return AgentResponse(
                    content="You do not have any active orders to cancel.",
                    action_taken="Cancellation Check",
                    suggested_replies=["View my recent orders", "Help"]
                )

        # Attempt pre-cancellation check (confirmed=False)
        result = tool_cancel_order(order_id=order_id, customer_id=customer_id, confirmed=False)

        if result.get("requires_confirmation"):
            return AgentResponse(
                content=result["message"],
                action_taken="Cancellation Confirmation Required",
                tool_used="tool_cancel_order",
                tool_data=result,
                pending_confirmation={
                    "type": "cancel_order",
                    "order_id": order_id
                },
                suggested_replies=["Confirm Cancel", "No, keep order"]
            )

        return AgentResponse(
            content=result["message"],
            action_taken="Cancellation Policy Check",
            tool_used="tool_cancel_order",
            tool_data=result,
            suggested_replies=["Check my recent orders", "Refund policy", "Speak to a human"]
        )

    def _handle_refund_request(
        self,
        clean_msg: str,
        customer_id: str,
        order_id: Optional[str]
    ) -> AgentResponse:
        """Handle refund eligibility evaluation using business rules and RAG citation."""
        lower = clean_msg.lower()

        # Classify specific refund issue type
        if "missing" in lower:
            issue_type = "missing_item"
        elif "wrong" in lower or "incorrect" in lower:
            issue_type = "wrong_item"
        elif "cancel" in lower:
            issue_type = "cancellation"
        elif "duplicate" in lower or "debited twice" in lower:
            issue_type = "duplicate_payment"
        elif "quality" in lower or "stale" in lower or "spill" in lower:
            issue_type = "food_quality"
        else:
            issue_type = "general_refund"

        # If no order ID was provided, ask the customer politely for their Order ID
        if not order_id:
            cust_orders = order_service.get_customer_orders(customer_id)
            if cust_orders:
                recent_ids = [o.get("order_id") if isinstance(o, dict) else o.order_id for o in cust_orders[:3]]
                id_list_str = ", ".join(f"`{oid}`" for oid in recent_ids)
                return AgentResponse(
                    content=(
                        f"I'd be happy to help you with your refund! Could you please share your **Order ID**? "
                        f"(Your recent orders on file include: {id_list_str}). "
                        f"Please also let me know what went wrong with the meal so I can assist you right away."
                    ),
                    action_taken="Refund Assistance - Order ID Needed",
                    suggested_replies=[f"Refund for {recent_ids[0]}", "What is your refund policy?", "Speak to a human"]
                )
            else:
                return AgentResponse(
                    content="To check refund eligibility, please specify your **Order ID** (for example, `ORD1001`).",
                    action_taken="Refund Assistance - Order ID Needed",
                    suggested_replies=["View my recent orders", "What is your refund policy?"]
                )

        res = tool_check_refund_eligibility(
            order_id=order_id,
            customer_id=customer_id,
            issue_type=issue_type
        )

        eval_res = res.get("evaluation", {})
        eligible = eval_res.get("eligible", False)
        refund_amt = eval_res.get("refund_amount", 0.0)
        reason = eval_res.get("reason", "")
        # Clean any internal code references from reason
        clean_reason = reason.replace("POL-", "")

        if eligible:
            content = (
                f"Good news! Your order **{order_id}** is eligible for a refund of **₹{refund_amt:.2f}** ({eval_res.get('refund_percentage', 0):.0f}%).\n\n"
                f"**Details:** {clean_reason}\n\n"
                "We have initiated this refund to your original payment method. The credit typically reflects within 3 to 5 business days."
            )
        else:
            content = (
                f"I checked order **{order_id}** for you, but unfortunately it isn't eligible for an automated refund at this time.\n\n"
                f"**Details:** {clean_reason}\n\n"
                "If you feel this is in error or have additional photos or details to share, please let me know and I will gladly connect you with a customer support supervisor to review your request."
            )

        return AgentResponse(
            content=content,
            action_taken="Refund Inquiry",
            tool_used="tool_check_refund_eligibility",
            tool_data=res,
            sources=[],
            suggested_replies=[f"Create support ticket for {order_id}", "Speak to a human", "Check order status"]
        )

    def _handle_complaint_request(
        self,
        clean_msg: str,
        customer_id: str,
        order_id: Optional[str]
    ) -> AgentResponse:
        """Handle customer complaints by logging a formal SupportTicket."""
        category = "Customer Complaint"
        if "cold" in clean_msg.lower() or "quality" in clean_msg.lower():
            category = "Food Quality"
        elif "late" in clean_msg.lower() or "delay" in clean_msg.lower():
            category = "Late Delivery"
        elif "rider" in clean_msg.lower() or "driver" in clean_msg.lower():
            category = "Delivery Partner Feedback"

        ticket_res = tool_create_ticket(
            customer_id=customer_id,
            issue_category=category,
            description=clean_msg,
            order_id=order_id,
            priority="Medium"
        )

        return AgentResponse(
            content=ticket_res["message"],
            action_taken="Support Ticket Creation",
            tool_used="tool_create_ticket",
            tool_data=ticket_res,
            suggested_replies=["Speak to a human", "Check my recent orders"]
        )

    def _handle_human_escalation(
        self,
        clean_msg: str,
        customer_id: str,
        order_id: Optional[str]
    ) -> AgentResponse:
        """Handle human support escalation handoff."""
        escalation_res = tool_escalate_to_human(
            customer_id=customer_id,
            reason=clean_msg,
            order_id=order_id
        )

        return AgentResponse(
            content=escalation_res["message"],
            action_taken="Human Support Escalation",
            tool_used="tool_escalate_to_human",
            tool_data=escalation_res,
            suggested_replies=["View my recent orders", "What is your refund policy?"]
        )

    def _handle_policy_inquiry(self, clean_msg: str) -> AgentResponse:
        """Answer policy questions via the grounded RAG pipeline, or fall back to general AI for external queries."""
        rag_res = tool_search_policies(clean_msg)

        # If policy chunks were retrieved with high similarity, return grounded company response
        if rag_res.get("grounded", False):
            action_name = "Refund Policy Inquiry" if "refund" in clean_msg.lower() else "Support Inquiry"
            return AgentResponse(
                content=rag_res["answer"],
                action_taken=action_name,
                tool_used="tool_search_policies",
                sources=[],
                suggested_replies=["Track my order", "What is your refund policy?", "Can I cancel my order?"]
            )

        # If question is not covered by internal food policies (e.g. math, science, general trivia),
        # delegate to general AI handler to answer accurately and helpfully
        return self._handle_general_query(clean_msg)

    def _handle_greeting(self, customer_id: str) -> AgentResponse:
        """Provide a warm, natural customer welcome."""
        content = (
            "Hello! Welcome to **FoodieHelp Customer Support** 👋\n\n"
            "I'm here to help you exclusively with your food orders, partner restaurants, refunds, and delivery policies! You can ask me to:\n"
            "• **Track an Order:** Check live kitchen preparation or delivery status.\n"
            "• **Restaurant & Menu Inquiries:** Explore partner restaurants and meal details.\n"
            "• **Cancel an Order:** Inquire about cancellation options and refund amounts.\n"
            "• **Report Issues & Refunds:** Get help with missing items, wrong dishes, or food quality.\n"
            "• **Delivery & Safety Policies:** Learn about packaging, tamper seals, delivery guarantees, or refund rules.\n"
            "• **Talk to an Agent:** Connect with a human customer support specialist.\n\n"
            "How can I assist you with your food order today?"
        )
        return AgentResponse(
            content=content,
            action_taken="Welcome",
            sources=[],
            suggested_replies=["Track my order", "What is your refund policy?", "Can I cancel an order in Preparing stage?"]
        )

    def _handle_general_query(self, clean_msg: str) -> AgentResponse:
        """
        Handle inquiries while strictly enforcing domain boundaries:
        Only answers Food, Restaurants, Orders, Refunds, Cancellations, and Company Policies.
        Politely declines external, non-food topics (like metals, general math, science, politics).
        """
        if gemini_service.is_available():
            system_instruction = (
                "You are FoodieHelp AI Assistant, dedicated EXCLUSIVELY to food delivery customer support, "
                "partner restaurants, food items, orders, refunds, cancellations, and company delivery/safety policies.\n\n"
                "CRITICAL SCOPE RESTRICTION:\n"
                "You MUST ONLY answer questions related to food, meals, partner restaurants, order tracking, "
                "refund eligibility, order cancellations, delivery terms, packaging/tamper seal standards, and support policies.\n"
                "If the user asks about ANY external, unrelated topic (such as metals, science, mathematics, history, politics, "
                "general trivia, coding, or non-food subjects like 'What is metals?'):\n"
                "You MUST politely and warmly decline, explaining that you only assist with FoodieHelp food orders, "
                "restaurants, refunds, cancellations, and delivery policies.\n"
                "Do NOT answer the external question. Guide the user back to how you can assist them with food, restaurants, refunds, or policies.\n"
                "Never mention internal document filenames, chunk IDs, or policy rule codes."
            )
            prompt = f"User asked: \"{clean_msg}\""
            res = gemini_service.generate_response(
                prompt=prompt,
                system_instruction=system_instruction,
                temperature=0.2
            )
            if res["success"]:
                return AgentResponse(
                    content=res["content"],
                    action_taken="Customer Support Inquiry",
                    sources=[],
                    suggested_replies=["Track my order", "What is your refund policy?", "View my recent orders"]
                )

        # Offline fallback for out-of-scope or general queries
        return AgentResponse(
            content=(
                "I am **FoodieHelp AI Assistant**, dedicated exclusively to assisting you with **food orders**, "
                "**partner restaurants**, **refunds**, **order cancellations**, and **company policies** (such as delivery, packaging, and safety).\n\n"
                "I cannot answer external or non-food-related questions. "
                "Please let me know how I can help you with your food order, restaurant, or delivery today!"
            ),
            action_taken="Support Inquiry",
            sources=[],
            suggested_replies=["Track my order", "What is your refund policy?", "View my recent orders"]
        )


# Global singleton instance
agent_orchestrator = AgentOrchestrator()

