"""
Agent Tools for FoodieHelp AI
Defines callable Python tools with strict argument validation for the agent orchestrator.
"""

from __future__ import annotations

import re
from typing import Dict, Any, Optional, List

from services.order_service import order_service
from services.refund_service import refund_service
from services.ticket_service import ticket_service
from rag.rag_pipeline import rag_pipeline


def validate_order_id_format(order_id: str) -> bool:
    """Validate that order_id resembles 'ORD1001' or similar alphanumeric code."""
    if not order_id:
        return False
    return bool(re.match(r"^[A-Za-z0-9\-_]{3,20}$", order_id.strip()))


def validate_customer_id_format(customer_id: str) -> bool:
    """Validate that customer_id resembles 'CUST001' or similar."""
    if not customer_id:
        return False
    return bool(re.match(r"^[A-Za-z0-9\-_]{3,20}$", customer_id.strip()))


# --- Tool Implementations ---

def tool_get_order_status(order_id: str, customer_id: str) -> Dict[str, Any]:
    """Retrieve the current live status and delivery estimate for a specific customer order."""
    if not validate_order_id_format(order_id):
        return {
            "success": False,
            "error": "INVALID_ARGUMENT",
            "message": f"'{order_id}' is not a valid Order ID format (e.g. ORD1001)."
        }
    if not validate_customer_id_format(customer_id):
        return {
            "success": False,
            "error": "INVALID_ARGUMENT",
            "message": "Invalid customer ID provided."
        }
    return order_service.get_order_status(order_id=order_id, customer_id=customer_id)


def tool_get_customer_orders(customer_id: str) -> Dict[str, Any]:
    """List all recent orders belonging to the customer."""
    if not validate_customer_id_format(customer_id):
        return {"success": False, "orders": [], "message": "Invalid customer ID."}
    orders = order_service.get_customer_orders(customer_id=customer_id)
    return {
        "success": True,
        "count": len(orders),
        "orders": orders,
        "message": f"Found {len(orders)} orders for customer {customer_id}."
    }


def tool_get_order_details(order_id: str, customer_id: str) -> Dict[str, Any]:
    """Retrieve full itemized bill and details of an order."""
    if not validate_order_id_format(order_id):
        return {"success": False, "error": "INVALID_ORDER_ID"}
    return order_service.get_order_details(order_id=order_id, customer_id=customer_id)


def tool_check_refund_eligibility(
    order_id: str,
    customer_id: str,
    issue_type: str,
    item_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Evaluate refund eligibility via deterministic Python business rules.
    RAG policy snippets are also retrieved for complete transparency.
    """
    if not validate_order_id_format(order_id):
        return {"success": False, "error": "INVALID_ORDER_ID", "message": "Please supply a valid order ID."}

    # Evaluate business rule
    result = refund_service.evaluate_refund(
        order_id=order_id,
        customer_id=customer_id,
        issue_type=issue_type,
        item_name=item_name
    )

    # Cross-reference with RAG policy for documentation
    rag_res = rag_pipeline.answer_question(f"What is the refund policy for {issue_type}?")

    return {
        "success": True,
        "evaluation": result.to_dict(),
        "policy_context": rag_res.get("context_used", ""),
        "policy_sources": rag_res.get("sources", [])
    }


def tool_cancel_order(
    order_id: str,
    customer_id: str,
    confirmed: bool = False,
    reason: Optional[str] = None
) -> Dict[str, Any]:
    """
    Simulate cancellation under business rules.
    Requires explicit customer confirmation if confirmed=False.
    """
    if not validate_order_id_format(order_id):
        return {"success": False, "error": "INVALID_ORDER_ID", "message": "Invalid order ID format."}

    res = order_service.process_cancellation(
        order_id=order_id,
        customer_id=customer_id,
        confirmed=confirmed,
        reason=reason
    )
    return {
        "success": res.success,
        "cancellation": res.to_dict(),
        "requires_confirmation": not confirmed and not res.success and res.refund_percentage > 0,
        "message": res.message
    }


def tool_create_ticket(
    customer_id: str,
    issue_category: str,
    description: str,
    order_id: Optional[str] = None,
    priority: str = "Medium"
) -> Dict[str, Any]:
    """Log a formal customer support complaint or request."""
    if not description or len(description.strip()) < 5:
        return {"success": False, "message": "Please provide a descriptive explanation of the issue."}

    ticket = ticket_service.create_support_ticket(
        customer_id=customer_id,
        issue_category=issue_category,
        description=description,
        order_id=order_id,
        priority=priority,
        status="Open"
    )
    return {
        "success": True,
        "ticket": ticket.to_dict(),
        "message": (
            f"🎫 **Support Ticket Created Successfully**\n\n"
            f"- **Ticket ID:** `{ticket.ticket_id}`\n"
            f"- **Category:** {ticket.issue_category}\n"
            f"- **Priority:** {ticket.priority}\n"
            f"- **Status:** {ticket.status}\n\n"
            "Our support team has been notified and will review your request."
        )
    }


def tool_escalate_to_human(
    customer_id: str,
    reason: str,
    order_id: Optional[str] = None
) -> Dict[str, Any]:
    """Escalate complicated or unresolved issue to human support executive."""
    return ticket_service.escalate_to_human(
        customer_id=customer_id,
        reason=reason,
        order_id=order_id
    )


def tool_search_policies(query: str) -> Dict[str, Any]:
    """Retrieve relevant grounded knowledge chunks and answer from policy docs."""
    return rag_pipeline.answer_question(question=query)

