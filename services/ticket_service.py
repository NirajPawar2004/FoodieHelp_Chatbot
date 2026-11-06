"""
Ticket Service for FoodieHelp AI
Manages support ticket generation, customer complaints, human escalations, and status tracking.
"""

from __future__ import annotations

import random
from typing import Optional, Dict, Any, List
from datetime import datetime

from database.db import (
    create_ticket as db_create_ticket,
    get_ticket_by_id,
    get_tickets_by_customer,
    get_all_tickets,
    update_ticket_status,
)
from database.models import SupportTicket


class TicketService:
    """Handles support ticket workflows, complaints, and human support handoffs."""

    @staticmethod
    def _generate_ticket_id() -> str:
        """Generate a human-readable unique support ticket ID."""
        return f"TIK-{random.randint(2000, 9999)}"

    @classmethod
    def create_support_ticket(
        cls,
        customer_id: str,
        issue_category: str,
        description: str,
        order_id: Optional[str] = None,
        priority: str = "Medium",
        status: str = "Open"
    ) -> SupportTicket:
        """Create and persist a new customer support ticket."""
        ticket_id = cls._generate_ticket_id()
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        clean_order_id = order_id.strip().upper() if order_id else None

        ticket = SupportTicket(
            ticket_id=ticket_id,
            customer_id=customer_id.strip(),
            order_id=clean_order_id,
            issue_category=issue_category.strip(),
            description=description.strip(),
            priority=priority.strip(),
            status=status.strip(),
            created_at=now_str,
            updated_at=None,
            resolution_notes=None
        )

        db_create_ticket(ticket)
        return ticket

    @classmethod
    def escalate_to_human(
        cls,
        customer_id: str,
        reason: str,
        order_id: Optional[str] = None,
        conversation_summary: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Human support escalation handoff:
        1. Formulates reason for escalation
        2. Creates support ticket with status='Escalated' and priority='High'
        3. Returns user-facing confirmation with Ticket ID
        """
        full_desc = f"Escalated Reason: {reason}"
        if conversation_summary:
            full_desc += f"\nContext: {conversation_summary}"

        ticket = cls.create_support_ticket(
            customer_id=customer_id,
            issue_category="Human Escalation",
            description=full_desc,
            order_id=order_id,
            priority="High",
            status="Escalated"
        )

        return {
            "escalated": True,
            "ticket_id": ticket.ticket_id,
            "status": ticket.status,
            "priority": ticket.priority,
            "message": (
                f"🤝 **Human Support Escalation Initiated**\n\n"
                f"I have transferred your case to our Senior Customer Support Team.\n\n"
                f"- **Ticket ID:** `{ticket.ticket_id}`\n"
                f"- **Status:** `{ticket.status}` (Priority: {ticket.priority})\n"
                f"- **Reason:** {reason}\n\n"
                "A human customer support executive will review your order details and contact you shortly. "
                "You can track this ticket anytime using your Ticket ID."
            )
        }

    @staticmethod
    def get_ticket(ticket_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve ticket details by ID."""
        ticket = get_ticket_by_id(ticket_id)
        return ticket.to_dict() if ticket else None

    @staticmethod
    def list_customer_tickets(customer_id: str) -> List[Dict[str, Any]]:
        """List all tickets logged by a customer."""
        tickets = get_tickets_by_customer(customer_id)
        return [t.to_dict() for t in tickets]

    @staticmethod
    def resolve_ticket(ticket_id: str, notes: str) -> bool:
        """Mark a ticket as Resolved with agent notes."""
        return update_ticket_status(ticket_id, "Resolved", resolution_notes=notes)


# Global singleton instance
ticket_service = TicketService()

