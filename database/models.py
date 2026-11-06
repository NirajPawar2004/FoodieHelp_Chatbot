"""
Data Models for FoodieHelp AI
Defines structured classes and type hints for customers, orders, tickets, and results.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any


@dataclass
class Customer:
    """Represents a customer in the FoodieHelp ecosystem."""
    customer_id: str
    name: str
    email: str
    phone: str
    address: str
    created_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OrderItem:
    """Represents an individual item inside an order."""
    item_name: str
    quantity: int
    price: float
    subtotal: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Order:
    """Represents a food delivery order."""
    order_id: str
    customer_id: str
    customer_name: str
    restaurant_name: str
    ordered_items: List[Dict[str, Any]]
    order_amount: float
    order_date_time: str
    payment_method: str
    order_status: str  # Placed, Preparing, Out for Delivery, Delivered, Cancelled
    estimated_delivery_time: str
    delivery_address: str
    cancellation_reason: Optional[str] = None
    cancelled_at: Optional[str] = None

    def items_as_json(self) -> str:
        """Serialize ordered items to a JSON string for database storage."""
        return json.dumps(self.ordered_items)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SupportTicket:
    """Represents a customer support issue or escalated case."""
    ticket_id: str
    customer_id: str
    issue_category: str  # Missing Item, Wrong Item, Cancellation, Delivery, Payment, Quality, General
    description: str
    priority: str  # Low, Medium, High, Critical
    status: str  # Open, In Progress, Resolved, Escalated
    created_at: str
    order_id: Optional[str] = None
    updated_at: Optional[str] = None
    resolution_notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RefundEligibilityResult:
    """Outcome of business-rule refund evaluation."""
    eligible: bool
    refund_percentage: float
    refund_amount: float
    reason: str
    applicable_rule: str
    required_next_action: str
    order_id: Optional[str] = None
    item_name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CancellationResult:
    """Outcome of order cancellation attempt."""
    success: bool
    order_id: str
    previous_status: str
    current_status: str
    refund_percentage: float
    refund_amount: float
    message: str
    cancelled_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

