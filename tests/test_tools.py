"""
Unit Tests for FoodieHelp AI Database, Order Management, and Tools
Tests order lookups, security boundaries, customer ownership, and cancellation flows.
"""

import pytest
import sqlite3
from pathlib import Path

from config.settings import settings
from database.db import (
    init_db,
    get_customer_by_id,
    get_all_customers,
    get_order_by_id,
    get_orders_by_customer,
    update_order_status,
    create_ticket,
    get_ticket_by_id,
    get_admin_metrics,
)
from database.models import Customer, Order, SupportTicket
from database.seed import seed_database
from services.order_service import order_service
from services.ticket_service import ticket_service
from agent.tools import (
    tool_get_order_status,
    tool_cancel_order,
    tool_create_ticket,
    validate_order_id_format,
)


@pytest.fixture(scope="module", autouse=True)
def setup_test_db():
    """Ensure database is seeded with fresh sample data before running tests."""
    seed_database(force_reseed=True)


def test_database_initialization_and_seeding():
    """Verify that 20 customers and 55 orders are loaded properly in SQLite."""
    customers = get_all_customers()
    assert len(customers) >= 20

    cust = get_customer_by_id("CUST001")
    assert cust is not None
    assert cust.name == "Aarav Sharma"

    metrics = get_admin_metrics()
    assert metrics["total_customers"] >= 20
    assert metrics["total_orders"] >= 50
    assert metrics["total_tickets"] >= 5


def test_order_lookup_success():
    """Test looking up an order with correct customer ownership."""
    order = get_order_by_id("ORD1001")
    assert order is not None

    # Customer CUST001 owns ORD1001
    status_resp = order_service.get_order_status(order_id="ORD1001", customer_id="CUST001")
    assert status_resp["success"] is True
    assert status_resp["order_id"] == "ORD1001"
    assert "restaurant_name" in status_resp


def test_invalid_order_id_lookup():
    """Test looking up a non-existent order ID."""
    status_resp = order_service.get_order_status(order_id="ORD99999", customer_id="CUST001")
    assert status_resp["success"] is False
    assert status_resp["error"] == "ORDER_NOT_FOUND"


def test_customer_ownership_security_boundary():
    """Verify that a customer cannot view another customer's order."""
    # ORD1001 belongs to CUST001, so CUST002 should be rejected
    status_resp = order_service.get_order_status(order_id="ORD1001", customer_id="CUST002")
    assert status_resp["success"] is False
    assert status_resp["error"] == "UNAUTHORIZED_ACCESS"


def test_order_id_format_validator():
    """Test regex order ID format sanitizer."""
    assert validate_order_id_format("ORD1001") is True
    assert validate_order_id_format("ord-204") is True
    assert validate_order_id_format("SELECT * FROM orders") is False
    assert validate_order_id_format("") is False


def test_cancellation_confirmation_gate():
    """Verify that cancellation requires explicit customer confirmation before state change."""
    # Order ORD1001 is in 'Placed' stage
    # First attempt without confirmation: should return pre-cancellation warning
    pre_check = order_service.process_cancellation(
        order_id="ORD1001",
        customer_id="CUST001",
        confirmed=False
    )
    assert pre_check.success is False
    assert pre_check.refund_percentage == 100.0
    assert "Confirmation Required" in pre_check.message

    # Verify status in database remains 'Placed'
    order_before = get_order_by_id("ORD1001")
    assert order_before.order_status == "Placed"

    # Second attempt with confirmation: should succeed and update DB
    confirmed_res = order_service.process_cancellation(
        order_id="ORD1001",
        customer_id="CUST001",
        confirmed=True
    )
    assert confirmed_res.success is True
    assert confirmed_res.current_status == "Cancelled"

    # Verify order is now Cancelled in DB
    order_after = get_order_by_id("ORD1001")
    assert order_after.order_status == "Cancelled"
    assert order_after.cancelled_at is not None


def test_cancellation_rejected_for_out_for_delivery():
    """Verify that in-transit orders cannot be cancelled."""
    # ORD1004 is 'Out for Delivery' for CUST004
    res = order_service.process_cancellation(
        order_id="ORD1004",
        customer_id="CUST004",
        confirmed=True
    )
    assert res.success is False
    assert "cannot be cancelled because it is already **Out for Delivery**" in res.message


def test_ticket_creation_and_escalation():
    """Verify ticket logging and human escalation handoff."""
    ticket = ticket_service.create_support_ticket(
        customer_id="CUST003",
        issue_category="App Bug",
        description="Delivery tracking map was not refreshing.",
        priority="Low"
    )
    assert ticket.ticket_id.startswith("TIK-")
    assert ticket.status == "Open"

    fetched = get_ticket_by_id(ticket.ticket_id)
    assert fetched is not None
    assert fetched.description == "Delivery tracking map was not refreshing."

    # Test human escalation
    esc = ticket_service.escalate_to_human(
        customer_id="CUST003",
        reason="Repeated delivery partner mismatch",
        order_id="ORD1003"
    )
    assert esc["escalated"] is True
    assert esc["status"] == "Escalated"
    assert esc["priority"] == "High"


def test_orchestrator_requests_without_explicit_order_id():
    """Verify orchestrator handles generic refund and cancel queries gracefully without order_id."""
    from agent.orchestrator import agent_orchestrator

    # 1. Refund query without explicit order_id
    resp_refund = agent_orchestrator.process_message(
        user_message="An item was missing from my order. Can I get a refund?",
        customer_id="CUST001"
    )
    assert resp_refund is not None
    assert "Refund" in resp_refund.action_taken

    # 2. Cancel query without explicit order_id
    resp_cancel = agent_orchestrator.process_message(
        user_message="I want to cancel my order",
        customer_id="CUST001"
    )
    assert resp_cancel is not None
    assert resp_cancel.action_taken in ["Cancellation Confirmation Required", "Order Identification Needed"]

