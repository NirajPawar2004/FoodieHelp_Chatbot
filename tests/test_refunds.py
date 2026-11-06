"""
Unit Tests for FoodieHelp AI Refund Assistant (Module E)
Tests deterministic business logic for missing items, wrong items, cancellations, and quality defects.
"""

import pytest
from database.seed import seed_database
from services.refund_service import refund_service
from services.order_service import order_service
from database.db import get_order_by_id


@pytest.fixture(scope="module", autouse=True)
def setup_db():
    seed_database(force_reseed=False)


def test_missing_item_refund_within_window():
    """Missing item on Delivered order within 24 hours should be eligible for 100% item refund."""
    # Find a delivered order belonging to CUST006
    orders = order_service.get_customer_orders("CUST006")
    delivered = [o for o in orders if o["order_status"] == "Delivered"]
    assert len(delivered) > 0
    order_id = delivered[0]["order_id"]

    res = refund_service.evaluate_refund(
        order_id=order_id,
        customer_id="CUST006",
        issue_type="missing_item",
        hours_since_delivery=3.5
    )

    assert res.eligible is True
    assert res.refund_percentage == 100.0
    assert res.refund_amount > 0.0
    assert "POL-MIS-004" in res.applicable_rule


def test_missing_item_expired_window():
    """Missing item reported after 24 hours should be denied."""
    orders = order_service.get_customer_orders("CUST006")
    delivered = [o for o in orders if o["order_status"] == "Delivered"]
    order_id = delivered[0]["order_id"]

    res = refund_service.evaluate_refund(
        order_id=order_id,
        customer_id="CUST006",
        issue_type="missing_item",
        hours_since_delivery=28.0  # Expired
    )

    assert res.eligible is False
    assert res.refund_amount == 0.0
    assert "expired" in res.reason.lower()


def test_missing_item_on_undelivered_order():
    """Missing item claimed before delivery should be rejected."""
    # ORD1002 is 'Placed'
    res = refund_service.evaluate_refund(
        order_id="ORD1002",
        customer_id="CUST002",
        issue_type="missing_item"
    )

    assert res.eligible is False
    assert "not 'Delivered'" in res.reason


def test_critical_dietary_violation_wrong_item():
    """Dietary violation (e.g. non-veg delivered to vegetarian) gets 100% full order refund."""
    orders = order_service.get_customer_orders("CUST007")
    delivered = [o for o in orders if o["order_status"] == "Delivered"]
    order_id = delivered[0]["order_id"]
    order_amt = delivered[0]["order_amount"]

    res = refund_service.evaluate_refund(
        order_id=order_id,
        customer_id="CUST007",
        issue_type="wrong_item",
        is_dietary_violation=True,
        hours_since_delivery=1.0
    )

    assert res.eligible is True
    assert res.refund_percentage == 100.0
    assert res.refund_amount == order_amt
    assert "CRITICAL DIETARY VIOLATION" in res.reason


def test_duplicate_payment_reversal():
    """Duplicate payment simulation returns 100% automatic reversal."""
    res = refund_service.evaluate_refund(
        order_id="ORD1003",
        customer_id="CUST003",
        issue_type="duplicate_payment"
    )

    assert res.eligible is True
    assert res.refund_percentage == 100.0
    assert "POL-REF-001 Section 2E" in res.applicable_rule


def test_food_quality_window():
    """Food quality reported within 4 hours is eligible; over 4 hours is denied."""
    orders = order_service.get_customer_orders("CUST008")
    delivered = [o for o in orders if o["order_status"] == "Delivered"]
    order_id = delivered[0]["order_id"]

    # Within 4 hours
    res_valid = refund_service.evaluate_refund(
        order_id=order_id,
        customer_id="CUST008",
        issue_type="food_quality",
        hours_since_delivery=2.0
    )
    assert res_valid.eligible is True

    # After 4 hours
    res_expired = refund_service.evaluate_refund(
        order_id=order_id,
        customer_id="CUST008",
        issue_type="food_quality",
        hours_since_delivery=6.0
    )
    assert res_expired.eligible is False
    assert "4 hours" in res_expired.reason


def test_unauthorized_refund_access():
    """Attempting refund evaluation on an order belonging to another customer is blocked."""
    res = refund_service.evaluate_refund(
        order_id="ORD1003",
        customer_id="CUST008",  # Does not own ORD1003
        issue_type="missing_item"
    )
    assert res.eligible is False
    assert "Access Denied" in res.reason

