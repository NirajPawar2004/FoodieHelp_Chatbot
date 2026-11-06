"""
Database Seeder for FoodieHelp AI
Initializes the SQLite schema and seeds 20 customers, 55 orders, and realistic support tickets.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

from config.settings import settings
from database.db import get_connection, init_db


def seed_database(db_path: Optional[Path] = None, force_reseed: bool = False) -> None:
    """Populate the database with fictional customers, orders, and tickets."""
    init_db(db_path)

    with get_connection(db_path) as conn:
        cursor = conn.cursor()

        # Check existing count
        existing_customers = cursor.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
        if existing_customers > 0 and not force_reseed:
            print("Database already contains data. Skipping reseed (use force_reseed=True to overwrite).")
            return

        if force_reseed:
            cursor.execute("DELETE FROM support_tickets")
            cursor.execute("DELETE FROM orders")
            cursor.execute("DELETE FROM customers")

        # 1. Load and insert customers
        customers_path = settings.SAMPLE_CUSTOMERS_FILE
        if not customers_path.exists():
            raise FileNotFoundError(f"Customer dataset not found at {customers_path}")

        with open(customers_path, "r", encoding="utf-8") as f:
            customers_data = json.load(f)

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for c in customers_data:
            cursor.execute(
                """
                INSERT OR REPLACE INTO customers (customer_id, name, email, phone, address, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (c["customer_id"], c["name"], c["email"], c["phone"], c["address"], c.get("created_at", now_str))
            )

        # 2. Load and insert orders
        orders_path = settings.SAMPLE_ORDERS_FILE
        if not orders_path.exists():
            raise FileNotFoundError(f"Orders dataset not found at {orders_path}")

        with open(orders_path, "r", encoding="utf-8") as f:
            orders_data = json.load(f)

        for o in orders_data:
            cursor.execute(
                """
                INSERT OR REPLACE INTO orders (
                    order_id, customer_id, customer_name, restaurant_name,
                    ordered_items, order_amount, order_date_time, payment_method,
                    order_status, estimated_delivery_time, delivery_address,
                    cancellation_reason, cancelled_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    o["order_id"],
                    o["customer_id"],
                    o["customer_name"],
                    o["restaurant_name"],
                    json.dumps(o["ordered_items"]),
                    float(o["order_amount"]),
                    o["order_date_time"],
                    o["payment_method"],
                    o["order_status"],
                    o["estimated_delivery_time"],
                    o["delivery_address"],
                    o.get("cancellation_reason"),
                    o.get("cancelled_at")
                )
            )

        # 3. Seed initial realistic support tickets for admin KPI visualization
        initial_tickets = [
            (
                "TIK-1001", "CUST001", "ORD1001", "Late Delivery",
                "Order delayed by over 45 minutes beyond delivery window due to rain.",
                "Medium", "Resolved", (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d %H:%M:%S"),
                (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S"),
                "Issued INR 100 late delivery compensation coupon to FoodiePay Wallet."
            ),
            (
                "TIK-1002", "CUST003", "ORD1003", "Missing Item",
                "Customer reported missing Garlic Naan from their dinner order.",
                "High", "Resolved", (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S"),
                (datetime.now() - timedelta(hours=18)).strftime("%Y-%m-%d %H:%M:%S"),
                "Verified missing item with restaurant. Refunded INR 75 to UPI account."
            ),
            (
                "TIK-1003", "CUST005", "ORD1005", "Wrong Item",
                "Ordered Veg Hakka Noodles but received Chicken Noodles (Dietary violation).",
                "Critical", "Escalated", (datetime.now() - timedelta(hours=3)).strftime("%Y-%m-%d %H:%M:%S"),
                (datetime.now() - timedelta(hours=2)).strftime("%Y-%m-%d %H:%M:%S"),
                "Escalated to Senior Quality Lead. Restaurant partner notified of audit."
            ),
            (
                "TIK-1004", "CUST007", "ORD1007", "Cancellation Inquiry",
                "Customer wanted to know if cancellation fee applies while preparing.",
                "Low", "Open", (datetime.now() - timedelta(minutes=40)).strftime("%Y-%m-%d %H:%M:%S"),
                None, None
            ),
            (
                "TIK-1005", "CUST010", "ORD1010", "Payment Issue",
                "Amount deducted twice during UPI transaction checkout.",
                "High", "In Progress", (datetime.now() - timedelta(hours=5)).strftime("%Y-%m-%d %H:%M:%S"),
                None, "Duplicate payment under verification with payment gateway."
            )
        ]

        for t in initial_tickets:
            cursor.execute(
                """
                INSERT OR REPLACE INTO support_tickets (
                    ticket_id, customer_id, order_id, issue_category,
                    description, priority, status, created_at, updated_at, resolution_notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                t
            )

        conn.commit()
        print(f"Database seeded successfully with {len(customers_data)} customers, {len(orders_data)} orders, and {len(initial_tickets)} tickets.")


if __name__ == "__main__":
    seed_database(force_reseed=True)

