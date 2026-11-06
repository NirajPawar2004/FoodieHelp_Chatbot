"""
SQLite Database Layer for FoodieHelp AI
Manages thread-safe SQLite interactions, schema initialization, and queries.
"""

from __future__ import annotations

import sqlite3
import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

from config.settings import settings
from database.models import Customer, Order, SupportTicket


def get_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Open a connection to SQLite database with foreign keys enabled."""
    target_path = Path(db_path) if db_path else settings.DATABASE_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(target_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db(db_path: Optional[Path] = None) -> None:
    """Initialize database tables for customers, orders, and tickets."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()

        # Customers table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS customers (
                customer_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT NOT NULL,
                phone TEXT NOT NULL,
                address TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
        """)

        # Orders table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS orders (
                order_id TEXT PRIMARY KEY,
                customer_id TEXT NOT NULL,
                customer_name TEXT NOT NULL,
                restaurant_name TEXT NOT NULL,
                ordered_items TEXT NOT NULL,
                order_amount REAL NOT NULL,
                order_date_time TEXT NOT NULL,
                payment_method TEXT NOT NULL,
                order_status TEXT NOT NULL,
                estimated_delivery_time TEXT NOT NULL,
                delivery_address TEXT NOT NULL,
                cancellation_reason TEXT,
                cancelled_at TEXT,
                FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
            );
        """)

        # Support Tickets table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS support_tickets (
                ticket_id TEXT PRIMARY KEY,
                customer_id TEXT NOT NULL,
                order_id TEXT,
                issue_category TEXT NOT NULL,
                description TEXT NOT NULL,
                priority TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT,
                resolution_notes TEXT,
                FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
            );
        """)

        conn.commit()


# --- Customer Queries ---

def get_customer_by_id(customer_id: str, db_path: Optional[Path] = None) -> Optional[Customer]:
    """Retrieve customer by their unique ID."""
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT customer_id, name, email, phone, address, created_at FROM customers WHERE customer_id = ?",
            (customer_id.strip(),)
        ).fetchone()
        if not row:
            return None
        return Customer(
            customer_id=row["customer_id"],
            name=row["name"],
            email=row["email"],
            phone=row["phone"],
            address=row["address"],
            created_at=row["created_at"]
        )


def get_all_customers(db_path: Optional[Path] = None) -> List[Customer]:
    """Retrieve all customers sorted by customer_id."""
    with get_connection(db_path) as conn:
        rows = conn.execute("SELECT customer_id, name, email, phone, address, created_at FROM customers ORDER BY customer_id").fetchall()
        return [
            Customer(
                customer_id=r["customer_id"],
                name=r["name"],
                email=r["email"],
                phone=r["phone"],
                address=r["address"],
                created_at=r["created_at"]
            )
            for r in rows
        ]


# --- Order Queries ---

def _row_to_order(row: sqlite3.Row) -> Order:
    """Helper to convert sqlite3.Row to an Order instance."""
    items = []
    try:
        items = json.loads(row["ordered_items"])
    except (json.JSONDecodeError, TypeError):
        items = []

    return Order(
        order_id=row["order_id"],
        customer_id=row["customer_id"],
        customer_name=row["customer_name"],
        restaurant_name=row["restaurant_name"],
        ordered_items=items,
        order_amount=float(row["order_amount"]),
        order_date_time=row["order_date_time"],
        payment_method=row["payment_method"],
        order_status=row["order_status"],
        estimated_delivery_time=row["estimated_delivery_time"],
        delivery_address=row["delivery_address"],
        cancellation_reason=row["cancellation_reason"],
        cancelled_at=row["cancelled_at"]
    )


def get_order_by_id(order_id: str, db_path: Optional[Path] = None) -> Optional[Order]:
    """Retrieve order by order_id."""
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM orders WHERE UPPER(order_id) = UPPER(?)",
            (order_id.strip(),)
        ).fetchone()
        if not row:
            return None
        return _row_to_order(row)


def get_orders_by_customer(customer_id: str, db_path: Optional[Path] = None) -> List[Order]:
    """Retrieve all orders placed by a specific customer."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM orders WHERE customer_id = ? ORDER BY order_date_time DESC",
            (customer_id.strip(),)
        ).fetchall()
        return [_row_to_order(r) for r in rows]


def get_all_orders(db_path: Optional[Path] = None) -> List[Order]:
    """Retrieve all orders in the system."""
    with get_connection(db_path) as conn:
        rows = conn.execute("SELECT * FROM orders ORDER BY order_date_time DESC").fetchall()
        return [_row_to_order(r) for r in rows]


def update_order_status(
    order_id: str,
    new_status: str,
    cancellation_reason: Optional[str] = None,
    db_path: Optional[Path] = None
) -> bool:
    """Update order status (e.g. to Cancelled) and record timestamp."""
    with get_connection(db_path) as conn:
        cancelled_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S") if new_status.lower() == "cancelled" else None
        cursor = conn.execute(
            """
            UPDATE orders
            SET order_status = ?, cancellation_reason = ?, cancelled_at = ?
            WHERE UPPER(order_id) = UPPER(?)
            """,
            (new_status, cancellation_reason, cancelled_at, order_id.strip())
        )
        conn.commit()
        return cursor.rowcount > 0


# --- Support Ticket Queries ---

def _row_to_ticket(row: sqlite3.Row) -> SupportTicket:
    """Helper to convert sqlite3.Row to SupportTicket."""
    return SupportTicket(
        ticket_id=row["ticket_id"],
        customer_id=row["customer_id"],
        order_id=row["order_id"],
        issue_category=row["issue_category"],
        description=row["description"],
        priority=row["priority"],
        status=row["status"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        resolution_notes=row["resolution_notes"]
    )


def create_ticket(ticket: SupportTicket, db_path: Optional[Path] = None) -> str:
    """Insert a new support ticket into the database."""
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO support_tickets (
                ticket_id, customer_id, order_id, issue_category,
                description, priority, status, created_at, updated_at, resolution_notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ticket.ticket_id,
                ticket.customer_id,
                ticket.order_id,
                ticket.issue_category,
                ticket.description,
                ticket.priority,
                ticket.status,
                ticket.created_at,
                ticket.updated_at,
                ticket.resolution_notes
            )
        )
        conn.commit()
        return ticket.ticket_id


def get_ticket_by_id(ticket_id: str, db_path: Optional[Path] = None) -> Optional[SupportTicket]:
    """Retrieve ticket by ticket_id."""
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM support_tickets WHERE UPPER(ticket_id) = UPPER(?)",
            (ticket_id.strip(),)
        ).fetchone()
        if not row:
            return None
        return _row_to_ticket(row)


def get_all_tickets(db_path: Optional[Path] = None) -> List[SupportTicket]:
    """Retrieve all tickets sorted newest first."""
    with get_connection(db_path) as conn:
        rows = conn.execute("SELECT * FROM support_tickets ORDER BY created_at DESC").fetchall()
        return [_row_to_ticket(r) for r in rows]


def get_tickets_by_customer(customer_id: str, db_path: Optional[Path] = None) -> List[SupportTicket]:
    """Retrieve tickets associated with a customer."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM support_tickets WHERE customer_id = ? ORDER BY created_at DESC",
            (customer_id.strip(),)
        ).fetchall()
        return [_row_to_ticket(r) for r in rows]


def update_ticket_status(
    ticket_id: str,
    new_status: str,
    resolution_notes: Optional[str] = None,
    db_path: Optional[Path] = None
) -> bool:
    """Update ticket status and resolution notes."""
    with get_connection(db_path) as conn:
        updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor = conn.execute(
            """
            UPDATE support_tickets
            SET status = ?, resolution_notes = ?, updated_at = ?
            WHERE UPPER(ticket_id) = UPPER(?)
            """,
            (new_status, resolution_notes, updated_at, ticket_id.strip())
        )
        conn.commit()
        return cursor.rowcount > 0


# --- Admin Metrics Query ---

def get_admin_metrics(db_path: Optional[Path] = None) -> Dict[str, Any]:
    """Compute live counts and distributions from SQLite data for admin KPI cards."""
    with get_connection(db_path) as conn:
        total_customers = conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
        total_orders = conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
        total_cancelled = conn.execute("SELECT COUNT(*) FROM orders WHERE order_status = 'Cancelled'").fetchone()[0]

        total_tickets = conn.execute("SELECT COUNT(*) FROM support_tickets").fetchone()[0]
        open_tickets = conn.execute("SELECT COUNT(*) FROM support_tickets WHERE status IN ('Open', 'Escalated')").fetchone()[0]
        resolved_tickets = conn.execute("SELECT COUNT(*) FROM support_tickets WHERE status = 'Resolved'").fetchone()[0]

        # Order status distribution
        status_rows = conn.execute("SELECT order_status, COUNT(*) as cnt FROM orders GROUP BY order_status").fetchall()
        order_status_counts = {r["order_status"]: r["cnt"] for r in status_rows}

        # Ticket category distribution
        cat_rows = conn.execute("SELECT issue_category, COUNT(*) as cnt FROM support_tickets GROUP BY issue_category").fetchall()
        ticket_category_counts = {r["issue_category"]: r["cnt"] for r in cat_rows}

        # Ticket priority distribution
        pri_rows = conn.execute("SELECT priority, COUNT(*) as cnt FROM support_tickets GROUP BY priority").fetchall()
        ticket_priority_counts = {r["priority"]: r["cnt"] for r in pri_rows}

        return {
            "total_customers": total_customers,
            "total_orders": total_orders,
            "total_cancelled": total_cancelled,
            "total_tickets": total_tickets,
            "open_tickets": open_tickets,
            "resolved_tickets": resolved_tickets,
            "order_status_counts": order_status_counts,
            "ticket_category_counts": ticket_category_counts,
            "ticket_priority_counts": ticket_priority_counts
        }

