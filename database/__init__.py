"""Database module for FoodieHelp AI."""
from database.db import (
    init_db,
    get_connection,
    get_customer_by_id,
    get_all_customers,
    get_order_by_id,
    get_orders_by_customer,
    get_all_orders,
    update_order_status,
    create_ticket,
    get_ticket_by_id,
    get_all_tickets,
    get_tickets_by_customer,
    update_ticket_status,
    get_admin_metrics,
)
from database.models import (
    Customer,
    Order,
    OrderItem,
    SupportTicket,
    RefundEligibilityResult,
    CancellationResult,
)

__all__ = [
    "init_db",
    "get_connection",
    "get_customer_by_id",
    "get_all_customers",
    "get_order_by_id",
    "get_orders_by_customer",
    "get_all_orders",
    "update_order_status",
    "create_ticket",
    "get_ticket_by_id",
    "get_all_tickets",
    "get_tickets_by_customer",
    "update_ticket_status",
    "get_admin_metrics",
    "Customer",
    "Order",
    "OrderItem",
    "SupportTicket",
    "RefundEligibilityResult",
    "CancellationResult",
]

