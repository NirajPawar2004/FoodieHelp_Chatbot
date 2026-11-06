"""
Order Service for FoodieHelp AI
Provides order lookup, customer ownership verification, and simulated order cancellation.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional
from datetime import datetime

from database.db import (
    get_order_by_id,
    get_orders_by_customer,
    update_order_status,
    get_customer_by_id,
)
from database.models import Order, CancellationResult


class OrderService:
    """Manages order inquiries, security validation, and status transitions."""

    @staticmethod
    def get_order_status(order_id: str, customer_id: str) -> Dict[str, Any]:
        """
        Lookup order status with customer ownership verification.
        Returns status, restaurant, estimated delivery time, and security check outcome.
        """
        clean_order_id = order_id.strip().upper()
        clean_cust_id = customer_id.strip()

        order = get_order_by_id(clean_order_id)
        if not order:
            return {
                "success": False,
                "error": "ORDER_NOT_FOUND",
                "message": f"No order found matching ID '{clean_order_id}'. Please check your order ID."
            }

        # Ownership validation
        if order.customer_id != clean_cust_id:
            return {
                "success": False,
                "error": "UNAUTHORIZED_ACCESS",
                "message": (
                    f"Order '{clean_order_id}' does not belong to your customer profile ({clean_cust_id}). "
                    "For privacy and security, you can only view your own orders."
                )
            }

        return {
            "success": True,
            "order_id": order.order_id,
            "restaurant_name": order.restaurant_name,
            "order_status": order.order_status,
            "order_amount": order.order_amount,
            "estimated_delivery_time": order.estimated_delivery_time,
            "order_date_time": order.order_date_time,
            "items_count": len(order.ordered_items),
            "message": (
                f"Order **{order.order_id}** from **{order.restaurant_name}** is currently **{order.order_status}**. "
                f"Estimated delivery: **{order.estimated_delivery_time}**."
            )
        }

    @staticmethod
    def get_order_details(order_id: str, customer_id: str) -> Dict[str, Any]:
        """Retrieve full details of an order with customer validation."""
        clean_order_id = order_id.strip().upper()
        clean_cust_id = customer_id.strip()

        order = get_order_by_id(clean_order_id)
        if not order:
            return {
                "success": False,
                "error": "ORDER_NOT_FOUND",
                "message": f"Order '{clean_order_id}' was not found in the system."
            }

        if order.customer_id != clean_cust_id:
            return {
                "success": False,
                "error": "UNAUTHORIZED_ACCESS",
                "message": f"Access denied: Order '{clean_order_id}' belongs to another customer."
            }

        return {
            "success": True,
            "order": order.to_dict(),
            "message": f"Retrieved full details for order {clean_order_id}."
        }

    @staticmethod
    def get_customer_orders(customer_id: str) -> List[Dict[str, Any]]:
        """List all orders associated with a customer."""
        orders = get_orders_by_customer(customer_id.strip())
        return [o.to_dict() for o in orders]

    @staticmethod
    def process_cancellation(
        order_id: str,
        customer_id: str,
        confirmed: bool = False,
        reason: Optional[str] = None
    ) -> CancellationResult:
        """
        Simulate order cancellation under strict business policies:
        - 'Placed': 100% refund, cancellable.
        - 'Preparing': 50% refund (50% kitchen preparation fee), cancellable with confirmation.
        - 'Out for Delivery': Cannot be cancelled (rider en route).
        - 'Delivered': Cannot be cancelled (already completed).
        - 'Cancelled': Already cancelled.
        Requires explicit customer confirmation before updating the database.
        """
        clean_order_id = order_id.strip().upper()
        clean_cust_id = customer_id.strip()

        order = get_order_by_id(clean_order_id)
        if not order:
            return CancellationResult(
                success=False,
                order_id=clean_order_id,
                previous_status="Unknown",
                current_status="Unknown",
                refund_percentage=0.0,
                refund_amount=0.0,
                message=f"Order '{clean_order_id}' was not found in our records."
            )

        if order.customer_id != clean_cust_id:
            return CancellationResult(
                success=False,
                order_id=clean_order_id,
                previous_status=order.order_status,
                current_status=order.order_status,
                refund_percentage=0.0,
                refund_amount=0.0,
                message=f"Access denied: Order '{clean_order_id}' does not belong to you."
            )

        current_status = order.order_status

        # 1. Check if already cancelled
        if current_status == "Cancelled":
            return CancellationResult(
                success=False,
                order_id=clean_order_id,
                previous_status="Cancelled",
                current_status="Cancelled",
                refund_percentage=0.0,
                refund_amount=0.0,
                message=f"Order {clean_order_id} was already cancelled previously on {order.cancelled_at or 'record'}."
            )

        # 2. Check if Out for Delivery or Delivered (Strictly cannot cancel)
        if current_status == "Out for Delivery":
            return CancellationResult(
                success=False,
                order_id=clean_order_id,
                previous_status=current_status,
                current_status=current_status,
                refund_percentage=0.0,
                refund_amount=0.0,
                message=(
                    f"Order {clean_order_id} cannot be cancelled because it is already **Out for Delivery**. "
                    "Our delivery partner is already on the road with your meal."
                )
            )

        if current_status == "Delivered":
            return CancellationResult(
                success=False,
                order_id=clean_order_id,
                previous_status=current_status,
                current_status=current_status,
                refund_percentage=0.0,
                refund_amount=0.0,
                message=(
                    f"Order {clean_order_id} cannot be cancelled because it has already been **Delivered**. "
                    "If you experienced an issue with missing or wrong food, please request a refund or raise a support ticket."
                )
            )

        # 3. Determine refund eligibility by stage
        if current_status == "Placed":
            refund_pct = 100.0
            refund_amt = round(order.order_amount, 2)
            policy_notice = (
                f"Your order is currently **Placed** (not yet accepted by restaurant kitchen). "
                f"Under our Cancellation Policy, you are eligible for a **100% full refund** of **₹{refund_amt:.2f}**."
            )
        elif current_status == "Preparing":
            refund_pct = 50.0
            refund_amt = round(order.order_amount * 0.50, 2)
            policy_notice = (
                f"Your order is currently **Preparing** in the restaurant kitchen. "
                f"Under our Cancellation Policy, a 50% kitchen preparation fee applies. "
                f"You will receive a **50% partial refund** of **₹{refund_amt:.2f}**."
            )
        else:
            return CancellationResult(
                success=False,
                order_id=clean_order_id,
                previous_status=current_status,
                current_status=current_status,
                refund_percentage=0.0,
                refund_amount=0.0,
                message=f"Order status '{current_status}' is not eligible for cancellation."
            )

        # 4. Enforce explicit customer confirmation
        if not confirmed:
            return CancellationResult(
                success=False,
                order_id=clean_order_id,
                previous_status=current_status,
                current_status=current_status,
                refund_percentage=refund_pct,
                refund_amount=refund_amt,
                message=(
                    f"{policy_notice}\n\n"
                    f"⚠️ **Confirmation Required:** Do you want to proceed with cancelling order **{clean_order_id}**? "
                    "Please reply with **'Confirm Cancel'** or click the confirmation button."
                )
            )

        # 5. Customer confirmed: execute state-changing cancellation
        cancellation_msg = reason or f"Customer requested cancellation during '{current_status}' stage."
        success = update_order_status(clean_order_id, "Cancelled", cancellation_reason=cancellation_msg)

        if not success:
            return CancellationResult(
                success=False,
                order_id=clean_order_id,
                previous_status=current_status,
                current_status=current_status,
                refund_percentage=0.0,
                refund_amount=0.0,
                message="Failed to update order status in database. Please try again."
            )

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return CancellationResult(
            success=True,
            order_id=clean_order_id,
            previous_status=current_status,
            current_status="Cancelled",
            refund_percentage=refund_pct,
            refund_amount=refund_amt,
            message=(
                f"✅ **Order {clean_order_id} has been successfully cancelled.**\n\n"
                f"- **Previous Stage:** {current_status}\n"
                f"- **Refund Entitlement:** {refund_pct:.0f}% (₹{refund_amt:.2f})\n"
                f"- **Refund Destination:** Original payment method ({order.payment_method})\n"
                f"- **Cancellation Time:** {now_str}"
            ),
            cancelled_at=now_str
        )


# Global singleton instance
order_service = OrderService()

