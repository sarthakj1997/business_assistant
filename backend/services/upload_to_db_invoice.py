from sqlalchemy.orm import Session
from database.setup_db import SessionLocal
from database.models.users import User
from database.models.invoice import Invoice
from database.models.line_items import LineItem
from backend.services.process_invoice import InvoiceSummary
from datetime import datetime

def save_invoice_to_db(invoice_data: InvoiceSummary, user_id: int) -> int:
    db = SessionLocal()
    try:
        # Create invoice
        invoice = Invoice(
            user_id=user_id,
            vendor_name=invoice_data.vendor_name,
            vendor_address=invoice_data.vendor_address,
            customer_name=invoice_data.customer_name,
            customer_address=invoice_data.customer_address,
            invoice_number=invoice_data.invoice_number,
            invoice_date=datetime.strptime(invoice_data.invoice_date, "%d-%m-%Y").date(),
            due_date=datetime.strptime(invoice_data.due_date, "%d-%m-%Y").date() if invoice_data.due_date else None,
            purchase_order=invoice_data.purchase_order,
            currency=invoice_data.currency,
            subtotal=invoice_data.subtotal,
            tax=invoice_data.tax,
            shipping=invoice_data.shipping,
            total=invoice_data.total,
            payment_terms=invoice_data.payment_terms,
            payment_method=invoice_data.payment_method,
            confidence_score=invoice_data.confidence_score
        )
        
        db.add(invoice)
        db.flush()
        
        # Create line items
        for item in invoice_data.items:
            line_item = LineItem(
                invoice_id=invoice.id,
                description=item.description,
                quantity=item.quantity,
                unit_price=item.unit_price,
                line_total=item.line_total
            )
            db.add(line_item)
        
        db.commit()
        return invoice.id
    finally:
        db.close()
