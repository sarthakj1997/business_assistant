from sqlalchemy.orm import Session
from database.setup_db import SessionLocal
from database.models.users import User
from database.models.invoice import Invoice
from database.models.line_items import LineItem
from backend.services.process_invoice import InvoiceSummary
from backend.services.rag_embedding_invoice import store_invoice_vectors
from datetime import datetime

def save_invoice_to_db(invoice_data: InvoiceSummary, user_id: int) -> int:
    db = SessionLocal()
    try:
        # Create invoice
        invoice = Invoice(
            user_id=user_id,
            order_id=invoice_data.order_id,
            customer_id=invoice_data.customer_id,
            invoice_date=datetime.strptime(invoice_data.invoice_date, "%Y-%m-%d").date(),
            contact_name=invoice_data.contact_name,
            address=invoice_data.address,
            city=invoice_data.city,
            postal_code=invoice_data.postal_code,
            country=invoice_data.country,
            customer_phone=invoice_data.customer_phone,
            customer_fax=invoice_data.customer_fax,
            total_price=invoice_data.total_price,
            confidence_score=invoice_data.confidence_score
        )
        
        db.add(invoice)
        db.flush()
        
        # Create line items
        for item in invoice_data.items:
            line_item = LineItem(
                invoice_id=invoice.id,
                product_id=item.product_id,
                product_name=item.product_name,
                quantity=item.quantity,
                unit_price=item.unit_price,
                line_total=item.line_total,
                confidence_score=item.confidence_score
            )
            db.add(line_item)
        
        db.commit()
        store_invoice_vectors(invoice_data, invoice.id, user_id)
        return invoice.id
    finally:
        db.close()
