from sqlalchemy import Column, Integer, String, Float, Date, Text, ForeignKey
from sqlalchemy.orm import relationship
from database.models import Base

class Invoice(Base):
    __tablename__ = "invoices"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    vendor_name = Column(String, nullable=False)
    vendor_address = Column(String)
    customer_name = Column(String)
    customer_address = Column(String)
    invoice_number = Column(String, nullable=False)
    invoice_date = Column(Date, nullable=False)
    due_date = Column(Date)
    purchase_order = Column(String)
    currency = Column(String, nullable=False)
    subtotal = Column(Float)
    tax = Column(Float)
    shipping = Column(Float)
    total = Column(Float, nullable=False)
    payment_terms = Column(String)
    payment_method = Column(String)
    confidence_score = Column(Float, nullable=False)
    raw_text = Column(Text)

    user = relationship("User", back_populates="invoices")
    line_items = relationship("LineItem", back_populates="invoice")