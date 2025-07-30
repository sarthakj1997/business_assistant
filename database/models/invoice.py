from sqlalchemy import Column, Integer, String, Float, Date, Text, ForeignKey
from sqlalchemy.orm import relationship
from database.models import Base

class Invoice(Base):
    __tablename__ = "invoices"

    id               = Column(Integer, primary_key=True)
    user_id          = Column(Integer, ForeignKey("users.id"))
    order_id         = Column(String, nullable=False, unique=True)
    customer_id      = Column(String)
    invoice_date     = Column(Date, nullable=False)
    contact_name     = Column(String)
    address          = Column(String)
    city             = Column(String)
    postal_code      = Column(String)
    country          = Column(String)
    customer_phone   = Column(String)
    customer_fax     = Column(String)
    total_price      = Column(Float, nullable=False)
    confidence_score = Column(Float, nullable=False)
    raw_text         = Column(Text)

    user       = relationship("User", back_populates="invoices")
    line_items = relationship(
        "LineItem",
        back_populates="invoice"    
    )
