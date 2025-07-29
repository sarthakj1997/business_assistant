from sqlalchemy import Column, Integer, String, Float, ForeignKey
from sqlalchemy.orm import relationship
from database.models import Base

class LineItem(Base):
    __tablename__ = "line_items"

    id = Column(Integer, primary_key=True)
    invoice_id = Column(Integer, ForeignKey("invoices.id"))
    description = Column(String, nullable=False)
    quantity = Column(Float)
    unit_price = Column(Float)
    line_total = Column(Float, nullable=False)

    invoice = relationship("Invoice", back_populates="line_items")