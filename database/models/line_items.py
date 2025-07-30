from sqlalchemy import Column, Integer, String, Float, ForeignKey, Text
from sqlalchemy.orm import relationship
from database.models import Base

class LineItem(Base):
    __tablename__ = "line_items"

    id               = Column(Integer, primary_key=True)
    invoice_id       = Column(
                         Integer,
                         ForeignKey("invoices.id", ondelete="CASCADE"),
                         nullable=False
                      )
    product_id       = Column(String)
    product_name     = Column(Text, nullable=False)
    quantity         = Column(Integer)
    unit_price       = Column(Float)
    line_total       = Column(Float)
    confidence_score = Column(Float)

    invoice = relationship("Invoice", back_populates="line_items")