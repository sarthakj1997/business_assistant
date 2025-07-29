from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import relationship
from database.models import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False)

    invoices = relationship("Invoice", back_populates="user")