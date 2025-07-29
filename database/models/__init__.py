from sqlalchemy.orm import declarative_base
Base = declarative_base()

from database.models.users import User
from database.models.invoice import Invoice
from database.models.line_items import LineItem