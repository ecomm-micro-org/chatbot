import uuid

from sqlalchemy import Column, Integer, String
from sqlalchemy.dialects.postgresql import UUID

from src.db.db import Base


class Product(Base):
    __tablename__ = "products"
