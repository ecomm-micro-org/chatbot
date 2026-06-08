from sqlalchemy import Column
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql.sqltypes import BigInteger, Boolean, Numeric, Text

from src.db.db import Base


class Product(Base):
    __tablename__ = "products"

    id = Column(BigInteger, primary_key=True)
    name = Column(Text, nullable=False)
    price = Column(Numeric(10, 2), nullable=False)
    original_price = Column(Numeric(10, 2), nullable=False)
    image = Column(Text, nullable=False)
    category = Column(Text, nullable=False)
    description = Column(Text, nullable=False)
    rating = Column(Numeric(3, 2), nullable=True)
    reviews = Column(BigInteger)
    stock = Column(BigInteger, default=0, nullable=False)
    in_stock = Column(Boolean, default=False, nullable=False)
    tags = Column(JSONB, default=list, nullable=True)
