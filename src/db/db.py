import logging
import os
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, declarative_base, sessionmaker

DSN = os.getenv("DSN")
if not DSN:
    logging.error("DSN environment variable is not set")
    raise RuntimeError("DSN environment variable is not set")


engine = create_engine(
    DSN,
    pool_pre_ping=True,
    pool_recycle=1800,
    echo=False,
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
)

Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """Yield a DB session and guarantee cleanup."""
    db: Session = SessionLocal()
    try:
        yield db
        db.commit()
    except SQLAlchemyError as e:
        db.rollback()
        logging.error(f"Database error: {e}", exc_info=True)
        raise
    finally:
        db.close()
