import os

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from internals.logger import logger

DSN = os.getenv("DSN")
if DSN is None:
    logger.info("unable to find DSN in env")
    os._exit(1)

engine = create_engine(DSN)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

Base = declarative_base()


def get_db():
    db = SessionLocal()

    try:
        yield db
    except Exception as e:
        logger.info(f"{str(e)}")
    finally:
        db.close()
