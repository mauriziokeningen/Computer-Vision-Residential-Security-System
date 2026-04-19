import os
import asyncio
import logging
from pathlib import Path
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import OperationalError
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[2]
ENV_FILE = BASE_DIR / ".env"
ENV_EXAMPLE_FILE = BASE_DIR / ".env.example"

if ENV_FILE.exists():
    load_dotenv(ENV_FILE)
elif ENV_EXAMPLE_FILE.exists():
    load_dotenv(ENV_EXAMPLE_FILE)

logger = logging.getLogger("DatabaseSession")

SQLALCHEMY_DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://admin:admin@127.0.0.1:5433/security_db"
)

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


async def init_db(retries: int = 20, delay: float = 2.0) -> None:
    last_error = None

    for attempt in range(1, retries + 1):
        try:
            with engine.begin() as connection:
                connection.execute(text("SELECT 1"))

                if engine.dialect.name == "postgresql":
                    connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                    connection.execute(text("CREATE EXTENSION IF NOT EXISTS pgcrypto"))

            from src.database import models  # noqa: F401
            Base.metadata.create_all(bind=engine)

            logger.info("Database initialized successfully.")
            return

        except OperationalError as exc:
            last_error = exc
            logger.warning(
                "Database not ready yet (%s/%s). Retrying in %.1f seconds...",
                attempt,
                retries,
                delay,
            )
            await asyncio.sleep(delay)

        except Exception:
            logger.exception("Database initialization failed.")
            raise

    logger.exception("Database initialization failed after all retries.")
    raise last_error


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()