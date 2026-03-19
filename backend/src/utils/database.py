"""
Database connection manager for PostgreSQL.
Uses psycopg2 with a SimpleConnectionPool for efficient connection reuse.
"""
import os
import logging
from contextlib import contextmanager
from psycopg2 import pool

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Database")

# Configuration (mirrors docker-compose.yml defaults)
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5432")),
    "dbname": os.getenv("DB_NAME", "security_db"),
    "user": os.getenv("DB_USER", "admin"),
    "password": os.getenv("DB_PASSWORD", "admin"),
}

# Connection Pool (Lazy Initialization)
_pool: pool.SimpleConnectionPool | None = None


def _get_pool() -> pool.SimpleConnectionPool:
    """Creates the connection pool on first use (Singleton pattern)."""
    global _pool
    if _pool is None or _pool.closed:
        try:
            _pool = pool.SimpleConnectionPool(minconn=1, maxconn=10, **DB_CONFIG)
            logger.info(f"Connection pool created for {DB_CONFIG['dbname']}@{DB_CONFIG['host']}")
        except Exception as e:
            logger.error(f"Failed to create connection pool: {e}")
            raise
    return _pool


@contextmanager
def get_connection():
    """
    Context manager that yields a database connection from the pool.
    Automatically returns the connection when the block exits.

    Usage:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT ...")
    """
    conn = _get_pool().getconn()
    try:
        yield conn
    finally:
        _get_pool().putconn(conn)


def close_pool() -> None:
    """Gracefully closes all connections in the pool. Call on application shutdown."""
    global _pool
    if _pool and not _pool.closed:
        _pool.closeall()
        logger.info("Database connection pool closed.")