import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv

# 1. Load environment variables from the .env file
load_dotenv()

# Get the database URL, or use a development fallback if it fails
SQLALCHEMY_DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql://admin:admin@localhost:5432/security_db"
)

# 2. The Engine: Handles physical communication and the Connection Pool with PostgreSQL
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    pool_size=5,        # Keeps 5 open connections ready for use (ideal for local RAM)
    max_overflow=10     # Allows up to 10 extra connections during traffic spikes
)

# 3. The Session Factory: Creates individual database transactions
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 4. The Base Class: All models (tables) we create will inherit from this
Base = declarative_base()

# 5. Dependency Injection: Function FastAPI will use to provide a database session per request
def get_db():
    """
    Database session generator.
    Ensures the connection is properly closed even if a server error occurs.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()