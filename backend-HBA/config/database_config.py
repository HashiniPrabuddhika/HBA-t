from typing import Generator
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session, declarative_base
from sqlalchemy.pool import QueuePool
from config.app_config import get_settings

# Get settings instance
settings = get_settings()

# Create declarative base for models
Base = declarative_base()


class DatabaseConfig:
    """Database configuration with connection pooling settings"""
    
    # Connection settings
    DATABASE_URL: str = settings.DATABASE_URL
    
    # Pool settings
    POOL_SIZE: int = 5
    MAX_OVERFLOW: int = 10
    POOL_TIMEOUT: int = 30
    POOL_RECYCLE: int = 3600
    POOL_PRE_PING: bool = True
    
    # Debug settings
    ECHO: bool = settings.DEBUG
    ECHO_POOL: bool = False


def create_db_engine():

    engine = create_engine(
        DatabaseConfig.DATABASE_URL,
        poolclass=QueuePool,
        pool_size=DatabaseConfig.POOL_SIZE,
        max_overflow=DatabaseConfig.MAX_OVERFLOW,
        pool_timeout=DatabaseConfig.POOL_TIMEOUT,
        pool_recycle=DatabaseConfig.POOL_RECYCLE,
        pool_pre_ping=DatabaseConfig.POOL_PRE_PING,
        echo=DatabaseConfig.ECHO,
        echo_pool=DatabaseConfig.ECHO_POOL,
    )
    
    # Add event listeners for connection management
    @event.listens_for(engine, "connect")
    def receive_connect(dbapi_conn, connection_record):
        """Set connection parameters on new connections"""
        # For MySQL: set charset and timezone
        cursor = dbapi_conn.cursor()
        cursor.execute("SET NAMES utf8mb4")
        cursor.execute("SET time_zone = '+00:00'")
        cursor.close()
    
    return engine


engine = create_db_engine()

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    expire_on_commit=False 
)


def get_db() -> Generator[Session, None, None]:
  
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        db.rollback()
        raise
    finally:
        db.close()


def test_connection() -> bool:
   
    try:
        with engine.connect() as connection:
            connection.execute("SELECT 1")
            print("✅ Database connection successful!")
            return True
    except Exception as e:
        print("❌ Database connection failed!")
        print(f"Error: {e}")
        return False


def init_db():
   
    try:
        Base.metadata.create_all(bind=engine)
        print("✅ Database tables created successfully!")
    except Exception as e:
        print("❌ Failed to create database tables!")
        print(f"Error: {e}")
        raise


def drop_all_tables():
  
    if not settings.DEBUG:
        raise RuntimeError("Cannot drop tables in production mode!")
    
    Base.metadata.drop_all(bind=engine)
    print("⚠️ All tables dropped!")


if __name__ == "__main__":
    test_connection()