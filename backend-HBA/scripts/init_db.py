import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import inspect
from utils.database import engine, Base
from utils.logger import setup_logger
from models import (
    MRBSArea,
    MRBSRoom,
    MRBSUser,
    MRBSModule,
    MRBSEntry,
    MRBSRepeat,
    MRBSSwapRequest
)

logger = setup_logger(__name__)


def check_table_exists(table_name: str) -> bool:
    """Check if a table exists in the database"""
    inspector = inspect(engine)
    return table_name in inspector.get_table_names()


def create_all_tables():
    """Create all database tables"""
    try:
        logger.info("Creating database tables...")
        Base.metadata.create_all(bind=engine)
        logger.info("All tables created successfully!")
        
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        logger.info(f"Existing tables: {', '.join(tables)}")
        
        return True
    except Exception as e:
        logger.error(f"Error creating tables: {e}")
        return False


def verify_schema():
    """Verify database schema"""
    try:
        inspector = inspect(engine)
        
        expected_tables = [
            'mrbs_area',
            'mrbs_room',
            'mrbs_users',
            'mrbs_module',
            'mrbs_entry',
            'mrbs_repeat',
            'swap_requests'
        ]
        
        existing_tables = inspector.get_table_names()
        
        logger.info("Verifying database schema...")
        
        for table in expected_tables:
            if table in existing_tables:
                columns = inspector.get_columns(table)
                logger.info(f"✓ Table '{table}' exists with {len(columns)} columns")
            else:
                logger.warning(f"✗ Table '{table}' is missing!")
        
        missing_tables = set(expected_tables) - set(existing_tables)
        if missing_tables:
            logger.warning(f"Missing tables: {', '.join(missing_tables)}")
            return False
        
        logger.info("Schema verification completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error verifying schema: {e}")
        return False


def create_indexes():
    """Create additional indexes for performance"""
    try:
        logger.info("Creating additional indexes...")
        
        with engine.connect() as conn:
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_entry_room_time ON mrbs_entry(room_id, start_time, end_time)",
                "CREATE INDEX IF NOT EXISTS idx_entry_user_time ON mrbs_entry(create_by, start_time)",
                "CREATE INDEX IF NOT EXISTS idx_swap_status_time ON swap_requests(status, created_at)",
            ]
            
            for idx_sql in indexes:
                try:
                    conn.execute(idx_sql)
                    logger.info(f"Created index: {idx_sql.split('idx_')[1].split(' ')[0]}")
                except Exception as e:
                    logger.warning(f"Index creation failed (may already exist): {e}")
            
            conn.commit()
        
        logger.info("Index creation completed!")
        return True
        
    except Exception as e:
        logger.error(f"Error creating indexes: {e}")
        return False


def main():
    """Main initialization function"""
    logger.info("=" * 60)
    logger.info("Database Initialization Script")
    logger.info("=" * 60)
    
    try:
        logger.info("Testing database connection...")
        with engine.connect() as conn:
            logger.info("✓ Database connection successful!")
        
        if not create_all_tables():
            logger.error("Failed to create tables")
            sys.exit(1)
        
        if not verify_schema():
            logger.error("Schema verification failed")
            sys.exit(1)
        
        if not create_indexes():
            logger.warning("Some indexes could not be created")
        
        logger.info("=" * 60)
        logger.info("Database initialization completed successfully!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()