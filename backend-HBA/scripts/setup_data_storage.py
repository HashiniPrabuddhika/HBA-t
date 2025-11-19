import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from data import EmbeddingManager, ModelManager, AnalyticsManager
from recommendtion.recommendations.utils.cache_manager import CacheManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_directory_structure(base_path: str = "data"):
    logger.info("Setting up directory structure...")
    directories = [
        f"{base_path}/embeddings/rooms", f"{base_path}/embeddings/users", 
        f"{base_path}/embeddings/bookings", f"{base_path}/embeddings/metadata",
        f"{base_path}/models/clustering", f"{base_path}/models/embedding",
        f"{base_path}/models/time_series", f"{base_path}/models/collaborative",
        f"{base_path}/models/metadata", f"{base_path}/cache",
        f"{base_path}/analytics/events", f"{base_path}/analytics/aggregates",
        f"{base_path}/analytics/reports", f"{base_path}/analytics/exports"
    ]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")
    logger.info("Directory structure setup complete!")

def create_config_files(base_path: str = "data"):
    logger.info("Creating configuration files...")
    try:
        cache_config = f"""# Cache Configuration
CACHE_DB_PATH = "{base_path}/cache/recommendations_cache.db"
CACHE_TTL = 3600  # Cache TTL in seconds (1 hour)
CACHE_CLEANUP_INTERVAL = 3600  # Cleanup interval in seconds
"""
        config_dir = Path("config")
        config_dir.mkdir(exist_ok=True)
        with open(config_dir / "cache_config.py", 'w') as f:
            f.write(cache_config)
        logger.info("Configuration files created successfully!")
    except Exception as e:
        logger.error(f"Error creating configuration files: {e}")
        raise

def initialize_managers(base_path: str = "data"):
    logger.info("Initializing data managers...")
    try:
        logger.info("Initializing EmbeddingManager...")
        embedding_mgr = EmbeddingManager(f"{base_path}/embeddings")
        logger.info("EmbeddingManager initialized successfully")
        
        logger.info("Initializing ModelManager...")
        model_mgr = ModelManager(f"{base_path}/models")
        logger.info("ModelManager initialized successfully")
        
        logger.info("Initializing SQLite CacheManager...")
        cache_mgr = CacheManager()
        logger.info("SQLite CacheManager initialized successfully")
        
        logger.info("Initializing AnalyticsManager...")
        analytics_mgr = AnalyticsManager(f"{base_path}/analytics")
        logger.info("AnalyticsManager initialized successfully")
        
        logger.info("Testing cache functionality...")
        test_key = "setup_test"
        test_value = {"timestamp": datetime.now().isoformat(), "status": "setup_complete"}
        
        import asyncio
        async def test_cache():
            await cache_mgr.set(test_key, test_value)
            retrieved = await cache_mgr.get(test_key)
            if retrieved:
                logger.info("Cache test successful!")
            else:
                logger.warning("Cache test failed - could not retrieve test data")
            await cache_mgr.delete(test_key)
        
        asyncio.run(test_cache())
        
        embedding_mgr.close()
        model_mgr.close()
        if hasattr(cache_mgr, 'close'):
            cache_mgr.close()
        analytics_mgr.close()
        
        logger.info("All managers initialized successfully!")
    except Exception as e:
        logger.error(f"Error initializing managers: {e}")
        raise

def create_sample_data():
    logger.info("Creating sample data...")
    try:
        import numpy as np
        from data import BookingEvent, RecommendationEvent
        
        embedding_mgr = EmbeddingManager()
        analytics_mgr = AnalyticsManager()
        cache_mgr = CacheManager()
        
        logger.info("Creating sample room embeddings...")
        for room_id in range(1, 6):
            sample_embedding = np.random.rand(384)
            sample_features = {
                'capacity': int(np.random.randint(5, 50)),
                'has_projector': bool(np.random.choice([True, False])),
                'has_whiteboard': bool(np.random.choice([True, False])),
                'floor': int(np.random.randint(1, 5)),
                'building': f"Building_{chr(65 + np.random.randint(0, 3))}"
            }
            embedding_mgr.save_room_embedding(room_id=room_id, embedding=sample_embedding, features=sample_features)
        
        logger.info("Creating sample user embeddings...")
        for user_id in range(1, 11):
            sample_embedding = np.random.rand(384)
            sample_preferences = {
                'preferred_capacity': int(np.random.randint(5, 30)),
                'needs_projector': bool(np.random.choice([True, False])),
                'preferred_floor': int(np.random.randint(1, 5)),
                'booking_frequency': float(np.random.uniform(0.5, 5.0))
            }
            embedding_mgr.save_user_embedding(user_id=user_id, embedding=sample_embedding, preferences=sample_preferences)
        
        logger.info("Creating sample booking events...")
        base_date = datetime.now() - timedelta(days=30)
        for i in range(100):
            event_date = base_date + timedelta(days=np.random.randint(0, 60))
            booking_date = event_date + timedelta(days=np.random.randint(1, 14))
            start_hour = np.random.randint(8, 18)
            duration = np.random.choice([30, 60, 90, 120])
            end_hour = start_hour + (duration // 60)
            
            booking_event = BookingEvent(
                event_id=f"sample_booking_{i}",
                user_id=int(np.random.randint(1, 11)),
                room_id=int(np.random.randint(1, 6)),
                event_type=np.random.choice(['booking_created', 'booking_created', 'booking_created', 'booking_cancelled', 'booking_modified']),
                timestamp=event_date.isoformat(),
                booking_date=booking_date.strftime('%Y-%m-%d'),
                start_time=f"{start_hour:02d}:00",
                end_time=f"{end_hour:02d}:00",
                duration_minutes=int(duration),
                metadata={
                    'source': np.random.choice(['web_app', 'mobile_app', 'api']),
                    'booking_type': np.random.choice(['meeting', 'presentation', 'workshop'])
                }
            )
            analytics_mgr.log_booking_event(booking_event)
        
        logger.info("Creating sample recommendation events...")
        for i in range(50):
            event_date = base_date + timedelta(days=np.random.randint(0, 30))
            rec_event = RecommendationEvent(
                event_id=f"sample_rec_{i}",
                user_id=int(np.random.randint(1, 11)),
                recommendation_type=np.random.choice(['alternative_time', 'alternative_room', 'proactive_suggestions']),
                recommended_items=[
                    {'room_id': int(np.random.randint(1, 6)), 'score': float(np.random.uniform(0.5, 1.0))},
                    {'room_id': int(np.random.randint(1, 6)), 'score': float(np.random.uniform(0.3, 0.8))}
                ],
                timestamp=event_date.isoformat(),
                accepted=bool(np.random.choice([True, False], p=[0.3, 0.7])),
                accepted_item_id=int(np.random.randint(1, 6)) if np.random.random() < 0.3 else None,
                response_time_ms=int(np.random.randint(50, 500)),
                context={'original_request': {'date': '2024-03-20', 'time': '14:00', 'duration': 60}}
            )
            analytics_mgr.log_recommendation_event(rec_event)
        
        logger.info("Creating sample cache entries...")
        import asyncio
        async def create_cache_samples():
            sample_cache_data = [
                ("user_profile_1", {"user_id": 1, "preferences": {"capacity": 20}, "last_updated": datetime.now().isoformat()}),
                ("room_similarity_2_3", {"similarity_score": 0.85, "features_match": ["projector", "capacity"]}),
                ("recent_bookings_user_5", [{"room_id": 2, "date": "2024-03-15"}, {"room_id": 4, "date": "2024-03-10"}]),
                ("recommendation_cache_user_3", {"recommendations": [{"room_id": 1, "score": 0.9}], "generated_at": datetime.now().isoformat()})
            ]
            for key, value in sample_cache_data:
                await cache_mgr.set(key, value)
            logger.info(f"Created {len(sample_cache_data)} sample cache entries")
        
        asyncio.run(create_cache_samples())
        
        embedding_mgr.close()
        analytics_mgr.close()
        if hasattr(cache_mgr, 'close'):
            cache_mgr.close()
        
        logger.info("Sample data created successfully!")
    except Exception as e:
        logger.error(f"Error creating sample data: {e}")
        raise

def verify_setup(base_path: str = "data"):
    logger.info("Verifying setup...")
    try:
        required_paths = [f"{base_path}/embeddings", f"{base_path}/models", f"{base_path}/cache", f"{base_path}/analytics"]
        for path in required_paths:
            if not Path(path).exists():
                raise FileNotFoundError(f"Required directory missing: {path}")
        
        db_files = [
            f"{base_path}/embeddings/metadata/embeddings_metadata.db",
            f"{base_path}/models/metadata/models_metadata.db",
            f"{base_path}/analytics/analytics.db"
        ]
        for db_file in db_files:
            if not Path(db_file).exists():
                raise FileNotFoundError(f"Required database missing: {db_file}")
        
        cache_db_path = f"{base_path}/cache/recommendations_cache.db"
        if Path(cache_db_path).exists():
            logger.info(f"Cache database found: {cache_db_path}")
        else:
            logger.info("Cache database will be created on first use")
        
        embedding_mgr = EmbeddingManager(f"{base_path}/embeddings")
        model_mgr = ModelManager(f"{base_path}/models")
        cache_mgr = CacheManager()
        analytics_mgr = AnalyticsManager(f"{base_path}/analytics")
        
        stats = embedding_mgr.get_embedding_stats()
        logger.info(f"Embedding stats: {stats}")
        model_stats = model_mgr.get_storage_stats()
        logger.info(f"Model stats: {model_stats}")
        
        import asyncio
        async def test_cache_stats():
            cache_stats = await cache_mgr.get_cache_stats() if hasattr(cache_mgr, 'get_cache_stats') else {"type": "SQLite", "status": "available"}
            logger.info(f"Cache stats: {cache_stats}")
            test_key = "verification_test"
            test_value = {"verification": True, "timestamp": datetime.now().isoformat()}
            await cache_mgr.set(test_key, test_value)
            retrieved = await cache_mgr.get(test_key)
            if retrieved and retrieved.get("verification"):
                logger.info("Cache operations verified successfully!")
            else:
                logger.warning("Cache operations test failed")
            await cache_mgr.delete(test_key)
        
        asyncio.run(test_cache_stats())
        
        analytics_summary = analytics_mgr.get_analytics_summary()
        logger.info(f"Analytics summary: {analytics_summary}")
        
        embedding_mgr.close()
        model_mgr.close()
        if hasattr(cache_mgr, 'close'):
            cache_mgr.close()
        analytics_mgr.close()
        
        logger.info("Setup verification completed successfully!")
    except Exception as e:
        logger.error(f"Setup verification failed: {e}")
        raise

def main():
    logger.info("Starting data storage system setup...")
    try:
        base_path = os.getenv('DATA_BASE_PATH', 'data')
        setup_directory_structure(base_path)
        create_config_files(base_path)
        initialize_managers(base_path)
        
        create_sample = os.getenv('CREATE_SAMPLE_DATA', 'true').lower() == 'true'
        if create_sample:
            create_sample_data()
        
        verify_setup(base_path)
        
        logger.info("Data storage system setup completed successfully!")
        logger.info(f"Base path: {Path(base_path).absolute()}")
        
        print("1. Configure cache TTL settings in recommendation_config.py")
        print("2. Set up backup procedures (databases included)")
        print("3. Configure retention policies")
        print("4. Set up monitoring and alerts")
        print("5. Consider cache warming strategies for better performance")
        
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()