import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
import logging
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

@dataclass
class RecommendationConfig:
    """Configuration class for the recommendation system"""
    
    # Database Configuration
    database_url: str = field(default_factory=lambda: os.getenv('DATABASE_URL'))
    cache_db_path: str = field(default_factory=lambda: os.getenv('CACHE_DB_PATH', 'cache/recommendations_cache.db'))
    vector_db_path: str = field(default_factory=lambda: os.getenv('VECTOR_DB_PATH', 'cache/vector_store.db'))
    analytics_db_path: str = field(default_factory=lambda: os.getenv('ANALYTICS_DB_PATH', './data/analytics/analytics.db'))
    
    # Database Connection Pool Settings 
    mysql_pool_size: int = 5
    mysql_max_overflow: int = 10
    mysql_pool_timeout: int = 30
    mysql_pool_recycle: int = 3600
    mysql_echo: bool = False
   
    # SQLite-specific cache settings 
    sqlite_cache_size_kb: int = 10000
    sqlite_timeout: float = 30.0
    enable_wal_mode: bool = True
    sqlite_memory_limit_mb: int = 100
    
    # Cache Configuration
    cache_ttl_default: int = 1800
    cache_ttl_user_preferences: int = 3600
    cache_ttl_room_features: int = 7200
    cache_ttl_recommendations: int = 300
    cache_max_size: int = 10000
    cache_cleanup_interval: int = 3600
    
    # Cache retention policies 
    recommendation_cache_ttl_hours: int = 24
    user_profile_cache_ttl_hours: int = 168
    room_similarity_cache_ttl_hours: int = 72
    analytics_cache_ttl_hours: int = 48
    
    # Cache optimization settings
    enable_cache_compression: bool = True
    enable_cache_statistics: bool = True
    auto_vacuum_enabled: bool = True
    lru_eviction_enabled: bool = True
    max_cache_size_mb: int = 500
    use_file_fallback: bool = True
    
    # Recommendation Engine Configuration
    max_recommendations: int = field(default_factory=lambda: int(os.getenv('MAX_ALTERNATIVES', '10')))
    min_score_threshold: float = 0.1
    min_confidence_score: float = field(default_factory=lambda: float(os.getenv('CONFIDENCE_THRESHOLD', '0.5')))
    diversity_factor: float = 0.3
    similarity_threshold: float = field(default_factory=lambda: float(os.getenv('SIMILARITY_THRESHOLD', '0.7')))
    
    # Vector Store Configuration
    max_search_results: int = 50
    vector_index_refresh_interval: int = 3600
    vector_db_path_env: str = field(default_factory=lambda: os.getenv('VECTOR_DB_PATH', './data/embeddings'))
   
    # Feature Extraction Configuration
    feature_vector_dimension: int = 128
    user_feature_dimension: int = 100
    room_feature_dimension: int = 80
    
    # Embedding configuration
    embedding_model_name: str = field(default_factory=lambda: os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2'))
    embedding_dimension: int = 384
    vector_dimension: int = 384
    
    # Analytics Configuration
    analytics_window_days: int = field(default_factory=lambda: int(os.getenv('ANALYTICS_WINDOW_DAYS', '30')))
    min_booking_history: int = field(default_factory=lambda: int(os.getenv('MIN_BOOKINGS_FOR_PATTERN', '3')))
    
    # Business Logic Configuration
    business_start_hour: int = field(default_factory=lambda: int(os.getenv('BUSINESS_START_HOUR', '7')))
    business_end_hour: int = field(default_factory=lambda: int(os.getenv('BUSINESS_END_HOUR', '21')))
    time_slot_minutes: int = field(default_factory=lambda: int(os.getenv('TIME_SLOT_MINUTES', '30')))
    
    # Model Paths 
    clustering_model_path: str = field(default_factory=lambda: os.getenv('CLUSTERING_MODEL_PATH', './data/models/clustering_model.pkl'))
    user_embedding_path: str = field(default_factory=lambda: os.getenv('USER_EMBEDDING_PATH', './data/embeddings/users'))
    room_embedding_path: str = field(default_factory=lambda: os.getenv('ROOM_EMBEDDING_PATH', './data/embeddings/rooms'))
     
    # Database Table Names 
    room_table_name: str = "mrbs_room"
    entry_table_name: str = "mrbs_entry"
    repeat_table_name: str = "mrbs_repeat"
    
    ENABLE_ML_FEATURES = True
    ENABLE_LLM_FEATURES = True
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    VECTOR_DB_PATH = "./data/embeddings"
    ML_MODEL_PATH = "./data/models"
    
    # Performance settings
    request_timeout: int = 30
    max_concurrent_requests: int = 100
    
    # Strategy Weights
    strategy_weights: Dict[str, float] = field(default_factory=lambda: {
        'alternative_room': 0.4, 'alternative_time': 0.3, 'collaborative_filtering': 0.2, 'content_based': 0.1
    })
    
    # Alternative Room Strategy Configuration
    alt_room_config: Dict[str, Any] = field(default_factory=lambda: {
        'capacity_tolerance': 0.2, 'equipment_match_weight': 0.4, 'location_preference_weight': 0.3,
        'availability_weight': 0.3, 'max_floor_difference': 2, 'same_building_bonus': 0.1
    })
    
    # Alternative Time Strategy Configuration
    alt_time_config: Dict[str, Any] = field(default_factory=lambda: {
        'time_slot_duration': 30, 'max_time_shift_hours': 4, 'preferred_hours_start': 8, 'preferred_hours_end': 18,
        'weekend_penalty': 0.2, 'early_morning_penalty': 0.3, 'late_evening_penalty': 0.3,
        'lunch_time_penalty': 0.1, 'same_day_bonus': 0.2
    })
    
    # Collaborative Filtering Configuration
    collaborative_config: Dict[str, Any] = field(default_factory=lambda: {
        'min_common_bookings': 3, 'user_similarity_threshold': 0.3, 'max_similar_users': 20,
        'temporal_decay_factor': 0.95, 'booking_weight_threshold': 0.1
    })
    
    # Content-Based Filtering Configuration
    content_based_config: Dict[str, Any] = field(default_factory=lambda: {
        'feature_weights': {'capacity': 0.25, 'equipment': 0.30, 'location': 0.20, 'amenities': 0.15, 'room_quality': 0.10},
        'categorical_similarity_weight': 0.6, 'numerical_similarity_weight': 0.4, 'location_radius_km': 0.5
    })
    
    # Feature flags
    enable_proactive_recommendations: bool = True
    enable_smart_scheduling: bool = True
    enable_collaborative_filtering: bool = True
     
    # Directory paths 
    cache_base_path: str = field(default_factory=lambda: os.getenv('CACHE_BASE_PATH', './data/cache'))
    main_db_path: str = field(default_factory=lambda: os.getenv('MAIN_DB_PATH', './data/recommendations.db'))
    chroma_persist_directory: Optional[str] = field(default_factory=lambda: os.getenv('CHROMA_PERSIST_DIRECTORY', './data/chroma_db'))
    
    # Backup and recovery 
    auto_backup_enabled: bool = True
    backup_interval_hours: int = 24
    backup_retention_days: int = 7
    backup_path: str = field(default_factory=lambda: os.getenv('BACKUP_PATH', './data/backups'))
    
    # Equipment and amenity types 
    equipment_types: List[str] = field(default_factory=lambda: [
        'projector', 'whiteboard', 'tv', 'microphone', 'camera', 'computer', 'phone', 'speakers', 'screen', 'flip_chart'
    ])
    amenity_types: List[str] = field(default_factory=lambda: [
        'wifi', 'ac', 'heating', 'natural_light', 'quiet', 'private', 'accessible', 'parking', 'kitchen', 'printer'
    ])
    
    # Time slot configuration 
    time_slot_start_hour: int = 6
    time_slot_end_hour: int = 22
    time_slot_interval_minutes: int = 30
    
    # Environment-specific overrides
    def __post_init__(self):
        """Post-initialization to handle environment-specific configurations"""
        env = os.getenv('ENVIRONMENT', 'development').lower()
        if env == 'testing': self._apply_testing_config()
        elif env == 'production': self._apply_production_config()
        elif env == 'development': self._apply_development_config()
    
    def _apply_development_config(self):
        """Apply development-specific settings"""
        self.cache_ttl_default = 300
        self.cache_ttl_recommendations = 60
        self.sqlite_memory_limit_mb = 50
        self.max_cache_size_mb = 200
        self.enable_cache_statistics = True
        self.cache_cleanup_interval = 300
        self.auto_backup_enabled = False
        self.mysql_echo = True
        logger.info("Applied development configuration")
    
    def _apply_testing_config(self):
        """Apply testing-specific settings"""
        self.cache_ttl_default = 10
        self.cache_ttl_recommendations = 5
        self.sqlite_memory_limit_mb = 10
        self.max_cache_size_mb = 50
        self.cache_db_path = "./test_data/cache/test_cache.db"
        self.main_db_path = "./test_data/test_recommendations.db"
        self.analytics_db_path = "./test_data/analytics/test_analytics.db"
        self.vector_db_path = "./test_data/cache/test_vector_store.db"
        self.chroma_persist_directory = "./test_data/chroma_db"
        self.auto_backup_enabled = False
        self.enable_cache_statistics = False
        self.cache_cleanup_interval = 60
        self.mysql_echo = False
        logger.info("Applied testing configuration")
    
    def _apply_production_config(self):
        """Apply production-specific settings"""
        self.cache_ttl_default = 1800
        self.sqlite_memory_limit_mb = 200
        self.max_cache_size_mb = 1000
        self.cache_base_path = "/app/data/cache"
        self.main_db_path = "/app/data/recommendations.db"
        self.cache_db_path = "/app/data/cache/cache.db"
        self.analytics_db_path = "/app/data/analytics/analytics.db"
        self.vector_db_path = "/app/data/cache/vector_store.db"
        self.chroma_persist_directory = "/app/data/chroma_db"
        self.backup_path = "/app/data/backups"
        self.auto_backup_enabled = True
        self.enable_cache_statistics = True
        self.cache_cleanup_interval = 1800
        logger.info("Applied production configuration")
    
    def get_database_urls(self) -> Dict[str, str]:
        """Get database URLs for different purposes"""
        return {"main": self.database_url, "cache": f"sqlite:///{self.cache_db_path}",
                "analytics": f"sqlite:///{self.analytics_db_path}", "vector": f"sqlite:///{self.vector_db_path}"}
    
    def get_mysql_engine_kwargs(self) -> Dict[str, Any]:
        """Get MySQL engine configuration parameters"""
        return {"pool_size": self.mysql_pool_size, "max_overflow": self.mysql_max_overflow,
                "pool_timeout": self.mysql_pool_timeout, "pool_recycle": self.mysql_pool_recycle,
                "echo": self.mysql_echo, "pool_pre_ping": True}
    
    def get_table_names(self) -> Dict[str, str]:
        """Get table names for database queries"""
        return {"rooms": self.room_table_name, "entries": self.entry_table_name, "repeats": self.repeat_table_name}
    
    def get_business_hours_config(self) -> Dict[str, Any]:
        """Get business hours configuration"""
        return {"start_hour": self.business_start_hour, "end_hour": self.business_end_hour, "time_slot_minutes": self.time_slot_minutes}
    
    def validate_mysql_connection(self) -> bool:
        """Validate MySQL connection parameters"""
        try:
            from sqlalchemy import create_engine
            engine = create_engine(self.database_url, **self.get_mysql_engine_kwargs())
            with engine.connect() as conn: conn.execute("SELECT 1")
            engine.dispose()
            return True
        except Exception as e:
            logger.error(f"MySQL connection validation failed: {e}")
            return False
    
    def ensure_directories(self):
        """Create necessary directories if they don't exist."""
        paths = [Path(p).parent if p.endswith('.db') or p.endswith('.pkl') else Path(p) for p in [
            self.cache_db_path, self.analytics_db_path, self.vector_db_path, self.vector_db_path_env,
            self.user_embedding_path, self.room_embedding_path, self.clustering_model_path,
            self.cache_base_path, self.main_db_path
        ]]
        if self.chroma_persist_directory: paths.append(Path(self.chroma_persist_directory))
        if self.auto_backup_enabled: paths.append(Path(self.backup_path))
        
        for path in paths: path.mkdir(parents=True, exist_ok=True)
        logger.info("Ensured all necessary directories exist")

    def get_cache_config(self) -> Dict[str, Any]:
        """Get cache-specific configuration as a dictionary."""
        return {"base_path": self.cache_base_path, "sqlite_memory_limit_mb": self.sqlite_memory_limit_mb,
                "use_file_fallback": self.use_file_fallback, "sqlite_cache_size_kb": self.sqlite_cache_size_kb,
                "sqlite_timeout": self.sqlite_timeout, "enable_wal_mode": self.enable_wal_mode,
                "cleanup_interval": self.cache_cleanup_interval, "enable_compression": self.enable_cache_compression,
                "enable_statistics": self.enable_cache_statistics, "auto_vacuum_enabled": self.auto_vacuum_enabled,
                "lru_eviction_enabled": self.lru_eviction_enabled, "max_cache_size_mb": self.max_cache_size_mb,
                "cache_ttl_default": self.cache_ttl_default, "max_size": self.cache_max_size}
    
    def get_ttl_config(self) -> Dict[str, int]:
        """Get TTL configuration for different cache types."""
        return {"default": self.cache_ttl_default, "recommendations": self.cache_ttl_recommendations,
                "user_preferences": self.cache_ttl_user_preferences, "room_features": self.cache_ttl_room_features,
                "recommendation_hours": self.recommendation_cache_ttl_hours, "user_profile_hours": self.user_profile_cache_ttl_hours,
                "room_similarity_hours": self.room_similarity_cache_ttl_hours, "analytics_hours": self.analytics_cache_ttl_hours}
    
    def get_db_paths(self) -> Dict[str, Path]:
        """Get all database file paths."""
        return {"main_db": Path(self.main_db_path), "cache_db": Path(self.cache_db_path),
                "analytics_db": Path(self.analytics_db_path), "vector_db": Path(self.vector_db_path),
                "chroma_db": Path(self.chroma_persist_directory) if self.chroma_persist_directory else None}
    
    def get_feature_config(self) -> Dict[str, Any]:
        """Get feature extraction configuration."""
        return {"user_feature_dimension": self.user_feature_dimension, "room_feature_dimension": self.room_feature_dimension,
                "equipment_types": self.equipment_types, "amenity_types": self.amenity_types,
                "time_slot_config": {"start_hour": self.time_slot_start_hour, "end_hour": self.time_slot_end_hour,
                                   "interval_minutes": self.time_slot_interval_minutes}}
    
    def get_vector_config(self) -> Dict[str, Any]:
        """Get vector store configuration."""
        return {"db_path": self.vector_db_path, "embedding_model": self.embedding_model_name,
                "embedding_dimension": self.embedding_dimension, "similarity_threshold": self.similarity_threshold,
                "max_search_results": self.max_search_results, "index_refresh_interval": self.vector_index_refresh_interval}
    
    def get_strategy_config(self, strategy_name: str) -> Dict[str, Any]:
        """Get configuration for a specific recommendation strategy."""
        configs = {'alternative_room': self.alt_room_config, 'alternative_time': self.alt_time_config,
                  'collaborative_filtering': self.collaborative_config, 'content_based': self.content_based_config}
        return configs.get(strategy_name, {})
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return list of warnings/errors."""
        warnings = []
        if self.sqlite_memory_limit_mb > self.max_cache_size_mb:
            warnings.append(f"sqlite_memory_limit_mb ({self.sqlite_memory_limit_mb}) exceeds max_cache_size_mb ({self.max_cache_size_mb})")
        if self.cache_ttl_default > 3600:
            warnings.append(f"cache_ttl_default ({self.cache_ttl_default}s) is quite high, consider lowering for more responsive caching")
        
        weight_sum = sum(self.strategy_weights.values())
        if abs(weight_sum - 1.0) > 0.01:
            warnings.append(f"Strategy weights sum to {weight_sum:.3f}, should sum to 1.0")
        if not Path(self.cache_base_path).is_absolute():
            warnings.append(f"cache_base_path ({self.cache_base_path}) is relative, consider using absolute path for production")
        if self.cache_cleanup_interval < 300:
            warnings.append(f"cache_cleanup_interval ({self.cache_cleanup_interval}s) is very frequent, may impact performance")
        if self.user_feature_dimension < 50:
            warnings.append(f"user_feature_dimension ({self.user_feature_dimension}) might be too small for complex user patterns")
        if self.room_feature_dimension < 50:
            warnings.append(f"room_feature_dimension ({self.room_feature_dimension}) might be too small for complex room features")
        return warnings
    
    def to_dict(self) -> Dict[str, Any]:
        config_dict = {}
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            config_dict[field_name] = str(value) if isinstance(value, Path) else value
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'RecommendationConfig':
        """Create configuration from dictionary."""
        valid_fields = {field.name for field in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_fields}
        return cls(**filtered_dict)
    
    def __repr__(self) -> str:
        """String representation of the configuration."""
        return (f"RecommendationConfig(max_recommendations={self.max_recommendations}, "
                f"cache_ttl_default={self.cache_ttl_default}, embedding_model={self.embedding_model_name}, "
                f"vector_dimension={self.vector_dimension})")


class DatabaseManager:
    """Manager for handling multiple database connections"""
    
    def __init__(self, config: RecommendationConfig):
        self.config = config
        self._engines = {}
        self._sessions = {}
    
    def get_main_engine(self):
        """Get MySQL engine for main data"""
        if "main" not in self._engines:
            from sqlalchemy import create_engine
            self._engines["main"] = create_engine(self.config.database_url, **self.config.get_mysql_engine_kwargs())
        return self._engines["main"]
    
    def get_cache_engine(self):
        """Get SQLite engine for caching"""
        if "cache" not in self._engines:
            from sqlalchemy import create_engine
            cache_url = f"sqlite:///{self.config.cache_db_path}"
            self._engines["cache"] = create_engine(cache_url, connect_args={"timeout": self.config.sqlite_timeout})
        return self._engines["cache"]
    
    def get_main_session(self):
        """Get session for main MySQL database"""
        from sqlalchemy.orm import sessionmaker
        if "main" not in self._sessions:
            Session = sessionmaker(bind=self.get_main_engine())
            self._sessions["main"] = Session()
        return self._sessions["main"]
    
    def close_all(self):
        """Close all database connections"""
        for session in self._sessions.values(): session.close()
        for engine in self._engines.values(): engine.dispose()
        self._sessions.clear()
        self._engines.clear()


class ConfigFactory:
    
    @staticmethod
    def create_config(environment: str = None) -> RecommendationConfig:
        """Create configuration for specified environment."""
        if environment is None: environment = os.getenv('ENVIRONMENT', 'development')
        
        original_env = os.getenv('ENVIRONMENT')
        os.environ['ENVIRONMENT'] = environment.lower()
        
        try: return RecommendationConfig()
        finally:
            if original_env is not None: os.environ['ENVIRONMENT'] = original_env
            else: os.environ.pop('ENVIRONMENT', None)
    
    @staticmethod
    def development() -> RecommendationConfig: return ConfigFactory.create_config('development')
    
    @staticmethod
    def testing() -> RecommendationConfig: return ConfigFactory.create_config('testing')
    
    @staticmethod
    def production() -> RecommendationConfig: return ConfigFactory.create_config('production')


# Example usage and validation
if __name__ == "__main__":
    config = RecommendationConfig()
    print("Database URLs:")
    for name, url in config.get_database_urls().items(): print(f"  {name}: {url}")
    
    print(f"\nTable Names: {config.get_table_names()}")
    print(f"Business Hours: {config.get_business_hours_config()}")
    
    print("✓ MySQL connection successful" if config.validate_mysql_connection() else "✗ MySQL connection failed")
    
    db_manager = DatabaseManager(config)
    try:
        main_session = db_manager.get_main_session()
        print("✓ Database manager initialized successfully")
    except Exception as e:
        print(f"✗ Database manager failed: {e}")
    finally:
        db_manager.close_all()
   
    config.ensure_directories()
    
    warnings = config.validate_config()
    if warnings:
        print("Configuration warnings:")
        for warning in warnings: print(f"  - {warning}")
    else:
        print("Configuration validated successfully!")
    
    dev_config = ConfigFactory.development()
    prod_config = ConfigFactory.production()
    test_config = ConfigFactory.testing()
    
    for strategy in config.strategy_weights.keys():
        strategy_config = config.get_strategy_config(strategy)

@dataclass
class Config:
    # Database
    db_url: str = field(default_factory=lambda: os.getenv('DATABASE_URL'))
    cache_ttl: int = 1800
    max_recommendations: int = 10
    
    # Strategy weights
    weights: Dict[str, float] = field(default_factory=lambda: {
        'alt_room': 0.4, 'alt_time': 0.3, 'collaborative': 0.2, 'content': 0.1
    })
    
    # Paths
    cache_path: str = './data/cache.db'
    vector_path: str = './data/vectors.db'