import os
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()

@dataclass
class RecommendationConfig:
    database_url: str = field(default_factory=lambda: os.getenv('DATABASE_URL'))
    
    cache_db_path: str = field(default_factory=lambda: os.getenv('CACHE_DB_PATH', './data/cache/recommendations_cache.db'))
    vector_db_path: str = field(default_factory=lambda: os.getenv('VECTOR_DB_PATH', './data/embeddings'))
    analytics_db_path: str = field(default_factory=lambda: os.getenv('ANALYTICS_DB_PATH', './data/analytics/analytics.db'))
    
    cache_ttl_default: int = 1800
    cache_ttl_recommendations: int = 300
    max_recommendations: int = field(default_factory=lambda: int(os.getenv('MAX_ALTERNATIVES', '10')))
    similarity_threshold: float = field(default_factory=lambda: float(os.getenv('SIMILARITY_THRESHOLD', '0.7')))
    min_confidence_score: float = field(default_factory=lambda: float(os.getenv('CONFIDENCE_THRESHOLD', '0.5')))
    
    embedding_model_name: str = field(default_factory=lambda: os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2'))
    
    business_start_hour: int = field(default_factory=lambda: int(os.getenv('BUSINESS_START_HOUR', '7')))
    business_end_hour: int = field(default_factory=lambda: int(os.getenv('BUSINESS_END_HOUR', '21')))
    time_slot_minutes: int = field(default_factory=lambda: int(os.getenv('TIME_SLOT_MINUTES', '30')))
    
    analytics_window_days: int = field(default_factory=lambda: int(os.getenv('ANALYTICS_WINDOW_DAYS', '30')))
    min_booking_history: int = field(default_factory=lambda: int(os.getenv('MIN_BOOKINGS_FOR_PATTERN', '3')))
    
    clustering_model_path: str = field(default_factory=lambda: os.getenv('CLUSTERING_MODEL_PATH', './data/models/clustering_model.pkl'))
    user_embedding_path: str = field(default_factory=lambda: os.getenv('USER_EMBEDDING_PATH', './data/embeddings/users'))
    room_embedding_path: str = field(default_factory=lambda: os.getenv('ROOM_EMBEDDING_PATH', './data/embeddings/rooms'))
    
    def ensure_directories(self):
        paths = [
            Path(self.cache_db_path).parent,
            Path(self.analytics_db_path).parent,
            Path(self.vector_db_path),
            Path(self.clustering_model_path).parent,
            Path(self.user_embedding_path),
            Path(self.room_embedding_path)
        ]
        for path in paths:
            path.mkdir(parents=True, exist_ok=True)
    
    def get_strategy_weights(self) -> Dict[str, float]:
        return {
            'alternative_room': 0.4,
            'alternative_time': 0.3,
            'smart_scheduling': 0.2,
            'proactive': 0.1
        }

_config_instance = None

def get_recommendation_config() -> RecommendationConfig:
    global _config_instance
    if _config_instance is None:
        _config_instance = RecommendationConfig()
        _config_instance.ensure_directories()
    return _config_instance