# data/__init__.py

from .embeddings import EmbeddingManager
from .models import ModelManager, ModelMetadata
from .cache import CacheManager, CacheEntry
from .analytics import AnalyticsManager, BookingEvent, RecommendationEvent

__all__ = [
    'EmbeddingManager',
    'ModelManager', 'ModelMetadata',
    'CacheManager', 'CacheEntry',
    'AnalyticsManager', 'BookingEvent', 'RecommendationEvent'
]

# Version information
__version__ = "1.0.0"

# Configuration constants
DEFAULT_EMBEDDING_DIMENSIONS = 384
DEFAULT_CACHE_TTL_HOURS = 24
DEFAULT_MODEL_RETENTION_DAYS = 90
DEFAULT_ANALYTICS_RETENTION_DAYS = 365

