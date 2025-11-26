import json
import pickle
import sqlite3
import logging
import threading
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import numpy as np
from contextlib import contextmanager
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class CacheKeyType(Enum):
    USER_PREFERENCES = "user_pref"
    ROOM_SIMILARITIES = "room_sim"
    BOOKING_PATTERNS = "booking_pat"
    RECOMMENDATIONS = "recommendations"
    ANALYTICS = "analytics"
    EMBEDDINGS = "embeddings"
    ML_MODELS = "ml_models"
    AVAILABILITY = "availability"
    TIME_SERIES = "time_series"
    COLLABORATIVE_FILTER = "collab_filter"

@dataclass
class CacheConfig:
    database_path: str = "cache.db"
    max_connections: int = 10
    timeout: float = 30.0
    check_same_thread: bool = False
    isolation_level: Optional[str] = None
    
    # TTL settings
    default_ttl: int = 3600
    user_preferences_ttl: int = 86400
    room_similarities_ttl: int = 43200
    recommendations_ttl: int = 1800
    analytics_ttl: int = 7200
    availability_ttl: int = 300
    embeddings_ttl: int = 604800
    ml_models_ttl: int = 259200
    
    # Maintenance
    cleanup_interval: int = 3600
    vacuum_interval: int = 86400

class CacheManager:
    def __init__(self, config: CacheConfig = None):
        self.config = config or CacheConfig()
        self._local = threading.local()
        self._lock = threading.RLock()
        self._last_cleanup = datetime.now()
        self._last_vacuum = datetime.now()
        self._initialize_database()
        
    def _get_connection(self) -> sqlite3.Connection:
        if not hasattr(self._local, 'connection'):
            try:
                Path(self.config.database_path).parent.mkdir(parents=True, exist_ok=True)
                
                conn = sqlite3.connect(
                    self.config.database_path,
                    timeout=self.config.timeout,
                    check_same_thread=self.config.check_same_thread,
                    isolation_level=self.config.isolation_level
                )
                
                for pragma in ["PRAGMA journal_mode=WAL", "PRAGMA synchronous=NORMAL", 
                             "PRAGMA cache_size=10000", "PRAGMA temp_store=MEMORY"]:
                    conn.execute(pragma)
                
                self._local.connection = conn
                
            except sqlite3.Error as e:
                logger.error(f"Failed to create SQLite connection: {e}")
                raise
                
        return self._local.connection
        
    def _initialize_database(self):
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cache_entries (
                    cache_key TEXT PRIMARY KEY,
                    key_type TEXT NOT NULL,
                    value_data BLOB NOT NULL,
                    value_type TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            for idx in [("idx_key_type", "key_type"), ("idx_expires_at", "expires_at"), ("idx_last_accessed", "last_accessed")]:
                cursor.execute(f"CREATE INDEX IF NOT EXISTS {idx[0]} ON cache_entries({idx[1]})")
            
            conn.commit()
            logger.info("SQLite cache database initialized successfully")
            
        except sqlite3.Error as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    @contextmanager
    def _handle_db_errors(self):
        try:
            yield
        except sqlite3.Error as e:
            logger.error(f"SQLite operation failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in cache operation: {e}")
            raise
    
    def _generate_cache_key(self, key_type: CacheKeyType, *args, **kwargs) -> str:
        key_parts = [key_type.value]
        
        for arg in args:
            if isinstance(arg, (dict, list)):
                key_parts.append(hashlib.md5(json.dumps(arg, sort_keys=True).encode()).hexdigest()[:8])
            else:
                key_parts.append(str(arg))
        
        if kwargs:
            key_parts.append(hashlib.md5(json.dumps(kwargs, sort_keys=True).encode()).hexdigest()[:8])
        
        return ":".join(key_parts)
    
    def _get_ttl_for_key_type(self, key_type: CacheKeyType) -> int:
        ttl_mapping = {
            CacheKeyType.USER_PREFERENCES: self.config.user_preferences_ttl,
            CacheKeyType.ROOM_SIMILARITIES: self.config.room_similarities_ttl,
            CacheKeyType.RECOMMENDATIONS: self.config.recommendations_ttl,
            CacheKeyType.ANALYTICS: self.config.analytics_ttl,
            CacheKeyType.AVAILABILITY: self.config.availability_ttl,
            CacheKeyType.EMBEDDINGS: self.config.embeddings_ttl,
            CacheKeyType.ML_MODELS: self.config.ml_models_ttl,
        }
        return ttl_mapping.get(key_type, self.config.default_ttl)
    
    def _serialize_value(self, value: Any) -> Tuple[bytes, str]:
        if isinstance(value, (dict, list)):
            return json.dumps(value, default=str).encode('utf-8'), 'json'
        elif isinstance(value, np.ndarray):
            return pickle.dumps(value), 'pickle'
        elif isinstance(value, (int, float)):
            return str(value).encode('utf-8'), 'numeric'
        elif isinstance(value, str):
            return value.encode('utf-8'), 'string'
        elif isinstance(value, bool):
            return str(value).encode('utf-8'), 'boolean'
        else:
            return pickle.dumps(value), 'pickle'
    
    def _deserialize_value(self, data: bytes, value_type: str) -> Any:
        if value_type == 'json':
            return json.loads(data.decode('utf-8'))
        elif value_type == 'pickle':
            return pickle.loads(data)
        elif value_type == 'numeric':
            value_str = data.decode('utf-8')
            return int(value_str) if '.' not in value_str else float(value_str)
        elif value_type == 'string':
            return data.decode('utf-8')
        elif value_type == 'boolean':
            return data.decode('utf-8') == 'True'
        else:
            return pickle.loads(data)
    
    def _cleanup_expired_entries(self):
        if (datetime.now() - self._last_cleanup).seconds < self.config.cleanup_interval:
            return
            
        try:
            with self._handle_db_errors():
                conn = self._get_connection()
                cursor = conn.cursor()
                cursor.execute("DELETE FROM cache_entries WHERE expires_at < CURRENT_TIMESTAMP")
                deleted_count = cursor.rowcount
                conn.commit()
                
                if deleted_count > 0:
                    logger.debug(f"Cleaned up {deleted_count} expired cache entries")
                
                self._last_cleanup = datetime.now()
                
        except Exception as e:
            logger.error(f"Failed to cleanup expired entries: {e}")
    
    def _vacuum_database(self):
        if (datetime.now() - self._last_vacuum).seconds < self.config.vacuum_interval:
            return
            
        try:
            with self._handle_db_errors():
                self._get_connection().execute("VACUUM")
                logger.debug("Database vacuumed successfully")
                self._last_vacuum = datetime.now()
        except Exception as e:
            logger.error(f"Failed to vacuum database: {e}")
    
    def set(self, key_type: CacheKeyType, value: Any, ttl: Optional[int] = None, *args, **kwargs) -> bool:
        try:
            with self._handle_db_errors():
                cache_key = self._generate_cache_key(key_type, *args, **kwargs)
                ttl = ttl or self._get_ttl_for_key_type(key_type)
                
                value_data, value_type = self._serialize_value(value)
                expires_at = datetime.now() + timedelta(seconds=ttl)
                
                conn = self._get_connection()
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO cache_entries 
                    (cache_key, key_type, value_data, value_type, expires_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (cache_key, key_type.value, value_data, value_type, expires_at))
                
                conn.commit()
                logger.debug(f"Cached key: {cache_key} with TTL: {ttl}")
                self._cleanup_expired_entries()
                return True
                
        except Exception as e:
            logger.error(f"Failed to set cache key {key_type.value}: {e}")
            return False
    
    def get(self, key_type: CacheKeyType, default: Any = None, *args, **kwargs) -> Any:
        try:
            with self._handle_db_errors():
                cache_key = self._generate_cache_key(key_type, *args, **kwargs)
                
                conn = self._get_connection()
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT value_data, value_type, expires_at 
                    FROM cache_entries 
                    WHERE cache_key = ? AND expires_at > CURRENT_TIMESTAMP
                """, (cache_key,))
                
                result = cursor.fetchone()
                if result is None:
                    return default
                
                value_data, value_type, expires_at = result
                
                cursor.execute("""
                    UPDATE cache_entries 
                    SET access_count = access_count + 1, last_accessed = CURRENT_TIMESTAMP
                    WHERE cache_key = ?
                """, (cache_key,))
                
                conn.commit()
                return self._deserialize_value(value_data, value_type)
                        
        except Exception as e:
            logger.error(f"Failed to get cache key {key_type.value}: {e}")
            return default
    
    def delete(self, key_type: CacheKeyType, *args, **kwargs) -> bool:
        try:
            with self._handle_db_errors():
                cache_key = self._generate_cache_key(key_type, *args, **kwargs)
                conn = self._get_connection()
                cursor = conn.cursor()
                cursor.execute("DELETE FROM cache_entries WHERE cache_key = ?", (cache_key,))
                deleted = cursor.rowcount > 0
                conn.commit()
                
                if deleted:
                    logger.debug(f"Deleted cache key: {cache_key}")
                return deleted
                
        except Exception as e:
            logger.error(f"Failed to delete cache key {key_type.value}: {e}")
            return False
    
    def exists(self, key_type: CacheKeyType, *args, **kwargs) -> bool:
        try:
            with self._handle_db_errors():
                cache_key = self._generate_cache_key(key_type, *args, **kwargs)
                conn = self._get_connection()
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT 1 FROM cache_entries 
                    WHERE cache_key = ? AND expires_at > CURRENT_TIMESTAMP
                """, (cache_key,))
                return cursor.fetchone() is not None
        except Exception as e:
            logger.error(f"Failed to check cache key existence {key_type.value}: {e}")
            return False
    
    def increment(self, key_type: CacheKeyType, amount: int = 1, *args, **kwargs) -> Optional[int]:
        try:
            with self._handle_db_errors():
                with self._lock:
                    current_value = self.get(key_type, 0, *args, **kwargs)
                    if not isinstance(current_value, (int, float)):
                        current_value = 0
                    
                    new_value = int(current_value) + amount
                    return new_value if self.set(key_type, new_value, None, *args, **kwargs) else None
        except Exception as e:
            logger.error(f"Failed to increment cache key {key_type.value}: {e}")
            return None
    
    def set_multiple(self, key_type: CacheKeyType, mapping: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        try:
            with self._handle_db_errors():
                ttl = ttl or self._get_ttl_for_key_type(key_type)
                expires_at = datetime.now() + timedelta(seconds=ttl)
                
                conn = self._get_connection()
                cursor = conn.cursor()
                
                batch_data = []
                for suffix, value in mapping.items():
                    cache_key = self._generate_cache_key(key_type, suffix)
                    value_data, value_type = self._serialize_value(value)
                    batch_data.append((cache_key, key_type.value, value_data, value_type, expires_at))
                
                cursor.executemany("""
                    INSERT OR REPLACE INTO cache_entries 
                    (cache_key, key_type, value_data, value_type, expires_at)
                    VALUES (?, ?, ?, ?, ?)
                """, batch_data)
                
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to set multiple cache keys {key_type.value}: {e}")
            return False
    
    def get_multiple(self, key_type: CacheKeyType, suffixes: List[str]) -> Dict[str, Any]:
        try:
            with self._handle_db_errors():
                cache_keys = [self._generate_cache_key(key_type, suffix) for suffix in suffixes]
                conn = self._get_connection()
                cursor = conn.cursor()
                
                placeholders = ','.join('?' * len(cache_keys))
                cursor.execute(f"""
                    SELECT cache_key, value_data, value_type 
                    FROM cache_entries 
                    WHERE cache_key IN ({placeholders}) 
                    AND expires_at > CURRENT_TIMESTAMP
                """, cache_keys)
                
                results = cursor.fetchall()
                key_to_suffix = {self._generate_cache_key(key_type, suffix): suffix for suffix in suffixes}
                
                result = {}
                for cache_key, value_data, value_type in results:
                    if cache_key in key_to_suffix:
                        suffix = key_to_suffix[cache_key]
                        result[suffix] = self._deserialize_value(value_data, value_type)
                
                return result
        except Exception as e:
            logger.error(f"Failed to get multiple cache keys {key_type.value}: {e}")
            return {}
    
    def flush_by_pattern(self, pattern: str) -> int:
        try:
            with self._handle_db_errors():
                sql_pattern = pattern.replace('*', '%').replace('?', '_')
                conn = self._get_connection()
                cursor = conn.cursor()
                cursor.execute("DELETE FROM cache_entries WHERE cache_key LIKE ?", (sql_pattern,))
                deleted = cursor.rowcount
                conn.commit()
                
                if deleted > 0:
                    logger.info(f"Deleted {deleted} keys matching pattern: {pattern}")
                return deleted
        except Exception as e:
            logger.error(f"Failed to flush keys by pattern {pattern}: {e}")
            return 0
    
    def flush_by_key_type(self, key_type: CacheKeyType) -> int:
        try:
            with self._handle_db_errors():
                conn = self._get_connection()
                cursor = conn.cursor()
                cursor.execute("DELETE FROM cache_entries WHERE key_type = ?", (key_type.value,))
                deleted = cursor.rowcount
                conn.commit()
                
                if deleted > 0:
                    logger.info(f"Deleted {deleted} keys of type: {key_type.value}")
                return deleted
        except Exception as e:
            logger.error(f"Failed to flush keys by type {key_type.value}: {e}")
            return 0
    
    def get_cache_stats(self) -> Dict[str, Any]:
        try:
            with self._handle_db_errors():
                conn = self._get_connection()
                cursor = conn.cursor()
                
                cursor.execute("SELECT COUNT(*) FROM cache_entries")
                total_entries = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM cache_entries WHERE expires_at <= CURRENT_TIMESTAMP")
                expired_entries = cursor.fetchone()[0]
                
                cursor.execute("""
                    SELECT key_type, COUNT(*) 
                    FROM cache_entries 
                    WHERE expires_at > CURRENT_TIMESTAMP
                    GROUP BY key_type
                """)
                entries_by_type = dict(cursor.fetchall())
                
                cursor.execute("SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()")
                db_size = cursor.fetchone()[0]
                
                cursor.execute("SELECT AVG(access_count) FROM cache_entries WHERE access_count > 0")
                avg_access = cursor.fetchone()[0] or 0
                
                return {
                    'total_entries': total_entries,
                    'active_entries': total_entries - expired_entries,
                    'expired_entries': expired_entries,
                    'entries_by_type': entries_by_type,
                    'database_size_bytes': db_size,
                    'database_size_mb': round(db_size / (1024 * 1024), 2),
                    'average_access_count': round(avg_access, 2)
                }
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {}
    
    def health_check(self) -> bool:
        try:
            cursor = self._get_connection().cursor()
            cursor.execute("SELECT 1")
            return cursor.fetchone()[0] == 1
        except Exception as e:
            logger.error(f"SQLite health check failed: {e}")
            return False
    
    def optimize_database(self):
        try:
            with self._handle_db_errors():
                conn = self._get_connection()
                self._cleanup_expired_entries()
                conn.execute("ANALYZE")
                self._vacuum_database()
                logger.info("Database optimization completed")
        except Exception as e:
            logger.error(f"Failed to optimize database: {e}")
    
    def close(self):
        try:
            if hasattr(self._local, 'connection'):
                self._local.connection.close()
                delattr(self._local, 'connection')
            logger.info("SQLite connections closed")
        except Exception as e:
            logger.error(f"Error closing SQLite connection: {e}")

class RecommendationCacheManager(CacheManager):
    def cache_user_preferences(self, user_id: str, preferences: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        return self.set(CacheKeyType.USER_PREFERENCES, preferences, ttl, user_id)
    
    def get_user_preferences(self, user_id: str) -> Optional[Dict[str, Any]]:
        return self.get(CacheKeyType.USER_PREFERENCES, None, user_id)
    
    def cache_room_similarities(self, room_id: str, similarities: Dict[str, float], ttl: Optional[int] = None) -> bool:
        return self.set(CacheKeyType.ROOM_SIMILARITIES, similarities, ttl, room_id)
    
    def get_room_similarities(self, room_id: str) -> Optional[Dict[str, float]]:
        return self.get(CacheKeyType.ROOM_SIMILARITIES, None, room_id)
    
    def cache_recommendations(self, user_id: str, context: str, recommendations: List[Dict], ttl: Optional[int] = None) -> bool:
        return self.set(CacheKeyType.RECOMMENDATIONS, recommendations, ttl, user_id, context)
    
    def get_recommendations(self, user_id: str, context: str) -> Optional[List[Dict]]:
        return self.get(CacheKeyType.RECOMMENDATIONS, None, user_id, context)
    
    def cache_booking_patterns(self, pattern_key: str, patterns: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        return self.set(CacheKeyType.BOOKING_PATTERNS, patterns, ttl, pattern_key)
    
    def get_booking_patterns(self, pattern_key: str) -> Optional[Dict[str, Any]]:
        return self.get(CacheKeyType.BOOKING_PATTERNS, None, pattern_key)
    
    def cache_room_availability(self, room_id: str, date: str, availability: List[Dict], ttl: Optional[int] = None) -> bool:
        return self.set(CacheKeyType.AVAILABILITY, availability, ttl, room_id, date)
    
    def get_room_availability(self, room_id: str, date: str) -> Optional[List[Dict]]:
        return self.get(CacheKeyType.AVAILABILITY, None, room_id, date)
    
    def invalidate_user_cache(self, user_id: str) -> int:
        patterns = [
            f"{CacheKeyType.USER_PREFERENCES.value}:{user_id}:%",
            f"{CacheKeyType.RECOMMENDATIONS.value}:{user_id}:%"
        ]
        return sum(self.flush_by_pattern(pattern) for pattern in patterns)
    
    def invalidate_room_cache(self, room_id: str) -> int:
        patterns = [
            f"{CacheKeyType.ROOM_SIMILARITIES.value}:{room_id}:%",
            f"{CacheKeyType.AVAILABILITY.value}:{room_id}:%"
        ]
        return sum(self.flush_by_pattern(pattern) for pattern in patterns)

_cache_manager_instance = None

def get_cache_manager(config: CacheConfig = None) -> RecommendationCacheManager:
    global _cache_manager_instance
    
    if _cache_manager_instance is None or not _cache_manager_instance.health_check():
        _cache_manager_instance = RecommendationCacheManager(config)
    
    return _cache_manager_instance