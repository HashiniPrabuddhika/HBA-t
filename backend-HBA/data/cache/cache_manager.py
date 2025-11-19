import json
import pickle
import hashlib
import gzip
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import logging
import sqlite3
from dataclasses import dataclass
import threading
import time

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    cache_key: str
    user_id: int
    request_type: str
    created_at: str
    expires_at: str
    hit_count: int
    data_size_bytes: int
    compression_ratio: float

class CacheManager:
    def __init__(self, base_path: str = "data/cache", sqlite_memory_limit_mb: int = 100, use_file_fallback: bool = True):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.sqlite_memory_limit = sqlite_memory_limit_mb * 1024 * 1024
        self.use_file_fallback = use_file_fallback
        
        if use_file_fallback:
            for path in ["recommendations", "user_profiles", "room_similarities", "analytics"]:
                (self.base_path / path).mkdir(exist_ok=True)
                setattr(self, f"{path}_path", self.base_path / path)
        
        self._init_sqlite_cache()
        self._start_cleanup_thread()
    
    def _init_sqlite_cache(self):
        self.conn = sqlite3.connect(str(self.base_path / "cache.db"), check_same_thread=False, timeout=30.0)
        
        for pragma in ["journal_mode=WAL", "synchronous=NORMAL", "cache_size=10000", "temp_store=MEMORY"]:
            self.conn.execute(f"PRAGMA {pragma}")
        
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS cache_data (
                cache_key TEXT PRIMARY KEY, data BLOB, compressed_size INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
            
            CREATE TABLE IF NOT EXISTS cache_entries (
                cache_key TEXT PRIMARY KEY, user_id INTEGER, request_type TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, expires_at TIMESTAMP,
                hit_count INTEGER DEFAULT 0, data_size_bytes INTEGER, compression_ratio REAL,
                storage_type TEXT DEFAULT 'sqlite', last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
            
            CREATE TABLE IF NOT EXISTS cache_stats (
                date TEXT PRIMARY KEY, total_requests INTEGER DEFAULT 0, cache_hits INTEGER DEFAULT 0,
                cache_misses INTEGER DEFAULT 0, evictions INTEGER DEFAULT 0, storage_used_mb REAL DEFAULT 0);
            
            CREATE INDEX IF NOT EXISTS idx_user_id ON cache_entries(user_id);
            CREATE INDEX IF NOT EXISTS idx_request_type ON cache_entries(request_type);
            CREATE INDEX IF NOT EXISTS idx_expires_at ON cache_entries(expires_at);
            CREATE INDEX IF NOT EXISTS idx_last_accessed ON cache_entries(last_accessed);
            CREATE INDEX IF NOT EXISTS idx_storage_type ON cache_entries(storage_type);
        """)
        self.conn.commit()
        self._db_lock = threading.Lock()
    
    def _generate_cache_key(self, user_id: int, request_type: str, params: Dict[str, Any]) -> str:
        key_string = f"{user_id}:{request_type}:{json.dumps(params, sort_keys=True)}"
        return hashlib.sha256(key_string.encode()).hexdigest()
    
    def _compress_data(self, data: Any) -> Tuple[bytes, float]:
        original_data = pickle.dumps(data)
        compressed_data = gzip.compress(original_data)
        return compressed_data, len(compressed_data) / len(original_data)
    
    def _decompress_data(self, compressed_data: bytes) -> Any:
        return pickle.loads(gzip.decompress(compressed_data))
    
    def _should_store_in_file(self, data_size: int) -> bool:
        return self.use_file_fallback and data_size > (self.sqlite_memory_limit // 10)
    
    def _store_data(self, cache_key: str, compressed_data: bytes, category: str) -> str:
        storage_type = "sqlite"
        
        if self._should_store_in_file(len(compressed_data)):
            self._store_in_file(cache_key, compressed_data, category)
            storage_type = "file"
        else:
            with self._db_lock:
                try:
                    self.conn.execute("INSERT OR REPLACE INTO cache_data VALUES (?, ?, ?, CURRENT_TIMESTAMP)", 
                                    (cache_key, compressed_data, len(compressed_data)))
                    self.conn.commit()
                except sqlite3.Error as e:
                    logger.warning(f"SQLite storage failed, falling back to file: {e}")
                    if self.use_file_fallback:
                        self._store_in_file(cache_key, compressed_data, category)
                        storage_type = "file"
                    else:
                        raise
        return storage_type
    
    def _retrieve_data(self, cache_key: str, storage_type: str, category: str) -> Optional[bytes]:
        if storage_type == "sqlite":
            with self._db_lock:
                cursor = self.conn.execute("SELECT data FROM cache_data WHERE cache_key = ?", (cache_key,))
                result = cursor.fetchone()
                return result[0] if result else None
        else:
            return self._load_from_file(cache_key, category)
    
    def _store_in_file(self, cache_key: str, data: bytes, category: str):
        file_path = getattr(self, f"{category}_path", self.base_path / "analytics") / f"{cache_key}.gz"
        with open(file_path, 'wb') as f:
            f.write(data)
    
    def _load_from_file(self, cache_key: str, category: str) -> Optional[bytes]:
        file_path = getattr(self, f"{category}_path", self.base_path / "analytics") / f"{cache_key}.gz"
        if file_path.exists():
            with open(file_path, 'rb') as f:
                return f.read()
        return None
    
    def set_recommendation(self, user_id: int, request_type: str, params: Dict[str, Any], data: Any, ttl_hours: int = 24) -> str:
        try:
            cache_key = self._generate_cache_key(user_id, request_type, params)
            compressed_data, compression_ratio = self._compress_data(data)
            expires_at = datetime.now() + timedelta(hours=ttl_hours)
            
            storage_type = self._store_data(cache_key, compressed_data, "recommendations")
            
            with self._db_lock:
                self.conn.execute("""
                    INSERT OR REPLACE INTO cache_entries VALUES (?, ?, ?, CURRENT_TIMESTAMP, ?, 0, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (cache_key, user_id, request_type, expires_at, len(compressed_data), compression_ratio, storage_type))
                self.conn.commit()
            
            logger.debug(f"Cached recommendation for user {user_id}, key: {cache_key[:8]}... ({storage_type})")
            return cache_key
        except Exception as e:
            logger.error(f"Error caching recommendation: {e}")
            return ""
    
    def get_recommendation(self, user_id: int, request_type: str, params: Dict[str, Any]) -> Optional[Any]:
        try:
            cache_key = self._generate_cache_key(user_id, request_type, params)
            
            with self._db_lock:
                cursor = self.conn.execute("""
                    SELECT storage_type, expires_at FROM cache_entries WHERE cache_key = ? AND expires_at > ?
                """, (cache_key, datetime.now()))
                
                result = cursor.fetchone()
                if not result:
                    self._record_cache_miss()
                    return None
                
                storage_type, expires_at = result
            
            compressed_data = self._retrieve_data(cache_key, storage_type, "recommendations")
            
            if compressed_data is None:
                self._record_cache_miss()
                return None
            
            data = self._decompress_data(compressed_data)
            
            with self._db_lock:
                self.conn.execute("UPDATE cache_entries SET hit_count = hit_count + 1, last_accessed = ? WHERE cache_key = ?", 
                                (datetime.now(), cache_key))
                self.conn.commit()
            
            self._record_cache_hit()
            logger.debug(f"Cache hit for user {user_id}, key: {cache_key[:8]}...")
            return data
        except Exception as e:
            logger.error(f"Error retrieving cached recommendation: {e}")
            self._record_cache_miss()
            return None
    
    def cache_user_profile(self, user_id: int, profile_data: Dict[str, Any], ttl_hours: int = 168) -> str:
        cache_key = f"user_profile_{user_id}"
        compressed_data, compression_ratio = self._compress_data(profile_data)
        
        storage_type = self._store_data(cache_key, compressed_data, "user_profiles")
        expires_at = datetime.now() + timedelta(hours=ttl_hours)
        
        with self._db_lock:
            self.conn.execute("""
                INSERT OR REPLACE INTO cache_entries VALUES (?, ?, 'user_profile', CURRENT_TIMESTAMP, ?, 0, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (cache_key, user_id, expires_at, len(compressed_data), compression_ratio, storage_type))
            self.conn.commit()
        
        return cache_key
    
    def get_user_profile(self, user_id: int) -> Optional[Dict[str, Any]]:
        cache_key = f"user_profile_{user_id}"
        
        with self._db_lock:
            cursor = self.conn.execute("SELECT storage_type FROM cache_entries WHERE cache_key = ? AND expires_at > ?", 
                                     (cache_key, datetime.now()))
            result = cursor.fetchone()
            if not result:
                return None
            storage_type = result[0]
        
        compressed_data = self._retrieve_data(cache_key, storage_type, "user_profiles")
        
        if compressed_data:
            with self._db_lock:
                self.conn.execute("UPDATE cache_entries SET last_accessed = ? WHERE cache_key = ?", 
                                (datetime.now(), cache_key))
                self.conn.commit()
            return self._decompress_data(compressed_data)
        return None
    
    def cache_room_similarities(self, similarities: Dict[int, Dict[int, float]], ttl_hours: int = 72) -> str:
        cache_key = f"room_similarities_{datetime.now().strftime('%Y%m%d')}"
        compressed_data, compression_ratio = self._compress_data(similarities)
        
        storage_type = self._store_data(cache_key, compressed_data, "room_similarities")
        expires_at = datetime.now() + timedelta(hours=ttl_hours)
        
        with self._db_lock:
            self.conn.execute("""
                INSERT OR REPLACE INTO cache_entries VALUES (?, 0, 'room_similarities', CURRENT_TIMESTAMP, ?, 0, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (cache_key, expires_at, len(compressed_data), compression_ratio, storage_type))
            self.conn.commit()
        
        return cache_key
    
    def get_room_similarities(self) -> Optional[Dict[int, Dict[int, float]]]:
        with self._db_lock:
            cursor = self.conn.execute("""
                SELECT cache_key, storage_type FROM cache_entries 
                WHERE request_type = 'room_similarities' AND expires_at > ?
                ORDER BY created_at DESC LIMIT 1
            """, (datetime.now(),))
            
            result = cursor.fetchone()
            if not result:
                return None
            cache_key, storage_type = result
        
        compressed_data = self._retrieve_data(cache_key, storage_type, "room_similarities")
        
        if compressed_data:
            with self._db_lock:
                self.conn.execute("UPDATE cache_entries SET last_accessed = ? WHERE cache_key = ?", 
                                (datetime.now(), cache_key))
                self.conn.commit()
            return self._decompress_data(compressed_data)
        return None
    
    def invalidate_user_cache(self, user_id: int):
        try:
            with self._db_lock:
                cursor = self.conn.execute("SELECT cache_key, storage_type FROM cache_entries WHERE user_id = ?", (user_id,))
                entries_to_delete = cursor.fetchall()
                
                for cache_key, storage_type in entries_to_delete:
                    if storage_type == "sqlite":
                        self.conn.execute("DELETE FROM cache_data WHERE cache_key = ?", (cache_key,))
                    
                    if self.use_file_fallback:
                        for category in ["recommendations", "user_profiles"]:
                            file_path = getattr(self, f"{category}_path", None)
                            if file_path:
                                (file_path / f"{cache_key}.gz").unlink(missing_ok=True)
                
                self.conn.execute("DELETE FROM cache_entries WHERE user_id = ?", (user_id,))
                self.conn.commit()
            
            logger.info(f"Invalidated cache for user {user_id}")
        except Exception as e:
            logger.error(f"Error invalidating user cache: {e}")
    
    def _record_cache_hit(self):
        today = datetime.now().strftime('%Y-%m-%d')
        with self._db_lock:
            self.conn.execute("INSERT OR IGNORE INTO cache_stats (date) VALUES (?)", (today,))
            self.conn.execute("UPDATE cache_stats SET total_requests = total_requests + 1, cache_hits = cache_hits + 1 WHERE date = ?", (today,))
            self.conn.commit()
    
    def _record_cache_miss(self):
        today = datetime.now().strftime('%Y-%m-%d')
        with self._db_lock:
            self.conn.execute("INSERT OR IGNORE INTO cache_stats (date) VALUES (?)", (today,))
            self.conn.execute("UPDATE cache_stats SET total_requests = total_requests + 1, cache_misses = cache_misses + 1 WHERE date = ?", (today,))
            self.conn.commit()
    
    def _start_cleanup_thread(self):
        def cleanup_worker():
            while True:
                try:
                    time.sleep(3600)
                    self.cleanup_expired_entries()
                    self._vacuum_database()
                except Exception as e:
                    logger.error(f"Error in cleanup thread: {e}")
        
        threading.Thread(target=cleanup_worker, daemon=True).start()
    
    def cleanup_expired_entries(self):
        try:
            with self._db_lock:
                cursor = self.conn.execute("SELECT cache_key, storage_type FROM cache_entries WHERE expires_at < ?", (datetime.now(),))
                expired_entries = cursor.fetchall()
                
                for cache_key, storage_type in expired_entries:
                    if storage_type == "sqlite":
                        self.conn.execute("DELETE FROM cache_data WHERE cache_key = ?", (cache_key,))
                    
                    if self.use_file_fallback:
                        for category in ["recommendations", "user_profiles", "room_similarities", "analytics"]:
                            file_path = getattr(self, f"{category}_path", None)
                            if file_path:
                                (file_path / f"{cache_key}.gz").unlink(missing_ok=True)
                
                self.conn.execute("DELETE FROM cache_entries WHERE expires_at < ?", (datetime.now(),))
                
                if expired_entries:
                    today = datetime.now().strftime('%Y-%m-%d')
                    self.conn.execute("INSERT OR IGNORE INTO cache_stats (date) VALUES (?)", (today,))
                    self.conn.execute("UPDATE cache_stats SET evictions = evictions + ? WHERE date = ?", (len(expired_entries), today))
                
                self.conn.commit()
            
            if expired_entries:
                logger.info(f"Cleaned up {len(expired_entries)} expired cache entries")
        except Exception as e:
            logger.error(f"Error during cache cleanup: {e}")
    
    def _vacuum_database(self):
        try:
            with self._db_lock:
                cursor = self.conn.execute("PRAGMA freelist_count")
                free_pages = cursor.fetchone()[0]
                
                if free_pages > 1000:
                    logger.info("Performing database vacuum...")
                    self.conn.execute("VACUUM")
                    logger.info("Database vacuum completed")
        except Exception as e:
            logger.error(f"Error during database vacuum: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        try:
            with self._db_lock:
                cursor = self.conn.execute("""
                    SELECT COUNT(*), SUM(data_size_bytes), AVG(hit_count), AVG(compression_ratio),
                           SUM(CASE WHEN storage_type = 'sqlite' THEN 1 ELSE 0 END),
                           SUM(CASE WHEN storage_type = 'file' THEN 1 ELSE 0 END)
                    FROM cache_entries WHERE expires_at > ?
                """, (datetime.now(),))
                
                count, total_size, avg_hits, avg_compression, sqlite_count, file_count = cursor.fetchone()
                
                cursor = self.conn.execute("SELECT page_count * page_size FROM pragma_page_count(), pragma_page_size()")
                db_size = cursor.fetchone()[0]
                
                cursor = self.conn.execute("""
                    SELECT SUM(total_requests), SUM(cache_hits), SUM(cache_misses), SUM(evictions)
                    FROM cache_stats WHERE date >= ?
                """, ((datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'),))
                
                total_req, total_hits, total_misses, total_evictions = cursor.fetchone()
                hit_rate = (total_hits / total_req) * 100 if total_req else 0
                
                return {
                    'active_entries': count or 0, 'total_size_mb': (total_size or 0) / (1024 * 1024),
                    'database_size_mb': db_size / (1024 * 1024), 'sqlite_entries': sqlite_count or 0,
                    'file_entries': file_count or 0, 'avg_hit_count': avg_hits or 0,
                    'avg_compression_ratio': avg_compression or 0, 'hit_rate_percent': hit_rate,
                    'total_requests_7d': total_req or 0, 'total_evictions_7d': total_evictions or 0,
                    'storage_backend': 'sqlite'
                }
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {}
    
    def close(self):
        if hasattr(self, 'conn'):
            try:
                self.conn.execute("VACUUM")
                self.conn.close()
            except:
                pass

# Example usage
if __name__ == "__main__":
    cache_manager = CacheManager(base_path="data/cache", sqlite_memory_limit_mb=50, use_file_fallback=True)
    
    test_data = {"recommendations": [1, 2, 3, 4, 5], "scores": [0.9, 0.8, 0.7, 0.6, 0.5]}
    
    cache_key = cache_manager.set_recommendation(
        user_id=123, request_type="similar_rooms", 
        params={"location": "Sri Lanka", "budget": 5000}, 
        data=test_data, ttl_hours=24
    )
    
    print(f"Cached data with key: {cache_key}")
    
    retrieved_data = cache_manager.get_recommendation(
        user_id=123, request_type="similar_rooms", 
        params={"location": "Sri Lanka", "budget": 5000}
    )
    
    print(f"Retrieved data: {retrieved_data}")
    print(f"Cache stats: {cache_manager.get_cache_stats()}")
    
    cache_manager.close()