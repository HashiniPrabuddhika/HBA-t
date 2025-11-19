import pickle
import numpy as np
from typing import Dict, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
from pathlib import Path
import sqlite3
import hashlib

logger = logging.getLogger(__name__)

class EmbeddingManager:
    def __init__(self, base_path: str = "data/embeddings"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        for name in ["rooms", "users", "bookings", "metadata"]:
            path = self.base_path / name
            path.mkdir(exist_ok=True)
            setattr(self, f"{name}_path", path)
        
        self.conn = sqlite3.connect(str(self.metadata_path / "embeddings_metadata.db"), check_same_thread=False)
        
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS room_embeddings (
                room_id INTEGER PRIMARY KEY, embedding_hash TEXT, created_at TIMESTAMP,
                updated_at TIMESTAMP, feature_version TEXT, dimensions INTEGER, file_path TEXT);
            
            CREATE TABLE IF NOT EXISTS user_embeddings (
                user_id INTEGER PRIMARY KEY, embedding_hash TEXT, created_at TIMESTAMP,
                updated_at TIMESTAMP, feature_version TEXT, dimensions INTEGER, file_path TEXT);
            
            CREATE TABLE IF NOT EXISTS booking_embeddings (
                booking_id INTEGER PRIMARY KEY, embedding_hash TEXT, created_at TIMESTAMP,
                updated_at TIMESTAMP, feature_version TEXT, dimensions INTEGER, file_path TEXT);
            
            CREATE TABLE IF NOT EXISTS embedding_versions (
                version_id TEXT PRIMARY KEY, model_name TEXT, created_at TIMESTAMP, description TEXT);
        """)
        self.conn.commit()
    
    def _save_embedding(self, table: str, id_field: str, id_val: int, embedding: np.ndarray, 
                       data: Dict[str, Any], version: str = "v1.0") -> bool:
        try:
            hash_val = hashlib.md5(embedding.tobytes()).hexdigest()
            file_path = getattr(self, f"{table}_path") / f"{table[:-1]}_{id_val}_{hash_val}.pkl"
            
            with open(file_path, 'wb') as f:
                pickle.dump({**data, 'embedding': embedding, id_field: id_val, 'created_at': datetime.now().isoformat()}, f)
            
            now = datetime.now()
            self.conn.execute(f"""
                INSERT OR REPLACE INTO {table}_embeddings 
                ({id_field}, embedding_hash, created_at, updated_at, feature_version, dimensions, file_path)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (id_val, hash_val, now, now, version, len(embedding), str(file_path)))
            self.conn.commit()
            
            logger.info(f"Saved {table[:-1]} embedding for {id_field} {id_val}")
            return True
        except Exception as e:
            logger.error(f"Error saving {table[:-1]} embedding: {e}")
            return False
    
    def _load_embedding(self, table: str, id_field: str, id_val: int) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        try:
            cursor = self.conn.execute(f"SELECT file_path FROM {table}_embeddings WHERE {id_field} = ?", (id_val,))
            result = cursor.fetchone()
            
            if not result or not Path(result[0]).exists():
                return None
            
            with open(result[0], 'rb') as f:
                data = pickle.load(f)
                return data['embedding'], {k: v for k, v in data.items() if k not in ['embedding', id_field, 'created_at']}
        except Exception as e:
            logger.error(f"Error loading {table[:-1]} embedding: {e}")
            return None
    
    def save_room_embedding(self, room_id: int, embedding: np.ndarray, features: Dict[str, Any], version: str = "v1.0") -> bool:
        return self._save_embedding("rooms", "room_id", room_id, embedding, {'features': features}, version)
    
    def load_room_embedding(self, room_id: int) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        return self._load_embedding("rooms", "room_id", room_id)
    
    def save_user_embedding(self, user_id: int, embedding: np.ndarray, preferences: Dict[str, Any], version: str = "v1.0") -> bool:
        return self._save_embedding("users", "user_id", user_id, embedding, {'preferences': preferences}, version)
    
    def load_user_embedding(self, user_id: int) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        return self._load_embedding("users", "user_id", user_id)
    
    def save_booking_embedding(self, booking_id: int, embedding: np.ndarray, context: Dict[str, Any], version: str = "v1.0") -> bool:
        return self._save_embedding("bookings", "booking_id", booking_id, embedding, {'context': context}, version)
    
    def get_all_room_embeddings(self) -> Dict[int, np.ndarray]:
        return self._get_all_embeddings("rooms", "room_id")
    
    def get_all_user_embeddings(self) -> Dict[int, np.ndarray]:
        return self._get_all_embeddings("users", "user_id")
    
    def _get_all_embeddings(self, table: str, id_field: str) -> Dict[int, np.ndarray]:
        embeddings = {}
        try:
            cursor = self.conn.execute(f"SELECT {id_field}, file_path FROM {table}_embeddings")
            
            for entity_id, file_path in cursor.fetchall():
                try:
                    with open(file_path, 'rb') as f:
                        embeddings[entity_id] = pickle.load(f)['embedding']
                except Exception as e:
                    logger.warning(f"Could not load embedding for {id_field} {entity_id}: {e}")
            
            return embeddings
        except Exception as e:
            logger.error(f"Error getting all {table[:-1]} embeddings: {e}")
            return {}
    
    def cleanup_old_embeddings(self, days_old: int = 30):
        try:
            cutoff_date = datetime.now() - timedelta(days=days_old)
            
            cursor = self.conn.execute("""
                SELECT file_path FROM room_embeddings WHERE created_at < ?
                UNION ALL SELECT file_path FROM user_embeddings WHERE created_at < ?
                UNION ALL SELECT file_path FROM booking_embeddings WHERE created_at < ?
            """, (cutoff_date, cutoff_date, cutoff_date))
            
            old_files = cursor.fetchall()
            
            for (file_path,) in old_files:
                Path(file_path).unlink(missing_ok=True)
            
            for table in ["room_embeddings", "user_embeddings", "booking_embeddings"]:
                self.conn.execute(f"DELETE FROM {table} WHERE created_at < ?", (cutoff_date,))
            
            self.conn.commit()
            logger.info(f"Cleaned up {len(old_files)} old embedding files")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def get_embedding_stats(self) -> Dict[str, Any]:
        try:
            stats = {}
            
            for table in ["room_embeddings", "user_embeddings", "booking_embeddings"]:
                cursor = self.conn.execute(f"SELECT COUNT(*) FROM {table}")
                stats[table] = cursor.fetchone()[0]
            
            total_size = sum(f.stat().st_size for path in [self.rooms_path, self.users_path, self.bookings_path] 
                           for f in path.rglob("*.pkl"))
            
            stats['total_disk_usage_mb'] = total_size / (1024 * 1024)
            return stats
        except Exception as e:
            logger.error(f"Error getting embedding stats: {e}")
            return {}
    
    def close(self):
        if hasattr(self, 'conn'):
            self.conn.close()