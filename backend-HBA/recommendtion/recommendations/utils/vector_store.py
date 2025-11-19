import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import sqlite3
import json
import os
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime, timedelta
import logging
import tempfile

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = os.getenv('VECTOR_DB_PATH', 'cache/vector_store.db')
        self.db_path = self._ensure_valid_db_path(db_path)
        self._init_database()
        self._room_vectors = {}
        self._last_cache_update = None
        self._cache_ttl = timedelta(minutes=30)
        logger.info(f"VectorStore initialized with database: {self.db_path}")
    
    def _ensure_valid_db_path(self, db_path: str) -> str:
        try:
            path = Path(db_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            test_file = path.parent / '.test_write'
            try:
                test_file.touch()
                test_file.unlink()
                return str(path)
            except PermissionError:
                logger.warning(f"Cannot write to {path.parent}, using temp directory")
                temp_dir = Path(tempfile.gettempdir()) / 'vector_store'
                temp_dir.mkdir(exist_ok=True)
                return str(temp_dir / 'vector_store.db')
        except Exception as e:
            logger.warning(f"Error setting up database path {db_path}: {e}")
            temp_dir = Path(tempfile.gettempdir()) / 'vector_store'
            temp_dir.mkdir(exist_ok=True)
            fallback_path = temp_dir / 'vector_store.db'
            logger.info(f"Using fallback database path: {fallback_path}")
            return str(fallback_path)
    
    def _init_database(self):
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("SELECT 1")
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS room_vectors (
                    room_id TEXT PRIMARY KEY,
                    room_name TEXT,
                    description TEXT,
                    capacity INTEGER,
                    features TEXT,
                    embedding BLOB,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_room_capacity ON room_vectors(capacity)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_room_updated ON room_vectors(updated_at)')
            conn.commit()
            conn.close()
            logger.info("Vector store database initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize vector store database: {e}")
            raise
    
    def add_room_vector(self, room_data: Dict[str, Any], embedding: np.ndarray = None) -> bool:
        try:
            room_id = room_data.get('id') or room_data.get('room_id')
            if not room_id:
                logger.error("Room data must contain 'id' or 'room_id'")
                return False
            if embedding is None:
                embedding = self._generate_room_embedding(room_data)
            room_name = room_data.get('name', '')
            description = room_data.get('description', '')
            capacity = room_data.get('capacity', 0)
            features = json.dumps(room_data.get('features', []))
            metadata = json.dumps({
                'equipment': room_data.get('equipment', []),
                'location': room_data.get('location', ''),
                'floor': room_data.get('floor'),
                'amenities': room_data.get('amenities', [])
            })
            embedding_bytes = embedding.astype(np.float64).tobytes()
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO room_vectors
                (room_id, room_name, description, capacity, features, 
                 embedding, metadata, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (room_id, room_name, description, capacity, features, embedding_bytes, metadata))
            conn.commit()
            conn.close()
            self._room_vectors[room_id] = {
                'embedding': embedding,
                'room_data': room_data,
                'updated_at': datetime.now()
            }
            logger.info(f"Added room vector for room {room_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to add room vector: {e}")
            return False
    
    def _generate_room_embedding(self, room_data: Dict[str, Any]) -> np.ndarray:
        try:
            from ..models.embedding_model import EmbeddingModel
            embedding_model = EmbeddingModel()
            text_parts = []
            if room_data.get('name'):
                text_parts.append(room_data['name'])
            if room_data.get('description'):
                text_parts.append(room_data['description'])
            if room_data.get('equipment'):
                text_parts.append(f"Equipment: {', '.join(room_data['equipment'])}")
            if room_data.get('amenities'):
                text_parts.append(f"Amenities: {', '.join(room_data['amenities'])}")
            room_text = '. '.join(text_parts)
            return embedding_model.get_room_embedding(room_text)
        except ImportError:
            logger.warning("EmbeddingModel not available, using simple hash-based embedding")
            return self._simple_room_embedding(room_data)
    
    def _simple_room_embedding(self, room_data: Dict[str, Any]) -> np.ndarray:
        features = np.zeros(128, dtype=np.float64)
        capacity = room_data.get('capacity', 0)
        features[0] = min(capacity / 50.0, 1.0)
        equipment = room_data.get('equipment', [])
        equipment_mapping = {
            'projector': 1, 'whiteboard': 2, 'tv': 3, 'microphone': 4,
            'camera': 5, 'computer': 6, 'phone': 7, 'speakers': 8
        }
        for eq in equipment:
            if eq in equipment_mapping:
                idx = equipment_mapping[eq]
                if idx < 20:
                    features[idx] = 1.0
        amenities = room_data.get('amenities', [])
        amenity_mapping = {
            'wifi': 20, 'ac': 21, 'heating': 22, 'natural_light': 23,
            'quiet': 24, 'private': 25, 'accessible': 26
        }
        for amenity in amenities:
            if amenity in amenity_mapping:
                idx = amenity_mapping[amenity]
                if idx < 50:
                    features[idx] = 1.0
        location = room_data.get('location', '')
        if location:
            location_hash = abs(hash(location)) % 50
            if 50 + location_hash < 100:
                features[50 + location_hash] = 1.0
        room_id = room_data.get('id', room_data.get('room_id', ''))
        if room_id:
            room_hash = abs(hash(room_id))
            np.random.seed(room_hash % 2147483647)
            features[100:] = np.random.normal(0, 0.1, 28)
        return features
    
    def search_similar_rooms(self, query: str, top_k: int = 5, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        try:
            query_embedding = self._generate_query_embedding(query)
            room_vectors = self._get_all_room_vectors(filters)
            if not room_vectors:
                logger.warning("No room vectors found in database")
                return []
            similarities = []
            for room_id, room_info in room_vectors.items():
                try:
                    similarity = cosine_similarity(
                        query_embedding.reshape(1, -1),
                        room_info['embedding'].reshape(1, -1)
                    )[0][0]
                    similarities.append({
                        'room_id': room_id,
                        'similarity': float(similarity),
                        'room_data': room_info['room_data']
                    })
                except Exception as e:
                    logger.warning(f"Error calculating similarity for room {room_id}: {e}")
                    continue
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            results = similarities[:top_k]
            logger.info(f"Found {len(results)} similar rooms for query: {query}")
            return results
        except Exception as e:
            logger.error(f"Failed to search similar rooms: {e}")
            return []
    
    def _generate_query_embedding(self, query: str) -> np.ndarray:
        try:
            from ..models.embedding_model import EmbeddingModel
            embedding_model = EmbeddingModel()
            return embedding_model.get_room_embedding(query)
        except ImportError:
            return self._simple_query_embedding(query)
    
    def _simple_query_embedding(self, query: str) -> np.ndarray:
        features = np.zeros(128, dtype=np.float64)
        query_lower = query.lower()
        equipment_keywords = {
            'projector': 1, 'whiteboard': 2, 'tv': 3, 'microphone': 4,
            'camera': 5, 'computer': 6, 'phone': 7, 'speakers': 8,
            'presentation': 1, 'display': 3, 'audio': 7, 'video': 5
        }
        for keyword, idx in equipment_keywords.items():
            if keyword in query_lower and idx < 20:
                features[idx] = 1.0
        if 'large' in query_lower or 'big' in query_lower:
            features[0] = 0.8
        elif 'small' in query_lower:
            features[0] = 0.2
        elif 'medium' in query_lower:
            features[0] = 0.5
        amenity_keywords = {
            'wifi': 20, 'air conditioning': 21, 'quiet': 24, 
            'private': 25, 'natural light': 23
        }
        for keyword, idx in amenity_keywords.items():
            if keyword in query_lower and idx < 50:
                features[idx] = 1.0
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        return features
    
    def _get_all_room_vectors(self, filters: Dict[str, Any] = None) -> Dict[str, Dict]:
        try:
            if (self._last_cache_update and 
                datetime.now() - self._last_cache_update < self._cache_ttl and
                not filters):
                return self._room_vectors
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            query = "SELECT * FROM room_vectors"
            params = []
            if filters:
                conditions = []
                if 'min_capacity' in filters:
                    conditions.append("capacity >= ?")
                    params.append(filters['min_capacity'])
                if 'max_capacity' in filters:
                    conditions.append("capacity <= ?")
                    params.append(filters['max_capacity'])
                if conditions:
                    query += " WHERE " + " AND ".join(conditions)
            cursor.execute(query, params)
            rows = cursor.fetchall()
            conn.close()
            room_vectors = {}
            for row in rows:
                try:
                    room_id = row[0]
                    embedding_bytes = row[5]
                    if embedding_bytes:
                        embedding = np.frombuffer(embedding_bytes, dtype=np.float64)
                    else:
                        logger.warning(f"No embedding data for room {room_id}")
                        continue
                    room_data = {
                        'id': room_id,
                        'name': row[1] or '',
                        'description': row[2] or '',
                        'capacity': row[3] or 0,
                        'features': json.loads(row[4]) if row[4] else [],
                        'metadata': json.loads(row[6]) if row[6] else {}
                    }
                    room_vectors[room_id] = {
                        'embedding': embedding,
                        'room_data': room_data,
                        'updated_at': datetime.fromisoformat(row[8]) if row[8] else datetime.now()
                    }
                except Exception as e:
                    logger.warning(f"Error processing room vector for room {row[0] if row else 'unknown'}: {e}")
                    continue
            if not filters:
                self._room_vectors = room_vectors
                self._last_cache_update = datetime.now()
            return room_vectors
        except Exception as e:
            logger.error(f"Failed to get room vectors: {e}")
            return {}
    
    def get_room_vector(self, room_id: str) -> Optional[np.ndarray]:
        try:
            room_vectors = self._get_all_room_vectors()
            room_info = room_vectors.get(room_id)
            if room_info:
                return room_info['embedding']
            return None
        except Exception as e:
            logger.error(f"Failed to get room vector for {room_id}: {e}")
            return None
    
    def remove_room_vector(self, room_id: str) -> bool:
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM room_vectors WHERE room_id = ?", (room_id,))
            deleted_count = cursor.rowcount
            conn.commit()
            conn.close()
            if room_id in self._room_vectors:
                del self._room_vectors[room_id]
            logger.info(f"Removed room vector for {room_id}")
            return deleted_count > 0
        except Exception as e:
            logger.error(f"Failed to remove room vector for {room_id}: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM room_vectors")
            total_rooms = cursor.fetchone()[0]
            cursor.execute("SELECT AVG(capacity), MIN(capacity), MAX(capacity) FROM room_vectors WHERE capacity > 0")
            capacity_stats = cursor.fetchone()
            conn.close()
            return {
                'total_rooms': total_rooms,
                'avg_capacity': capacity_stats[0] if capacity_stats[0] else 0,
                'min_capacity': capacity_stats[1] if capacity_stats[1] else 0,
                'max_capacity': capacity_stats[2] if capacity_stats[2] else 0,
                'cache_size': len(self._room_vectors),
                'last_cache_update': self._last_cache_update.isoformat() if self._last_cache_update else None,
                'database_path': self.db_path
            }
        except Exception as e:
            logger.error(f"Failed to get vector store stats: {e}")
            return {'error': str(e)}
    
    def clear_cache(self):
        self._room_vectors.clear()
        self._last_cache_update = None
        logger.info("Vector store cache cleared")
    
    def test_connection(self) -> bool:
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM room_vectors")
            count = cursor.fetchone()[0]
            conn.close()
            logger.info(f"Vector store connection test successful. Found {count} room vectors.")
            return True
        except Exception as e:
            logger.error(f"Vector store connection test failed: {e}")
            return False
    
    def close(self):
        self.clear_cache()
        logger.info("Vector store closed")