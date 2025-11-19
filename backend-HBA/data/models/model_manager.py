# data/models/model_manager.py
import os
import json
import pickle
import joblib
import numpy as np
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from pathlib import Path
import logging
import sqlite3
import hashlib
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class ModelMetadata:
    """Metadata for stored ML models."""
    model_id: str
    model_type: str
    version: str
    created_at: str
    updated_at: str
    performance_metrics: Dict[str, float]
    hyperparameters: Dict[str, Any]
    training_data_hash: str
    file_path: str
    file_size_mb: float
    description: str = ""

class ModelManager:
    """Manages storage and versioning of trained ML models."""
    
    def __init__(self, base_path: str = "data/models"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize paths
        paths = ["clustering", "embedding", "time_series", "collaborative", "metadata"]
        for p in paths:
            setattr(self, f"{p}_path", self.base_path / p)
            getattr(self, f"{p}_path").mkdir(exist_ok=True)
        
        self._init_metadata_db()
    
    def _init_metadata_db(self):
        """Initialize SQLite database for model metadata."""
        db_path = self.metadata_path / "models_metadata.db"
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS models (
                model_id TEXT PRIMARY KEY, model_type TEXT NOT NULL, version TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL, updated_at TIMESTAMP NOT NULL,
                performance_metrics TEXT, hyperparameters TEXT, training_data_hash TEXT,
                file_path TEXT NOT NULL, file_size_mb REAL, description TEXT, is_active BOOLEAN DEFAULT 1
            );
            CREATE TABLE IF NOT EXISTS model_versions (
                model_type TEXT, version TEXT, model_id TEXT, created_at TIMESTAMP, is_latest BOOLEAN DEFAULT 0,
                PRIMARY KEY (model_type, version), FOREIGN KEY (model_id) REFERENCES models (model_id)
            );
            CREATE TABLE IF NOT EXISTS training_logs (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT, model_id TEXT, training_started TIMESTAMP,
                training_completed TIMESTAMP, status TEXT, error_message TEXT,
                FOREIGN KEY (model_id) REFERENCES models (model_id)
            );
        """)
        self.conn.commit()
    
    def save_model(self, model: Any, model_type: str, version: str, performance_metrics: Dict[str, float],
                   hyperparameters: Dict[str, Any], training_data_hash: str, description: str = "") -> str:
        """Save a trained model with metadata."""
        try:
            model_id = f"{model_type}_{version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            path_map = {"clustering": self.clustering_path, "embedding": self.embedding_path,
                       "time_series": self.time_series_path, "collaborative": self.collaborative_path}
            file_path = path_map.get(model_type, self.base_path) / f"{model_id}.pkl"
            
            joblib.dump(model, file_path) if hasattr(model, 'fit') else pickle.dump(model, open(file_path, 'wb'))
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            
            metadata = ModelMetadata(model_id, model_type, version, datetime.now().isoformat(),
                                   datetime.now().isoformat(), performance_metrics, hyperparameters,
                                   training_data_hash, str(file_path), file_size_mb, description)
            
            self.conn.execute("""INSERT INTO models (model_id, model_type, version, created_at, updated_at,
                performance_metrics, hyperparameters, training_data_hash, file_path, file_size_mb, description)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", 
                (metadata.model_id, metadata.model_type, metadata.version, metadata.created_at,
                 metadata.updated_at, json.dumps(metadata.performance_metrics), json.dumps(metadata.hyperparameters),
                 metadata.training_data_hash, metadata.file_path, metadata.file_size_mb, metadata.description))
            
            self.conn.execute("""INSERT OR REPLACE INTO model_versions (model_type, version, model_id, created_at, is_latest)
                VALUES (?, ?, ?, ?, 1)""", (model_type, version, model_id, datetime.now()))
            
            self.conn.execute("""UPDATE model_versions SET is_latest = 0 
                WHERE model_type = ? AND version != ? AND model_id != ?""", (model_type, version, model_id))
            
            self.conn.commit()
            logger.info(f"Saved model {model_id} of type {model_type}")
            return model_id
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    def load_model(self, model_id: str) -> Optional[Any]:
        """Load a model by its ID."""
        try:
            cursor = self.conn.execute("SELECT file_path FROM models WHERE model_id = ? AND is_active = 1", (model_id,))
            result = cursor.fetchone()
            if not result: return None
            
            file_path = Path(result[0])
            if not file_path.exists(): return None
            
            try: return joblib.load(file_path)
            except: return pickle.load(open(file_path, 'rb'))
        except Exception as e:
            logger.error(f"Error loading model {model_id}: {e}")
            return None
    
    def load_latest_model(self, model_type: str) -> Optional[Any]:
        """Load the latest version of a model type."""
        try:
            cursor = self.conn.execute("""SELECT m.model_id, m.file_path FROM models m
                JOIN model_versions mv ON m.model_id = mv.model_id
                WHERE m.model_type = ? AND mv.is_latest = 1 AND m.is_active = 1
                ORDER BY m.created_at DESC LIMIT 1""", (model_type,))
            
            result = cursor.fetchone()
            if not result: return None
            
            file_path = Path(result[1])
            if not file_path.exists(): return None
            
            try: return joblib.load(file_path)
            except: return pickle.load(open(file_path, 'rb'))
        except Exception as e:
            logger.error(f"Error loading latest model for type {model_type}: {e}")
            return None
    
    def get_model_metadata(self, model_id: str) -> Optional[ModelMetadata]:
        """Get metadata for a specific model."""
        try:
            cursor = self.conn.execute("""SELECT model_id, model_type, version, created_at, updated_at,
                performance_metrics, hyperparameters, training_data_hash, file_path, file_size_mb, description
                FROM models WHERE model_id = ?""", (model_id,))
            result = cursor.fetchone()
            return ModelMetadata(*result[:5], json.loads(result[5] or "{}"), json.loads(result[6] or "{}"),
                               *result[7:10], result[10] or "") if result else None
        except Exception as e:
            logger.error(f"Error getting model metadata: {e}")
            return None
    
    def list_models(self, model_type: Optional[str] = None) -> List[ModelMetadata]:
        """List all models, optionally filtered by type."""
        try:
            query = """SELECT model_id, model_type, version, created_at, updated_at, performance_metrics,
                hyperparameters, training_data_hash, file_path, file_size_mb, description
                FROM models WHERE is_active = 1"""
            params = ()
            if model_type:
                query += " AND model_type = ?"
                params = (model_type,)
            query += " ORDER BY created_at DESC"
            
            cursor = self.conn.execute(query, params)
            return [ModelMetadata(*row[:5], json.loads(row[5] or "{}"), json.loads(row[6] or "{}"),
                                 *row[7:10], row[10] or "") for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []
    
    def delete_model(self, model_id: str) -> bool:
        """Soft delete a model (mark as inactive)."""
        try:
            self.conn.execute("UPDATE models SET is_active = 0 WHERE model_id = ?", (model_id,))
            self.conn.commit()
            logger.info(f"Marked model {model_id} as inactive")
            return True
        except Exception as e:
            logger.error(f"Error deleting model {model_id}: {e}")
            return False
    
    def cleanup_old_models(self, days_old: int = 90, keep_latest: int = 3):
        """Clean up old models, keeping the latest N versions of each type."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_old)
            cursor = self.conn.execute("SELECT DISTINCT model_type FROM models")
            model_types = [row[0] for row in cursor.fetchall()]
            
            for model_type in model_types:
                cursor = self.conn.execute("""SELECT model_id FROM models 
                    WHERE model_type = ? AND is_active = 1 ORDER BY created_at DESC LIMIT ?""", 
                    (model_type, keep_latest))
                keep_models = {row[0] for row in cursor.fetchall()}
                
                cursor = self.conn.execute("""SELECT model_id, file_path FROM models 
                    WHERE model_type = ? AND created_at < ? AND is_active = 1""", (model_type, cutoff_date))
                
                for model_id, file_path in cursor.fetchall():
                    if model_id not in keep_models:
                        Path(file_path).unlink(missing_ok=True)
                        self.conn.execute("UPDATE models SET is_active = 0 WHERE model_id = ?", (model_id,))
            
            self.conn.commit()
            logger.info("Completed model cleanup")
        except Exception as e:
            logger.error(f"Error during model cleanup: {e}")
    
    def get_model_performance_history(self, model_type: str) -> List[Dict[str, Any]]:
        """Get performance history for a model type."""
        try:
            cursor = self.conn.execute("""SELECT version, created_at, performance_metrics 
                FROM models WHERE model_type = ? AND is_active = 1 ORDER BY created_at ASC""", (model_type,))
            return [{'version': v, 'created_at': c, 'metrics': json.loads(m or "{}")} 
                   for v, c, m in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error getting performance history: {e}")
            return []
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        try:
            cursor = self.conn.execute("""SELECT model_type, COUNT(*), SUM(file_size_mb), AVG(file_size_mb)
                FROM models WHERE is_active = 1 GROUP BY model_type""")
            
            stats = {}
            total_models, total_size = 0, 0
            
            for model_type, count, total_mb, avg_mb in cursor.fetchall():
                stats[model_type] = {'count': count, 'total_size_mb': total_mb or 0, 'avg_size_mb': avg_mb or 0}
                total_models += count
                total_size += (total_mb or 0)
            
            stats['totals'] = {'total_models': total_models, 'total_size_mb': total_size}
            return stats
        except Exception as e:
            logger.error(f"Error getting storage stats: {e}")
            return {}
    
    def close(self):
        """Close database connection."""
        if hasattr(self, 'conn'): self.conn.close()