from .embedding_model import EmbeddingModel
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import numpy as np
from typing import List, Dict, Any, Optional
import logging
import datetime
from datetime import timedelta

logger = logging.getLogger(__name__)

class EnhancedEmbeddingModel(EmbeddingModel):
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", persist_directory: str = "./data/embeddings"):
        super().__init__(model_name, persist_directory)
        self._initialize_advanced_components()
    
    def _initialize_advanced_components(self):
        try:
            self.behavioral_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
            self.chroma_client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=self.persist_directory))
            
            self.user_collection = self._get_or_create_collection("user_behaviors")
            self.room_collection = self._get_or_create_collection("room_features")
            self.booking_collection = self._get_or_create_collection("booking_patterns")
            
            logger.info("Advanced ML components initialized successfully")
        except Exception as e:
            logger.warning(f"Could not initialize advanced components: {e}, using fallback")
            self.behavioral_model = None
            self.chroma_client = None
    
    def _get_or_create_collection(self, name: str):
        try:
            return self.chroma_client.get_collection(name=name)
        except:
            return self.chroma_client.create_collection(name=name)
    
    def _calculate_duration_from_times(self, start_time: str, end_time: str) -> float:
        """Calculate duration in hours from start and end times"""
        try:
            if 'T' in start_time:  # ISO format
                start_dt = datetime.datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                end_dt = datetime.datetime.fromisoformat(end_time.replace('Z', '+00:00'))
            else:  # Time only format
                start_dt = datetime.datetime.strptime(start_time, '%H:%M')
                end_dt = datetime.datetime.strptime(end_time, '%H:%M')
                if end_dt < start_dt:  # Next day
                    end_dt += timedelta(days=1)
            
            duration = (end_dt - start_dt).total_seconds() / 3600
            return max(duration, 0.5)  # Minimum 30 minutes
        except:
            return 1.0  # Default 1 hour
    
    async def create_advanced_user_embedding(self, user_data: Dict[str, Any]) -> np.ndarray:
        try:
            base_embedding = self.create_user_embedding(user_data)
            
            if self.behavioral_model:
                behavioral_text = self._create_behavioral_description(user_data)
                behavioral_embedding = self.behavioral_model.encode(behavioral_text)
                combined_embedding = np.concatenate([base_embedding, behavioral_embedding])
            else:
                combined_embedding = base_embedding
            
            if self.chroma_client and self.user_collection:
                self.user_collection.upsert(
                    embeddings=[combined_embedding.tolist()],
                    documents=[self._create_user_description(user_data)],
                    ids=[f"user_{user_data.get('user_id', 'unknown')}"],
                    metadatas={"user_id": user_data.get('user_id'), "last_updated": str(datetime.datetime.now())}
                )
            
            return combined_embedding
        except Exception as e:
            logger.error(f"Error creating advanced user embedding: {e}")
            return self.create_user_embedding(user_data)
    
    def _create_behavioral_description(self, user_data: Dict[str, Any]) -> str:
        patterns = user_data.get('booking_patterns', {})
        history = user_data.get('booking_history', [])
        
        timing_pattern = self._analyze_timing_patterns(history)
        room_pattern = self._analyze_room_selection_patterns(history)
        duration_pattern = self._analyze_duration_patterns(history)
        
        return f"Booking timing: {timing_pattern} Room selection: {room_pattern} Duration: {duration_pattern} " \
               f"Frequency: {patterns.get('frequency', 'occasional')} " \
               f"Advance booking: {patterns.get('advance_booking_days', 1)} days"
    
    def _analyze_timing_patterns(self, history: List[Dict]) -> str:
        if not history: return "insufficient data"
        
        morning_count = sum(1 for booking in history 
                           if self._get_hour_from_time(booking.get('start_time', '')) in range(8, 12))
        afternoon_count = sum(1 for booking in history 
                             if self._get_hour_from_time(booking.get('start_time', '')) in range(12, 17))
        
        total = len(history)
        if morning_count / total > 0.6: return "morning preference"
        elif afternoon_count / total > 0.6: return "afternoon preference"
        else: return "flexible timing"
    
    def _get_hour_from_time(self, time_str: str) -> int:
        try:
            if 'T' in time_str:  # ISO format
                return datetime.datetime.fromisoformat(time_str.replace('Z', '+00:00')).hour
            elif ':' in time_str:  # Time format
                return int(time_str.split(':')[0])
            return 12  # Default noon
        except:
            return 12
    
    def _analyze_room_selection_patterns(self, history: List[Dict]) -> str:
        if not history: return "no pattern"
        
        room_counts = {}
        for booking in history:
            room = booking.get('room_name', 'unknown')
            room_counts[room] = room_counts.get(room, 0) + 1
        
        if len(room_counts) == 1: return "single room preference"
        elif len(room_counts) <= 3: return "few preferred rooms"
        else: return "diverse room usage"
    
    def _analyze_duration_patterns(self, history: List[Dict]) -> str:
        if not history: return "unknown duration preference"
        
        durations = []
        for booking in history:
            start = booking.get('start_time', '')
            end = booking.get('end_time', '')
            if start and end:
                duration = self._calculate_duration_from_times(start, end)
                durations.append(duration)
        
        if not durations: return "unknown duration"
        
        avg_duration = sum(durations) / len(durations)
        if avg_duration < 0.75: return "short meetings preference"
        elif avg_duration > 2.5: return "long sessions preference"
        else: return "standard duration preference"
    
    async def get_duration_based_recommendations(self, user_id: str, requested_duration: float) -> Dict[str, Any]:
        """Get recommendations based on user's typical duration patterns"""
        try:
            if not self.user_collection:
                return {"recommendations": [], "insights": []}
            
            # Get user's historical duration preferences
            user_embedding = await self._get_user_embedding_from_db(user_id)
            similar_bookings = self.booking_collection.query(
                query_embeddings=[user_embedding],
                n_results=10,
                where={"duration_hours": {"$gte": requested_duration * 0.5, "$lte": requested_duration * 1.5}}
            )
            
            insights = []
            if requested_duration > 2:
                insights.append("Consider booking larger rooms for longer sessions")
            if requested_duration < 1:
                insights.append("Quick meetings work well in smaller spaces")
            
            return {
                "recommended_duration_adjustment": self._suggest_duration_adjustment(user_id, requested_duration),
                "insights": insights,
                "similar_bookings": len(similar_bookings.get('ids', [{}])[0]) if similar_bookings else 0
            }
        except Exception as e:
            logger.error(f"Error getting duration recommendations: {e}")
            return {"recommendations": [], "insights": []}
    
    def _suggest_duration_adjustment(self, user_id: str, requested_duration: float) -> Dict[str, Any]:
        """Suggest duration adjustments based on user patterns"""
        # This would analyze user's typical meeting durations and suggest optimal times
        return {
            "suggested_duration": requested_duration,
            "confidence": 0.7,
            "reason": "Based on your typical meeting patterns"
        }
    
    async def find_similar_users_advanced(self, user_id: str, n_results: int = 5) -> List[Dict]:
        if self.chroma_client and self.user_collection:
            try:
                user_embedding = await self._get_user_embedding_from_db(user_id)
                results = self.user_collection.query(
                    query_embeddings=[user_embedding],
                    n_results=n_results + 1
                )
                
                similar_users = []
                for i, (id_, distance, metadata) in enumerate(zip(
                    results['ids'][0], results['distances'][0], results['metadatas'][0])):
                    if id_ != f"user_{user_id}":
                        similar_users.append({
                            'user_id': metadata.get('user_id'), 
                            'similarity_score': 1 - distance, 
                            'metadata': metadata
                        })
                
                return similar_users[:n_results]
            except Exception as e:
                logger.error(f"Error in advanced similarity search: {e}")
                return []
        else:
            return []
    
    async def _get_user_embedding_from_db(self, user_id: str) -> List[float]:
        try:
            results = self.user_collection.get(ids=[f"user_{user_id}"])
            if results['embeddings']: 
                return results['embeddings'][0]
            else: 
                return [0.0] * 768
        except: 
            return [0.0] * 768
    
    def store_booking_pattern(self, booking_data: Dict[str, Any]):
        """Store booking patterns with proper duration calculation"""
        try:
            if not self.booking_collection:
                return
                
            start_time = booking_data.get('start_time', '')
            end_time = booking_data.get('end_time', '')
            duration = self._calculate_duration_from_times(start_time, end_time)
            
            # Create embedding for this booking pattern
            pattern_text = f"Room: {booking_data.get('room_name')} Duration: {duration}h " \
                          f"Time: {self._get_hour_from_time(start_time)} Purpose: {booking_data.get('purpose', '')}"
            
            if self.behavioral_model:
                embedding = self.behavioral_model.encode(pattern_text)
                
                self.booking_collection.upsert(
                    embeddings=[embedding.tolist()],
                    documents=[pattern_text],
                    ids=[f"booking_{booking_data.get('booking_id', 'unknown')}"],
                    metadatas={
                        "user_id": booking_data.get('user_id'),
                        "duration_hours": duration,
                        "room_name": booking_data.get('room_name'),
                        "hour_of_day": self._get_hour_from_time(start_time)
                    }
                )
        except Exception as e:
            logger.error(f"Error storing booking pattern: {e}")