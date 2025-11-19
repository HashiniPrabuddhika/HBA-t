import numpy as np
from typing import List, Dict, Any, Optional
import logging
import os
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class EmbeddingModel:
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", 
                 persist_directory: str = "./data/embeddings"):
        logger.info(f"Initializing EmbeddingModel with model: {model_name}")
        
        self.embeddings = None
        self.model_name = "fallback"
        
        try:
            # Try langchain-huggingface first (newer)
            try:
                from langchain_huggingface import HuggingFaceEmbeddings
                self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
                self.model_name = model_name
                logger.info("✓ Using langchain-huggingface package")
            except ImportError:
                # Fall back to langchain_community
                try:
                    import warnings
                    warnings.filterwarnings('ignore', category=DeprecationWarning)
                    from langchain_community.embeddings import HuggingFaceEmbeddings
                    self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
                    self.model_name = model_name
                    logger.info("✓ Using langchain_community package")
                except ImportError:
                    logger.warning("LangChain packages not available")
                    raise
                    
        except Exception as e:
            logger.warning(f"Failed to load embedding model: {e}, using fallback")
            self.embeddings = None
            self.model_name = "fallback"
        
        os.makedirs(persist_directory, exist_ok=True)
        self.persist_directory = persist_directory
        self._room_embeddings = {}
        self._user_embeddings = {}
        self._booking_embeddings = {}
        
        logger.info(f"EmbeddingModel initialized in {'full' if self.embeddings else 'fallback'} mode")
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text"""
        try:
            if self.embeddings is not None:
                embedding = self.embeddings.embed_query(text)
                return np.array(embedding, dtype=np.float32)
            else:
                # Fallback: deterministic hash-based embedding
                import hashlib
                hash_obj = hashlib.md5(text.encode())
                hash_int = int(hash_obj.hexdigest(), 16)
                np.random.seed(hash_int % (2**32))
                embedding = np.random.normal(0, 1, 384).astype(np.float32)
                return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return np.zeros(384, dtype=np.float32)
    
    def create_room_embedding(self, room_data: Dict[str, Any]) -> np.ndarray:
        """Create embedding for room data"""
        try:
            name = room_data.get('name', room_data.get('room_name', 'Unknown Room'))
            capacity = room_data.get('capacity', 0)
            description = room_data.get('description', '')
            features = room_data.get('features', [])
            location = room_data.get('location', '')
            equipment = room_data.get('equipment', [])
            room_type = room_data.get('type', 'general')
            
            room_text = (
                f"Room Name: {name} Type: {room_type} Capacity: {capacity} people "
                f"Location: {location} Description: {description} "
                f"Features: {', '.join(features) if features else 'None'} "
                f"Equipment: {', '.join(equipment) if equipment else 'None'} "
                f"Suitable for: {self._infer_suitable_activities(room_data)}"
            )
            
            embedding = self._get_embedding(room_text)
            logger.debug(f"Created embedding for room: {name}")
            return embedding
            
        except Exception as e:
            logger.error(f"Error creating room embedding: {e}")
            return np.zeros(384, dtype=np.float32)
    
    def create_user_embedding(self, user_data: Dict[str, Any]) -> np.ndarray:
        """Create embedding for user preferences"""
        try:
            booking_patterns = user_data.get('booking_patterns', {})
            preferred_rooms = user_data.get('preferred_rooms', [])
            common_times = user_data.get('common_times', [])
            department = user_data.get('department', 'unknown')
            role = user_data.get('role', 'staff')
            
            avg_duration = booking_patterns.get('avg_duration', 1.0)
            frequency = booking_patterns.get('frequency', 'occasional')
            preferred_capacity = booking_patterns.get('preferred_capacity', 10)
            common_purposes = booking_patterns.get('common_purposes', [])
            
            user_text = (
                f"Department: {department} Role: {role} "
                f"Preferred rooms: {', '.join(preferred_rooms) if preferred_rooms else 'No preference'} "
                f"Common times: {', '.join(common_times) if common_times else 'Flexible'} "
                f"Avg duration: {avg_duration} hours Frequency: {frequency} "
                f"Preferred capacity: {preferred_capacity} people "
                f"Purposes: {', '.join(common_purposes) if common_purposes else 'General meetings'} "
                f"Style: {self._infer_booking_style(booking_patterns)}"
            )
            
            embedding = self._get_embedding(user_text)
            logger.debug(f"Created embedding for user with {len(preferred_rooms)} preferred rooms")
            return embedding
            
        except Exception as e:
            logger.error(f"Error creating user embedding: {e}")
            return np.zeros(384, dtype=np.float32)
    
    def _infer_suitable_activities(self, room_data: Dict[str, Any]) -> str:
        """Infer suitable activities based on room characteristics"""
        capacity = room_data.get('capacity', 0)
        features = room_data.get('features', [])
        equipment = room_data.get('equipment', [])
        
        activities = []
        if capacity <= 4:
            activities.append("small meetings")
        elif capacity <= 12:
            activities.append("team meetings")
        else:
            activities.append("large presentations")
        
        if 'projector' in equipment or 'screen' in equipment:
            activities.append("presentations")
        if 'whiteboard' in equipment:
            activities.append("brainstorming")
        if 'video_conference' in equipment:
            activities.append("video calls")
        
        return ', '.join(activities) if activities else 'general meetings'
    
    def _infer_booking_style(self, booking_patterns: Dict[str, Any]) -> str:
        """Infer booking style from patterns"""
        frequency = booking_patterns.get('frequency', 'occasional')
        avg_duration = booking_patterns.get('avg_duration', 1.0)
        
        if frequency == 'daily' and avg_duration < 1:
            return 'frequent short meetings'
        elif frequency == 'daily' and avg_duration >= 2:
            return 'daily long sessions'
        elif frequency == 'weekly':
            return 'regular weekly meetings'
        else:
            return 'occasional bookings'
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            if a.size == 0 or b.size == 0:
                return 0.0
            dot_product = np.dot(a, b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return float(dot_product / (norm_a * norm_b))
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {e}")
            return 0.0
    
    def health_check(self) -> Dict[str, Any]:
        """Check embedding model health"""
        try:
            test_embedding = self._get_embedding("test")
            
            return {
                'status': 'healthy',
                'embedding_dimension': len(test_embedding) if test_embedding is not None else 0,
                'model': self.model_name,
                'fallback_mode': self.embeddings is None,
                'room_embeddings': len(self._room_embeddings),
                'user_embeddings': len(self._user_embeddings)
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }


class EnhancedEmbeddingModel(EmbeddingModel):
    """Enhanced embedding model with behavioral analysis"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 persist_directory: str = "./data/embeddings"):
        super().__init__(model_name, persist_directory)
        self.behavioral_model = None
        self._initialize_advanced_components()
    
    def _initialize_advanced_components(self):
        """Initialize advanced components if available"""
        try:
            from sentence_transformers import SentenceTransformer
            self.behavioral_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
            logger.info("✓ Advanced behavioral model initialized")
        except ImportError:
            logger.info("sentence-transformers not available, using base model only")
            self.behavioral_model = None
        except Exception as e:
            logger.warning(f"Could not initialize behavioral model: {e}, using fallback")
            self.behavioral_model = None
    
    def create_enhanced_user_embedding(self, user_data: Dict[str, Any]) -> np.ndarray:
        """Create enhanced user embedding with behavioral analysis"""
        try:
            # Get base embedding
            base_embedding = self.create_user_embedding(user_data)
            
            # If behavioral model available, enhance the embedding
            if self.behavioral_model:
                behavioral_text = self._create_behavioral_description(user_data)
                behavioral_embedding = self.behavioral_model.encode(behavioral_text)
                combined_embedding = np.concatenate([base_embedding, behavioral_embedding])
                logger.debug("Created enhanced embedding with behavioral analysis")
                return combined_embedding
            else:
                # Return base embedding if behavioral model not available
                return base_embedding
                
        except Exception as e:
            logger.error(f"Error creating enhanced user embedding: {e}")
            return self.create_user_embedding(user_data)
    
    def _create_behavioral_description(self, user_data: Dict[str, Any]) -> str:
        """Create behavioral description from user data"""
        patterns = user_data.get('booking_patterns', {})
        history = user_data.get('booking_history', [])
        
        timing_pattern = self._analyze_timing_patterns(history)
        room_pattern = self._analyze_room_selection_patterns(history)
        duration_pattern = self._analyze_duration_patterns(history)
        
        return (
            f"Booking timing: {timing_pattern} "
            f"Room selection: {room_pattern} "
            f"Duration: {duration_pattern} "
            f"Frequency: {patterns.get('frequency', 'occasional')} "
            f"Advance booking: {patterns.get('advance_booking_days', 1)} days"
        )
    
    def _analyze_timing_patterns(self, history: List[Dict]) -> str:
        """Analyze user's timing preferences"""
        if not history:
            return "insufficient data"
        
        morning_count = sum(
            1 for booking in history
            if self._get_hour_from_time(booking.get('start_time', '')) in range(8, 12)
        )
        afternoon_count = sum(
            1 for booking in history
            if self._get_hour_from_time(booking.get('start_time', '')) in range(12, 17)
        )
        
        total = len(history)
        if morning_count / total > 0.6:
            return "morning preference"
        elif afternoon_count / total > 0.6:
            return "afternoon preference"
        else:
            return "flexible timing"
    
    def _get_hour_from_time(self, time_str: str) -> int:
        """Extract hour from time string"""
        try:
            if 'T' in time_str:
                return datetime.fromisoformat(time_str.replace('Z', '+00:00')).hour
            elif ':' in time_str:
                return int(time_str.split(':')[0])
            return 12
        except:
            return 12
    
    def _analyze_room_selection_patterns(self, history: List[Dict]) -> str:
        """Analyze room selection patterns"""
        if not history:
            return "no pattern"
        
        room_counts = {}
        for booking in history:
            room = booking.get('room_name', 'unknown')
            room_counts[room] = room_counts.get(room, 0) + 1
        
        if len(room_counts) == 1:
            return "single room preference"
        elif len(room_counts) <= 3:
            return "few preferred rooms"
        else:
            return "diverse room usage"
    
    def _analyze_duration_patterns(self, history: List[Dict]) -> str:
        """Analyze duration preferences"""
        if not history:
            return "unknown duration preference"
        
        durations = []
        for booking in history:
            start = booking.get('start_time', '')
            end = booking.get('end_time', '')
            if start and end:
                duration = self._calculate_duration_from_times(start, end)
                durations.append(duration)
        
        if not durations:
            return "unknown duration"
        
        avg_duration = sum(durations) / len(durations)
        if avg_duration < 0.75:
            return "short meetings preference"
        elif avg_duration > 2.5:
            return "long sessions preference"
        else:
            return "standard duration preference"
    
    def _calculate_duration_from_times(self, start_time: str, end_time: str) -> float:
        """Calculate duration in hours between start and end times"""
        try:
            if 'T' in start_time:
                start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
            else:
                start_dt = datetime.strptime(start_time, '%H:%M')
                end_dt = datetime.strptime(end_time, '%H:%M')
                if end_dt < start_dt:
                    end_dt += timedelta(days=1)
            
            duration = (end_dt - start_dt).total_seconds() / 3600
            return max(duration, 0.5)
        except:
            return 1.0