import logging
import math
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
from collections import defaultdict, Counter
from dataclasses import dataclass
from enum import Enum

from sqlalchemy.orm import Session
from sqlalchemy import func, and_

from .time_utils import TimeUtils
from ..data.cache_manager import CacheManager

from models.room import MRBSRoom, MRBSArea
from models.booking import MRBSEntry

logger = logging.getLogger(__name__)


class SimilarityType(Enum):
    """Types of similarity calculations"""
    ROOM_FEATURES = "room_features"
    ROOM_USAGE = "room_usage"
    TIME_PATTERNS = "time_patterns"
    USER_BEHAVIOR = "user_behavior"
    BOOKING_CONTEXT = "booking_context"


@dataclass
class SimilarityScore:
    """Similarity score between two entities"""
    entity1_id: Any
    entity2_id: Any
    similarity_score: float
    similarity_type: SimilarityType
    contributing_factors: Dict[str, float]
    confidence: float
    calculated_at: datetime


@dataclass
class RoomProfile:
    """Room profile with calculated features"""
    room_id: int
    room_name: str
    capacity: int
    area_id: int
    area_name: str
    description: str
    usage_frequency: float
    average_booking_duration: float
    peak_usage_hours: List[int]
    common_users: Set[str]
    booking_purposes: List[str]
    utilization_rate: float
    feature_vector: List[float]
    usage_vector: List[float]


@dataclass
class TimeSlotProfile:
    """Time slot profile with usage patterns"""
    start_hour: int
    end_hour: int
    day_of_week: int
    duration_hours: float
    popularity_score: float
    conflict_probability: float
    typical_users: Set[str]
    common_purposes: List[str]
    seasonal_usage: Dict[str, float]


class SimilarityEngine:
    """Calculate similarities between rooms, time slots, and booking patterns"""
    
    def __init__(self, db: Session, cache_manager: CacheManager):
        self.db = db
        self.cache_manager = cache_manager
        self.time_utils = TimeUtils()
        
        self.room_similarity_weights = {
            'capacity': 0.25,
            'area': 0.15,
            'features': 0.20,
            'usage_patterns': 0.25,
            'user_overlap': 0.15
        }
        
        self.time_similarity_weights = {
            'hour_proximity': 0.30,
            'day_similarity': 0.20,
            'duration_match': 0.25,
            'usage_patterns': 0.25
        }
        
        self.cache_ttl = 3600
    
    # ==================== Room Similarity ====================
    
    def calculate_room_similarity(
        self,
        room1_id: int,
        room2_id: int,
        context: Optional[Dict] = None
    ) -> SimilarityScore:
        """Calculate similarity between two rooms"""
        try:
            cache_key = f"room_similarity:{min(room1_id, room2_id)}:{max(room1_id, room2_id)}"
            if cached_score := self.cache_manager.get(cache_key):
                return cached_score
            
            room1_profile = self._get_room_profile(room1_id)
            room2_profile = self._get_room_profile(room2_id)
            
            if not room1_profile or not room2_profile:
                return self._create_empty_similarity_score(
                    room1_id, room2_id, SimilarityType.ROOM_FEATURES
                )
            
            factors = {
                'capacity': self._calculate_capacity_similarity(room1_profile, room2_profile),
                'area': self._calculate_area_similarity(room1_profile, room2_profile),
                'features': self._calculate_feature_similarity(room1_profile, room2_profile),
                'usage_patterns': self._calculate_usage_similarity(room1_profile, room2_profile),
                'user_overlap': self._calculate_user_overlap_similarity(room1_profile, room2_profile)
            }
            
            total_score = sum(
                score * self.room_similarity_weights[factor]
                for factor, score in factors.items()
            )
            
            confidence = self._calculate_room_similarity_confidence(room1_profile, room2_profile)
            
            similarity_score = SimilarityScore(
                entity1_id=room1_id,
                entity2_id=room2_id,
                similarity_score=min(max(total_score, 0.0), 1.0),
                similarity_type=SimilarityType.ROOM_FEATURES,
                contributing_factors=factors,
                confidence=confidence,
                calculated_at=datetime.now()
            )
            
            self.cache_manager.set(cache_key, similarity_score, ttl=self.cache_ttl)
            return similarity_score
            
        except Exception as e:
            logger.error(f"Error calculating room similarity: {str(e)}")
            return self._create_empty_similarity_score(
                room1_id, room2_id, SimilarityType.ROOM_FEATURES
            )
    
    def find_similar_rooms(
        self,
        target_room_id: int,
        limit: int = 10,
        min_similarity: float = 0.3,
        context: Optional[Dict] = None
    ) -> List[SimilarityScore]:
        """Find rooms similar to the target room"""
        try:
            rooms = self.db.query(MRBSRoom).filter(
                and_(
                    MRBSRoom.disabled == False,
                    MRBSRoom.id != target_room_id
                )
            ).all()
            
            similar_rooms = []
            for room in rooms:
                similarity = self.calculate_room_similarity(target_room_id, room.id, context)
                if similarity.similarity_score >= min_similarity:
                    similar_rooms.append(similarity)
            
            similar_rooms.sort(key=lambda x: x.similarity_score, reverse=True)
            return similar_rooms[:limit]
            
        except Exception as e:
            logger.error(f"Error finding similar rooms: {str(e)}")
            return []
    
    def find_best_alternative_room(
        self,
        original_room_id: int,
        booking_context: Dict[str, Any]
    ) -> Optional[int]:
        """Find the best alternative room based on similarity and context"""
        try:
            similar_rooms = self.find_similar_rooms(
                original_room_id,
                limit=20,
                min_similarity=0.2,
                context=booking_context
            )
            
            if not similar_rooms:
                return None
            
            scored_alternatives = []
            
            for similarity_score in similar_rooms:
                room_id = similarity_score.entity2_id
                score = similarity_score.similarity_score
                
                # Area preference bonus
                if booking_context.get('prefer_same_area', False):
                    original_room = self.db.query(MRBSRoom).filter(
                        MRBSRoom.id == original_room_id
                    ).first()
                    alternative_room = self.db.query(MRBSRoom).filter(
                        MRBSRoom.id == room_id
                    ).first()
                    
                    if (original_room and alternative_room and
                        original_room.area_id == alternative_room.area_id):
                        score *= 1.2
                
                # User history bonus
                if user_id := booking_context.get('user_id'):
                    user_bookings = self.db.query(MRBSEntry).filter(
                        and_(
                            MRBSEntry.create_by == user_id,
                            MRBSEntry.room_id == room_id
                        )
                    ).count()
                    
                    if user_bookings > 0:
                        score *= 1.1
                
                scored_alternatives.append((room_id, score))
            
            scored_alternatives.sort(key=lambda x: x[1], reverse=True)
            return scored_alternatives[0][0] if scored_alternatives else None
            
        except Exception as e:
            logger.error(f"Error finding best alternative room: {str(e)}")
            return None
    
    def get_room_similarity_matrix(self, room_ids: List[int]) -> Dict[Tuple[int, int], float]:
        """Calculate similarity matrix for a set of rooms"""
        try:
            similarity_matrix = {}
            
            for i, room1_id in enumerate(room_ids):
                for j, room2_id in enumerate(room_ids):
                    if i < j:
                        similarity = self.calculate_room_similarity(room1_id, room2_id)
                        similarity_matrix[(room1_id, room2_id)] = similarity.similarity_score
                        similarity_matrix[(room2_id, room1_id)] = similarity.similarity_score
                    elif i == j:
                        similarity_matrix[(room1_id, room2_id)] = 1.0
            
            return similarity_matrix
            
        except Exception as e:
            logger.error(f"Error calculating similarity matrix: {str(e)}")
            return {}
    
    # ==================== Time Similarity ====================
    
    def calculate_time_similarity(
        self,
        time1: datetime,
        time2: datetime,
        duration1: float,
        duration2: float,
        context: Optional[Dict] = None
    ) -> SimilarityScore:
        """Calculate similarity between two time slots"""
        try:
            profile1 = self._get_time_slot_profile(time1, duration1)
            profile2 = self._get_time_slot_profile(time2, duration2)
            
            factors = {
                'hour_proximity': self._calculate_hour_proximity(time1, time2),
                'day_similarity': self._calculate_day_similarity(time1, time2),
                'duration_match': self._calculate_duration_similarity(duration1, duration2),
                'usage_patterns': self._calculate_time_usage_similarity(profile1, profile2)
            }
            
            total_score = sum(
                score * self.time_similarity_weights[factor]
                for factor, score in factors.items()
            )
            
            confidence = self._calculate_time_similarity_confidence(profile1, profile2)
            
            return SimilarityScore(
                entity1_id=f"{time1.isoformat()}_{duration1}",
                entity2_id=f"{time2.isoformat()}_{duration2}",
                similarity_score=min(max(total_score, 0.0), 1.0),
                similarity_type=SimilarityType.TIME_PATTERNS,
                contributing_factors=factors,
                confidence=confidence,
                calculated_at=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error calculating time similarity: {str(e)}")
            return self._create_empty_similarity_score(
                f"{time1.isoformat()}_{duration1}",
                f"{time2.isoformat()}_{duration2}",
                SimilarityType.TIME_PATTERNS
            )
    
    def find_similar_time_slots(
        self,
        target_time: datetime,
        target_duration: float,
        search_window_hours: int = 48,
        limit: int = 10,
        min_similarity: float = 0.3
    ) -> List[SimilarityScore]:
        """Find time slots similar to the target time"""
        try:
            similar_slots = []
            
            start_search = target_time - timedelta(hours=search_window_hours // 2)
            end_search = target_time + timedelta(hours=search_window_hours // 2)
            current_time = start_search.replace(minute=0, second=0, microsecond=0)
            
            while current_time <= end_search:
                if current_time != target_time:
                    similarity = self.calculate_time_similarity(
                        target_time, current_time,
                        target_duration, target_duration
                    )
                    
                    if similarity.similarity_score >= min_similarity:
                        similar_slots.append(similarity)
                
                current_time += timedelta(hours=1)
            
            similar_slots.sort(key=lambda x: x.similarity_score, reverse=True)
            return similar_slots[:limit]
            
        except Exception as e:
            logger.error(f"Error finding similar time slots: {str(e)}")
            return []
    
    # ==================== User Similarity ====================
    
    def calculate_user_booking_similarity(
        self,
        user1_id: str,
        user2_id: str
    ) -> SimilarityScore:
        """Calculate similarity between two users' booking patterns"""
        try:
            user1_bookings = self._get_user_booking_history(user1_id)
            user2_bookings = self._get_user_booking_history(user2_id)
            
            if not user1_bookings or not user2_bookings:
                return self._create_empty_similarity_score(
                    user1_id, user2_id, SimilarityType.USER_BEHAVIOR
                )
            
            factors = {
                'room_preferences': self._calculate_room_preference_similarity(
                    user1_bookings, user2_bookings
                ),
                'time_preferences': self._calculate_time_preference_similarity(
                    user1_bookings, user2_bookings
                ),
                'duration_patterns': self._calculate_duration_preference_similarity(
                    user1_bookings, user2_bookings
                ),
                'booking_frequency': self._calculate_booking_frequency_similarity(
                    user1_bookings, user2_bookings
                )
            }
            
            total_score = sum(factors.values()) / len(factors)
            confidence = min(
                len(user1_bookings) / 20.0,
                len(user2_bookings) / 20.0,
                1.0
            )
            
            return SimilarityScore(
                entity1_id=user1_id,
                entity2_id=user2_id,
                similarity_score=min(max(total_score, 0.0), 1.0),
                similarity_type=SimilarityType.USER_BEHAVIOR,
                contributing_factors=factors,
                confidence=confidence,
                calculated_at=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error calculating user similarity: {str(e)}")
            return self._create_empty_similarity_score(
                user1_id, user2_id, SimilarityType.USER_BEHAVIOR
            )
    
    # ==================== Booking Similarity ====================
    
    def calculate_booking_similarity(
        self,
        booking1: Dict[str, Any],
        booking2: Dict[str, Any]
    ) -> SimilarityScore:
        """Calculate similarity between two booking requests/patterns"""
        try:
            factors = {}
            
            # Room similarity
            if 'room_id' in booking1 and 'room_id' in booking2:
                room_sim = self.calculate_room_similarity(
                    booking1['room_id'], booking2['room_id']
                )
                factors['room_similarity'] = room_sim.similarity_score
            else:
                factors['room_similarity'] = 0.0
            
            # Time similarity
            if 'start_time' in booking1 and 'start_time' in booking2:
                time1 = booking1['start_time']
                time2 = booking2['start_time']
                
                if isinstance(time1, (int, float)):
                    time1 = datetime.fromtimestamp(time1)
                if isinstance(time2, (int, float)):
                    time2 = datetime.fromtimestamp(time2)
                
                duration1 = booking1.get('duration', 1.0)
                duration2 = booking2.get('duration', 1.0)
                
                time_sim = self.calculate_time_similarity(
                    time1, time2, duration1, duration2
                )
                factors['time_similarity'] = time_sim.similarity_score
            else:
                factors['time_similarity'] = 0.0
            
            # User similarity
            if 'user_id' in booking1 and 'user_id' in booking2:
                if booking1['user_id'] == booking2['user_id']:
                    factors['user_similarity'] = 1.0
                else:
                    user_sim = self.calculate_user_booking_similarity(
                        booking1['user_id'], booking2['user_id']
                    )
                    factors['user_similarity'] = user_sim.similarity_score
            else:
                factors['user_similarity'] = 0.0
            
            # Purpose similarity
            if 'purpose' in booking1 and 'purpose' in booking2:
                purpose1 = (booking1['purpose'] or '').lower()
                purpose2 = (booking2['purpose'] or '').lower()
                
                if purpose1 == purpose2:
                    factors['purpose_similarity'] = 1.0
                elif purpose1 and purpose2:
                    words1 = set(purpose1.split())
                    words2 = set(purpose2.split())
                    overlap = len(words1 & words2)
                    union = len(words1 | words2)
                    factors['purpose_similarity'] = overlap / union if union > 0 else 0.0
                else:
                    factors['purpose_similarity'] = 0.0
            else:
                factors['purpose_similarity'] = 0.0
            
            weights = {
                'room_similarity': 0.3,
                'time_similarity': 0.3,
                'user_similarity': 0.2,
                'purpose_similarity': 0.2
            }
            
            total_score = sum(factors[factor] * weights[factor] for factor in factors)
            confidence = sum(1 for score in factors.values() if score > 0) / len(factors)
            
            return SimilarityScore(
                entity1_id=str(booking1),
                entity2_id=str(booking2),
                similarity_score=min(max(total_score, 0.0), 1.0),
                similarity_type=SimilarityType.BOOKING_CONTEXT,
                contributing_factors=factors,
                confidence=confidence,
                calculated_at=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error calculating booking similarity: {str(e)}")
            return self._create_empty_similarity_score(
                str(booking1), str(booking2), SimilarityType.BOOKING_CONTEXT
            )
    
    # ==================== Utility Methods ====================
    
    def clear_similarity_cache(self):
        """Clear all similarity-related cache entries"""
        try:
            self.cache_manager.clear_pattern("room_similarity:*")
            self.cache_manager.clear_pattern("room_profile:*")
            logger.info("Similarity cache cleared")
        except Exception as e:
            logger.error(f"Error clearing similarity cache: {str(e)}")
    
    def get_similarity_statistics(self) -> Dict[str, Any]:
        """Get statistics about similarity calculations"""
        try:
            total_rooms = self.db.query(MRBSRoom).filter(
                MRBSRoom.disabled == False
            ).count()
            
            sample_rooms = self.db.query(MRBSRoom).filter(
                MRBSRoom.disabled == False
            ).limit(10).all()
            
            sample_similarities = []
            for i, room1 in enumerate(sample_rooms):
                for room2 in sample_rooms[i+1:]:
                    similarity = self.calculate_room_similarity(room1.id, room2.id)
                    sample_similarities.append(similarity.similarity_score)
            
            return {
                'total_rooms': total_rooms,
                'sample_similarities_count': len(sample_similarities),
                'average_similarity': np.mean(sample_similarities) if sample_similarities else 0.0,
                'max_similarity': max(sample_similarities) if sample_similarities else 0.0,
                'min_similarity': min(sample_similarities) if sample_similarities else 0.0,
                'similarity_std': np.std(sample_similarities) if sample_similarities else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error getting similarity statistics: {str(e)}")
            return {}
    
    # ==================== Private Helper Methods ====================
    
    def _create_empty_similarity_score(
        self,
        entity1: Any,
        entity2: Any,
        similarity_type: SimilarityType
    ) -> SimilarityScore:
        """Create empty similarity score for error cases"""
        return SimilarityScore(
            entity1_id=entity1,
            entity2_id=entity2,
            similarity_score=0.0,
            similarity_type=similarity_type,
            contributing_factors={},
            confidence=0.0,
            calculated_at=datetime.now()
        )
    
    def _get_room_profile(self, room_id: int) -> Optional[RoomProfile]:
        """Get comprehensive room profile with usage statistics"""
        try:
            cache_key = f"room_profile:{room_id}"
            if cached_profile := self.cache_manager.get(cache_key):
                return cached_profile
            
            room = self.db.query(MRBSRoom).filter(MRBSRoom.id == room_id).first()
            if not room:
                return None
            
            area = self.db.query(MRBSArea).filter(MRBSArea.id == room.area_id).first()
            area_name = area.area_name if area else "Unknown"
            
            six_months_ago = datetime.now() - timedelta(days=180)
            six_months_timestamp = int(six_months_ago.timestamp())
            
            bookings = self.db.query(MRBSEntry).filter(
                and_(
                    MRBSEntry.room_id == room_id,
                    MRBSEntry.start_time >= six_months_timestamp
                )
            ).all()
            
            usage_frequency = len(bookings)
            total_hours = sum((b.end_time - b.start_time) / 3600 for b in bookings)
            avg_duration = total_hours / len(bookings) if bookings else 0.0
            
            hour_counts = Counter()
            common_users = set()
            booking_purposes = []
            
            for booking in bookings:
                start_dt = datetime.fromtimestamp(booking.start_time)
                hour_counts[start_dt.hour] += 1
                common_users.add(booking.create_by)
                if booking.name:
                    booking_purposes.append(booking.name)
            
            peak_hours = [hour for hour, _ in hour_counts.most_common(3)]
            
            total_possible_hours = 180 * 12
            utilization_rate = total_hours / total_possible_hours if total_possible_hours > 0 else 0.0
            
            features = self._extract_room_features(room.description or "")
            feature_vector = self._create_feature_vector(room, features)
            usage_vector = self._create_usage_vector(bookings)
            
            profile = RoomProfile(
                room_id=room.id,
                room_name=room.room_name,
                capacity=room.capacity,
                area_id=room.area_id,
                area_name=area_name,
                description=room.description or "",
                usage_frequency=usage_frequency,
                average_booking_duration=avg_duration,
                peak_usage_hours=peak_hours,
                common_users=common_users,
                booking_purposes=booking_purposes,
                utilization_rate=utilization_rate,
                feature_vector=feature_vector,
                usage_vector=usage_vector
            )
            
            self.cache_manager.set(cache_key, profile, ttl=self.cache_ttl)
            return profile
            
        except Exception as e:
            logger.error(f"Error getting room profile for room {room_id}: {str(e)}")
            return None
    
    def _get_time_slot_profile(
        self,
        time: datetime,
        duration: float
    ) -> TimeSlotProfile:
        """Get time slot profile with usage patterns"""
        try:
            start_hour = time.hour
            end_hour = (time.hour + int(duration)) % 24
            day_of_week = time.weekday()
            
            similar_slots = self.db.query(MRBSEntry).filter(
                and_(
                    func.extract('hour', func.from_unixtime(MRBSEntry.start_time)) == start_hour,
                    func.extract('dow', func.from_unixtime(MRBSEntry.start_time)) == day_of_week
                )
            ).limit(100).all()
            
            popularity_score = len(similar_slots) / 100.0
            typical_users = set(booking.create_by for booking in similar_slots)
            common_purposes = [booking.name for booking in similar_slots if booking.name]
            conflict_probability = min(popularity_score * 1.5, 1.0)
            
            seasonal_usage = {
                'spring': 0.25,
                'summer': 0.25,
                'fall': 0.25,
                'winter': 0.25
            }
            
            return TimeSlotProfile(
                start_hour=start_hour,
                end_hour=end_hour,
                day_of_week=day_of_week,
                duration_hours=duration,
                popularity_score=popularity_score,
                conflict_probability=conflict_probability,
                typical_users=typical_users,
                common_purposes=common_purposes,
                seasonal_usage=seasonal_usage
            )
            
        except Exception as e:
            logger.error(f"Error getting time slot profile: {str(e)}")
            return TimeSlotProfile(
                start_hour=time.hour,
                end_hour=(time.hour + int(duration)) % 24,
                day_of_week=time.weekday(),
                duration_hours=duration,
                popularity_score=0.0,
                conflict_probability=0.0,
                typical_users=set(),
                common_purposes=[],
                seasonal_usage={}
            )
    
    def _calculate_capacity_similarity(
        self,
        room1: RoomProfile,
        room2: RoomProfile
    ) -> float:
        """Calculate similarity based on room capacity"""
        if room1.capacity == 0 or room2.capacity == 0:
            return 0.5
        
        max_capacity = max(room1.capacity, room2.capacity)
        min_capacity = min(room1.capacity, room2.capacity)
        
        if max_capacity == 0:
            return 1.0
        
        ratio = min_capacity / max_capacity
        return 2 / (1 + math.exp(-4 * ratio)) - 1
    
    def _calculate_area_similarity(
        self,
        room1: RoomProfile,
        room2: RoomProfile
    ) -> float:
        """Calculate similarity based on room area/location"""
        return 1.0 if room1.area_id == room2.area_id else 0.0
    
    def _calculate_feature_similarity(
        self,
        room1: RoomProfile,
        room2: RoomProfile
    ) -> float:
        """Calculate similarity based on room features"""
        if not room1.feature_vector or not room2.feature_vector:
            return 0.5
        return self._cosine_similarity(room1.feature_vector, room2.feature_vector)
    
    def _calculate_usage_similarity(
        self,
        room1: RoomProfile,
        room2: RoomProfile
    ) -> float:
        """Calculate similarity based on usage patterns"""
        if not room1.usage_vector or not room2.usage_vector:
            return 0.5
        return self._cosine_similarity(room1.usage_vector, room2.usage_vector)
    
    def _calculate_user_overlap_similarity(
        self,
        room1: RoomProfile,
        room2: RoomProfile
    ) -> float:
        """Calculate similarity based on common users"""
        if not room1.common_users or not room2.common_users:
            return 0.0
        
        intersection = len(room1.common_users & room2.common_users)
        union = len(room1.common_users | room2.common_users)
        return intersection / union if union > 0 else 0.0
    
    def _calculate_hour_proximity(
        self,
        time1: datetime,
        time2: datetime
    ) -> float:
        """Calculate similarity based on hour proximity"""
        hour_diff = abs(time1.hour - time2.hour)
        hour_diff = min(hour_diff, 24 - hour_diff)
        return max(0, 1 - hour_diff / 12.0)
    
    def _calculate_day_similarity(
        self,
        time1: datetime,
        time2: datetime
    ) -> float:
        """Calculate similarity based on day of week"""
        if time1.weekday() == time2.weekday():
            return 1.0
        
        weekdays1 = time1.weekday() < 5
        weekdays2 = time2.weekday() < 5
        
        return 0.5 if weekdays1 == weekdays2 else 0.2
    
    def _calculate_duration_similarity(
        self,
        duration1: float,
        duration2: float
    ) -> float:
        """Calculate similarity based on booking duration"""
        if duration1 == 0 or duration2 == 0:
            return 0.5
        
        ratio = min(duration1, duration2) / max(duration1, duration2)
        return ratio
    
    def _calculate_time_usage_similarity(
        self,
        profile1: TimeSlotProfile,
        profile2: TimeSlotProfile
    ) -> float:
        """Calculate similarity based on time slot usage patterns"""
        popularity_sim = 1 - abs(profile1.popularity_score - profile2.popularity_score)
        
        if profile1.typical_users and profile2.typical_users:
            user_overlap = (
                len(profile1.typical_users & profile2.typical_users) /
                len(profile1.typical_users | profile2.typical_users)
            )
        else:
            user_overlap = 0.0
        
        return (popularity_sim + user_overlap) / 2
    
    def _extract_room_features(self, description: str) -> List[str]:
        """Extract features from room description"""
        if not description:
            return []
        
        features = []
        description_lower = description.lower()
        
        feature_keywords = {
            'projector': ['projector', 'projection'],
            'whiteboard': ['whiteboard', 'board'],
            'tv': ['tv', 'television', 'screen'],
            'ac': ['ac', 'air conditioning', 'aircon'],
            'wifi': ['wifi', 'wireless'],
            'video_conference': ['video', 'conference', 'zoom', 'teams'],
            'phone': ['phone', 'telephone'],
            'windows': ['window', 'natural light'],
            'kitchen': ['kitchen', 'pantry', 'coffee'],
            'parking': ['parking', 'garage']
        }
        
        for feature, keywords in feature_keywords.items():
            if any(keyword in description_lower for keyword in keywords):
                features.append(feature)
        
        return features
    
    def _create_feature_vector(
        self,
        room: MRBSRoom,
        features: List[str]
    ) -> List[float]:
        """Create feature vector for room"""
        feature_dims = [
            'projector', 'whiteboard', 'tv', 'ac', 'wifi',
            'video_conference', 'phone', 'windows', 'kitchen', 'parking'
        ]
        
        vector = [1.0 if dim in features else 0.0 for dim in feature_dims]
        vector.append(min(room.capacity / 50.0, 1.0))
        
        return vector
    
    def _create_usage_vector(self, bookings: List[MRBSEntry]) -> List[float]:
        """Create usage pattern vector"""
        if not bookings:
            return [0.0] * 10
        
        hour_counts = [0] * 24
        for booking in bookings:
            hour = datetime.fromtimestamp(booking.start_time).hour
            hour_counts[hour] += 1
        
        periods = [
            hour_counts[0:3], hour_counts[3:6], hour_counts[6:9], hour_counts[9:12],
            hour_counts[12:15], hour_counts[15:18], hour_counts[18:21], hour_counts[21:24]
        ]
        
        total_bookings = len(bookings)
        vector = [
            sum(period) / total_bookings if total_bookings > 0 else 0.0
            for period in periods
        ]
        
        durations = [(b.end_time - b.start_time) / 3600 for b in bookings]
        avg_duration = sum(durations) / len(durations) if durations else 0.0
        
        vector.append(min(avg_duration / 8.0, 1.0))
        vector.append(min(len(bookings) / 100.0, 1.0))
        
        return vector
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _get_user_booking_history(self, user_id: str) -> List[MRBSEntry]:
        """Get user's booking history"""
        return self.db.query(MRBSEntry).filter(
            MRBSEntry.create_by == user_id
        ).order_by(MRBSEntry.timestamp.desc()).limit(200).all()
    
    def _calculate_room_preference_similarity(
        self,
        bookings1: List[MRBSEntry],
        bookings2: List[MRBSEntry]
    ) -> float:
        """Calculate similarity in room preferences between users"""
        rooms1 = set(b.room_id for b in bookings1)
        rooms2 = set(b.room_id for b in bookings2)
        
        if not rooms1 or not rooms2:
            return 0.0
        
        intersection = len(rooms1 & rooms2)
        union = len(rooms1 | rooms2)
        return intersection / union if union > 0 else 0.0
    
    def _calculate_time_preference_similarity(
        self,
        bookings1: List[MRBSEntry],
        bookings2: List[MRBSEntry]
    ) -> float:
        """Calculate similarity in time preferences between users"""
        hours1 = [datetime.fromtimestamp(b.start_time).hour for b in bookings1]
        hours2 = [datetime.fromtimestamp(b.start_time).hour for b in bookings2]
        
        if not hours1 or not hours2:
            return 0.0
        
        hour_dist1 = [0] * 24
        hour_dist2 = [0] * 24
        
        for hour in hours1:
            hour_dist1[hour] += 1
        for hour in hours2:
            hour_dist2[hour] += 1
        
        total1 = sum(hour_dist1)
        total2 = sum(hour_dist2)
        
        if total1 == 0 or total2 == 0:
            return 0.0
        
        norm_dist1 = [count / total1 for count in hour_dist1]
        norm_dist2 = [count / total2 for count in hour_dist2]
        
        return self._cosine_similarity(norm_dist1, norm_dist2)
    
    def _calculate_duration_preference_similarity(
        self,
        bookings1: List[MRBSEntry],
        bookings2: List[MRBSEntry]
    ) -> float:
        """Calculate similarity in duration preferences between users"""
        durations1 = [(b.end_time - b.start_time) / 3600 for b in bookings1]
        durations2 = [(b.end_time - b.start_time) / 3600 for b in bookings2]
        
        if not durations1 or not durations2:
            return 0.0
        
        avg_duration1 = sum(durations1) / len(durations1)
        avg_duration2 = sum(durations2) / len(durations2)
        
        max_duration = max(avg_duration1, avg_duration2)
        min_duration = min(avg_duration1, avg_duration2)
        
        if max_duration == 0:
            return 1.0
        
        return min_duration / max_duration
    
    def _calculate_booking_frequency_similarity(
        self,
        bookings1: List[MRBSEntry],
        bookings2: List[MRBSEntry]
    ) -> float:
        """Calculate similarity in booking frequency patterns"""
        if not bookings1 or not bookings2:
            return 0.0
        
        dates1 = [datetime.fromtimestamp(b.start_time).date() for b in bookings1]
        dates2 = [datetime.fromtimestamp(b.start_time).date() for b in bookings2]
        
        if not dates1 or not dates2:
            return 0.0
        
        range1 = max((max(dates1) - min(dates1)).days, 1)
        range2 = max((max(dates2) - min(dates2)).days, 1)
        
        freq1 = len(bookings1) * 7 / range1
        freq2 = len(bookings2) * 7 / range2
        
        max_freq = max(freq1, freq2)
        min_freq = min(freq1, freq2)
        
        if max_freq == 0:
            return 1.0
        
        return min_freq / max_freq
    
    def _calculate_room_similarity_confidence(
        self,
        room1: RoomProfile,
        room2: RoomProfile
    ) -> float:
        """Calculate confidence in room similarity score"""
        factors = []
        
        if room1.usage_frequency > 10 and room2.usage_frequency > 10:
            factors.append(0.8)
        elif room1.usage_frequency > 5 and room2.usage_frequency > 5:
            factors.append(0.6)
        else:
            factors.append(0.3)
        
        factors.append(0.9 if room1.feature_vector and room2.feature_vector else 0.4)
        factors.append(0.7 if room1.description and room2.description else 0.3)
        
        return sum(factors) / len(factors)
    
    def _calculate_time_similarity_confidence(
        self,
        profile1: TimeSlotProfile,
        profile2: TimeSlotProfile
    ) -> float:
        """Calculate confidence in time similarity score"""
        factors = []
        
        if len(profile1.typical_users) > 5 and len(profile2.typical_users) > 5:
            factors.append(0.8)
        elif len(profile1.typical_users) > 2 and len(profile2.typical_users) > 2:
            factors.append(0.6)
        else:
            factors.append(0.3)
        
        factors.append(0.7 if profile1.common_purposes and profile2.common_purposes else 0.4)
        
        return sum(factors) / len(factors)