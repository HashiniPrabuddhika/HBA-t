import logging
import math
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
from collections import defaultdict, Counter
from dataclasses import dataclass
from enum import Enum
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_

from ..utils.time_utils import TimeUtils
from ..data.cache_manager import CacheManager
from src.models import MRBSEntry, MRBSRoom, MRBSArea

logger = logging.getLogger(__name__)

class SimilarityType(Enum):
    ROOM_FEATURES = "room_features"
    ROOM_USAGE = "room_usage"
    TIME_PATTERNS = "time_patterns"
    USER_BEHAVIOR = "user_behavior"
    BOOKING_CONTEXT = "booking_context"

@dataclass
class SimilarityScore:
    entity1_id: Any
    entity2_id: Any
    similarity_score: float
    similarity_type: SimilarityType
    contributing_factors: Dict[str, float]
    confidence: float
    calculated_at: datetime

@dataclass
class RoomProfile:
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
    def __init__(self, db: Session, cache_manager: CacheManager):
        self.db = db
        self.cache_manager = cache_manager
        self.time_utils = TimeUtils()
        self.room_similarity_weights = {'capacity': 0.25, 'area': 0.15, 'features': 0.20, 'usage_patterns': 0.25, 'user_overlap': 0.15}
        self.time_similarity_weights = {'hour_proximity': 0.30, 'day_similarity': 0.20, 'duration_match': 0.25, 'usage_patterns': 0.25}
        self.cache_ttl = 3600
    
    def calculate_room_similarity(self, room1_id: int, room2_id: int, context: Optional[Dict] = None) -> SimilarityScore:
        try:
            cache_key = f"room_similarity:{min(room1_id, room2_id)}:{max(room1_id, room2_id)}"
            cached_score = self.cache_manager.get(cache_key)
            if cached_score: return cached_score
            
            room1_profile = self._get_room_profile(room1_id)
            room2_profile = self._get_room_profile(room2_id)
            
            if not room1_profile or not room2_profile:
                return self._create_empty_similarity_score(room1_id, room2_id, SimilarityType.ROOM_FEATURES)
            
            factors = {
                'capacity': self._calculate_capacity_similarity(room1_profile, room2_profile),
                'area': self._calculate_area_similarity(room1_profile, room2_profile),
                'features': self._calculate_feature_similarity(room1_profile, room2_profile),
                'usage_patterns': self._calculate_usage_similarity(room1_profile, room2_profile),
                'user_overlap': self._calculate_user_overlap_similarity(room1_profile, room2_profile)
            }
            
            total_score = sum(score * self.room_similarity_weights[factor] for factor, score in factors.items())
            confidence = self._calculate_room_similarity_confidence(room1_profile, room2_profile)
            
            similarity_score = SimilarityScore(
                entity1_id=room1_id, entity2_id=room2_id,
                similarity_score=min(max(total_score, 0.0), 1.0),
                similarity_type=SimilarityType.ROOM_FEATURES,
                contributing_factors=factors, confidence=confidence, calculated_at=datetime.now()
            )
            
            self.cache_manager.set(cache_key, similarity_score, ttl=self.cache_ttl)
            return similarity_score
        except Exception as e:
            logger.error(f"Error calculating room similarity: {str(e)}")
            return self._create_empty_similarity_score(room1_id, room2_id, SimilarityType.ROOM_FEATURES)
    
    def calculate_time_similarity(self, time1: datetime, time2: datetime, duration1: float, duration2: float, context: Optional[Dict] = None) -> SimilarityScore:
        try:
            profile1 = self._get_time_slot_profile(time1, duration1)
            profile2 = self._get_time_slot_profile(time2, duration2)
            
            factors = {
                'hour_proximity': self._calculate_hour_proximity(time1, time2),
                'day_similarity': self._calculate_day_similarity(time1, time2),
                'duration_match': self._calculate_duration_similarity(duration1, duration2),
                'usage_patterns': self._calculate_time_usage_similarity(profile1, profile2)
            }
            
            total_score = sum(score * self.time_similarity_weights[factor] for factor, score in factors.items())
            confidence = self._calculate_time_similarity_confidence(profile1, profile2)
            
            return SimilarityScore(
                entity1_id=f"{time1.isoformat()}_{duration1}",
                entity2_id=f"{time2.isoformat()}_{duration2}",
                similarity_score=min(max(total_score, 0.0), 1.0),
                similarity_type=SimilarityType.TIME_PATTERNS,
                contributing_factors=factors, confidence=confidence, calculated_at=datetime.now()
            )
        except Exception as e:
            logger.error(f"Error calculating time similarity: {str(e)}")
            return self._create_empty_similarity_score(f"{time1.isoformat()}_{duration1}", f"{time2.isoformat()}_{duration2}", SimilarityType.TIME_PATTERNS)
    
    def find_similar_rooms(self, target_room_id: int, limit: int = 10, min_similarity: float = 0.3, context: Optional[Dict] = None) -> List[SimilarityScore]:
        try:
            rooms = self.db.query(MRBSRoom).filter(and_(MRBSRoom.disabled == False, MRBSRoom.id != target_room_id)).all()
            similar_rooms = [self.calculate_room_similarity(target_room_id, room.id, context) 
                           for room in rooms if self.calculate_room_similarity(target_room_id, room.id, context).similarity_score >= min_similarity]
            return sorted(similar_rooms, key=lambda x: x.similarity_score, reverse=True)[:limit]
        except Exception as e:
            logger.error(f"Error finding similar rooms: {str(e)}")
            return []
    
    def find_similar_time_slots(self, target_time: datetime, target_duration: float, search_window_hours: int = 48, limit: int = 10, min_similarity: float = 0.3) -> List[SimilarityScore]:
        try:
            start_search = target_time - timedelta(hours=search_window_hours // 2)
            end_search = target_time + timedelta(hours=search_window_hours // 2)
            current_time = start_search.replace(minute=0, second=0, microsecond=0)
            
            similar_slots = []
            while current_time <= end_search:
                if current_time != target_time:
                    similarity = self.calculate_time_similarity(target_time, current_time, target_duration, target_duration)
                    if similarity.similarity_score >= min_similarity:
                        similar_slots.append(similarity)
                current_time += timedelta(hours=1)
            
            return sorted(similar_slots, key=lambda x: x.similarity_score, reverse=True)[:limit]
        except Exception as e:
            logger.error(f"Error finding similar time slots: {str(e)}")
            return []
    
    def calculate_user_booking_similarity(self, user1_id: str, user2_id: str) -> SimilarityScore:
        try:
            user1_bookings = self._get_user_booking_history(user1_id)
            user2_bookings = self._get_user_booking_history(user2_id)
            
            if not user1_bookings or not user2_bookings:
                return self._create_empty_similarity_score(user1_id, user2_id, SimilarityType.USER_BEHAVIOR)
            
            factors = {
                'room_preferences': self._calculate_room_preference_similarity(user1_bookings, user2_bookings),
                'time_preferences': self._calculate_time_preference_similarity(user1_bookings, user2_bookings),
                'duration_patterns': self._calculate_duration_preference_similarity(user1_bookings, user2_bookings),
                'booking_frequency': self._calculate_booking_frequency_similarity(user1_bookings, user2_bookings)
            }
            
            total_score = sum(factors.values()) / len(factors)
            confidence = min(len(user1_bookings) / 20.0, len(user2_bookings) / 20.0, 1.0)
            
            return SimilarityScore(
                entity1_id=user1_id, entity2_id=user2_id,
                similarity_score=min(max(total_score, 0.0), 1.0),
                similarity_type=SimilarityType.USER_BEHAVIOR,
                contributing_factors=factors, confidence=confidence, calculated_at=datetime.now()
            )
        except Exception as e:
            logger.error(f"Error calculating user similarity: {str(e)}")
            return self._create_empty_similarity_score(user1_id, user2_id, SimilarityType.USER_BEHAVIOR)
    
    def get_room_similarity_matrix(self, room_ids: List[int]) -> Dict[Tuple[int, int], float]:
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
    
    def find_best_alternative_room(self, original_room_id: int, booking_context: Dict[str, Any]) -> Optional[int]:
        try:
            similar_rooms = self.find_similar_rooms(original_room_id, limit=20, min_similarity=0.2, context=booking_context)
            if not similar_rooms: return None
            
            scored_alternatives = []
            for similarity_score in similar_rooms:
                room_id = similarity_score.entity2_id
                score = similarity_score.similarity_score
                
                if booking_context.get('prefer_same_area', False):
                    orig_room = self.db.query(MRBSRoom).filter(MRBSRoom.id == original_room_id).first()
                    alt_room = self.db.query(MRBSRoom).filter(MRBSRoom.id == room_id).first()
                    if orig_room and alt_room and orig_room.area_id == alt_room.area_id:
                        score *= 1.2
                
                if booking_context.get('user_id'):
                    user_bookings = self.db.query(MRBSEntry).filter(and_(MRBSEntry.create_by == booking_context['user_id'], MRBSEntry.room_id == room_id)).count()
                    if user_bookings > 0: score *= 1.1
                
                scored_alternatives.append((room_id, score))
            
            return max(scored_alternatives, key=lambda x: x[1])[0] if scored_alternatives else None
        except Exception as e:
            logger.error(f"Error finding best alternative room: {str(e)}")
            return None
    
    def calculate_booking_similarity(self, booking1: Dict[str, Any], booking2: Dict[str, Any]) -> SimilarityScore:
        try:
            factors = {}
            
            if 'room_id' in booking1 and 'room_id' in booking2:
                factors['room_similarity'] = self.calculate_room_similarity(booking1['room_id'], booking2['room_id']).similarity_score
            else:
                factors['room_similarity'] = 0.0
            
            if 'start_time' in booking1 and 'start_time' in booking2:
                time1 = datetime.fromtimestamp(booking1['start_time']) if isinstance(booking1['start_time'], (int, float)) else booking1['start_time']
                time2 = datetime.fromtimestamp(booking2['start_time']) if isinstance(booking2['start_time'], (int, float)) else booking2['start_time']
                factors['time_similarity'] = self.calculate_time_similarity(time1, time2, booking1.get('duration', 1.0), booking2.get('duration', 1.0)).similarity_score
            else:
                factors['time_similarity'] = 0.0
            
            if 'user_id' in booking1 and 'user_id' in booking2:
                factors['user_similarity'] = 1.0 if booking1['user_id'] == booking2['user_id'] else self.calculate_user_booking_similarity(booking1['user_id'], booking2['user_id']).similarity_score
            else:
                factors['user_similarity'] = 0.0
            
            if 'purpose' in booking1 and 'purpose' in booking2:
                p1, p2 = (booking1['purpose'] or '').lower(), (booking2['purpose'] or '').lower()
                if p1 == p2:
                    factors['purpose_similarity'] = 1.0
                elif p1 and p2:
                    w1, w2 = set(p1.split()), set(p2.split())
                    factors['purpose_similarity'] = len(w1 & w2) / len(w1 | w2) if w1 | w2 else 0.0
                else:
                    factors['purpose_similarity'] = 0.0
            else:
                factors['purpose_similarity'] = 0.0
            
            weights = {'room_similarity': 0.3, 'time_similarity': 0.3, 'user_similarity': 0.2, 'purpose_similarity': 0.2}
            total_score = sum(factors[f] * weights[f] for f in factors)
            confidence = sum(1 for s in factors.values() if s > 0) / len(factors)
            
            return SimilarityScore(
                entity1_id=str(booking1), entity2_id=str(booking2),
                similarity_score=min(max(total_score, 0.0), 1.0),
                similarity_type=SimilarityType.BOOKING_CONTEXT,
                contributing_factors=factors, confidence=confidence, calculated_at=datetime.now()
            )
        except Exception as e:
            logger.error(f"Error calculating booking similarity: {str(e)}")
            return self._create_empty_similarity_score(str(booking1), str(booking2), SimilarityType.BOOKING_CONTEXT)
    
    def clear_similarity_cache(self):
        try:
            self.cache_manager.clear_pattern("room_similarity:*")
            self.cache_manager.clear_pattern("room_profile:*")
            logger.info("Similarity cache cleared")
        except Exception as e:
            logger.error(f"Error clearing similarity cache: {str(e)}")
    
    def get_similarity_statistics(self) -> Dict[str, Any]:
        try:
            total_rooms = self.db.query(MRBSRoom).filter(MRBSRoom.disabled == False).count()
            sample_rooms = self.db.query(MRBSRoom).filter(MRBSRoom.disabled == False).limit(10).all()
            
            similarities = [self.calculate_room_similarity(r1.id, r2.id).similarity_score 
                          for i, r1 in enumerate(sample_rooms) for r2 in sample_rooms[i+1:]]
            
            return {
                'total_rooms': total_rooms,
                'sample_similarities_count': len(similarities),
                'average_similarity': np.mean(similarities) if similarities else 0.0,
                'max_similarity': max(similarities) if similarities else 0.0,
                'min_similarity': min(similarities) if similarities else 0.0,
                'similarity_std': np.std(similarities) if similarities else 0.0
            }
        except Exception as e:
            logger.error(f"Error getting similarity statistics: {str(e)}")
            return {}
    
    # Helper methods (condensed)
    def _create_empty_similarity_score(self, id1, id2, sim_type):
        return SimilarityScore(id1, id2, 0.0, sim_type, {}, 0.0, datetime.now())
    
    def _get_room_profile(self, room_id: int) -> Optional[RoomProfile]:
        try:
            cache_key = f"room_profile:{room_id}"
            if cached := self.cache_manager.get(cache_key): return cached
            
            room = self.db.query(MRBSRoom).filter(MRBSRoom.id == room_id).first()
            if not room: return None
            
            area = self.db.query(MRBSArea).filter(MRBSArea.id == room.area_id).first()
            bookings = self.db.query(MRBSEntry).filter(and_(MRBSEntry.room_id == room_id, MRBSEntry.start_time >= int((datetime.now() - timedelta(days=180)).timestamp()))).all()
            
            usage_freq = len(bookings)
            total_hours = sum((b.end_time - b.start_time) / 3600 for b in bookings)
            avg_duration = total_hours / len(bookings) if bookings else 0.0
            
            hour_counts = Counter(datetime.fromtimestamp(b.start_time).hour for b in bookings)
            peak_hours = [h for h, c in hour_counts.most_common(3)]
            common_users = set(b.create_by for b in bookings)
            purposes = [b.name for b in bookings if b.name]
            utilization_rate = total_hours / (180 * 12) if total_hours else 0.0
            
            features = self._extract_room_features(room.description or "")
            feature_vector = self._create_feature_vector(room, features)
            usage_vector = self._create_usage_vector(bookings)
            
            profile = RoomProfile(room.id, room.room_name, room.capacity, room.area_id, area.area_name if area else "Unknown", room.description or "", usage_freq, avg_duration, peak_hours, common_users, purposes, utilization_rate, feature_vector, usage_vector)
            self.cache_manager.set(cache_key, profile, ttl=self.cache_ttl)
            return profile
        except Exception as e:
            logger.error(f"Error getting room profile: {str(e)}")
            return None
    
    def _get_time_slot_profile(self, time: datetime, duration: float) -> TimeSlotProfile:
        try:
            similar_slots = self.db.query(MRBSEntry).filter(and_(func.extract('hour', func.from_unixtime(MRBSEntry.start_time)) == time.hour, func.extract('dow', func.from_unixtime(MRBSEntry.start_time)) == time.weekday())).limit(100).all()
            return TimeSlotProfile(
                time.hour, (time.hour + int(duration)) % 24, time.weekday(), duration,
                len(similar_slots) / 100.0, min(len(similar_slots) / 100.0 * 1.5, 1.0),
                set(b.create_by for b in similar_slots), [b.name for b in similar_slots if b.name],
                {'spring': 0.25, 'summer': 0.25, 'fall': 0.25, 'winter': 0.25}
            )
        except Exception as e:
            logger.error(f"Error getting time slot profile: {str(e)}")
            return TimeSlotProfile(time.hour, (time.hour + int(duration)) % 24, time.weekday(), duration, 0.0, 0.0, set(), [], {})
    
    def _calculate_capacity_similarity(self, r1: RoomProfile, r2: RoomProfile) -> float:
        if r1.capacity == 0 or r2.capacity == 0: return 0.5
        ratio = min(r1.capacity, r2.capacity) / max(r1.capacity, r2.capacity)
        return 2 / (1 + math.exp(-4 * ratio)) - 1
    
    def _calculate_area_similarity(self, r1: RoomProfile, r2: RoomProfile) -> float:
        return 1.0 if r1.area_id == r2.area_id else 0.0
    
    def _calculate_feature_similarity(self, r1: RoomProfile, r2: RoomProfile) -> float:
        return self._cosine_similarity(r1.feature_vector, r2.feature_vector) if r1.feature_vector and r2.feature_vector else 0.5
    
    def _calculate_usage_similarity(self, r1: RoomProfile, r2: RoomProfile) -> float:
        return self._cosine_similarity(r1.usage_vector, r2.usage_vector) if r1.usage_vector and r2.usage_vector else 0.5
    
    def _calculate_user_overlap_similarity(self, r1: RoomProfile, r2: RoomProfile) -> float:
        if not r1.common_users or not r2.common_users: return 0.0
        return len(r1.common_users & r2.common_users) / len(r1.common_users | r2.common_users)
    
    def _calculate_hour_proximity(self, t1: datetime, t2: datetime) -> float:
        diff = min(abs(t1.hour - t2.hour), 24 - abs(t1.hour - t2.hour))
        return max(0, 1 - diff / 12.0)
    
    def _calculate_day_similarity(self, t1: datetime, t2: datetime) -> float:
        if t1.weekday() == t2.weekday(): return 1.0
        return 0.5 if (t1.weekday() < 5) == (t2.weekday() < 5) else 0.2
    
    def _calculate_duration_similarity(self, d1: float, d2: float) -> float:
        return 0.5 if d1 == 0 or d2 == 0 else min(d1, d2) / max(d1, d2)
    
    def _calculate_time_usage_similarity(self, p1: TimeSlotProfile, p2: TimeSlotProfile) -> float:
        pop_sim = 1 - abs(p1.popularity_score - p2.popularity_score)
        user_overlap = len(p1.typical_users & p2.typical_users) / len(p1.typical_users | p2.typical_users) if p1.typical_users and p2.typical_users else 0.0
        return (pop_sim + user_overlap) / 2
    
    def _extract_room_features(self, desc: str) -> List[str]:
        if not desc: return []
        features, desc_lower = [], desc.lower()
        keywords = {'projector': ['projector'], 'whiteboard': ['whiteboard'], 'tv': ['tv', 'screen'], 'ac': ['ac', 'air'], 'wifi': ['wifi'], 'video_conference': ['video', 'zoom'], 'phone': ['phone'], 'windows': ['window'], 'kitchen': ['kitchen'], 'parking': ['parking']}
        return [f for f, kws in keywords.items() if any(kw in desc_lower for kw in kws)]
    
    def _create_feature_vector(self, room: MRBSRoom, features: List[str]) -> List[float]:
        dims = ['projector', 'whiteboard', 'tv', 'ac', 'wifi', 'video_conference', 'phone', 'windows', 'kitchen', 'parking']
        return [1.0 if d in features else 0.0 for d in dims] + [min(room.capacity / 50.0, 1.0)]
    
    def _create_usage_vector(self, bookings: List[MRBSEntry]) -> List[float]:
        if not bookings: return [0.0] * 10
        hour_counts = [0] * 24
        for b in bookings: hour_counts[datetime.fromtimestamp(b.start_time).hour] += 1
        periods = [hour_counts[i:i+3] for i in range(0, 24, 3)]
        vector = [sum(p) / len(bookings) for p in periods]
        durations = [(b.end_time - b.start_time) / 3600 for b in bookings]
        return vector + [min(sum(durations) / len(durations) / 8.0, 1.0), min(len(bookings) / 100.0, 1.0)]
    
    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        if not v1 or not v2 or len(v1) != len(v2): return 0.0
        dot = sum(a * b for a, b in zip(v1, v2))
        norm1, norm2 = math.sqrt(sum(a * a for a in v1)), math.sqrt(sum(b * b for b in v2))
        return dot / (norm1 * norm2) if norm1 and norm2 else 0.0
    
    def _get_user_booking_history(self, user_id: str) -> List[MRBSEntry]:
        return self.db.query(MRBSEntry).filter(MRBSEntry.create_by == user_id).order_by(MRBSEntry.timestamp.desc()).limit(200).all()
    
    def _calculate_room_preference_similarity(self, b1: List[MRBSEntry], b2: List[MRBSEntry]) -> float:
        r1, r2 = set(b.room_id for b in b1), set(b.room_id for b in b2)
        return len(r1 & r2) / len(r1 | r2) if r1 and r2 and (r1 | r2) else 0.0
    
    def _calculate_time_preference_similarity(self, b1: List[MRBSEntry], b2: List[MRBSEntry]) -> float:
        if not b1 or not b2: return 0.0
        h1, h2 = [datetime.fromtimestamp(b.start_time).hour for b in b1], [datetime.fromtimestamp(b.start_time).hour for b in b2]
        d1, d2 = [0] * 24, [0] * 24
        for h in h1: d1[h] += 1
        for h in h2: d2[h] += 1
        t1, t2 = sum(d1), sum(d2)
        return self._cosine_similarity([c / t1 for c in d1], [c / t2 for c in d2]) if t1 and t2 else 0.0
    
    def _calculate_duration_preference_similarity(self, b1: List[MRBSEntry], b2: List[MRBSEntry]) -> float:
        if not b1 or not b2: return 0.0
        avg1 = sum((b.end_time - b.start_time) / 3600 for b in b1) / len(b1)
        avg2 = sum((b.end_time - b.start_time) / 3600 for b in b2) / len(b2)
        return min(avg1, avg2) / max(avg1, avg2) if max(avg1, avg2) else 1.0
    
    def _calculate_booking_frequency_similarity(self, b1: List[MRBSEntry], b2: List[MRBSEntry]) -> float:
        if not b1 or not b2: return 0.0
        dates1 = [datetime.fromtimestamp(b.start_time).date() for b in b1]
        dates2 = [datetime.fromtimestamp(b.start_time).date() for b in b2]
        range1 = max((max(dates1) - min(dates1)).days, 1)
        range2 = max((max(dates2) - min(dates2)).days, 1)
        freq1, freq2 = len(b1) * 7 / range1, len(b2) * 7 / range2
        return min(freq1, freq2) / max(freq1, freq2) if max(freq1, freq2) else 1.0
    
    def _calculate_room_similarity_confidence(self, r1: RoomProfile, r2: RoomProfile) -> float:
        factors = []
        factors.append(0.8 if r1.usage_frequency > 10 and r2.usage_frequency > 10 else 0.6 if r1.usage_frequency > 5 and r2.usage_frequency > 5 else 0.3)
        factors.append(0.9 if r1.feature_vector and r2.feature_vector else 0.4)
        factors.append(0.7 if r1.description and r2.description else 0.3)
        return sum(factors) / len(factors)
    
    def _calculate_time_similarity_confidence(self, p1: TimeSlotProfile, p2: TimeSlotProfile) -> float:
        factors = []
        factors.append(0.8 if len(p1.typical_users) > 5 and len(p2.typical_users) > 5 else 0.6 if len(p1.typical_users) > 2 and len(p2.typical_users) > 2 else 0.3)
        factors.append(0.7 if p1.common_purposes and p2.common_purposes else 0.4)
        return sum(factors) / len(factors)