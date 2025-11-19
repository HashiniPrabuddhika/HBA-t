import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict, Counter
import numpy as np
from dataclasses import dataclass
from enum import Enum
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_

try:
    from ..models.embedding_model import EmbeddingModel
except ImportError:
    EmbeddingModel = None

try:
    from ..utils.time_utils import TimeUtils
except ImportError:
    class TimeUtils:
        pass

try:
    from ..utils.metrics import RecommendationMetrics
except ImportError:
    class RecommendationMetrics:
        pass

try:
    from ..data.cache_manager import CacheManager
except ImportError:
    class CacheManager:
        def get(self, key): return None
        def set(self, key, value, ttl=None): pass

try:
    from src.models import MRBSEntry, MRBSRoom, MRBSRepeat
except ImportError:
    class MRBSEntry:
        def __init__(self):
            self.create_by = None
            self.timestamp = datetime.now()
            self.start_time = 0
            self.end_time = 0
            self.room_id = None
            self.room = None
            self.repeat_id = None
    
    class MRBSRoom:
        def __init__(self):
            self.id = None
            self.capacity = 10
            self.description = ""
    
    class MRBSRepeat:
        def __init__(self):
            self.id = None

logger = logging.getLogger(__name__)

class PreferenceType(Enum):
    TIME_SLOT = "time_slot"
    ROOM_TYPE = "room_type"
    DURATION = "duration"
    CAPACITY = "capacity"
    FEATURES = "features"
    LOCATION = "location"
    RECURRENCE = "recurrence"

class LearningStrategy(Enum):
    IMPLICIT = "implicit"
    EXPLICIT = "explicit"
    HYBRID = "hybrid"

@dataclass
class PreferenceScore:
    attribute: str
    value: Any
    score: float
    confidence: float
    last_updated: datetime
    source: str

@dataclass
class UserProfile:
    user_id: str
    preferences: Dict[PreferenceType, List[PreferenceScore]]
    booking_patterns: Dict[str, Any]
    interaction_history: List[Dict[str, Any]]
    created_at: datetime
    updated_at: datetime

class PreferenceLearner:
    from ..models.embedding_model import EmbeddingModel
    def __init__(self, db: Session, embedding_model: Optional[EmbeddingModel] = None, cache_manager: Optional[CacheManager] = None):
        self.db = db
        self.embedding_model = embedding_model or (EmbeddingModel() if EmbeddingModel else None)
        self.cache_manager = cache_manager or CacheManager()
        self.time_utils = TimeUtils()
        self.metrics = RecommendationMetrics()
        self.decay_factor = 0.95
        self.min_interactions = 5
        self.confidence_threshold = 0.6
        self.tables_exist = self._check_tables_exist()
        
    def _check_tables_exist(self) -> bool:
        try:
            self.db.query(MRBSEntry).limit(1).all()
            return True
        except Exception as e:
            logger.warning(f"Database tables not available, running in mock mode: {e}")
            return False
        
    def learn_user_preferences(self, user_id: str, strategy: LearningStrategy = LearningStrategy.HYBRID) -> UserProfile:
        try:
            cached_profile = self._get_cached_profile(user_id)
            if cached_profile and self._is_profile_fresh(cached_profile):
                return cached_profile
            
            if not self.tables_exist:
                return self._get_mock_profile(user_id)
            
            if strategy == LearningStrategy.IMPLICIT:
                preferences = self._learn_implicit_preferences(user_id)
            elif strategy == LearningStrategy.EXPLICIT:
                preferences = self._learn_explicit_preferences(user_id)
            else:
                implicit_prefs = self._learn_implicit_preferences(user_id)
                explicit_prefs = self._learn_explicit_preferences(user_id)
                preferences = self._merge_preferences(implicit_prefs, explicit_prefs)
            
            patterns = self._extract_booking_patterns(user_id)
            interactions = self._get_interaction_history(user_id)
            
            profile = UserProfile(user_id=user_id, preferences=preferences, booking_patterns=patterns,
                                interaction_history=interactions, created_at=datetime.now(), updated_at=datetime.now())
            
            self._cache_profile(profile)
            self._update_user_preferences_db(profile)
            logger.info(f"Learned preferences for user {user_id} using {strategy.value} strategy")
            return profile
            
        except Exception as e:
            logger.error(f"Error learning preferences for user {user_id}: {str(e)}")
            return self._get_default_profile(user_id)
    
    def _get_mock_profile(self, user_id: str) -> UserProfile:
        mock_preferences = {
            PreferenceType.TIME_SLOT: [
                PreferenceScore("preferred_hour", 10, 0.8, 0.7, datetime.now(), "mock"),
                PreferenceScore("preferred_day_of_week", 1, 0.6, 0.6, datetime.now(), "mock")
            ],
            PreferenceType.ROOM_TYPE: [PreferenceScore("room_size_category", "medium", 0.7, 0.8, datetime.now(), "mock")],
            PreferenceType.DURATION: [PreferenceScore("preferred_duration", 2.0, 0.9, 0.8, datetime.now(), "mock")]
        }
        
        mock_patterns = {
            'total_bookings': 15, 'booking_frequency': {'per_week': 2.5, 'per_month': 10},
            'peak_usage_times': {'peak_hours': [10, 14, 16], 'peak_days': [1, 2, 3]},
            'typical_advance_booking': 24.0, 'cancellation_rate': 0.1, 'preferred_booking_duration': 2.0
        }
        
        return UserProfile(user_id=user_id, preferences=mock_preferences, booking_patterns=mock_patterns,
                         interaction_history=[], created_at=datetime.now(), updated_at=datetime.now())
    
    def _learn_implicit_preferences(self, user_id: str) -> Dict[PreferenceType, List[PreferenceScore]]:
        preferences = defaultdict(list)
        
        if not self.tables_exist:
            return dict(preferences)
        
        try:
            bookings = self.db.query(MRBSEntry).filter(MRBSEntry.create_by == user_id).order_by(MRBSEntry.timestamp.desc()).limit(500).all()
            
            if len(bookings) < self.min_interactions:
                logger.info(f"Insufficient booking history for user {user_id}")
                return dict(preferences)
            
            self._learn_time_preferences(bookings, preferences)
            self._learn_room_preferences(bookings, preferences)
            self._learn_duration_preferences(bookings, preferences)
            self._learn_capacity_preferences(bookings, preferences)
            self._learn_feature_preferences(bookings, preferences)
            self._learn_recurrence_preferences(bookings, preferences)
            
        except Exception as e:
            logger.error(f"Error learning implicit preferences: {e}")
        
        return dict(preferences)
    
    def _learn_explicit_preferences(self, user_id: str) -> Dict[PreferenceType, List[PreferenceScore]]:
        preferences = defaultdict(list)
        logger.info(f"Explicit preference learning not available without feedback tables for user {user_id}")
        return dict(preferences)
    
    def _learn_time_preferences(self, bookings: List[MRBSEntry], preferences: Dict[PreferenceType, List[PreferenceScore]]):
        time_slots = []
        
        for booking in bookings:
            try:
                start_time = datetime.fromtimestamp(booking.start_time)
                time_slots.append({'hour': start_time.hour, 'day_of_week': start_time.weekday(), 'timestamp': booking.timestamp})
            except (AttributeError, ValueError) as e:
                logger.warning(f"Invalid booking time data: {e}")
                continue
        
        if not time_slots:
            return
        
        hour_counts = Counter([slot['hour'] for slot in time_slots])
        total_bookings = len(time_slots)
        
        for hour, count in hour_counts.most_common():
            score = count / total_bookings
            confidence = min(score * 2, 1.0)
            
            if score > 0.1:
                preferences[PreferenceType.TIME_SLOT].append(
                    PreferenceScore("preferred_hour", hour, score, confidence, datetime.now(), "booking"))
        
        dow_counts = Counter([slot['day_of_week'] for slot in time_slots])
        
        for dow, count in dow_counts.most_common():
            score = count / total_bookings
            confidence = min(score * 2, 1.0)
            
            if score > 0.1:
                preferences[PreferenceType.TIME_SLOT].append(
                    PreferenceScore("preferred_day_of_week", dow, score, confidence, datetime.now(), "booking"))
    
    def _learn_room_preferences(self, bookings: List[MRBSEntry], preferences: Dict[PreferenceType, List[PreferenceScore]]):
        room_usage = defaultdict(int)
        room_types = defaultdict(int)
        
        for booking in bookings:
            try:
                room = booking.room
                if room:
                    room_usage[room.id] += 1
                    capacity = getattr(room, 'capacity', 10)
                    room_type = "small" if capacity <= 5 else "medium" if capacity <= 15 else "large"
                    room_types[room_type] += 1
            except AttributeError:
                continue
        
        total_bookings = len(bookings)
        
        if total_bookings == 0:
            return
        
        for room_type, count in room_types.items():
            score = count / total_bookings
            confidence = min(score * 1.5, 1.0)
            
            if score > 0.15:
                preferences[PreferenceType.ROOM_TYPE].append(
                    PreferenceScore("room_size_category", room_type, score, confidence, datetime.now(), "booking"))
        
        for room_id, count in room_usage.items():
            score = count / total_bookings
            confidence = min(score * 3, 1.0)
            
            if score > 0.2:
                preferences[PreferenceType.ROOM_TYPE].append(
                    PreferenceScore("preferred_room_id", room_id, score, confidence, datetime.now(), "booking"))
    
    def _learn_duration_preferences(self, bookings: List[MRBSEntry], preferences: Dict[PreferenceType, List[PreferenceScore]]):
        durations = []
        
        for booking in bookings:
            try:
                duration = (booking.end_time - booking.start_time) / 3600
                if duration > 0:
                    durations.append(duration)
            except (AttributeError, TypeError):
                continue
        
        if not durations:
            return
        
        duration_counts = Counter([round(d * 2) / 2 for d in durations])
        total_bookings = len(durations)
        avg_duration = np.mean(durations)
        
        for duration, count in duration_counts.most_common():
            score = count / total_bookings
            confidence = min(score * 2, 1.0)
            
            if score > 0.1 and duration > 0:
                preferences[PreferenceType.DURATION].append(
                    PreferenceScore("preferred_duration", duration, score, confidence, datetime.now(), "booking"))
        
        preferences[PreferenceType.DURATION].append(
            PreferenceScore("average_duration", avg_duration, 1.0, 0.8, datetime.now(), "booking"))
    
    def _learn_capacity_preferences(self, bookings: List[MRBSEntry], preferences: Dict[PreferenceType, List[PreferenceScore]]):
        capacities = []
        
        for booking in bookings:
            try:
                if booking.room and hasattr(booking.room, 'capacity') and booking.room.capacity:
                    capacities.append(booking.room.capacity)
            except AttributeError:
                continue
        
        if not capacities:
            return
        
        avg_capacity = np.mean(capacities)
        capacity_ranges = {"small": (0, 10), "medium": (11, 25), "large": (26, 50), "extra_large": (51, float('inf'))}
        
        range_counts = defaultdict(int)
        for capacity in capacities:
            for range_name, (min_cap, max_cap) in capacity_ranges.items():
                if min_cap <= capacity <= max_cap:
                    range_counts[range_name] += 1
                    break
        
        total_bookings = len(capacities)
        
        for range_name, count in range_counts.items():
            score = count / total_bookings
            confidence = min(score * 1.5, 1.0)
            
            if score > 0.15:
                preferences[PreferenceType.CAPACITY].append(
                    PreferenceScore("capacity_range", range_name, score, confidence, datetime.now(), "booking"))
        
        preferences[PreferenceType.CAPACITY].append(
            PreferenceScore("average_capacity", avg_capacity, 1.0, 0.7, datetime.now(), "booking"))
    
    def _learn_feature_preferences(self, bookings: List[MRBSEntry], preferences: Dict[PreferenceType, List[PreferenceScore]]):
        room_features = {}
        
        for booking in bookings:
            try:
                if booking.room and hasattr(booking.room, 'description') and booking.room.description:
                    description = booking.room.description.lower()
                    features = []
                    
                    feature_keywords = {
                        'projector': ['projector', 'projection'], 'whiteboard': ['whiteboard', 'board'],
                        'tv': ['tv', 'television', 'screen'], 'ac': ['ac', 'air conditioning', 'aircon'],
                        'wifi': ['wifi', 'wireless'], 'video_conference': ['video', 'conference', 'zoom', 'teams'],
                        'phone': ['phone', 'telephone']
                    }
                    
                    for feature, keywords in feature_keywords.items():
                        if any(keyword in description for keyword in keywords):
                            features.append(feature)
                    
                    if features:
                        room_features[booking.room_id] = features
            except AttributeError:
                continue
        
        feature_counts = defaultdict(int)
        total_bookings = len(bookings)
        
        for room_id, features in room_features.items():
            bookings_with_room = sum(1 for b in bookings if getattr(b, 'room_id', None) == room_id)
            for feature in features:
                feature_counts[feature] += bookings_with_room
        
        for feature_name, count in feature_counts.items():
            if total_bookings > 0:
                score = count / total_bookings
                confidence = min(score * 1.5, 1.0)
                
                if score > 0.1:
                    preferences[PreferenceType.FEATURES].append(
                        PreferenceScore("room_feature", feature_name, score, confidence, datetime.now(), "booking"))
    
    def _learn_recurrence_preferences(self, bookings: List[MRBSEntry], preferences: Dict[PreferenceType, List[PreferenceScore]]):
        try:
            recurring_bookings = [b for b in bookings if getattr(b, 'repeat_id', None) is not None]
            one_time_bookings = [b for b in bookings if getattr(b, 'repeat_id', None) is None]
            total_bookings = len(bookings)
            
            if total_bookings == 0:
                return
            
            recurring_ratio = len(recurring_bookings) / total_bookings
            one_time_ratio = len(one_time_bookings) / total_bookings
            
            if recurring_ratio > 0.1:
                preferences[PreferenceType.RECURRENCE].append(
                    PreferenceScore("booking_type", "recurring", recurring_ratio, min(recurring_ratio * 2, 1.0), datetime.now(), "booking"))
            
            if one_time_ratio > 0.1:
                preferences[PreferenceType.RECURRENCE].append(
                    PreferenceScore("booking_type", "one_time", one_time_ratio, min(one_time_ratio * 2, 1.0), datetime.now(), "booking"))
        except Exception as e:
            logger.warning(f"Error learning recurrence preferences: {e}")
    
    def _merge_preferences(self, implicit_prefs: Dict, explicit_prefs: Dict) -> Dict:
        merged = defaultdict(list)
        implicit_weight = 0.7
        explicit_weight = 0.3
        
        all_types = set(implicit_prefs.keys()) | set(explicit_prefs.keys())
        
        for pref_type in all_types:
            implicit_scores = implicit_prefs.get(pref_type, [])
            explicit_scores = explicit_prefs.get(pref_type, [])
            
            for score in implicit_scores:
                merged[pref_type].append(PreferenceScore(
                    score.attribute, score.value, score.score * implicit_weight,
                    score.confidence * implicit_weight, score.last_updated, f"implicit_{score.source}"))
            
            for score in explicit_scores:
                merged[pref_type].append(PreferenceScore(
                    score.attribute, score.value, score.score * explicit_weight,
                    score.confidence * explicit_weight, score.last_updated, f"explicit_{score.source}"))
        
        return dict(merged)
    
    def update_preferences_from_feedback(self, user_id: str, feedback: Dict[str, Any]):
        try:
            profile = self.learn_user_preferences(user_id)
            new_preferences = self._process_new_feedback(feedback)
            
            for pref_type, new_scores in new_preferences.items():
                if pref_type in profile.preferences:
                    profile.preferences[pref_type].extend(new_scores)
                else:
                    profile.preferences[pref_type] = new_scores
            
            profile.updated_at = datetime.now()
            self._cache_profile(profile)
            self._update_user_preferences_db(profile)
            logger.info(f"Updated preferences for user {user_id} based on feedback")
            
        except Exception as e:
            logger.error(f"Error updating preferences from feedback: {str(e)}")
    
    def get_preference_strength(self, user_id: str, preference_type: PreferenceType, attribute: str, value: Any) -> float:
        try:
            profile = self.learn_user_preferences(user_id)
            
            if preference_type not in profile.preferences:
                return 0.0
            
            for score in profile.preferences[preference_type]:
                if score.attribute == attribute and score.value == value:
                    return score.score * score.confidence
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error getting preference strength: {str(e)}")
            return 0.0
    
    def _extract_booking_patterns(self, user_id: str) -> Dict[str, Any]:
        if not self.tables_exist:
            return {
                'total_bookings': 5, 'booking_frequency': {'per_week': 1.5, 'per_month': 6},
                'peak_usage_times': {'peak_hours': [10, 14], 'peak_days': [1, 2]},
                'typical_advance_booking': 24.0, 'cancellation_rate': 0.1, 'preferred_booking_duration': 2.0
            }
        
        try:
            bookings = self.db.query(MRBSEntry).filter(MRBSEntry.create_by == user_id).order_by(MRBSEntry.timestamp.desc()).limit(200).all()
            
            if not bookings:
                return {}
            
            return {
                'total_bookings': len(bookings),
                'booking_frequency': self._calculate_booking_frequency(bookings),
                'peak_usage_times': self._find_peak_usage_times(bookings),
                'typical_advance_booking': self._calculate_advance_booking_time(bookings),
                'cancellation_rate': self._calculate_cancellation_rate(user_id),
                'preferred_booking_duration': self._calculate_preferred_duration(bookings)
            }
        except Exception as e:
            logger.error(f"Error extracting booking patterns: {e}")
            return {}
    
    def _calculate_booking_frequency(self, bookings: List[MRBSEntry]) -> Dict[str, float]:
        if not bookings:
            return {}
        
        try:
            dates = [datetime.fromtimestamp(b.start_time).date() for b in bookings if hasattr(b, 'start_time')]
            
            if not dates:
                return {}
            
            date_range = (max(dates) - min(dates)).days
            
            if date_range == 0:
                return {'per_week': len(bookings), 'per_month': len(bookings)}
            
            bookings_per_week = len(bookings) * 7 / date_range
            bookings_per_month = len(bookings) * 30 / date_range
            
            return {'per_week': round(bookings_per_week, 2), 'per_month': round(bookings_per_month, 2)}
        except Exception as e:
            logger.warning(f"Error calculating booking frequency: {e}")
            return {}
    
    def _find_peak_usage_times(self, bookings: List[MRBSEntry]) -> Dict[str, List[int]]:
        try:
            hours = []
            days = []
            
            for booking in bookings:
                try:
                    start_time = datetime.fromtimestamp(booking.start_time)
                    hours.append(start_time.hour)
                    days.append(start_time.weekday())
                except (AttributeError, ValueError):
                    continue
            
            if not hours or not days:
                return {'peak_hours': [], 'peak_days': []}
            
            hour_counts = Counter(hours)
            day_counts = Counter(days)
            
            return {
                'peak_hours': [hour for hour, count in hour_counts.most_common(3)],
                'peak_days': [day for day, count in day_counts.most_common(3)]
            }
        except Exception as e:
            logger.warning(f"Error finding peak usage times: {e}")
            return {'peak_hours': [], 'peak_days': []}
    
    def _calculate_advance_booking_time(self, bookings: List[MRBSEntry]) -> float:
        try:
            advance_times = []
            
            for booking in bookings:
                try:
                    booking_time = booking.timestamp
                    start_time = datetime.fromtimestamp(booking.start_time)
                    advance_hours = (start_time - booking_time).total_seconds() / 3600
                    
                    if advance_hours > 0:
                        advance_times.append(advance_hours)
                except (AttributeError, ValueError):
                    continue
            
            return np.median(advance_times) if advance_times else 24.0
        except Exception as e:
            logger.warning(f"Error calculating advance booking time: {e}")
            return 24.0
    
    def _calculate_cancellation_rate(self, user_id: str) -> float:
        return 0.1
    
    def _calculate_preferred_duration(self, bookings: List[MRBSEntry]) -> float:
        try:
            durations = []
            for booking in bookings:
                try:
                    duration = (booking.end_time - booking.start_time) / 3600
                    if duration > 0:
                        durations.append(duration)
                except (AttributeError, TypeError):
                    continue
            
            return np.median(durations) if durations else 2.0
        except Exception as e:
            logger.warning(f"Error calculating preferred duration: {e}")
            return 2.0
    
    def _get_interaction_history(self, user_id: str) -> List[Dict[str, Any]]:
        return []
    
    def _get_cached_profile(self, user_id: str) -> Optional[UserProfile]:
        try:
            return self.cache_manager.get(f"user_profile:{user_id}")
        except Exception as e:
            logger.warning(f"Error getting cached profile: {e}")
            return None
    
    def _cache_profile(self, profile: UserProfile):
        try:
            self.cache_manager.set(f"user_profile:{profile.user_id}", profile, ttl=3600)
        except Exception as e:
            logger.warning(f"Error caching profile: {e}")
    
    def _is_profile_fresh(self, profile: UserProfile) -> bool:
        try:
            return (datetime.now() - profile.updated_at).total_seconds() < 1800
        except Exception:
            return False
    
    def _update_user_preferences_db(self, profile: UserProfile):
        try:
            logger.debug(f"Preference storage skipped - no UserPreference table available")
        except Exception as e:
            logger.error(f"Error updating preferences in database: {str(e)}")
            if hasattr(self, 'db') and self.db:
                try:
                    self.db.rollback()
                except Exception:
                    pass
    
    def _get_default_profile(self, user_id: str) -> UserProfile:
        return UserProfile(user_id=user_id, preferences={}, booking_patterns={}, interaction_history=[],
                         created_at=datetime.now(), updated_at=datetime.now())
    
    def _process_new_feedback(self, feedback: Dict[str, Any]) -> Dict[PreferenceType, List[PreferenceScore]]:
        preferences = defaultdict(list)
        
        try:
            if 'time_preference' in feedback:
                preferences[PreferenceType.TIME_SLOT].append(
                    PreferenceScore("feedback_time_preference", feedback['time_preference'], 0.8, 0.7, datetime.now(), "feedback"))
            
            if 'room_features' in feedback:
                for feature in feedback['room_features']:
                    preferences[PreferenceType.FEATURES].append(
                        PreferenceScore("room_feature", feature, 0.9, 0.8, datetime.now(), "feedback"))
        except Exception as e:
            logger.error(f"Error processing feedback: {e}")
        
        return dict(preferences)