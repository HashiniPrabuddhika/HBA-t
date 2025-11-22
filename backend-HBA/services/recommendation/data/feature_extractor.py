import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
from collections import defaultdict, Counter
import re

logger = logging.getLogger(__name__)

class FeatureExtractor:
    
    def __init__(self):
        self.user_features_cache = {}
        self.room_features_cache = {}
        self.time_slots = [f"{h:02d}:{m:02d}" for h in range(6, 22) for m in [0, 30]]
        self.equipment_types = ['projector', 'whiteboard', 'tv', 'microphone', 'camera', 'computer', 'phone', 'speakers', 'screen', 'flip_chart']
        self.amenity_types = ['wifi', 'ac', 'heating', 'natural_light', 'quiet', 'private', 'accessible', 'parking', 'kitchen', 'printer']
        logger.info("FeatureExtractor initialized")
    
    def extract_user_features(self, user_data: Dict[str, Any], booking_history: List[Dict[str, Any]] = None) -> Dict[str, Any]:
       
        try:
            user_id = user_data.get('user_id') or user_data.get('id')
            if user_id in self.user_features_cache:
                return self.user_features_cache[user_id]
            
            features = {
                'user_id': user_id,
                'demographic_features': self._extract_demographic_features(user_data),
                'preference_features': self._extract_preference_features(user_data),
                'behavioral_features': self._extract_behavioral_features(booking_history or []),
                'temporal_features': self._extract_temporal_features(booking_history or []),
                'satisfaction_features': self._extract_satisfaction_features(booking_history or []),
                'usage_patterns': self._extract_usage_patterns(booking_history or [])
            }
            features['feature_vector'] = self._create_user_feature_vector(features)
            self.user_features_cache[user_id] = features
            logger.info(f"Extracted features for user {user_id}")
            return features
        except Exception as e:
            logger.error(f"Failed to extract user features: {e}")
            return {'error': str(e)}
    
    def _extract_demographic_features(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        return {k: user_data.get(k, 'unknown' if k in ['department', 'role', 'seniority_level', 'location'] else 0) 
                for k in ['department', 'role', 'seniority_level', 'team_size', 'location']}
    
    def _extract_preference_features(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        prefs = user_data.get('preferences', {})
        return {
            'preferred_capacity_min': prefs.get('capacity_min', 1),
            'preferred_capacity_max': prefs.get('capacity_max', 50),
            'preferred_duration_min': prefs.get('duration_min', 30),
            'preferred_duration_max': prefs.get('duration_max', 480),
            'required_equipment': prefs.get('equipment', []),
            'preferred_amenities': prefs.get('amenities', []),
            'preferred_floors': prefs.get('floors', []),
            'preferred_buildings': prefs.get('buildings', []),
            'avoid_rooms': prefs.get('avoid_rooms', []),
            'accessibility_needs': prefs.get('accessibility_needs', False)
        }
    
    def _extract_behavioral_features(self, booking_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not booking_history:
            return self._get_default_behavioral_features()
        
        total_bookings = len(booking_history)
        room_counts = Counter(b.get('room_id') for b in booking_history)
        durations = [(datetime.fromisoformat(b.get('end_time', '')) - datetime.fromisoformat(b.get('start_time', ''))).total_seconds() / 60 
                    for b in booking_history if b.get('start_time') and b.get('end_time')]
        advance_days = [max(0, (datetime.fromisoformat(b.get('start_time', '')) - datetime.fromisoformat(b.get('booking_time', ''))).total_seconds() / 86400) 
                       for b in booking_history if b.get('booking_time') and b.get('start_time')]
        
        return {
            'total_bookings': total_bookings,
            'avg_duration': np.mean(durations) if durations else 60,
            'std_duration': np.std(durations) if len(durations) > 1 else 0,
            'most_used_rooms': [r for r, _ in room_counts.most_common(5)],
            'room_diversity': len(room_counts),
            'avg_advance_booking_days': np.mean(advance_days) if advance_days else 1,
            'cancellation_rate': sum(1 for b in booking_history if b.get('status') == 'cancelled') / max(1, total_bookings),
            'no_show_rate': sum(1 for b in booking_history if b.get('attended') == False) / max(1, total_bookings),
            'booking_frequency': total_bookings / max(1, len(set(b.get('start_time', '')[:10] for b in booking_history)))
        }
    
    def _extract_temporal_features(self, booking_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not booking_history:
            return self._get_default_temporal_features()
        
        day_counts, hour_counts = defaultdict(int), defaultdict(int)
        morning = afternoon = evening = weekend = 0
        
        for b in booking_history:
            if b.get('start_time'):
                dt = datetime.fromisoformat(b['start_time'])
                day_counts[dt.strftime('%A')] += 1
                hour_counts[dt.hour] += 1
                if 6 <= dt.hour < 12: morning += 1
                elif 12 <= dt.hour < 18: afternoon += 1
                elif 18 <= dt.hour < 22: evening += 1
                if dt.weekday() >= 5: weekend += 1
        
        total = len(booking_history)
        return {
            'preferred_days': [d for d, _ in sorted(day_counts.items(), key=lambda x: x[1], reverse=True)[:3]],
            'preferred_hours': [h for h, _ in sorted(hour_counts.items(), key=lambda x: x[1], reverse=True)[:3]],
            'morning_preference': morning / max(1, total),
            'afternoon_preference': afternoon / max(1, total),
            'evening_preference': evening / max(1, total),
            'weekend_usage': weekend / max(1, total)
        }
    
    def _extract_satisfaction_features(self, booking_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not booking_history:
            return self._get_default_satisfaction_features()
        
        ratings = [b.get('rating') for b in booking_history if b.get('rating') is not None]
        issues = [issue for b in booking_history if b.get('issues') for issue in b['issues']]
        
        return {
            'avg_rating': np.mean(ratings) if ratings else 3.0,
            'rating_std': np.std(ratings) if len(ratings) > 1 else 0.5,
            'total_ratings': len(ratings),
            'positive_feedback_rate': sum(1 for b in booking_history if b.get('feedback_sentiment') == 'positive') / max(1, len(booking_history)),
            'negative_feedback_rate': sum(1 for b in booking_history if b.get('feedback_sentiment') == 'negative') / max(1, len(booking_history)),
            'common_issues': [i for i, _ in Counter(issues).most_common(5)]
        }
    
    def _extract_usage_patterns(self, booking_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not booking_history:
            return self._get_default_usage_patterns()
        
        equipment_usage = defaultdict(int)
        for b in booking_history:
            for eq in b.get('equipment_used', []):
                equipment_usage[eq] += 1
        
        purpose_counts = Counter(b.get('purpose', 'unknown') for b in booking_history)
        attendee_counts = [b.get('attendee_count', 1) for b in booking_history if b.get('attendee_count')]
        lead_times = [max(0, (datetime.fromisoformat(b['start_time']) - datetime.fromisoformat(b['booking_time'])).total_seconds() / 3600) 
                     for b in booking_history if b.get('booking_time') and b.get('start_time')]
        
        return {
            'equipment_preferences': dict(equipment_usage),
            'common_purposes': [p for p, _ in purpose_counts.most_common(5)],
            'avg_attendee_count': np.mean(attendee_counts) if attendee_counts else 1,
            'std_attendee_count': np.std(attendee_counts) if len(attendee_counts) > 1 else 0,
            'avg_lead_time_hours': np.mean(lead_times) if lead_times else 24,
            'planning_consistency': 1 - (np.std(lead_times) / max(1, np.mean(lead_times))) if lead_times else 0.5
        }
    
    def _create_user_feature_vector(self, features: Dict[str, Any]) -> np.ndarray:
        vector = np.zeros(100)
        try:
            behavioral = features.get('behavioral_features', {})
            temporal = features.get('temporal_features', {})
            satisfaction = features.get('satisfaction_features', {})
            preferences = features.get('preference_features', {})
            usage = features.get('usage_patterns', {})
            demographic = features.get('demographic_features', {})
            
            # Behavioral features (0-6)
            vector[0] = min(behavioral.get('total_bookings', 0) / 100, 1.0)
            vector[1] = min(behavioral.get('avg_duration', 60) / 480, 1.0)
            vector[2] = min(behavioral.get('room_diversity', 1) / 20, 1.0)
            vector[3] = min(behavioral.get('avg_advance_booking_days', 1) / 30, 1.0)
            vector[4] = behavioral.get('cancellation_rate', 0)
            vector[5] = behavioral.get('no_show_rate', 0)
            vector[6] = min(behavioral.get('booking_frequency', 1) / 10, 1.0)
            
            # Temporal features (7-17)
            vector[7:10] = [temporal.get(k, 0) for k in ['morning_preference', 'afternoon_preference', 'evening_preference']]
            vector[10] = temporal.get('weekend_usage', 0)
            
            day_mapping = {'Monday': 11, 'Tuesday': 12, 'Wednesday': 13, 'Thursday': 14, 'Friday': 15, 'Saturday': 16, 'Sunday': 17}
            for day in temporal.get('preferred_days', [])[:2]:
                if day in day_mapping:
                    vector[day_mapping[day]] = 1.0
            
            # Satisfaction features (18-21)
            vector[18] = satisfaction.get('avg_rating', 3.0) / 5.0
            vector[19] = min(satisfaction.get('rating_std', 0.5) / 2.0, 1.0)
            vector[20:22] = [satisfaction.get(k, 0) for k in ['positive_feedback_rate', 'negative_feedback_rate']]
            
            # Equipment and amenity preferences (22-45)
            for i, eq in enumerate(self.equipment_types[:10]):
                vector[26 + i] = 1.0 if eq in preferences.get('required_equipment', []) else 0.0
            for i, am in enumerate(self.amenity_types[:10]):
                vector[36 + i] = 1.0 if am in preferences.get('preferred_amenities', []) else 0.0
            
            # Usage patterns (46-59)
            vector[46] = min(usage.get('avg_attendee_count', 1) / 50, 1.0)
            vector[47] = min(usage.get('std_attendee_count', 0) / 20, 1.0)
            vector[48] = min(usage.get('avg_lead_time_hours', 24) / 168, 1.0)
            vector[49] = usage.get('planning_consistency', 0.5)
            
            # Equipment usage patterns (50-59)
            total_bookings = max(1, behavioral.get('total_bookings', 1))
            for i, eq in enumerate(self.equipment_types[:10]):
                vector[50 + i] = min(usage.get('equipment_preferences', {}).get(eq, 0) / total_bookings, 1.0)
            
            # Demographic features (60-79)
            vector[60 + hash(demographic.get('department', 'unknown')) % 10] = 1.0
            vector[70 + hash(demographic.get('role', 'unknown')) % 5] = 1.0
            vector[75] = min(demographic.get('team_size', 0) / 100, 1.0)
            
        except Exception as e:
            logger.error(f"Error creating user feature vector: {e}")
        
        return vector
    
    def extract_room_features(self, room_data: Dict[str, Any], usage_history: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        try:
            room_id = room_data.get('room_id') or room_data.get('id')
            if room_id in self.room_features_cache:
                return self.room_features_cache[room_id]
            
            features = {
                'room_id': room_id,
                'physical_features': self._extract_physical_features(room_data),
                'equipment_features': self._extract_equipment_features(room_data),
                'location_features': self._extract_location_features(room_data),
                'usage_features': self._extract_room_usage_features(usage_history or []),
                'availability_features': self._extract_availability_features(room_data, usage_history or []),
                'quality_features': self._extract_quality_features(usage_history or [])
            }
            features['feature_vector'] = self._create_room_feature_vector(features)
            self.room_features_cache[room_id] = features
            logger.info(f"Extracted features for room {room_id}")
            return features
        except Exception as e:
            logger.error(f"Failed to extract room features: {e}")
            return {'error': str(e)}
    
    def _extract_physical_features(self, room_data: Dict[str, Any]) -> Dict[str, Any]:
        return {k: room_data.get(k, v) for k, v in [('capacity', 0), ('area_sqm', 0), ('has_windows', False), 
                ('natural_light_rating', 0), ('noise_level', 3), ('temperature_control', 'none'), 
                ('furniture_type', 'standard'), ('room_shape', 'rectangular'), ('ceiling_height', 2.5)]}
    
    def _extract_equipment_features(self, room_data: Dict[str, Any]) -> Dict[str, Any]:
        equipment = room_data.get('equipment', [])
        return {
            'available_equipment': equipment,
            'equipment_count': len(equipment),
            'has_av_equipment': any(eq in equipment for eq in ['projector', 'tv', 'speakers', 'microphone']),
            'has_presentation_tools': any(eq in equipment for eq in ['projector', 'whiteboard', 'flip_chart']),
            'has_video_conferencing': any(eq in equipment for eq in ['camera', 'microphone', 'speakers']),
            'tech_level': self._calculate_tech_level(equipment),
            'equipment_age': room_data.get('equipment_age', 'unknown'),
            'maintenance_status': room_data.get('maintenance_status', 'good')
        }
    
    def _extract_location_features(self, room_data: Dict[str, Any]) -> Dict[str, Any]:
        return {k: room_data.get(k, v) for k, v in [('building', 'unknown'), ('floor', 0), ('wing', 'unknown'),
                ('distance_to_elevator', 0), ('distance_to_restroom', 0), ('distance_to_kitchen', 0),
                ('parking_nearby', False), ('public_transport_access', 'unknown'), 
                ('accessibility_features', []), ('is_accessible', len(room_data.get('accessibility_features', [])) > 0)]}
    
    def _extract_room_usage_features(self, usage_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not usage_history:
            return self._get_default_room_usage_features()
        
        durations = [(datetime.fromisoformat(b['end_time']) - datetime.fromisoformat(b['start_time'])).total_seconds() / 60 
                    for b in usage_history if b.get('start_time') and b.get('end_time')]
        hour_counts = Counter(datetime.fromisoformat(b['start_time']).hour for b in usage_history if b.get('start_time'))
        unique_users = len(set(b.get('user_id') for b in usage_history if b.get('user_id')))
        
        return {
            'total_bookings': len(usage_history),
            'avg_duration_minutes': np.mean(durations) if durations else 60,
            'std_duration_minutes': np.std(durations) if len(durations) > 1 else 0,
            'peak_hours': [h for h, _ in hour_counts.most_common(3)],
            'utilization_rate': min(len(usage_history) / (30 * 24), 1.0),
            'user_diversity': unique_users,
            'repeat_user_rate': (len(usage_history) - unique_users) / max(1, len(usage_history)),
            'popular_purposes': [p for p, _ in Counter(b.get('purpose', 'unknown') for b in usage_history).most_common(3)],
            'avg_attendee_count': np.mean([b.get('attendee_count', 1) for b in usage_history if b.get('attendee_count')]) or 1.0
        }
    
    def _extract_availability_features(self, room_data: Dict[str, Any], usage_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        busy_periods = self._identify_busy_periods(usage_history)
        return {
            'typical_busy_hours': busy_periods.get('hours', []),
            'typical_busy_days': busy_periods.get('days', []),
            'availability_score': 1.0 - min(len(usage_history) / (30 * 24), 1.0),
            'booking_lead_time_avg': np.mean([max(0, (datetime.fromisoformat(b['start_time']) - datetime.fromisoformat(b['booking_time'])).total_seconds() / 3600) 
                                             for b in usage_history if b.get('booking_time') and b.get('start_time')]) or 24.0,
            'cancellation_rate': sum(1 for b in usage_history if b.get('status') == 'cancelled') / max(1, len(usage_history)),
            'overbooking_incidents': sum(1 for b in usage_history if b.get('overbooking_incident', False))
        }
    
    def _extract_quality_features(self, usage_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not usage_history:
            return self._get_default_quality_features()
        
        ratings = [b.get('rating') for b in usage_history if b.get('rating') is not None]
        issues = [issue for b in usage_history if b.get('issues') for issue in b['issues']]
        
        return {
            'avg_rating': np.mean(ratings) if ratings else 3.0,
            'rating_std': np.std(ratings) if len(ratings) > 1 else 0.5,
            'total_ratings': len(ratings),
            'common_issues': [i for i, _ in Counter(issues).most_common(5)],
            'issue_rate': len(issues) / max(1, len(usage_history)),
            'maintenance_request_rate': sum(1 for b in usage_history if b.get('maintenance_requested', False)) / max(1, len(usage_history)),
            'satisfaction_trend': self._calculate_satisfaction_trend(usage_history)
        }
    
    def _create_room_feature_vector(self, features: Dict[str, Any]) -> np.ndarray:
        vector = np.zeros(80)
        try:
            physical = features.get('physical_features', {})
            equipment = features.get('equipment_features', {})
            location = features.get('location_features', {})
            usage = features.get('usage_features', {})
            quality = features.get('quality_features', {})
            availability = features.get('availability_features', {})
            
            # Physical features (0-5)
            vector[0] = min(physical.get('capacity', 0) / 100, 1.0)
            vector[1] = min(physical.get('area_sqm', 0) / 200, 1.0)
            vector[2] = 1.0 if physical.get('has_windows', False) else 0.0
            vector[3] = physical.get('natural_light_rating', 0) / 5.0
            vector[4] = physical.get('noise_level', 3) / 5.0
            vector[5] = min(physical.get('ceiling_height', 2.5) / 5.0, 1.0)
            
            # Equipment features (6-20)
            vector[6] = min(equipment.get('equipment_count', 0) / 20, 1.0)
            vector[7:11] = [1.0 if equipment.get(k, False) else 0.0 for k in ['has_av_equipment', 'has_presentation_tools', 'has_video_conferencing']] + [equipment.get('tech_level', 0) / 5.0]
            
            available_equipment = equipment.get('available_equipment', [])
            for i, eq in enumerate(self.equipment_types[:10]):
                vector[11 + i] = 1.0 if eq in available_equipment else 0.0
            
            # Location features (21-26)
            vector[21] = min(location.get('floor', 0) / 20, 1.0)
            vector[22:26] = [1.0 / max(1, location.get(k, 1)) for k in ['distance_to_elevator', 'distance_to_restroom', 'distance_to_kitchen']] + [1.0 if location.get('parking_nearby', False) else 0.0]
            vector[26] = 1.0 if location.get('is_accessible', False) else 0.0
            
            # Usage and quality features (27-41)
            vector[27] = min(usage.get('total_bookings', 0) / 1000, 1.0)
            vector[28] = min(usage.get('avg_duration_minutes', 60) / 480, 1.0)
            vector[29:38] = [usage.get('utilization_rate', 0), min(usage.get('user_diversity', 1) / 100, 1.0), usage.get('repeat_user_rate', 0),
                            min(usage.get('avg_attendee_count', 1) / 50, 1.0), quality.get('avg_rating', 3.0) / 5.0, min(quality.get('rating_std', 0.5) / 2.0, 1.0),
                            quality.get('issue_rate', 0), quality.get('maintenance_request_rate', 0), quality.get('satisfaction_trend', 0)]
            
            # Availability features (38-41)
            vector[38:42] = [availability.get('availability_score', 0.5), min(availability.get('booking_lead_time_avg', 24) / 168, 1.0),
                            availability.get('cancellation_rate', 0), min(availability.get('overbooking_incidents', 0) / 10, 1.0)]
        except Exception as e:
            logger.error(f"Error creating room feature vector: {e}")
        
        return vector
    
    # Default feature methods
    def _get_default_behavioral_features(self) -> Dict[str, Any]:
        return {'total_bookings': 0, 'avg_duration': 60, 'std_duration': 0, 'most_used_rooms': [], 
                'room_diversity': 0, 'avg_advance_booking_days': 1, 'cancellation_rate': 0, 'no_show_rate': 0, 'booking_frequency': 0}
    
    def _get_default_temporal_features(self) -> Dict[str, Any]:
        return {'preferred_days': [], 'preferred_hours': [], 'morning_preference': 0.33, 
                'afternoon_preference': 0.33, 'evening_preference': 0.33, 'weekend_usage': 0}
    
    def _get_default_satisfaction_features(self) -> Dict[str, Any]:
        return {'avg_rating': 3.0, 'rating_std': 0.5, 'total_ratings': 0, 'positive_feedback_rate': 0, 'negative_feedback_rate': 0, 'common_issues': []}
    
    def _get_default_usage_patterns(self) -> Dict[str, Any]:
        return {'equipment_preferences': {}, 'common_purposes': [], 'avg_attendee_count': 1, 
                'std_attendee_count': 0, 'avg_lead_time_hours': 24, 'planning_consistency': 0.5}
    
    def _get_default_room_usage_features(self) -> Dict[str, Any]:
        return {'total_bookings': 0, 'avg_duration_minutes': 60, 'std_duration_minutes': 0, 'peak_hours': [],
                'utilization_rate': 0, 'user_diversity': 0, 'repeat_user_rate': 0, 'popular_purposes': [], 'avg_attendee_count': 1}
    
    def _get_default_quality_features(self) -> Dict[str, Any]:
        return {'avg_rating': 3.0, 'rating_std': 0.5, 'total_ratings': 0, 'common_issues': [], 
                'issue_rate': 0, 'maintenance_request_rate': 0, 'satisfaction_trend': 0}
    
    # Utility methods
    def _calculate_tech_level(self, equipment: List[str]) -> float:
        tech_scores = {'projector': 1, 'tv': 1, 'speakers': 1, 'microphone': 2, 'camera': 2, 'computer': 2, 
                      'smartboard': 3, 'video_conference': 3, 'wireless_display': 3, 'touch_screen': 4, 'ai_assistant': 5}
        return min(sum(tech_scores.get(eq, 0) for eq in equipment) / max(1, len(equipment)), 5.0)
    
    def _identify_busy_periods(self, usage_history: List[Dict[str, Any]]) -> Dict[str, List]:
        if not usage_history:
            return {'hours': [], 'days': []}
        
        hour_counts = Counter(datetime.fromisoformat(b['start_time']).hour for b in usage_history if b.get('start_time'))
        day_counts = Counter(datetime.fromisoformat(b['start_time']).strftime('%A') for b in usage_history if b.get('start_time'))
        threshold = max(1, len(usage_history) * 0.1)
        
        return {'hours': [h for h, c in hour_counts.items() if c >= threshold],
                'days': [d for d, c in day_counts.items() if c >= threshold]}
    
    def _calculate_satisfaction_trend(self, usage_history: List[Dict[str, Any]]) -> float:
        ratings_with_time = [(datetime.fromisoformat(b['start_time']), b['rating']) 
                            for b in usage_history if b.get('rating') and b.get('start_time')]
        if len(ratings_with_time) < 2:
            return 0.0
        
        ratings_with_time.sort()
        ratings = [r for _, r in ratings_with_time]
        return np.clip(np.corrcoef(range(len(ratings)), ratings)[0, 1] if len(ratings) > 2 else (ratings[-1] - ratings[0]), -1.0, 1.0)
    
    def get_feature_importance(self, feature_type: str = 'user') -> Dict[str, float]:
        if feature_type == 'user':
            return {'total_bookings': 0.15, 'avg_duration': 0.12, 'preferred_equipment': 0.20, 
                   'satisfaction_rating': 0.18, 'temporal_patterns': 0.15, 'cancellation_rate': 0.10, 'room_diversity': 0.10}
        elif feature_type == 'room':
            return {'capacity': 0.20, 'equipment_available': 0.25, 'location_convenience': 0.15, 
                   'utilization_rate': 0.12, 'satisfaction_rating': 0.18, 'availability_score': 0.10}
        return {}
    
    def clear_cache(self):
        self.user_features_cache.clear()
        self.room_features_cache.clear()
        logger.info("Feature extractor caches cleared")