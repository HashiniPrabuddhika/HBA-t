from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict

from .base_engine import BaseRecommendationEngine
from .models.embeddings import EnhancedEmbeddingModel
from utils.logger import get_logger


logger = get_logger(__name__)


class MLRecommendationEngine(BaseRecommendationEngine):
    
    def __init__(self, db=None, config: Optional[Dict[str, Any]] = None):
        super().__init__(db, config)
        self._initialize_ml_components()
        self.scoring_weights = {
            'base_strategy': 0.6,
            'ml_similarity': 0.4
        }
    
    def _initialize_ml_components(self) -> None:
        try:
            self.enhanced_embeddings = EnhancedEmbeddingModel()
            logger.info("ML embedding model initialized successfully")
        except Exception as e:
            logger.warning(f"ML embedding model initialization failed: {e}")
            self.enhanced_embeddings = None
    
    def get_recommendations(
        self, 
        request_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        try:
            base_recommendations = super().get_recommendations(request_data)
            
            if not self.enhanced_embeddings:
                logger.info("ML enhancements unavailable, returning base recommendations")
                return self._ensure_time_fields_batch(base_recommendations, request_data)
            
            ml_enhanced = self._apply_ml_enhancements(request_data, base_recommendations)
            return self._ensure_time_fields_batch(ml_enhanced, request_data)
                
        except Exception as e:
            logger.error(f"ML recommendation pipeline failed: {e}", exc_info=True)
            fallback = super().get_recommendations(request_data)
            return self._ensure_time_fields_batch(fallback, request_data)
    
    def _apply_ml_enhancements(
        self, 
        request_data: Dict[str, Any], 
        base_recommendations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        try:
            user_id = str(request_data.get('user_id', 'unknown'))
            user_profile = self._build_user_profile(user_id, request_data)
            ml_insights = self._extract_ml_insights(user_profile)
            
            enhanced_recommendations = []
            for rec in base_recommendations:
                enhanced_rec = self._enhance_recommendation(
                    rec, 
                    ml_insights, 
                    user_profile
                )
                enhanced_recommendations.append(enhanced_rec)
            
            enhanced_recommendations.sort(
                key=lambda x: x.get('ml_enhanced_score', 0), 
                reverse=True
            )
            
            logger.info(
                f"ML enhanced {len(enhanced_recommendations)} recommendations "
                f"for user {user_id}"
            )
            return enhanced_recommendations
            
        except Exception as e:
            logger.error(f"ML enhancement process failed: {e}", exc_info=True)
            return base_recommendations
    
    def _enhance_recommendation(
        self,
        recommendation: Dict[str, Any],
        ml_insights: Dict[str, Any],
        user_profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        enhanced = recommendation.copy()
        
        base_score = recommendation.get('score', 0.5)
        ml_score = self._calculate_ml_score(recommendation, ml_insights, user_profile)
        
        enhanced['ml_score'] = ml_score
        enhanced['ml_enhanced_score'] = (
            self.scoring_weights['base_strategy'] * base_score +
            self.scoring_weights['ml_similarity'] * ml_score
        )
        enhanced['enhancement_type'] = 'ml_enhanced'
        enhanced['ml_insights'] = self._extract_recommendation_insights(
            recommendation, 
            ml_insights
        )
        
        return enhanced
    
    def _build_user_profile(
        self, 
        user_id: str, 
        request_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            booking_history = self.get_user_booking_history(
                request_data, 
                user_id, 
                days=90
            )
            
            room_frequency = defaultdict(int)
            time_patterns = defaultdict(int)
            duration_patterns = []
            
            for booking in booking_history:
                room_name = booking.get('room_name', '')
                if room_name:
                    room_frequency[room_name] += 1
                
                start_time = booking.get('start_time')
                if start_time:
                    hour = self._extract_hour(start_time)
                    time_patterns[hour] += 1
                
                duration = booking.get('duration_minutes')
                if duration:
                    duration_patterns.append(duration)
            
            preferred_rooms = [
                room for room, _ in sorted(
                    room_frequency.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:5]
            ]
            
            avg_duration = (
                sum(duration_patterns) / len(duration_patterns) 
                if duration_patterns else 60
            )
            
            return {
                'user_id': user_id,
                'booking_history': booking_history,
                'preferred_rooms': preferred_rooms,
                'room_frequency': dict(room_frequency),
                'time_patterns': dict(time_patterns),
                'avg_duration': avg_duration,
                'current_context': request_data,
                'booking_stats': {
                    'total_bookings': len(booking_history),
                    'unique_rooms': len(room_frequency),
                    'frequency_category': self._categorize_frequency(len(booking_history))
                }
            }
            
        except Exception as e:
            logger.error(f"User profile building failed for {user_id}: {e}")
            return {
                'user_id': user_id,
                'booking_history': [],
                'preferred_rooms': [],
                'room_frequency': {},
                'time_patterns': {},
                'avg_duration': 60,
                'current_context': request_data,
                'booking_stats': {}
            }
    
    def _extract_ml_insights(
        self, 
        user_profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            if not self.enhanced_embeddings:
                return self._default_ml_insights()
            
            similar_users = self._find_similar_users(user_profile)
            room_embeddings = self._compute_room_embeddings(user_profile)
            pattern_clusters = self._identify_pattern_clusters(user_profile)
            
            return {
                'similar_users': similar_users,
                'room_embeddings': room_embeddings,
                'pattern_clusters': pattern_clusters,
                'embedding_available': True,
                'confidence_level': self._calculate_confidence(user_profile)
            }
            
        except Exception as e:
            logger.error(f"ML insight extraction failed: {e}")
            return self._default_ml_insights()
    
    def _calculate_ml_score(
        self,
        recommendation: Dict[str, Any],
        ml_insights: Dict[str, Any],
        user_profile: Dict[str, Any]
    ) -> float:
        try:
            base_score = 0.5
            
            room_name = recommendation.get('suggestion', {}).get('room_name', '')
            
            if room_name in user_profile.get('preferred_rooms', []):
                preference_rank = user_profile['preferred_rooms'].index(room_name)
                base_score += (5 - preference_rank) * 0.05
            
            if ml_insights.get('embedding_available'):
                base_score += 0.1
            
            similar_users = ml_insights.get('similar_users', [])
            if similar_users:
                similarity_boost = min(len(similar_users) * 0.02, 0.15)
                base_score += similarity_boost
            
            room_embeddings = ml_insights.get('room_embeddings', {})
            if room_name in room_embeddings:
                embedding_score = room_embeddings[room_name].get('score', 0)
                base_score += embedding_score * 0.15
            
            confidence = ml_insights.get('confidence_level', 0.5)
            base_score *= (0.8 + 0.2 * confidence)
            
            return min(base_score, 1.0)
            
        except Exception as e:
            logger.error(f"ML score calculation failed: {e}")
            return 0.5
    
    def _find_similar_users(
        self, 
        user_profile: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        try:
            if not self.enhanced_embeddings:
                return []
            return []
        except Exception as e:
            logger.error(f"Similar user search failed: {e}")
            return []
    
    def _compute_room_embeddings(
        self, 
        user_profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            if not self.enhanced_embeddings:
                return {}
            return {}
        except Exception as e:
            logger.error(f"Room embedding computation failed: {e}")
            return {}
    
    def _identify_pattern_clusters(
        self, 
        user_profile: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        try:
            time_patterns = user_profile.get('time_patterns', {})
            if not time_patterns:
                return []
            
            clusters = []
            sorted_hours = sorted(time_patterns.items(), key=lambda x: x[1], reverse=True)
            
            if sorted_hours:
                top_hour, frequency = sorted_hours[0]
                clusters.append({
                    'type': 'time_preference',
                    'hour': top_hour,
                    'frequency': frequency,
                    'confidence': min(frequency / sum(time_patterns.values()), 1.0)
                })
            
            return clusters
            
        except Exception as e:
            logger.error(f"Pattern clustering failed: {e}")
            return []
    
    def _extract_recommendation_insights(
        self,
        recommendation: Dict[str, Any],
        ml_insights: Dict[str, Any]
    ) -> Dict[str, Any]:
        room_name = recommendation.get('suggestion', {}).get('room_name', '')
        
        insights = {
            'ml_confidence': ml_insights.get('confidence_level', 0.5),
            'embedding_match': room_name in ml_insights.get('room_embeddings', {}),
            'similar_user_preference': False
        }
        
        for user in ml_insights.get('similar_users', []):
            if room_name in user.get('preferred_rooms', []):
                insights['similar_user_preference'] = True
                break
        
        return insights
    
    def _calculate_confidence(
        self, 
        user_profile: Dict[str, Any]
    ) -> float:
        total_bookings = user_profile.get('booking_stats', {}).get('total_bookings', 0)
        
        if total_bookings == 0:
            return 0.3
        elif total_bookings < 5:
            return 0.5
        elif total_bookings < 15:
            return 0.7
        else:
            return 0.9
    
    def _categorize_frequency(self, booking_count: int) -> str:
        if booking_count == 0:
            return 'new'
        elif booking_count < 5:
            return 'occasional'
        elif booking_count < 15:
            return 'regular'
        else:
            return 'frequent'
    
    def _extract_hour(self, time_value: Any) -> int:
        try:
            if isinstance(time_value, str):
                dt = datetime.fromisoformat(time_value.replace('Z', '+00:00'))
                return dt.hour
            elif isinstance(time_value, datetime):
                return time_value.hour
            return 9
        except Exception:
            return 9
    
    def _default_ml_insights(self) -> Dict[str, Any]:
        return {
            'similar_users': [],
            'room_embeddings': {},
            'pattern_clusters': [],
            'embedding_available': False,
            'confidence_level': 0.3
        }
    
    def _ensure_time_fields(
        self, 
        recommendation: Dict[str, Any], 
        request_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        if 'suggestion' not in recommendation:
            recommendation['suggestion'] = {}
        
        start_time = request_data.get(
            'start_time', 
            datetime.now().isoformat()
        )
        end_time = request_data.get(
            'end_time', 
            (datetime.now() + timedelta(hours=1)).isoformat()
        )
        
        suggestion = recommendation['suggestion']
        if 'start_time' not in suggestion:
            suggestion['start_time'] = start_time
        if 'end_time' not in suggestion:
            suggestion['end_time'] = end_time
        
        return recommendation
    
    def _ensure_time_fields_batch(
        self,
        recommendations: List[Dict[str, Any]],
        request_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        return [
            self._ensure_time_fields(rec, request_data) 
            for rec in recommendations
        ]