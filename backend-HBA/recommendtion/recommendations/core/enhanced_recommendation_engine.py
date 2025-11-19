from .recommendation_engine import RecommendationEngine
from ..models.enhanced_embedding_model import EnhancedEmbeddingModel
from ..models.llm_processor import LLMRecommendationProcessor
from ..models.deepseek_integration import DeepSeekRecommendationProcessor
from typing import Dict, List, Any
import logging
import asyncio
from datetime import datetime, timedelta
import json
from collections import defaultdict


logger = logging.getLogger(__name__)

class EnhancedRecommendationEngine(RecommendationEngine):
    
    def __init__(self, db=None, config=None):
        super().__init__(db, config)
        self._initialize_enhanced_components()
        self.deepseek_processor = DeepSeekRecommendationProcessor()
    
    def _ensure_time_fields(self, recommendation: Dict[str, Any], request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure both start_time and end_time are present in recommendation"""
        if 'suggestion' not in recommendation:
            recommendation['suggestion'] = {}
        
        # Get original times from request
        start_time = request_data.get('start_time', datetime.now().isoformat())
        end_time = request_data.get('end_time', (datetime.now() + timedelta(hours=1)).isoformat())
        
        # Ensure times are in suggestion
        if 'start_time' not in recommendation['suggestion']:
            recommendation['suggestion']['start_time'] = start_time
        if 'end_time' not in recommendation['suggestion']:
            recommendation['suggestion']['end_time'] = end_time
        
        return recommendation
    
    async def get_llm_enhanced_recommendations(self, request_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        base_recommendations = super().get_recommendations(request_data)
        
        if self.deepseek_processor.deepseek:
            try:
                user_id = str(request_data.get('user_id', 'unknown'))
                user_data = self._prepare_user_data_for_llm(user_id, request_data)
                user_analysis = await self.deepseek_processor.analyze_user_booking_context(user_data)
                llm_room_recs = await self.deepseek_processor.generate_room_recommendations(request_data)
                
                enhanced_recommendations = []
                for rec in base_recommendations:
                    enhanced_rec = rec.copy()
                    enhanced_rec = self._ensure_time_fields(enhanced_rec, request_data)  # FIX: Add time fields
                    enhanced_rec['ai_explanation'] = await self.deepseek_processor.explain_recommendation(rec, user_data)
                    llm_boost = self._calculate_llm_confidence_boost(rec, user_analysis)
                    enhanced_rec['llm_enhanced_score'] = min(rec.get('score', 0.5) + llm_boost, 1.0)
                    enhanced_recommendations.append(enhanced_rec)
                
                # FIX: Ensure LLM generated recommendations also have time fields
                for llm_rec in llm_room_recs[:2]:
                    llm_rec = self._ensure_time_fields(llm_rec, request_data)
                    enhanced_recommendations.append(llm_rec)
                
                enhanced_recommendations.sort(key=lambda x: x.get('llm_enhanced_score', x.get('score', 0)), reverse=True)
                return enhanced_recommendations[:5]
                
            except Exception as e:
                logger.error(f"LLM enhancement failed: {e}, using base recommendations")
        
        # FIX: Ensure base recommendations have time fields
        return [self._ensure_time_fields(rec, request_data) for rec in base_recommendations]
    
    def _initialize_enhanced_components(self):
        try:
            self.enhanced_embeddings = EnhancedEmbeddingModel()
            logger.info("Enhanced embedding model initialized")
        except Exception as e:
            logger.warning(f"Could not initialize enhanced embeddings: {e}")
            self.enhanced_embeddings = None
        
        try:
            self.llm_processor = LLMRecommendationProcessor()
            logger.info("LLM processor initialized")
        except Exception as e:
            logger.warning(f"Could not initialize LLM processor: {e}")
            self.llm_processor = None
        
        self.scoring_weights = {'existing_strategies': 0.4, 'ml_similarity': 0.3, 'llm_context': 0.3}
    
    def get_enhanced_recommendations(self, request_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        try:
            base_recommendations = super().get_recommendations(request_data)
            
            if self.enhanced_embeddings or self.llm_processor:
                enhanced_recs = self._add_ml_llm_enhancements(request_data, base_recommendations)
            else:
                logger.info("Using base recommendations (ML/LLM not available)")
                enhanced_recs = base_recommendations
            
            # FIX: Ensure all recommendations have time fields
            return [self._ensure_time_fields(rec, request_data) for rec in enhanced_recs]
                
        except Exception as e:
            logger.error(f"Error in enhanced recommendations: {e}")
            base_recs = super().get_recommendations(request_data)
            return [self._ensure_time_fields(rec, request_data) for rec in base_recs]
    
    def _add_ml_llm_enhancements(self, request_data: Dict[str, Any], base_recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        try:
            user_id = str(request_data.get('user_id', 'unknown'))
            user_data = self._prepare_user_data_for_ml_llm(user_id, request_data)
            ml_insights = self._get_ml_insights(user_data) if self.enhanced_embeddings else {}
            llm_insights = self._get_llm_insights(user_data) if self.llm_processor else {}
            
            enhanced_recommendations = []
            for rec in base_recommendations:
                enhanced_rec = rec.copy()
                enhanced_rec = self._ensure_time_fields(enhanced_rec, request_data)  # FIX: Add time fields
                enhanced_rec['ml_score'] = self._calculate_ml_score(rec, ml_insights) if ml_insights else 0.5
                enhanced_rec['llm_score'] = self._calculate_llm_score(rec, llm_insights) if llm_insights else 0.5
                
                base_score = rec.get('score', 0.5)
                enhanced_rec['hybrid_score'] = (
                    self.scoring_weights['existing_strategies'] * base_score +
                    self.scoring_weights['ml_similarity'] * enhanced_rec['ml_score'] +
                    self.scoring_weights['llm_context'] * enhanced_rec['llm_score']
                )
                enhanced_rec['enhancement_type'] = 'ml_llm_enhanced'
                
                if llm_insights and 'explanations' in llm_insights:
                    room_name = rec.get('suggestion', {}).get('room_name', '')
                    enhanced_rec['ai_explanation'] = llm_insights['explanations'].get(room_name, rec.get('reason', 'Standard recommendation'))
                
                enhanced_recommendations.append(enhanced_rec)
            
            enhanced_recommendations.sort(key=lambda x: x['hybrid_score'], reverse=True)
            logger.info(f"Enhanced {len(enhanced_recommendations)} recommendations with ML/LLM")
            return enhanced_recommendations
            
        except Exception as e:
            logger.error(f"Error adding ML/LLM enhancements: {e}")
            return base_recommendations
    
    def _prepare_user_data_for_ml_llm(self, user_id: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            booking_history = self.get_user_booking_history(request_data, user_id, days=90)
            room_counts = {}
            for booking in booking_history:
                room = booking.get('room_name', '')
                room_counts[room] = room_counts.get(room, 0) + 1
            
            preferred_rooms = [room for room, count in sorted(room_counts.items(), key=lambda x: x[1], reverse=True)[:5]]
            
            return {
                'user_id': user_id, 'booking_history': booking_history, 'preferred_rooms': preferred_rooms,
                'current_context': request_data,
                'booking_patterns': {
                    'total_bookings': len(booking_history), 'unique_rooms': len(room_counts),
                    'frequency': 'regular' if len(booking_history) > 10 else 'occasional'
                }
            }
        except Exception as e:
            logger.error(f"Error preparing user data: {e}")
            return {'user_id': user_id, 'booking_history': [], 'preferred_rooms': [], 'current_context': request_data, 'booking_patterns': {}}
    
    def _get_ml_insights(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if not self.enhanced_embeddings:
                return {}
            return {'similar_users': [], 'user_embedding_available': True, 'similarity_recommendations': []}
        except Exception as e:
            logger.error(f"Error getting ML insights: {e}")
            return {}
    
    def _get_llm_insights(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if not self.llm_processor:
                return {}
            return {'room_recommendations': [], 'time_recommendations': [], 'patterns_observed': [], 'explanations': {}}
        except Exception as e:
            logger.error(f"Error getting LLM insights: {e}")
            return {}
    
    def _calculate_ml_score(self, recommendation: Dict[str, Any], ml_insights: Dict[str, Any]) -> float:
        try:
            base_score = 0.5
            if ml_insights.get('user_embedding_available'): base_score += 0.1
            if ml_insights.get('similar_users'): base_score += 0.2
            return min(base_score, 1.0)
        except Exception as e:
            logger.error(f"Error calculating ML score: {e}")
            return 0.5
    
    def _calculate_llm_score(self, recommendation: Dict[str, Any], llm_insights: Dict[str, Any]) -> float:
        try:
            base_score = 0.5
            room_name = recommendation.get('suggestion', {}).get('room_name', '')
            for llm_rec in llm_insights.get('room_recommendations', []):
                if llm_rec.get('room_name') == room_name:
                    base_score += llm_rec.get('confidence', 0.0) * 0.5
                    break
            return min(base_score, 1.0)
        except Exception as e:
            logger.error(f"Error calculating LLM score: {e}")
            return 0.5
    
    def get_ai_explanation(self, recommendation: Dict[str, Any], user_context: Dict[str, Any]) -> str:
        if self.llm_processor:
            try:
                return f"This recommendation is based on your booking patterns and preferences."
            except Exception as e:
                logger.error(f"Error getting AI explanation: {e}")
        return recommendation.get('reason', 'Recommended based on system analysis')