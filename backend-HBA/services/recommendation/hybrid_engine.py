from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
import logging
import asyncio

from .base_engine import BaseRecommendationEngine
from .models.embeddings import EnhancedEmbeddingModel
from .models.deepseek_processor import DeepSeekRecommendationProcessor

logger = logging.getLogger(__name__)


class HybridRecommendationEngine(BaseRecommendationEngine):
    
    def __init__(self, db: Session):
        super().__init__(db)
        self.db = db
        self._initialize_enhanced_components()
        self._configure_scoring()
        logger.info("HybridRecommendationEngine initialized")
    
    def _initialize_enhanced_components(self):
        """Initialize ML and LLM components with proper error handling"""
        # ML Component
        try:
            self.enhanced_embeddings = EnhancedEmbeddingModel()
            self.ml_available = True
            logger.info("✓ ML embeddings initialized")
        except Exception as e:
            logger.warning(f"ML Component unavailable: {e}")
            self.enhanced_embeddings = None
            self.ml_available = False
        
        # LLM Component
        try:
            self.deepseek_processor = DeepSeekRecommendationProcessor()
            self.llm_available = bool(self.deepseek_processor.deepseek)
            logger.info(f"✓ LLM Component {'initialized' if self.llm_available else 'unavailable'}")
        except Exception as e:
            logger.warning(f"LLM Component unavailable: {e}")
            self.deepseek_processor = None
            self.llm_available = False
    
    def _configure_scoring(self):
        """Configure scoring weights based on available components"""
        if self.ml_available and self.llm_available:
            self.mode = 'full_hybrid'
            self.base_weights = {
                'existing_system': 0.85,  # Base rules
                'ml_similarity': 0.075,   # ML embeddings
                'llm_context': 0.075      # LLM context
            }
        elif self.ml_available:
            self.mode = 'ml_enhanced'
            self.base_weights = {
                'existing_system': 0.9,
                'ml_similarity': 0.1,
                'llm_context': 0.0
            }
        elif self.llm_available:
            self.mode = 'llm_enhanced'
            self.base_weights = {
                'existing_system': 0.9,
                'ml_similarity': 0.0,
                'llm_context': 0.1
            }
        else:
            self.mode = 'standard'
            self.base_weights = {
                'existing_system': 1.0,
                'ml_similarity': 0.0,
                'llm_context': 0.0
            }
        
        logger.info(f"Engine mode: {self.mode}, Weights: {self.base_weights}")
    
    def get_recommendations(self, request_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate hybrid recommendations combining all three components
        """
        user_id = str(request_data.get('user_id', 'unknown'))
        logger.info(f"Generating hybrid recommendations for user {user_id} (mode: {self.mode})")
        
        try:
            # Parse time information
            start_time = request_data.get('start_time', '')
            end_time = request_data.get('end_time', '')
            
            try:
                start_dt = self._parse_datetime(start_time)
                end_dt = self._parse_datetime(end_time)
                user_duration_minutes = int((end_dt - start_dt).total_seconds() / 60)
            except Exception as e:
                logger.error(f"Time parsing error: {e}")
                start_dt = datetime.now()
                end_dt = start_dt + timedelta(hours=1)
                user_duration_minutes = 60
            
            # Get recommendations using appropriate async handling
            try:
                loop = asyncio.get_running_loop()
                recommendations = self._get_enhanced_recommendations_sync(request_data)
            except RuntimeError:
                # No event loop, create one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    recommendations = loop.run_until_complete(
                        self._get_enhanced_recommendations_async(request_data)
                    )
                finally:
                    loop.close()
            
            # Fix duration consistency
            validated_recommendations = self._validate_and_fix_durations(
                recommendations, user_duration_minutes, request_data
            )
            
            # Log final results
            self._log_final_scores(validated_recommendations)
            
            return validated_recommendations
            
        except Exception as e:
            logger.error(f"Hybrid recommendation error: {e}", exc_info=True)
            # Fallback to base recommendations
            base_recs = super().get_recommendations(request_data)
            return base_recs
    
    def _get_enhanced_recommendations_sync(self, request_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Synchronous enhancement of base recommendations"""
        # Step 1: Get base rule-based recommendations
        base_recs = super().get_recommendations(request_data)
        logger.info(f"Base recommendations: {len(base_recs)}")
        
        # Step 2: Prepare user context
        user_context = self._prepare_user_context_sync(request_data)
        
        # Step 3: Run ML analysis
        ml_scores = {}
        if self.ml_available:
            ml_scores = self._run_ml_analysis_sync(base_recs, user_context)
            logger.info(f"ML scores computed for {len(ml_scores)} recommendations")
        
        # Step 4: Run LLM analysis
        llm_scores = {}
        if self.llm_available:
            llm_scores = self._run_llm_analysis_sync(base_recs, user_context)
            logger.info(f"LLM scores computed for {len(llm_scores)} recommendations")
        
        # Step 5: Calculate final scores
        enhanced_recs = self._calculate_final_scores(base_recs, ml_scores, llm_scores)
        
        # Step 6: Sort by priority and score
        return self._sort_by_priority(enhanced_recs)[:8]
    
    async def _get_enhanced_recommendations_async(self, request_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Async enhancement of base recommendations"""
        base_recs = super().get_recommendations(request_data)
        logger.info(f"Base recommendations: {len(base_recs)}")
        
        user_context = await self._prepare_user_context(request_data)
        
        ml_scores = {}
        llm_scores = {}
        
        if self.ml_available:
            ml_scores = await self._run_ml_analysis(base_recs, user_context)
        
        if self.llm_available:
            llm_scores = await self._run_llm_analysis(base_recs, user_context)
        
        enhanced_recs = self._calculate_final_scores(base_recs, ml_scores, llm_scores)
        
        return self._sort_by_priority(enhanced_recs)[:8]
    
    def _prepare_user_context_sync(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare user context for scoring"""
        user_id = str(request_data.get('user_id', 'unknown'))
        return {
            'user_id': user_id,
            'request_data': request_data,
            'timestamp': datetime.now()
        }
    
    async def _prepare_user_context(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Async user context preparation"""
        return self._prepare_user_context_sync(request_data)
    
    def _run_ml_analysis_sync(self, recommendations: List[Dict], user_context: Dict) -> Dict[str, float]:
        """
        ML Component: Use embeddings to score recommendations
        """
        ml_scores = {}
        
        if not self.enhanced_embeddings or not self.ml_available:
            # Return neutral scores if ML unavailable
            for rec in recommendations:
                room_name = rec.get('suggestion', {}).get('room_name', '')
                if room_name:
                    ml_scores[room_name] = 0.5
            return ml_scores
        
        for rec in recommendations:
            room_name = rec.get('suggestion', {}).get('room_name', '')
            if room_name:
                try:
                    # Calculate ML score based on embeddings
                    ml_score = self._calculate_ml_score_sync(rec, user_context)
                    ml_scores[room_name] = ml_score
                except Exception as e:
                    logger.error(f"ML scoring error for {room_name}: {e}")
                    ml_scores[room_name] = 0.5
        
        return ml_scores
    
    async def _run_ml_analysis(self, recommendations: List[Dict], user_context: Dict) -> Dict[str, float]:
        """Async ML analysis"""
        return self._run_ml_analysis_sync(recommendations, user_context)
    
    def _run_llm_analysis_sync(self, recommendations: List[Dict], user_context: Dict) -> Dict[str, float]:
        """
        LLM Component: Use LLM for context-aware scoring
        """
        llm_scores = {}
        
        if not self.deepseek_processor or not self.llm_available:
            # Return neutral scores if LLM unavailable
            for rec in recommendations:
                room_name = rec.get('suggestion', {}).get('room_name', '')
                if room_name:
                    llm_scores[room_name] = 0.6
            return llm_scores
        
        for rec in recommendations:
            room_name = rec.get('suggestion', {}).get('room_name', '')
            if room_name:
                try:
                    # Calculate LLM score based on context understanding
                    llm_score = self._calculate_llm_score_sync(rec, user_context)
                    llm_scores[room_name] = llm_score
                except Exception as e:
                    logger.error(f"LLM scoring error for {room_name}: {e}")
                    llm_scores[room_name] = 0.6
        
        return llm_scores
    
    async def _run_llm_analysis(self, recommendations: List[Dict], user_context: Dict) -> Dict[str, float]:
        """Async LLM analysis"""
        return self._run_llm_analysis_sync(recommendations, user_context)
    
    def _calculate_ml_score_sync(self, rec: Dict, user_context: Dict) -> float:
        """
        Calculate ML-based score using embeddings
        
        This is where you'd use:
        - Room similarity embeddings
        - User preference embeddings
        - Historical booking patterns
        """
        base_score = 0.5
        
        # TODO: Implement actual embedding similarity calculations
        # Example: Compare room embedding with user's preferred room embeddings
        # room_embedding = self.enhanced_embeddings.get_room_embedding(room_name)
        # user_prefs = self.enhanced_embeddings.get_user_preferences(user_id)
        # similarity = cosine_similarity(room_embedding, user_prefs)
        
        return min(base_score, 1.0)
    
    def _calculate_llm_score_sync(self, rec: Dict, user_context: Dict) -> float:
        """
        Calculate LLM-based score using context understanding
        
        This is where you'd use:
        - Natural language understanding of booking purpose
        - Context-aware reasoning
        - Semantic matching
        """
        base_score = 0.6
        
        try:
            room_name = rec.get('suggestion', {}).get('room_name', '').lower()
            request_data = user_context.get('request_data', {})
            
            # Simple context matching (replace with actual LLM call)
            purpose = str(request_data.get('purpose', 'general')).lower()
            
            # Example: Boost score if room matches purpose
            if 'conference' in purpose and 'conference' in room_name:
                base_score += 0.05
            elif 'lecture' in purpose and 'lt' in room_name:
                base_score += 0.05
        except Exception:
            pass
        
        return min(base_score, 1.0)
    
    def _calculate_final_scores(self, recommendations: List[Dict], 
                                ml_scores: Dict, llm_scores: Dict) -> List[Dict]:
        """
        Combine scores from all three components using configured weights
        """
        enhanced_recs = []
        
        for rec in recommendations:
            room_name = rec.get('suggestion', {}).get('room_name', '')
            
            # Get individual scores
            base_score = rec.get('score', 0.5)  # Base rules score
            ml_score = ml_scores.get(room_name, 0.5)  # ML score
            llm_score = llm_scores.get(room_name, 0.5)  # LLM score
            
            # Calculate weighted final score
            final_score = (
                self.base_weights['existing_system'] * base_score +
                self.base_weights['ml_similarity'] * ml_score +
                self.base_weights['llm_context'] * llm_score
            )
            
            # Add scores to recommendation
            rec['final_score'] = final_score
            rec['ml_score'] = ml_score
            rec['llm_score'] = llm_score
            rec['score_breakdown'] = {
                'base': base_score,
                'ml': ml_score,
                'llm': llm_score,
                'final': final_score,
                'weights': self.base_weights
            }
            
            enhanced_recs.append(rec)
        
        return enhanced_recs
    
    def _sort_by_priority(self, recommendations: List[Dict]) -> List[Dict]:
        """Sort recommendations by type priority and final score"""
        priority_map = {
            'alternative_room': 1,
            'alternative_time': 2,
            'smart_scheduling': 3,
            'proactive': 4,
            'default': 5
        }
        
        return sorted(
            recommendations,
            key=lambda x: (
                priority_map.get(x.get('type', 'default'), 5),
                -x.get('final_score', 0)
            )
        )
    
    def _validate_and_fix_durations(self, recommendations: List[Dict[str, Any]], 
                                   expected_duration_minutes: int, 
                                   request_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Ensure all recommendations have consistent durations"""
        validated_recs = []
        
        for rec in recommendations:
            try:
                suggestion = rec.get('suggestion', {})
                
                if 'start_time' in suggestion:
                    start_time = self._parse_datetime(suggestion['start_time'])
                    corrected_end_time = start_time + timedelta(minutes=expected_duration_minutes)
                    suggestion['end_time'] = corrected_end_time.isoformat()
                    suggestion['duration_minutes'] = expected_duration_minutes
                    
                    rec['start_time'] = suggestion['start_time']
                    rec['end_time'] = suggestion['end_time']
                
                validated_recs.append(rec)
                
            except Exception as e:
                logger.error(f"Duration validation error: {e}")
                validated_recs.append(rec)
        
        return validated_recs
    
    def _log_final_scores(self, recommendations: List[Dict]):
        """Log final scoring results"""
        logger.info(f"Generated {len(recommendations)} hybrid recommendations")
        for i, rec in enumerate(recommendations[:5], 1):
            breakdown = rec.get('score_breakdown', {})
            logger.debug(
                f"Rank {i}: {rec.get('type')} - "
                f"{rec.get('suggestion', {}).get('room_name')} - "
                f"Final: {rec.get('final_score', 0):.3f} "
                f"(Base: {breakdown.get('base', 0):.3f}, "
                f"ML: {breakdown.get('ml', 0):.3f}, "
                f"LLM: {breakdown.get('llm', 0):.3f})"
            )
    
    def get_engine_status(self) -> Dict[str, Any]:
        """Get current engine status"""
        base_status = super().get_engine_status()
        return {
            **base_status,
            'engine_type': 'hybrid',
            'mode': self.mode,
            'ml_available': self.ml_available,
            'llm_available': self.llm_available,
            'weights': self.base_weights,
            'components': {
                'base_rules': True,
                'ml_embeddings': self.ml_available,
                'llm_context': self.llm_available
            }
        }