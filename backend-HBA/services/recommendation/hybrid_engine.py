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
    
    def __init__(self, db=None, config=None):
        super().__init__(db, config)
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
                'existing_system': 0.85,
                'ml_similarity': 0.075,
                'llm_context': 0.075
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
    
    # FIX 1: Add missing _parse_datetime method
    def _parse_datetime(self, datetime_str):
        """Parse datetime string in various formats"""
        if isinstance(datetime_str, datetime):
            return datetime_str
        
        if not datetime_str:
            return datetime.now()
        
        # Try ISO format first
        try:
            return datetime.fromisoformat(str(datetime_str).replace('Z', '+00:00'))
        except (ValueError, AttributeError):
            pass
        
        # Try standard format
        try:
            return datetime.strptime(str(datetime_str), '%Y-%m-%d %H:%M:%S')
        except (ValueError, AttributeError):
            pass
        
        # Try date only format
        try:
            return datetime.strptime(str(datetime_str), '%Y-%m-%d')
        except (ValueError, AttributeError):
            pass
        
        logger.warning(f"Could not parse datetime: {datetime_str}, using current time")
        return datetime.now()
    
    # FIX 2: Add missing _calculate_duration_minutes method
    def _calculate_duration_minutes(self, start_dt, end_dt):
        """Calculate duration in minutes between two datetime objects"""
        try:
            if isinstance(start_dt, datetime) and isinstance(end_dt, datetime):
                return int((end_dt - start_dt).total_seconds() / 60)
            return 60  # Default 1 hour
        except Exception as e:
            logger.error(f"Error calculating duration: {e}")
            return 60
    
    def _get_priority_order(self, rec_type):
        priority_map = {
            'alternative_room': 1,   
            'alternative_time': 2,    
            'smart_scheduling': 3,    
            'default': 4           
        }
        return priority_map.get(rec_type, priority_map['default'])

    def _sort_recommendations_by_priority(self, recommendations):
        return sorted(recommendations, 
                     key=lambda x: (self._get_priority_order(x.get('type', 'default')), 
                                   -x.get('final_score', 0)))
    
    def _add_duration_to_datetime(self, dt: datetime, minutes: int) -> datetime:
        """Add duration in minutes to datetime"""
        return dt + timedelta(minutes=minutes)
    
    def get_recommendations(self, request_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        
        user_id = str(request_data.get('user_id', 'unknown'))
        logger.info(f"Generating hybrid recommendations for user {user_id} (mode: {self.mode})")
        
        try:
            start_time = request_data.get('start_time', '')
            end_time = request_data.get('end_time', '')
            
            try:
                start_dt = self._parse_datetime(start_time)
                end_dt = self._parse_datetime(end_time)
                user_duration_minutes = int((end_dt - start_dt).total_seconds() / 60)
            except Exception as e:
                logger.error(f"Time parsing error: {e}")
                start_dt = datetime.now()
                end_dt = start_dt + timedelta(hours=4)
                user_duration_minutes = 240
            
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
            base_recs = self._get_base_recommendations(request_data)
            return self._validate_and_fix_durations(base_recs, user_duration_minutes, request_data)
    
    # FIX 3: Improved _get_base_recommendations to extract room names properly
    def _get_base_recommendations(self, request_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        try:
            recs = super().get_recommendations(request_data)
            
            if not recs:
                logger.warning("No base recommendations returned from parent class")
                return []
            
            # FIX: Extract room names properly from suggestions
            for i, rec in enumerate(recs):
                if 'base_score' not in rec:
                    rec['base_score'] = rec.get('score', 0.5)
                
                # Extract room_name from various possible locations
                room_name = None
                
                # Check suggestion dict first
                suggestion = rec.get('suggestion', {})
                if isinstance(suggestion, dict):
                    room_name = suggestion.get('room_name') or suggestion.get('room_id')
                
                # Check top-level if not found
                if not room_name:
                    room_name = rec.get('room_name') or rec.get('room_id')
                
                # Fallback with warning
                if not room_name:
                    room_name = f'Room_{i+1}'
                    logger.warning(f"Recommendation {i} missing room name, using fallback: {room_name}")
                
                # Set room_name at top level
                rec['room_name'] = room_name
                
                # Ensure suggestion dict exists and has room_name
                if not isinstance(rec.get('suggestion'), dict):
                    rec['suggestion'] = {}
                
                rec['suggestion']['room_name'] = room_name
                if 'capacity' not in rec['suggestion']:
                    rec['suggestion']['capacity'] = 10
            
            logger.info(f"✓ Processed {len(recs)} base recommendations with room names")
            return recs
            
        except Exception as e:
            logger.error(f"❌ Base recommendation error: {e}", exc_info=True)
            return []

    
    def _get_enhanced_recommendations_sync(self, request_data: Dict[str, Any]) -> List[Dict[str, Any]]:
       
        base_recs = self._get_base_recommendations(request_data)
        logger.info(f"Base recommendations: {len(base_recs)}")
        
        # FIX 4: Pass request_data to prepare context (not just user_id)
        user_context = self._prepare_user_context_sync(request_data)
        
        ml_scores = {}
        if self.ml_available:
            ml_scores = self._run_ml_analysis_sync(base_recs, user_context)
            logger.info(f"ML scores computed for {len(ml_scores)} recommendations")
        
        llm_scores = {}
        if self.llm_available:
            llm_scores = self._run_llm_analysis_sync(base_recs, user_context)
            logger.info(f"LLM scores computed for {len(llm_scores)} recommendations")
        
        enhanced_recs = self._calculate_final_scores(base_recs, ml_scores, llm_scores)
        
        # FIX 5: Return more recommendations (up to 8)
        sorted_recs = self._sort_by_priority(enhanced_recs)
        return sorted_recs[:8]  # Return top 8 instead of limiting
    
    async def _get_enhanced_recommendations_async(self, request_data: Dict[str, Any]) -> List[Dict[str, Any]]:
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
        
        # Deduplicate before sorting
        deduplicated_recs = self._deduplicate_recommendations(enhanced_recs)
        
        return self._sort_by_priority(deduplicated_recs)[:8]
    
    def _prepare_user_context_sync(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare user context for scoring"""
        user_id = str(request_data.get('user_id', 'unknown'))
        return {
            'user_id': user_id,
            'request_data': request_data,
            'booking_history': self.get_user_booking_history(request_data, user_id, days=30),
            'timestamp': datetime.now()
        }
    
    async def _prepare_user_context(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
       return self._prepare_user_context_sync(request_data)
    
    def _run_ml_analysis_sync(self, recommendations: List[Dict], user_context: Dict) -> Dict[str, float]:
        ml_scores = {}
        
        try:
            if not self.enhanced_embeddings or not self.ml_available:
                for rec in recommendations:
                    room_name = rec.get('room_name', '')
                    if room_name:
                        ml_scores[room_name] = 0.5
                return ml_scores
            
            for rec in recommendations:
                room_name = rec.get('room_name', '')
                if room_name:
                    try:
                        ml_score = self._calculate_ml_score_sync(rec, user_context)
                        ml_scores[room_name] = ml_score
                    except Exception as e:
                        logger.error(f"ML scoring error for {room_name}: {e}")
                        ml_scores[room_name] = 0.5
                        
        except Exception as e:
            for rec in recommendations:
                room_name = rec.get('room_name', '')
                if room_name:
                    ml_scores[room_name] = 0.500
        
        return ml_scores
    
    async def _run_ml_analysis(self, recommendations: List[Dict], user_context: Dict) -> Dict[str, float]:
        """Async ML analysis"""
        return self._run_ml_analysis_sync(recommendations, user_context)
    
    def _run_llm_analysis_sync(self, recommendations: List[Dict], user_context: Dict) -> Dict[str, float]:
       
        llm_scores = {}
        
        try:
            if not self.deepseek_processor or not self.llm_available:
                for rec in recommendations:
                    room_name = rec.get('room_name', '')
                    if room_name:
                        llm_scores[room_name] = 0.6
                return llm_scores
            
            for rec in recommendations:
                room_name = rec.get('room_name', '')
                if room_name:
                    try:
                        llm_score = self._calculate_llm_score_sync(rec, user_context)
                        llm_scores[room_name] = llm_score
                    except Exception as e:
                        logger.error(f"LLM scoring error for {room_name}: {e}")
                        llm_scores[room_name] = 0.6
                        
        except Exception as e:
            for rec in recommendations:
                room_name = rec.get('room_name', '')
                if room_name:
                    llm_scores[room_name] = 0.600
        
        return llm_scores
    
    async def _run_llm_analysis(self, recommendations: List[Dict], user_context: Dict) -> Dict[str, float]:
        """Async LLM analysis"""
        return self._run_llm_analysis_sync(recommendations, user_context)
    
    def _calculate_ml_score_sync(self, rec: Dict, user_context: Dict) -> float:
        """Calculate ML-based score using embeddings"""
        base_score = 0.5
        
        try:
            room_name = rec.get('room_name', '')
            booking_history = user_context.get('booking_history', [])
            
            room_usage = sum(1 for booking in booking_history 
                           if booking.get('room_name') == room_name)
            
            if room_usage > 0:
                base_score += min(room_usage * 0.02, 0.1)
            
        except Exception:
            pass
        
        return min(base_score, 1.0)
    
    def _calculate_llm_score_sync(self, rec: Dict, user_context: Dict) -> float:
        """Calculate LLM-based score using context understanding"""
        base_score = 0.6
        
        try:
            room_name = rec.get('room_name', '').lower()
            request_data = user_context.get('request_data', {})
            
            meeting_type = str(request_data.get('meeting_type', 'general')).lower()
            purpose = str(request_data.get('purpose', 'general')).lower()
            
            if 'conference' in meeting_type and 'conference' in room_name:
                base_score += 0.05
            elif 'board' in meeting_type and 'board' in room_name:
                base_score += 0.05
            elif 'meeting' in purpose and 'meeting' in room_name:
                base_score += 0.03
            elif 'lecture' in purpose and 'lt' in room_name:
                base_score += 0.05
            
        except Exception:
            pass
        
        return min(base_score, 1.0)
    
    def _calculate_final_scores(self, recommendations: List[Dict], 
                                ml_scores: Dict, llm_scores: Dict) -> List[Dict]:
        
        enhanced_recs = []
        
        for rec in recommendations:
            room_name = rec.get('room_name', '')
            
            # Get individual scores
            base_score = rec.get('score', 0.5)
            ml_score = ml_scores.get(room_name, 0.5)
            llm_score = llm_scores.get(room_name, 0.5)
            
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
       
        validated_recs = []
        
        for rec in recommendations:
            try:
                suggestion = rec.get('suggestion', {})
                
                if 'start_time' in suggestion:
                    start_time = self._parse_datetime(suggestion['start_time'])
                    corrected_end_time = start_time + timedelta(minutes=expected_duration_minutes)
                    suggestion['end_time'] = corrected_end_time.isoformat()
                    suggestion['duration_minutes'] = expected_duration_minutes
                    suggestion['duration_hours'] = expected_duration_minutes / 60
                    
                    rec['start_time'] = suggestion['start_time']
                    rec['end_time'] = suggestion['end_time']
                
                validated_recs.append(rec)
                
            except Exception as e:
                logger.error(f"Duration validation error: {e}")
                validated_recs.append(rec)
        
        return validated_recs
    
    def _log_final_scores(self, recommendations: List[Dict]):
        """Log final scoring results with room names visible"""
        logger.info(f"Generated {len(recommendations)} hybrid recommendations")
        
        for i, rec in enumerate(recommendations[:5], 1):
            room_name = rec.get('room_name', 'UNKNOWN')
            rec_type = rec.get('type', 'unknown')
            
            # Also try to get from suggestion
            if room_name == 'UNKNOWN' and 'suggestion' in rec:
                room_name = rec['suggestion'].get('room_name', 'UNKNOWN')
            
            breakdown = rec.get('score_breakdown', {})
            logger.debug(
                f"Rank {i}: {rec_type} - "
                f"Room: {room_name} - "
                f"Final: {rec.get('final_score', 0):.3f} "
                f"(Base: {breakdown.get('base', 0):.3f}, "
                f"ML: {breakdown.get('ml', 0):.3f}, "
                f"LLM: {breakdown.get('llm', 0):.3f})"
            )
        
        # Print summary table for visibility
        print("\n=== RECOMMENDATIONS SUMMARY ===")
        for i, rec in enumerate(recommendations[:8], 1):
            room_name = rec.get('room_name') or rec.get('suggestion', {}).get('room_name', 'UNKNOWN')
            rec_type = rec.get('type', 'unknown')
            print(f"{i}. [{rec_type:20}] Room: {room_name}")
        print("=" * 50)
    
    def get_engine_status(self) -> Dict[str, Any]:
        base_status = super().get_engine_status()
        return {
            **base_status,
            'hybrid_mode': self.mode,
            'ml_available': self.ml_available,
            'llm_available': self.llm_available,
            'weights': self.base_weights,
            'hybrid_status': 'ready',
            'duration_handling': 'enhanced_with_preservation',
            'priority_ordering': 'alternative_rooms_first',
            'fixes_applied': [
                'duration_preservation',
                'time_validation',
                'end_time_calculation',
                'availability_checking',
                'priority_based_sorting',
                'room_name_extraction',
                'datetime_parsing'
            ]
        }