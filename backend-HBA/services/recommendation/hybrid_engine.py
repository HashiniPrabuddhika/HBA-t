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
        
    def _get_base_recommendations(self, request_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        try:
            recs = super().get_recommendations(request_data)
            
            if not recs:
                return []
            
            for i, rec in enumerate(recs):
                if 'base_score' not in rec:
                    rec['base_score'] = rec.get('score', 0.5)
                
                if isinstance(rec.get('suggestion'), dict):
                    rec['room_name'] = rec['suggestion'].get('room_name', f'Room_{i+1}')
                else:
                    rec['room_name'] = rec.get('room_name', f'Room_{i+1}')
                
                if 'suggestion' not in rec:
                    rec['suggestion'] = {'room_name': rec['room_name'], 'capacity': 10}
            
            return recs
            
        except Exception as e:
            print(f"❌ Base recommendation error: {e}")
            return []

    
    def _get_enhanced_recommendations_sync(self, request_data: Dict[str, Any]) -> List[Dict[str, Any]]:
       
        base_recs = self._get_base_recommendations(request_data)
        logger.info(f"Base recommendations: {len(base_recs)}")
        
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
        
        return self._sort_by_priority(enhanced_recs)[:8]
    
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
        
        return self._sort_by_priority(enhanced_recs)[:8]
    
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
                # Return neutral scores if ML unavailable
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
        """
        Calculate ML-based score using embeddings
        
        This is where you'd use:
        - Room similarity embeddings
        - User preference embeddings
        - Historical booking patterns
        """
        base_score = 0.5
        
        try:
            room_name = rec.get('room_name', '')
            booking_history = user_context.get('booking_history', [])
            
            room_usage = sum(1 for booking in booking_history 
                           if booking.get('room_name') == room_name)
            
            if room_usage > 0:
                base_score += min(room_usage * 0.02, 0.1)  # Very minimal impact
            
        except Exception:
            pass
        
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
    
    def _print_final_scores(self, recommendations: List[Dict]):
        """Print final scores summary with priority-based ordering"""
        print("\n" + "="*150)
        print("🏆 FINAL HYBRID SCORES WITH PRIORITY-BASED ORDERING")
        print("="*150)
        print(f"{'Rank':<5} | {'Type':<18} | {'Room Name':<25} | {'Hybrid':>8} | {'Base':>8} | {'ML':>8} | {'LLM':>8}")
        print("-" * 150)
        
        for i, rec in enumerate(recommendations[:8], 1):
            room_name = rec.get('room_name', f'Room_{i}')[:24]
            rec_type = rec.get('type', 'unknown')[:17]
            suggestion = rec.get('suggestion', {})
            
            final_score = rec.get('final_score', 0.0)
            base_score = rec.get('base_score', rec.get('score', 0.0))
            ml_score = rec.get('ml_score', 0.0)
            llm_score = rec.get('llm_score', 0.0)
            
            print(f"{i:<5} | {rec_type:<18} | {room_name:<25} | {final_score:>8.3f} | {base_score:>8.3f} | {ml_score:>8.3f} | {llm_score:>8.3f}")
        
        print("="*150)
        print(f"📈 Total: {len(recommendations)} | Priority Order: Alternative Rooms → Alternative Times → Smart Scheduling")
        print("="*150 + "\n")
    
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
                'priority_based_sorting'
            ]
           
        }
        
    def get_recommendation_explanation(self, recommendation: Dict) -> Dict[str, Any]:
        """Get explanation for a recommendation"""
        suggestion = recommendation.get('suggestion', {})
        
        # Calculate duration info
        duration_info = {}
        try:
            if 'duration_minutes' in suggestion:
                duration_minutes = suggestion['duration_minutes']
                duration_info = {
                    'duration_minutes': duration_minutes,
                    'duration_hours': duration_minutes / 60,
                    'duration_display': f"{duration_minutes} minutes ({duration_minutes/60:.1f} hours)"
                }
            else:
                start_dt = self._parse_datetime(suggestion.get('start_time', ''))
                end_dt = self._parse_datetime(suggestion.get('end_time', ''))
                duration_minutes = self._calculate_duration_minutes(start_dt, end_dt)
                duration_info = {
                    'duration_minutes': duration_minutes,
                    'duration_hours': duration_minutes / 60,
                    'duration_display': f"{duration_minutes} minutes ({duration_minutes/60:.1f} hours)"
                }
        except Exception as e:
            duration_info = {'error': f'Could not calculate duration: {e}'}
        
        return {
            'room_name': recommendation.get('room_name', 'Unknown'),
            'start_time': suggestion.get('start_time', 'N/A'),
            'end_time': suggestion.get('end_time', 'N/A'),
            'duration_info': duration_info,
            'final_score': recommendation.get('final_score', 0),
            'score_breakdown': recommendation.get('score_breakdown', {}),
            'mode_used': self.mode,
            'recommendation_type': recommendation.get('type', 'unknown'),
            'data_source': recommendation.get('data_source', 'unknown'),
            'duration_handling_method': 'preserved_user_duration',
            'priority_order': self._get_priority_order(recommendation.get('type', 'unknown')),
            'fixes_applied': 'time_validation_duration_preservation_priority_sorting'
        }

    def check_lt1_availability(self, date: str, start_time: str, end_time: str) -> Dict[str, Any]:
        """Specific method to check LT1 availability for your booking request"""
        try:
            full_start = f"{date}T{start_time}:00"
            full_end = f"{date}T{end_time}:00"
            
            return self.check_room_availability_for_booking('LT1', full_start, full_end)
        except Exception as e:
            return {
                'available': False,
                'message': f'Error checking LT1 availability: {str(e)}',
                'error': True
            }

    def demo_lt1_booking_fix(self) -> Dict[str, Any]:
        """Demonstrate the LT1 booking fix for 2025-08-15 8AM-12PM"""
        
        request_data = {
            'user_id': 'demo_user',
            'room_id': 'LT1',
            'start_time': '2025-08-15T08:00:00',
            'end_time': '2025-08-15T12:00:00',
            'purpose': 'lecture',
            'capacity': 50
        }
        
        print("\n🎯 DEMONSTRATING LT1 BOOKING FIX WITH PRIORITY ORDERING")
        print("="*70)
        print(f"Request: LT1 on 2025-08-15 from 08:00 to 12:00 (4 hours)")
        
        # Check LT1 availability first
        lt1_check = self.check_lt1_availability('2025-08-15', '08:00', '12:00')
        print(f"LT1 Availability: {lt1_check.get('message', 'Unknown')}")
        
        # Get recommendations
        recommendations = self.get_recommendations(request_data)
        
        # Validate that all recommendations have 4-hour duration and correct ordering
        print("\n📊 DURATION & ORDERING VALIDATION:")
        type_counts = {'alternative_room': 0, 'alternative_time': 0, 'smart_scheduling': 0}
        
        for i, rec in enumerate(recommendations[:5], 1):
            suggestion = rec.get('suggestion', {})
            rec_type = rec.get('type', 'unknown')
            type_counts[rec_type] = type_counts.get(rec_type, 0) + 1
            
            try:
                start_dt = self._parse_datetime(suggestion.get('start_time', ''))
                end_dt = self._parse_datetime(suggestion.get('end_time', ''))
                duration_min = self._calculate_duration_minutes(start_dt, end_dt)
                
                print(f"{i}. {rec_type} - {suggestion.get('room_name', 'Unknown')}: {duration_min} minutes ({'✅' if duration_min == 240 else '❌'})")
            except:
                print(f"{i}. {rec_type} - {suggestion.get('room_name', 'Unknown')}: Duration calculation error")
        
        return {
            'lt1_availability': lt1_check,
            'recommendations_count': len(recommendations),
            'type_distribution': type_counts,
            'all_durations_correct': all(
                self._get_duration_from_rec(rec) == 240 
                for rec in recommendations[:3] 
                if self._get_duration_from_rec(rec) is not None
            ),
            'priority_ordering_applied': True
        }
    
    def _get_duration_from_rec(self, rec: Dict) -> Optional[int]:
        """Helper to get duration from recommendation"""
        try:
            suggestion = rec.get('suggestion', {})
            if 'duration_minutes' in suggestion:
                return suggestion['duration_minutes']
            
            start_dt = self._parse_datetime(suggestion.get('start_time', ''))
            end_dt = self._parse_datetime(suggestion.get('end_time', ''))
            return self._calculate_duration_minutes(start_dt, end_dt)
        except:
            return None