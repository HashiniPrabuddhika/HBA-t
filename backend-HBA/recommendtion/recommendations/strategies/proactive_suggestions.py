# recommendtion/recommendations/strategies/proactive_suggestions.py
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from ..data.analytics_processor import AnalyticsProcessor
from ..models.embedding_model import EmbeddingModel
from ..utils.time_utils import TimeUtils
from ..utils.metrics import RecommendationMetrics
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class BookingSuggestion:
    type: str
    room_name: str
    date: str
    start_time: str
    end_time: str
    confidence_score: float
    reason: str
    metadata: Dict[str, Any]

class ProactiveSuggestionStrategy:
    def __init__(self, db: Session):
        self.db = db
        self.analytics = AnalyticsProcessor(db)
        self.embedding_model = EmbeddingModel()
        self.time_utils = TimeUtils()
        self.metrics = RecommendationMetrics()
        self.max_suggestions = 5
        self.min_confidence_threshold = 0.3
        self.business_hours = {'start': '08:00', 'end': '20:00'}
        self.prediction_window_days = 14
    
    async def predict_future_bookings(self, user_id: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        try:
            user_patterns = await self.analytics.get_user_booking_patterns(user_id)
            if not user_patterns:
                return await self._suggest_default_bookings(user_id, context)
            
            suggestions = []
            strategies = [
                self._suggest_recurring_bookings,
                self._suggest_seasonal_bookings,
                self._suggest_collaborative_bookings,
                self._suggest_context_aware_bookings,
                self._suggest_gap_filling_bookings
            ]
            
            for strategy in strategies:
                try:
                    suggestions.extend(await strategy(user_id, user_patterns, context))
                except Exception as e:
                    logger.warning(f"Strategy failed: {strategy.__name__}: {str(e)}")
            
            filtered = self._filter_and_deduplicate_suggestions(suggestions)
            filtered.sort(key=lambda x: x['confidence_score'], reverse=True)
            
            await self.metrics.log_proactive_suggestions(user_id, len(filtered), context)
            return filtered[:self.max_suggestions]
            
        except Exception as e:
            logger.error(f"Error predicting bookings for {user_id}: {str(e)}")
            return await self._suggest_default_bookings(user_id, context)
    
    async def _suggest_recurring_bookings(self, user_id: str, user_patterns: Dict, context: Dict = None) -> List[Dict]:
        suggestions = []
        for pattern in user_patterns.get('recurring_patterns', []):
            try:
                last_date = datetime.strptime(pattern['last_booking_date'], "%Y-%m-%d")
                next_dates = self._calc_next_occurrences(last_date, pattern['frequency'], 3)
                
                for next_date in next_dates:
                    if next_date > datetime.now() and await self._check_room_availability(
                        pattern['room_name'], next_date.strftime("%Y-%m-%d"), 
                        pattern['start_time'], pattern['end_time']):
                        
                        suggestions.append({
                            'type': 'recurring', 'room_name': pattern['room_name'],
                            'date': next_date.strftime("%Y-%m-%d"),
                            'start_time': pattern['start_time'], 'end_time': pattern['end_time'],
                            'confidence_score': min(0.8 * pattern.get('consistency_score', 0.5), 1.0),
                            'reason': f'Based on your {pattern["frequency"]} pattern',
                            'metadata': {'pattern_strength': pattern.get('occurrences', 1)}
                        })
            except (ValueError, KeyError):
                continue
        return suggestions
    
    async def _suggest_seasonal_bookings(self, user_id: str, user_patterns: Dict, context: Dict = None) -> List[Dict]:
        suggestions = []
        seasonal_patterns = user_patterns.get('seasonal_patterns', {})
        current_month = datetime.now().month
        
        for pattern in [seasonal_patterns.get(str(current_month), {}), 
                       seasonal_patterns.get(self._get_season(current_month), {})]:
            if not pattern: continue
            
            for room in pattern.get('preferred_rooms', [])[:2]:
                for time_slot in pattern.get('preferred_times', [])[:2]:
                    dates = self._get_next_suitable_dates(pattern.get('preferred_weekdays', []), 7)
                    
                    for date in dates[:2]:
                        if await self._check_room_availability(room, date.strftime("%Y-%m-%d"),
                                                             time_slot['start_time'], time_slot['end_time']):
                            suggestions.append({
                                'type': 'seasonal', 'room_name': room, 'date': date.strftime("%Y-%m-%d"),
                                'start_time': time_slot['start_time'], 'end_time': time_slot['end_time'],
                                'confidence_score': min(0.6 * pattern.get('preference_score', 0.5), 0.85),
                                'reason': f'You typically book {room} in {datetime.now().strftime("%B")}',
                                'metadata': {'seasonal_factor': pattern.get('frequency', 1)}
                            })
        return suggestions
    
    async def _suggest_collaborative_bookings(self, user_id: str, user_patterns: Dict, context: Dict = None) -> List[Dict]:
        suggestions = []
        try:
            similar_users = await self.embedding_model.find_similar_users(user_id, n_results=5)
            
            for user in similar_users:
                if user['similarity_score'] < 0.3: continue
                
                similar_patterns = await self.analytics.get_user_booking_patterns(user['user_id'])
                for booking in similar_patterns.get('recent_bookings', [])[:3]:
                    try:
                        booking_date = datetime.strptime(booking['date'], "%Y-%m-%d")
                        suggested_dates = [booking_date + timedelta(days=d) for d in [7, 14]]
                        
                        for date in suggested_dates:
                            if date > datetime.now() and await self._check_room_availability(
                                booking['room_name'], date.strftime("%Y-%m-%d"),
                                booking['start_time'], booking['end_time']):
                                
                                suggestions.append({
                                    'type': 'collaborative', 'room_name': booking['room_name'],
                                    'date': date.strftime("%Y-%m-%d"),
                                    'start_time': booking['start_time'], 'end_time': booking['end_time'],
                                    'confidence_score': user['similarity_score'] * 0.7,
                                    'reason': 'Similar users have booked this recently',
                                    'metadata': {'similarity_score': user['similarity_score']}
                                })
                    except ValueError:
                        continue
        except Exception as e:
            logger.error(f"Collaborative filtering error: {str(e)}")
        return suggestions
    
    async def _suggest_context_aware_bookings(self, user_id: str, user_patterns: Dict, context: Dict = None) -> List[Dict]:
        suggestions = []
        if not context: return suggestions
        
        # Handle different context types
        handlers = [
            (lambda: context.get('recent_booking'), self._handle_recent_booking_context),
            (lambda: user_patterns.get('typically_available_times'), self._handle_time_preference_context),
            (lambda: user_patterns.get('preferred_rooms'), self._handle_room_preference_context),
            (lambda: context.get('workload_level'), self._handle_workload_context)
        ]
        
        for condition, handler in handlers:
            if condition():
                await handler(context, user_patterns, suggestions)
        
        return suggestions
    
    async def _suggest_gap_filling_bookings(self, user_id: str, user_patterns: Dict, context: Dict = None) -> List[Dict]:
        suggestions = []
        try:
            upcoming_bookings = await self.analytics.get_upcoming_bookings(user_id, days_ahead=14)
            if not upcoming_bookings: return suggestions
            
            gaps = self._identify_schedule_gaps(upcoming_bookings)
            preferred_duration = user_patterns.get('average_duration_minutes', 60)
            
            for gap in gaps:
                if gap['duration_minutes'] >= preferred_duration:
                    for room in user_patterns.get('preferred_rooms', [])[:3]:
                        if await self._check_room_availability(
                            room, gap['date'], gap['start_time'], 
                            self._add_minutes_to_time(gap['start_time'], preferred_duration)):
                            
                            suggestions.append({
                                'type': 'gap_filling', 'room_name': room, 'date': gap['date'],
                                'start_time': gap['start_time'],
                                'end_time': self._add_minutes_to_time(gap['start_time'], preferred_duration),
                                'confidence_score': self._calc_gap_confidence(gap, user_patterns),
                                'reason': 'Fill gap in your schedule',
                                'metadata': {'gap_duration': gap['duration_minutes']}
                            })
        except Exception as e:
            logger.error(f"Gap filling error: {str(e)}")
        return suggestions
    
    async def _suggest_default_bookings(self, user_id: str, context: Dict) -> List[Dict]:
        suggestions = []
        popular_slots = await self.analytics.get_popular_time_slots()
        popular_rooms = await self.analytics.get_popular_rooms()
        next_week_start = datetime.now() + timedelta(days=7)
        
        for i in range(3):
            date = next_week_start + timedelta(days=i)
            if popular_slots and popular_rooms:
                slot = popular_slots[i % len(popular_slots)]
                room = popular_rooms[i % len(popular_rooms)]
                
                if await self._check_room_availability(room['name'], date.strftime("%Y-%m-%d"),
                                                     slot['start_time'], slot['end_time']):
                    suggestions.append({
                        'type': 'default', 'room_name': room['name'], 'date': date.strftime("%Y-%m-%d"),
                        'start_time': slot['start_time'], 'end_time': slot['end_time'],
                        'confidence_score': 0.4, 'reason': 'Popular combination',
                        'metadata': {'popularity_score': slot.get('popularity', 0)}
                    })
        return suggestions
    
    # Helper methods (shortened)
    def _calc_next_occurrences(self, last_date: datetime, frequency: str, count: int) -> List[datetime]:
        delta_map = {'daily': 1, 'weekly': 7, 'monthly': 30, 'bi-weekly': 14}
        delta_days = delta_map.get(frequency, 7)
        return [last_date + timedelta(days=delta_days * (i+1)) for i in range(count)]
    
    def _get_season(self, month: int) -> str:
        return ['winter', 'winter', 'spring', 'spring', 'spring', 'summer', 
                'summer', 'summer', 'fall', 'fall', 'fall', 'winter'][month-1]
    
    def _get_next_suitable_dates(self, preferred_weekdays: List[int], days_ahead: int = 7) -> List[datetime]:
        dates = []
        current = datetime.now() + timedelta(days=1)
        for i in range(days_ahead * 2):
            date = current + timedelta(days=i)
            if not preferred_weekdays or date.weekday() in preferred_weekdays:
                dates.append(date)
                if len(dates) >= 3: break
        return dates
    
    def _identify_schedule_gaps(self, bookings: List[Dict]) -> List[Dict]:
        gaps = []
        bookings_by_date = {}
        for b in sorted(bookings, key=lambda x: (x['date'], x['start_time'])):
            bookings_by_date.setdefault(b['date'], []).append(b)
        
        for date, day_bookings in bookings_by_date.items():
            day_bookings.sort(key=lambda x: x['start_time'])
            for i in range(len(day_bookings) - 1):
                gap_mins = self._calc_time_diff(day_bookings[i]['end_time'], day_bookings[i+1]['start_time'])
                if gap_mins >= 30:
                    gaps.append({
                        'date': date, 'start_time': day_bookings[i]['end_time'],
                        'end_time': day_bookings[i+1]['start_time'], 'duration_minutes': gap_mins,
                        'type': 'between_meetings'
                    })
        return gaps
    
    def _calc_time_diff(self, start: str, end: str) -> int:
        try:
            return int((datetime.strptime(end, "%H:%M") - datetime.strptime(start, "%H:%M")).total_seconds() / 60)
        except: return 0
    
    def _add_minutes_to_time(self, time_str: str, minutes: int) -> str:
        try:
            return (datetime.strptime(time_str, "%H:%M") + timedelta(minutes=minutes)).strftime("%H:%M")
        except: return time_str
    
    def _calc_gap_confidence(self, gap: Dict, user_patterns: Dict) -> float:
        base = 0.5
        gap_duration = gap['duration_minutes']
        preferred_duration = user_patterns.get('average_duration_minutes', 60)
        duration_match = 1 - abs(gap_duration - preferred_duration) / preferred_duration
        confidence = base + (duration_match * 0.2)
        return min(confidence + (0.1 if gap['type'] == 'between_meetings' else 0), 0.7)
    
    async def _check_room_availability(self, room_name: str, date: str, start_time: str, end_time: str) -> bool:
        try:
            from src.availability_logic import check_availability
            return await check_availability(room_name=room_name, date=date, 
                                          start_time=start_time, end_time=end_time)
        except Exception as e:
            logger.error(f"Availability check error: {str(e)}")
            return False
    
    def _filter_and_deduplicate_suggestions(self, suggestions: List[Dict]) -> List[Dict]:
        filtered = [s for s in suggestions if s['confidence_score'] >= self.min_confidence_threshold]
        seen = set()
        deduplicated = []
        for s in filtered:
            key = (s['room_name'], s['date'], s['start_time'], s['end_time'])
            if key not in seen:
                seen.add(key)
                deduplicated.append(s)
        return deduplicated
    
    # Context handlers (simplified)
    async def _handle_recent_booking_context(self, context: Dict, user_patterns: Dict, suggestions: List):
        if rb := context.get('recent_booking'):
            if rb.get('room_type') == 'presentation':
                follow_up_start = self._add_minutes_to_time(rb.get('end_time', ''), 15)
                follow_up_end = self._add_minutes_to_time(follow_up_start, 60)
                
                if await self._check_room_availability('Discussion Room', rb['date'], 
                                                     follow_up_start, follow_up_end):
                    suggestions.append({
                        'type': 'context_aware', 'room_name': 'Discussion Room',
                        'date': rb['date'], 'start_time': follow_up_start, 'end_time': follow_up_end,
                        'confidence_score': 0.7, 'reason': 'Follow-up discussion after presentation',
                        'metadata': {'context_type': 'follow_up'}
                    })
    
    async def _handle_time_preference_context(self, context: Dict, user_patterns: Dict, suggestions: List):
        optimal_times = await self.analytics.get_optimal_booking_times()
        available_times = user_patterns.get('typically_available_times', [])
        next_week = datetime.now() + timedelta(days=7)
        
        for ot in optimal_times[:3]:
            if ot['time_slot'] in available_times and await self._check_room_availability(
                ot['room_name'], next_week.strftime("%Y-%m-%d"), ot['start_time'], ot['end_time']):
                suggestions.append({
                    'type': 'context_aware', 'room_name': ot['room_name'],
                    'date': next_week.strftime("%Y-%m-%d"),
                    'start_time': ot['start_time'], 'end_time': ot['end_time'],
                    'confidence_score': 0.65, 'reason': 'Optimal time slot',
                    'metadata': {'context_type': 'optimal_time'}
                })
    
    async def _handle_room_preference_context(self, context: Dict, user_patterns: Dict, suggestions: List):
        tomorrow = datetime.now() + timedelta(days=1)
        for room in user_patterns.get('preferred_rooms', [])[:2]:
            for time_slot in user_patterns.get('preferred_times', [])[:2]:
                if await self._check_room_availability(room, tomorrow.strftime("%Y-%m-%d"),
                                                     time_slot['start_time'], time_slot['end_time']):
                    suggestions.append({
                        'type': 'context_aware', 'room_name': room,
                        'date': tomorrow.strftime("%Y-%m-%d"),
                        'start_time': time_slot['start_time'], 'end_time': time_slot['end_time'],
                        'confidence_score': 0.6, 'reason': 'Based on preferences',
                        'metadata': {'context_type': 'preference_match'}
                    })
    
    async def _handle_workload_context(self, context: Dict, user_patterns: Dict, suggestions: List):
        workload = context.get('workload_level', 'medium')
        duration_map = {'high': 30, 'low': 120, 'medium': user_patterns.get('average_duration_minutes', 60)}
        duration = duration_map[workload]
        reason_map = {'high': 'Shorter meeting for busy schedule', 
                     'low': 'Extended session for light schedule',
                     'medium': 'Standard duration'}
        
        tomorrow = datetime.now() + timedelta(days=1)
        popular_rooms = await self.analytics.get_popular_rooms()
        
        for room in popular_rooms[:2]:
            start_time, end_time = '09:00', self._add_minutes_to_time('09:00', duration)
            if await self._check_room_availability(room['name'], tomorrow.strftime("%Y-%m-%d"), 
                                                 start_time, end_time):
                suggestions.append({
                    'type': 'context_aware', 'room_name': room['name'],
                    'date': tomorrow.strftime("%Y-%m-%d"),
                    'start_time': start_time, 'end_time': end_time,
                    'confidence_score': 0.55 if workload != 'medium' else 0.5,
                    'reason': reason_map[workload],
                    'metadata': {'context_type': 'workload_optimization', 'workload_level': workload}
                })
    
    # Alternative suggestions methods (condensed)
    async def generate_alternative_time_suggestions(self, user_id: str, requested_room: str, 
                                                   requested_date: str, requested_start_time: str, 
                                                   requested_end_time: str, context: Dict = None) -> List[Dict]:
        try:
            user_patterns = await self.analytics.get_user_booking_patterns(user_id)
            duration = self._calc_time_diff(requested_start_time, requested_end_time)
            date_obj = datetime.strptime(requested_date, "%Y-%m-%d")
            
            existing_bookings = await self.analytics.get_room_bookings_for_date(requested_room, requested_date)
            
            alternatives = []
            if not existing_bookings:
                alternatives = await self._suggest_optimal_times_for_room(requested_room, requested_date, duration, user_patterns)
            else:
                alternatives = await self._find_time_gaps_in_schedule(existing_bookings, requested_date, duration, user_patterns)
            
            if len(alternatives) < 3:
                future_alts = await self._suggest_future_date_alternatives(requested_room, date_obj, 
                                                                         requested_start_time, requested_end_time, user_patterns)
                alternatives.extend(future_alts)
            
            alternatives.sort(key=lambda x: x['confidence_score'], reverse=True)
            return alternatives[:5]
            
        except Exception as e:
            logger.error(f"Error generating alternatives: {str(e)}")
            return []
    
    async def get_proactive_suggestions_summary(self, user_id: str) -> Dict[str, Any]:
        try:
            user_patterns = await self.analytics.get_user_booking_patterns(user_id)
            suggestion_metrics = await self.metrics.get_user_suggestion_metrics(user_id)
            next_booking = await self._predict_next_booking_time(user_patterns)
            trending = await self.analytics.get_trending_booking_data()
            
            return {
                'user_id': user_id, 'summary_generated_at': datetime.now().isoformat(),
                'booking_patterns': {
                    'total_bookings': user_patterns.get('total_bookings', 0),
                    'recurring_patterns': len(user_patterns.get('recurring_patterns', [])),
                    'preferred_rooms': user_patterns.get('preferred_rooms', [])[:3],
                    'average_duration': user_patterns.get('average_duration_minutes', 60)
                },
                'suggestions_performance': {
                    'total_suggestions_made': suggestion_metrics.get('total_suggestions', 0),
                    'acceptance_rate': suggestion_metrics.get('acceptance_rate', 0.0)
                },
                'predictions': {'next_likely_booking': next_booking},
                'trending_insights': {
                    'popular_rooms_this_week': trending.get('popular_rooms', [])[:3],
                    'peak_booking_times': trending.get('peak_times', [])[:3]
                }
            }
        except Exception as e:
            logger.error(f"Summary error: {str(e)}")
            return {'user_id': user_id, 'error': 'Unable to generate summary'}
    
    def get_strategy_weights(self) -> Dict[str, float]:
        return {'recurring': 0.35, 'seasonal': 0.20, 'collaborative': 0.15, 
                'context_aware': 0.20, 'gap_filling': 0.10}
    
    async def cleanup_old_suggestions(self, days_old: int = 30) -> int:
        try:
            cutoff_date = datetime.now() - timedelta(days=days_old)
            deleted_count = await self.analytics.cleanup_old_suggestions(cutoff_date)
            logger.info(f"Cleaned up {deleted_count} old suggestions")
            return deleted_count
        except Exception as e:
            logger.error(f"Cleanup error: {str(e)}")
            return 0