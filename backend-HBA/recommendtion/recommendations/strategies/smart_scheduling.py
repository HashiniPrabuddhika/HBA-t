#recommendtion.recommendations.strategies.smart_scheduling.py
import asyncio
import json
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import logging
from dataclasses import dataclass
from enum import Enum
import aiosqlite
import pickle
import base64
from sqlalchemy.orm import Session
from src.database import get_db
from src.models import MRBSRoom, MRBSEntry, MRBSRepeat
from ..core.pattern_analyzer import PatternAnalyzer
from ..core.preference_learner import PreferenceLearner
from ..data.analytics_processor import AnalyticsProcessor
from ..utils.time_utils import TimeUtils
from ..utils.metrics import RecommendationMetrics

logger = logging.getLogger(__name__)

class OptimizationGoal(Enum):
    MINIMIZE_CONFLICTS = "minimize_conflicts"
    MAXIMIZE_EFFICIENCY = "maximize_efficiency"
    REDUCE_TRAVEL_TIME = "reduce_travel_time"
    BALANCE_WORKLOAD = "balance_workload"
    ENERGY_SAVINGS = "energy_savings"
    USER_SATISFACTION = "user_satisfaction"

@dataclass
class SchedulingConstraint:
    constraint_type: str
    priority: int
    parameters: Dict[str, Any]
    user_id: Optional[int] = None
    room_id: Optional[str] = None

@dataclass
class OptimizedSchedule:
    original_request: Dict[str, Any]
    optimized_slot: Dict[str, Any]
    optimization_score: float
    improvements: List[str]
    trade_offs: List[str]
    confidence: float
    alternative_slots: List[Dict[str, Any]]

class SQLiteCacheManager:
    def __init__(self, db_path: str = "cache.db"):
        self.db_path = db_path
        self.initialized = False
    
    async def initialize(self):
        if self.initialized: return
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""CREATE TABLE IF NOT EXISTS cache_entries (
                key TEXT PRIMARY KEY, value TEXT NOT NULL, expires_at REAL,
                created_at REAL DEFAULT (julianday('now')), updated_at REAL DEFAULT (julianday('now'))
            )""")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_expires_at ON cache_entries(expires_at)")
            await db.commit()
        self.initialized = True
    
    async def get(self, key: str) -> Optional[Any]:
        await self.initialize()
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("SELECT value, expires_at FROM cache_entries WHERE key = ?", (key,))
            row = await cursor.fetchone()
            if not row: return None
            
            value_str, expires_at = row
            if expires_at and expires_at < datetime.now().timestamp():
                await self.delete(key)
                return None
            
            try: return json.loads(value_str)
            except: 
                try: return pickle.loads(base64.b64decode(value_str))
                except: return value_str
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        await self.initialize()
        try: value_str = json.dumps(value)
        except: 
            try: value_str = base64.b64encode(pickle.dumps(value)).decode('utf-8')
            except: value_str = str(value)
        
        expires_at = (datetime.now() + timedelta(seconds=ttl)).timestamp() if ttl else None
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("INSERT OR REPLACE INTO cache_entries (key, value, expires_at, updated_at) VALUES (?, ?, ?, julianday('now'))", 
                           (key, value_str, expires_at))
            await db.commit()
        return True
    
    async def delete(self, key: str) -> bool:
        await self.initialize()
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
            await db.commit()
            return cursor.rowcount > 0
    
    async def clear_expired(self):
        await self.initialize()
        current_time = datetime.now().timestamp()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("DELETE FROM cache_entries WHERE expires_at IS NOT NULL AND expires_at < ?", (current_time,))
            await db.commit()
    
    async def get_stats(self) -> Dict[str, Any]:
        await self.initialize()
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("SELECT COUNT(*) FROM cache_entries")
            total = (await cursor.fetchone())[0]
            cursor = await db.execute("SELECT COUNT(*) FROM cache_entries WHERE expires_at IS NOT NULL AND expires_at < ?", 
                                    (datetime.now().timestamp(),))
            expired = (await cursor.fetchone())[0]
            cursor = await db.execute("SELECT SUM(LENGTH(value)) FROM cache_entries")
            size = (await cursor.fetchone())[0] or 0
            return {'total_entries': total, 'expired_entries': expired, 'active_entries': total - expired, 'cache_size_bytes': size}

class SmartSchedulingStrategy:
    def __init__(self, db: Session, db_session: Session = None, cache_db_path: str = "scheduling_cache.db"):
        self.db = db or db_session
        self.cache_manager = SQLiteCacheManager(cache_db_path)
        self.time_utils = TimeUtils()
        
        # Initialize components with error handling
        components = [
            ('pattern_analyzer', PatternAnalyzer, [self.db]),
            ('preference_learner', PreferenceLearner, []),
            ('analytics_processor', AnalyticsProcessor, [self.db]),
            ('metrics', RecommendationMetrics, [])
        ]
        
        for name, cls, args in components:
            try:
                setattr(self, name, cls(*args))
            except Exception as e:
                logger.warning(f"Could not initialize {name}: {e}")
                setattr(self, name, None)
        
        self.optimization_weights = {
            OptimizationGoal.MINIMIZE_CONFLICTS: 0.25, OptimizationGoal.MAXIMIZE_EFFICIENCY: 0.20,
            OptimizationGoal.REDUCE_TRAVEL_TIME: 0.15, OptimizationGoal.BALANCE_WORKLOAD: 0.15,
            OptimizationGoal.ENERGY_SAVINGS: 0.10, OptimizationGoal.USER_SATISFACTION: 0.15
        }
        
        asyncio.create_task(self._schedule_cache_cleanup())
    
    async def _schedule_cache_cleanup(self):
        while True:
            try:
                await asyncio.sleep(3600)
                await self.cache_manager.clear_expired()
                logger.info("Cache cleanup completed")
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")
    
    async def optimize_schedule(self, user_id: str, context: Dict[str, Any], 
                              current_bookings: List[Dict[str, Any]] = None,
                              user_bookings: List[Dict[str, Any]] = None,
                              availability_checker: callable = None) -> List[Dict[str, Any]]:
        try:
            logger.info(f"Starting schedule optimization for user {user_id}")
            
            if not self.pattern_analyzer or not self.analytics_processor:
                return self._create_mock_optimizations(user_id, context)
            
            optimizations = []
            optimization_methods = [
                (self._can_optimize_time(context), self._generate_time_optimization, user_bookings or []),
                (self._can_optimize_room(context), self._generate_room_optimization, current_bookings or []),
                (bool(user_bookings), self._generate_efficiency_optimization, user_bookings or [])
            ]
            
            for can_optimize, method, data in optimization_methods:
                if can_optimize:
                    result = await method(user_id, context, data)
                    if result: optimizations.append(result)
            
            logger.info(f"Generated {len(optimizations)} optimization recommendations")
            return optimizations
            
        except Exception as e:
            logger.error(f"Error in schedule optimization: {e}")
            return self._create_mock_optimizations(user_id, context)
    
    async def _generate_time_optimization(self, user_id: str, context: Dict[str, Any], user_bookings: List) -> Dict[str, Any]:
        try:
            patterns = await self._analyze_time_patterns(user_id, user_bookings) if user_bookings and self.pattern_analyzer else {'preferred_hours': [9, 10, 14, 15]}
            optimal_times = patterns.get('preferred_hours', [9, 10, 14, 15])
            
            current_hour = 12
            try:
                if context.get('start_time'):
                    current_hour = datetime.fromisoformat(context['start_time']).hour
            except: pass
            
            if current_hour not in optimal_times:
                suggested_hour = min(optimal_times, key=lambda x: abs(x - current_hour))
                return {
                    'type': 'time_optimization', 'score': 0.8,
                    'reason': f'Moving to {suggested_hour}:00 aligns with your productive hours',
                    'suggestion': {'optimization_type': 'time_shift', 'suggested_hour': suggested_hour, 'confidence': 0.8}
                }
            return None
        except Exception as e:
            logger.error(f"Error generating time optimization: {e}")
            return None
    
    async def _generate_room_optimization(self, user_id: str, context: Dict[str, Any], current_bookings: List) -> Dict[str, Any]:
        try:
            room_name = context.get('room_name', '')
            if 'large' in room_name.lower() or 'conference' in room_name.lower():
                return {
                    'type': 'room_optimization', 'score': 0.7,
                    'reason': 'Consider smaller room if fewer than 6 attendees',
                    'suggestion': {'optimization_type': 'room_size', 'recommended_capacity': 'small', 'confidence': 0.7}
                }
            return None
        except Exception as e:
            logger.error(f"Error generating room optimization: {e}")
            return None
    
    async def _generate_efficiency_optimization(self, user_id: str, context: Dict[str, Any], user_bookings: List) -> Dict[str, Any]:
        try:
            if len(user_bookings) > 10:
                return {
                    'type': 'efficiency_optimization', 'score': 0.65,
                    'reason': 'Consider batching meetings to create focus time blocks',
                    'suggestion': {'optimization_type': 'meeting_batching', 'recommendation': 'group_meetings', 'confidence': 0.65}
                }
            return None
        except Exception as e:
            logger.error(f"Error generating efficiency optimization: {e}")
            return None
    
    async def _analyze_time_patterns(self, user_id: str, bookings: List[Dict[str, Any]]) -> Dict[str, Any]:
        try:
            if not bookings: return {'preferred_hours': [9, 10, 14, 15]}
            
            hours = []
            for booking in bookings:
                try:
                    start_time = booking.get('start_time', '')
                    if isinstance(start_time, str):
                        dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                        hours.append(dt.hour)
                    elif hasattr(start_time, 'hour'):
                        hours.append(start_time.hour)
                except: continue
            
            if not hours: return {'preferred_hours': [9, 10, 14, 15]}
            
            from collections import Counter
            hour_counts = Counter(hours)
            preferred_hours = [hour for hour, count in hour_counts.most_common(4)]
            
            return {'preferred_hours': preferred_hours, 'total_meetings': len(bookings), 'hour_distribution': dict(hour_counts)}
        except Exception as e:
            logger.error(f"Error analyzing time patterns: {e}")
            return {'preferred_hours': [9, 10, 14, 15]}

    def _create_mock_optimizations(self, user_id: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        room_name = context.get("room_name", "unknown_room")
        start_time = context.get("start_time", "")
        end_time = context.get("end_time", "")
        
        return [
            {
                'type': 'time_optimization', 'score': 0.75,
                'reason': 'Suggest moving meeting 30 minutes earlier for better focus time',
                'suggestion': {'room_name': room_name, 'start_time': start_time, 'end_time': end_time, 
                              'optimization_type': 'time_shift', 'confidence': 0.75},
                'data_source': 'mock_optimization'
            },
            {
                'type': 'efficiency_optimization', 'score': 0.68,
                'reason': 'Combining with nearby meetings to reduce context switching',
                'suggestion': {'room_name': room_name, 'start_time': start_time, 'end_time': end_time, 
                              'optimization_type': 'batch_meetings', 'confidence': 0.68},
                'data_source': 'mock_optimization'
            }
        ]
    
    def _can_optimize_time(self, context: Dict[str, Any]) -> bool:
        return all(key in context for key in ["start_time", "end_time", "date"])
    
    def _can_optimize_room(self, context: Dict[str, Any]) -> bool:
        return "room_name" in context
    
    async def suggest_optimal_meeting_time(self, attendees: List[int], duration_minutes: int,
                                         room_requirements: Dict[str, Any] = None,
                                         preferred_time_range: Tuple[datetime, datetime] = None,
                                         optimization_goals: List[OptimizationGoal] = None) -> List[Dict[str, Any]]:
        try:
            logger.info(f"Finding optimal meeting time for {len(attendees)} attendees")
            
            cache_key = self._generate_cache_key("meeting_suggestions", attendees, duration_minutes, room_requirements, preferred_time_range)
            cached_result = await self.cache_manager.get(cache_key)
            if cached_result:
                logger.info("Returning cached meeting suggestions")
                return cached_result
            
            # Get availability and preferences
            attendee_data = {}
            for attendee_id in attendees:
                availability = await self._get_user_availability(attendee_id, preferred_time_range)
                preferences = await self._load_user_preferences(attendee_id)
                attendee_data[attendee_id] = {'availability': availability, 'preferences': preferences}
            
            # Find common slots and score them
            common_slots = self._find_common_available_slots(
                {aid: data['availability'] for aid, data in attendee_data.items()}, 
                duration_minutes, preferred_time_range
            )
            
            scored_slots = []
            for slot in common_slots:
                score = await self._score_meeting_slot(
                    slot, attendees, {aid: data['preferences'] for aid, data in attendee_data.items()},
                    room_requirements, optimization_goals or []
                )
                scored_slots.append({
                    'start_time': slot['start_time'].isoformat(),
                    'end_time': slot['end_time'].isoformat(),
                    'score': score, 'reasoning': self._generate_slot_reasoning(slot, score)
                })
            
            result = sorted(scored_slots, key=lambda x: x['score'], reverse=True)[:10]
            await self.cache_manager.set(cache_key, result, ttl=1800)
            return result
            
        except Exception as e:
            logger.error(f"Error suggesting optimal meeting time: {e}")
            return []
    
    async def optimize_room_utilization(self, rooms: List[str] = None, time_period_days: int = 7, 
                                      optimization_strategy: str = "balanced") -> Dict[str, Any]:
        try:
            logger.info(f"Optimizing room utilization for {time_period_days} days")
            
            cache_key = self._generate_cache_key("room_utilization", rooms, time_period_days, optimization_strategy)
            cached_result = await self.cache_manager.get(cache_key)
            if cached_result:
                logger.info("Returning cached room utilization analysis")
                return cached_result
            
            room_analytics = await self._get_room_usage_analytics(rooms, time_period_days)
            opportunities = await self._identify_utilization_opportunities(room_analytics, optimization_strategy)
            recommendations = await self._generate_utilization_recommendations(opportunities, room_analytics)
            
            result = {
                'current_utilization': room_analytics, 'opportunities': opportunities,
                'recommendations': recommendations, 
                'projected_improvements': self._calculate_projected_improvements(recommendations)
            }
            
            await self.cache_manager.set(cache_key, result, ttl=7200)
            return result
            
        except Exception as e:
            logger.error(f"Error optimizing room utilization: {e}")
            return {}
    
    async def schedule_recurring_meetings_optimally(self, meeting_template: Dict[str, Any],
                                                  recurrence_pattern: Dict[str, Any], 
                                                  duration_weeks: int = 12) -> List[Dict[str, Any]]:
        try:
            logger.info("Optimizing recurring meeting schedule")
            
            cache_key = self._generate_cache_key("recurring_meetings", meeting_template, recurrence_pattern, duration_weeks)
            cached_result = await self.cache_manager.get(cache_key)
            if cached_result: return cached_result
            
            meeting_instances = self._generate_recurring_instances(meeting_template, recurrence_pattern, duration_weeks)
            optimized_instances = []
            
            for instance in meeting_instances:
                optimized = await self._optimize_recurring_instance(instance, meeting_instances, optimized_instances)
                optimized_instances.append(optimized)
            
            await self.cache_manager.set(cache_key, optimized_instances, ttl=14400)
            return optimized_instances
            
        except Exception as e:
            logger.error(f"Error optimizing recurring meetings: {e}")
            return []
    
    # Utility and helper methods (condensed)
    def _generate_cache_key(self, prefix: str, *args) -> str:
        serialized_args = json.dumps(args, sort_keys=True, default=str)
        return f"{prefix}:{hashlib.md5(serialized_args.encode()).hexdigest()}"
    
    async def _load_user_patterns(self, user_id: int) -> Dict[str, Any]:
        cache_key = f"user_patterns:{user_id}"
        cached = await self.cache_manager.get(cache_key)
        if cached: return cached
        
        with next(get_db()) as db:
            bookings = db.query(MRBSEntry).filter(
                MRBSEntry.create_by == str(user_id),
                MRBSEntry.start_time >= datetime.now() - timedelta(days=90)
            ).all()
        
        patterns = self.pattern_analyzer.analyze_user_patterns(user_id, [booking.__dict__ for booking in bookings]) if self.pattern_analyzer else {}
        await self.cache_manager.set(cache_key, patterns, ttl=3600)
        return patterns
    
    async def _load_user_preferences(self, user_id: int) -> Dict[str, Any]:
        cache_key = f"user_preferences:{user_id}"
        cached = await self.cache_manager.get(cache_key)
        if cached: return cached
        
        preferences = self.preference_learner.learn_user_preferences(user_id) if self.preference_learner else {}
        await self.cache_manager.set(cache_key, preferences, ttl=7200)
        return preferences
    
    async def _get_schedule_context(self, user_id: int, time_horizon_hours: int) -> Dict[str, Any]:
        cache_key = f"schedule_context:{user_id}:{time_horizon_hours}"
        cached = await self.cache_manager.get(cache_key)
        if cached: return cached
        
        end_time = datetime.now() + timedelta(hours=time_horizon_hours)
        with next(get_db()) as db:
            upcoming_bookings = db.query(MRBSEntry).filter(
                MRBSEntry.create_by == str(user_id),
                MRBSEntry.start_time >= datetime.now(),
                MRBSEntry.start_time <= end_time
            ).all()
        
        context = {
            'upcoming_meetings': len(upcoming_bookings),
            'meeting_density': self._calculate_meeting_density(upcoming_bookings),
            'available_slots': self._identify_available_slots(upcoming_bookings, time_horizon_hours),
            'peak_hours': self._identify_peak_hours(upcoming_bookings)
        }
        
        await self.cache_manager.set(cache_key, context, ttl=900)
        return context
    
    def _find_common_available_slots(self, attendee_availability: Dict[int, List[Dict]], 
                                   duration_minutes: int, time_range: Tuple[datetime, datetime] = None) -> List[Dict[str, Any]]:
        if not attendee_availability: return []
        
        common_slots = []
        first_attendee = list(attendee_availability.keys())[0]
        base_slots = attendee_availability[first_attendee]
        
        for slot in base_slots:
            if all(self._is_time_available_for_user(slot['start_time'], slot['end_time'], availability) 
                   for availability in attendee_availability.values()):
                if (slot['end_time'] - slot['start_time']).total_seconds() >= duration_minutes * 60:
                    common_slots.append({
                        'start_time': slot['start_time'],
                        'end_time': slot['start_time'] + timedelta(minutes=duration_minutes)
                    })
        return common_slots
    
    async def _score_meeting_slot(self, slot: Dict[str, Any], attendees: List[int],
                                attendee_preferences: Dict[int, Dict], room_requirements: Dict[str, Any],
                                optimization_goals: List[OptimizationGoal]) -> float:
        total_score = 0.0
        
        scoring_methods = {
            OptimizationGoal.USER_SATISFACTION: lambda: sum(self._score_against_preferences(slot, attendee_preferences.get(aid, {})) 
                                                           for aid in attendees) / len(attendees),
            OptimizationGoal.MINIMIZE_CONFLICTS: lambda: asyncio.create_task(self._score_conflict_likelihood(slot, attendees)),
            OptimizationGoal.MAXIMIZE_EFFICIENCY: lambda: self._score_productivity_factors(slot),
            OptimizationGoal.REDUCE_TRAVEL_TIME: lambda: asyncio.create_task(self._score_travel_time(slot, attendees))
        }
        
        for goal in optimization_goals:
            if goal in scoring_methods:
                score_func = scoring_methods[goal]
                goal_score = await score_func() if asyncio.iscoroutine(score_func()) else score_func()
                weight = self.optimization_weights.get(goal, 0.1)
                total_score += goal_score * weight
        
        return min(max(total_score, 0.0), 1.0)
    
    def _score_against_preferences(self, slot: Dict[str, Any], preferences: Dict[str, Any]) -> float:
        score = 0.5
        slot_time = slot['start_time']
        
        # Check preferences
        if slot_time.hour in preferences.get('preferred_hours', []):
            score += 0.2
        if slot_time.weekday() in preferences.get('preferred_days', []):
            score += 0.1
        
        # Check blocked times
        for blocked in preferences.get('blocked_times', []):
            if self.time_utils.times_overlap(slot_time, slot['end_time'], blocked['start'], blocked['end']):
                score -= 0.3
        
        return max(0.0, min(1.0, score))
    
    def _calculate_meeting_density(self, bookings: List) -> Dict[str, float]:
        if not bookings: return {'daily': 0.0, 'hourly': 0.0}
        
        daily_counts, hourly_counts = {}, {}
        for booking in bookings:
            day_key, hour_key = booking.start_time.date(), booking.start_time.hour
            daily_counts[day_key] = daily_counts.get(day_key, 0) + 1
            hourly_counts[hour_key] = hourly_counts.get(hour_key, 0) + 1
        
        return {
            'daily': sum(daily_counts.values()) / len(daily_counts) if daily_counts else 0,
            'hourly': sum(hourly_counts.values()) / len(hourly_counts) if hourly_counts else 0
        }
    
    def _generate_slot_reasoning(self, slot: Dict[str, Any], score: float) -> List[str]:
        reasoning = []
        slot_time = slot['start_time']
        
        if 9 <= slot_time.hour <= 11:
            reasoning.append("Optimal morning focus time")
        elif 14 <= slot_time.hour <= 16:
            reasoning.append("Good afternoon collaboration time")
        
        if slot_time.weekday() < 5:
            reasoning.append("Scheduled during business days")
        
        if score > 0.8:
            reasoning.append("High compatibility with attendee preferences")
        elif score > 0.6:
            reasoning.append("Good compatibility with most attendees")
        
        return reasoning
    
    # Placeholder methods for missing functionality
    async def _get_user_availability(self, user_id: int, time_range: Tuple[datetime, datetime]) -> List[Dict]:
        return [{'start_time': datetime.now(), 'end_time': datetime.now() + timedelta(hours=1)}]
    
    def _is_time_available_for_user(self, start_time, end_time, availability) -> bool:
        return True
    
    async def _score_conflict_likelihood(self, slot, attendees) -> float:
        return 0.8
    
    def _score_productivity_factors(self, slot) -> float:
        hour = slot['start_time'].hour
        return 0.9 if 9 <= hour <= 11 or 14 <= hour <= 16 else 0.6
    
    async def _score_travel_time(self, slot, attendees) -> float:
        return 0.7
    
    def _identify_available_slots(self, bookings, time_horizon_hours) -> List:
        return []
    
    def _identify_peak_hours(self, bookings) -> List:
        return [9, 10, 11, 14, 15, 16]
    
    async def _get_room_usage_analytics(self, rooms, time_period_days) -> Dict:
        return {'utilization_rate': 0.6, 'peak_hours': [10, 14]}
    
    async def _identify_utilization_opportunities(self, analytics, strategy) -> List:
        return ['Redistribute peak hour bookings', 'Optimize room sizes']
    
    async def _generate_utilization_recommendations(self, opportunities, analytics) -> List:
        return [{'type': 'time_shift', 'description': 'Move 20% of 10am meetings to 11am'}]
    
    def _calculate_projected_improvements(self, recommendations) -> Dict:
        return {'utilization_increase': 0.15, 'conflict_reduction': 0.25}
    
    def _generate_recurring_instances(self, template, pattern, weeks) -> List:
        return [template for _ in range(weeks)]
    
    async def _optimize_recurring_instance(self, instance, all_instances, optimized) -> Dict:
        return instance
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        return await self.cache_manager.get_stats()
    
    async def clear_user_cache(self, user_id: int):
        patterns = [f"user_patterns:{user_id}", f"user_preferences:{user_id}", f"schedule_context:{user_id}"]
        for pattern in patterns:
            await self.cache_manager.delete(pattern)