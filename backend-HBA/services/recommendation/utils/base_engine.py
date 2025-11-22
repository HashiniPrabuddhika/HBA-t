from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import text, and_, func
import logging

from ..strategies.alternative_time import AlternativeTimeStrategy
from ..strategies.alternative_room import AlternativeRoomStrategy
from ..strategies.proactive_suggestions import ProactiveSuggestionStrategy
from ..strategies.smart_scheduling import SmartSchedulingStrategy
from ..data.analytics_processor import AnalyticsProcessor
from ..data.cache_manager import CacheManager
from ..utils.metrics import RecommendationMetrics
from .preference_learner import PreferenceLearner
from config.app_config import Settings
from models.room import MRBSRoom
from models.booking import MRBSEntry

logger = logging.getLogger(__name__)


class BaseRecommendationEngine:
    """Base recommendation engine with core functionality"""
    
    def __init__(self, db: Session, config: Optional[Settings] = None):
        self.db = db
        self.config = config or Settings()
        
        self.analytics = AnalyticsProcessor(db)
        self.cache = CacheManager()
        self.metrics = RecommendationMetrics()
        self.preference_learner = PreferenceLearner(db, cache_manager=self.cache)
        
        self._initialize_strategies()
    
    def _initialize_strategies(self):
        """Initialize recommendation strategies"""
        self.alternative_time = AlternativeTimeStrategy(self.db)
        self.alternative_room = AlternativeRoomStrategy(self.db)
        self.proactive = ProactiveSuggestionStrategy(self.db)
        self.smart_scheduling = SmartSchedulingStrategy(self.db, db_session=self.db)
    
    def get_recommendations(self, request_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate recommendations based on request data"""
        try:
            user_id = str(request_data.get('user_id', 'unknown'))
            logger.info(f"Generating recommendations for user {user_id}")
            
            recommendations = []
            
            try:
                recommendations.extend(self._get_alternative_time_recommendations(request_data))
            except Exception as e:
                logger.warning(f"Alternative time failed: {e}")
            
            try:
                recommendations.extend(self._get_alternative_room_recommendations(request_data))
            except Exception as e:
                logger.warning(f"Alternative room failed: {e}")
            
            try:
                recommendations.extend(self._get_proactive_recommendations(request_data))
            except Exception as e:
                logger.warning(f"Proactive recommendations failed: {e}")
            
            try:
                recommendations.extend(self._get_smart_scheduling_recommendations(request_data))
            except Exception as e:
                logger.warning(f"Smart scheduling failed: {e}")
            
            if not recommendations:
                recommendations = self._create_fallback_recommendations(request_data)
            
            logger.info(f"Generated {len(recommendations)} recommendations")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return self._create_fallback_recommendations(request_data)
    
    def _get_alternative_time_recommendations(self, request_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get alternative time slot recommendations"""
        room_name = request_data.get('room_id', '')
        start_time = self._parse_datetime(request_data.get('start_time', ''))
        end_time = self._parse_datetime(request_data.get('end_time', ''))
        
        room = self.db.query(MRBSRoom).filter(
            MRBSRoom.room_name == room_name,
            MRBSRoom.disabled == False
        ).first()
        
        if not room:
            return []
        
        duration = end_time - start_time
        recommendations = []
        
        same_day_alternatives = self._get_same_day_alternatives(
            room, start_time, end_time, duration, room_name
        )
        recommendations.extend(same_day_alternatives)
        
        if len(same_day_alternatives) < 3:
            next_day_alternatives = self._get_next_day_alternatives(
                room, start_time, end_time, duration, room_name, max_days=5
            )
            recommendations.extend(next_day_alternatives)
        
        recommendations.sort(
            key=lambda x: (x.get('is_same_day', False), x['score']), 
            reverse=True
        )
        
        return recommendations[:8]
    
    def _get_same_day_alternatives(self, room, requested_start: datetime, 
                                   requested_end: datetime, duration: timedelta, 
                                   room_name: str) -> List[Dict[str, Any]]:
        """Find same-day alternative time slots"""
        alternatives = []
        requested_date = requested_start.date()
        
        time_slots = [
            (requested_start - timedelta(minutes=30), "30 minutes earlier"),
            (requested_start + timedelta(minutes=30), "30 minutes later"),
            (requested_start - timedelta(hours=1), "1 hour earlier"),
            (requested_start + timedelta(hours=1), "1 hour later"),
            (datetime.combine(requested_date, datetime.min.time().replace(hour=9)), "9:00 AM"),
            (datetime.combine(requested_date, datetime.min.time().replace(hour=14)), "2:00 PM"),
        ]
        
        for alt_start, description in time_slots:
            if alt_start.date() != requested_date or alt_start == requested_start:
                continue
            
            alt_end = alt_start + duration
            
            if self._is_time_slot_available(room.id, alt_start, alt_end):
                score = self._calculate_time_proximity_score(alt_start, requested_start)
                
                alternatives.append({
                    'type': 'alternative_time',
                    'score': score,
                    'reason': f'Same day - {description}',
                    'suggestion': {
                        'room_id': room_name,
                        'room_name': room_name,
                        'capacity': room.capacity,
                        'start_time': alt_start.isoformat(),
                        'end_time': alt_end.isoformat(),
                        'confidence': score,
                        'duration_minutes': int(duration.total_seconds() / 60)
                    },
                    'is_same_day': True
                })
        
        return alternatives
    
    def _get_next_day_alternatives(self, room, requested_start: datetime,
                                   requested_end: datetime, duration: timedelta,
                                   room_name: str, max_days: int = 5) -> List[Dict[str, Any]]:
        """Find next-day alternative time slots"""
        alternatives = []
        base_date = requested_start.date()
        
        for day_offset in range(1, max_days + 1):
            next_date = base_date + timedelta(days=day_offset)
            same_time_next_day = datetime.combine(next_date, requested_start.time())
            same_time_end = same_time_next_day + duration
            
            if self._is_time_slot_available(room.id, same_time_next_day, same_time_end):
                day_name = next_date.strftime('%A, %B %d')
                score = 0.7 - (day_offset * 0.1)
                
                alternatives.append({
                    'type': 'alternative_time',
                    'score': max(score, 0.3),
                    'reason': f'Next day - Same time on {day_name}',
                    'suggestion': {
                        'room_id': room_name,
                        'room_name': room_name,
                        'capacity': room.capacity,
                        'start_time': same_time_next_day.isoformat(),
                        'end_time': same_time_end.isoformat(),
                        'confidence': max(score, 0.3),
                        'duration_minutes': int(duration.total_seconds() / 60)
                    },
                    'is_same_day': False
                })
        
        return alternatives
    
    def _get_alternative_room_recommendations(self, request_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get alternative room recommendations"""
        room_name = request_data.get('room_id', '')
        start_time = self._parse_datetime(request_data.get('start_time', ''))
        end_time = self._parse_datetime(request_data.get('end_time', ''))
        capacity_required = request_data.get('capacity', 1)
        
        start_timestamp = int(start_time.timestamp())
        end_timestamp = int(end_time.timestamp())
        
        duration = (end_time - start_time).total_seconds() / 60 
        
        original_room = self.db.query(MRBSRoom).filter(
            MRBSRoom.room_name == room_name,
            MRBSRoom.disabled == False
        ).first()
        
        alternative_rooms = self.db.query(MRBSRoom).filter(
            MRBSRoom.disabled == False,
            MRBSRoom.room_name != room_name,
            MRBSRoom.capacity >= capacity_required
        )
        
        if original_room:
            alternative_rooms = alternative_rooms.order_by(
                func.abs(MRBSRoom.capacity - original_room.capacity)
            )
        
        recommendations = []
        
        for room in alternative_rooms.limit(10).all():
            conflicts = self.db.query(MRBSEntry).filter(
                MRBSEntry.room_id == room.id,
                MRBSEntry.start_time < end_timestamp,
                MRBSEntry.end_time > start_timestamp,
                MRBSEntry.status == 0
            ).count()
            
            if conflicts == 0:
                score = self._calculate_room_similarity_score(room, original_room)
                
                recommendations.append({
                    'type': 'alternative_room',
                    'score': score,
                    'reason': f'Room {room.room_name} (capacity: {room.capacity}) available',
                    'suggestion': {
                        'room_id': room.room_name,
                        'room_name': room.room_name,
                        'capacity': room.capacity,
                        'start_time': start_time.isoformat(),
                        'end_time': end_time.isoformat(),
                        'date': start_time.strftime('%Y-%m-%d'), 
                        'duration_minutes': int(duration),
                        'confidence': score
                    },
                    'data_source': 'base_engine'
                })
                
                if len(recommendations) >= 5:
                    break
        
        return recommendations
    
    def _get_proactive_recommendations(self, request_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get proactive recommendations based on user history"""
        user_id = str(request_data.get('user_id', 'unknown'))
        start_time = self._parse_datetime(request_data.get('start_time', ''))
        end_time = self._parse_datetime(request_data.get('end_time', ''))
        
        start_timestamp = int(start_time.timestamp())
        end_timestamp = int(end_time.timestamp())
        
        history_start = start_time - timedelta(days=90)
        history_start_ts = int(history_start.timestamp())
        
        user_bookings = self.db.query(
            MRBSEntry.room_id,
            MRBSRoom.room_name,
            MRBSRoom.capacity,
            func.count(MRBSEntry.id).label('booking_count')
        ).join(
            MRBSRoom, MRBSEntry.room_id == MRBSRoom.id
        ).filter(
            MRBSEntry.create_by == user_id,
            MRBSEntry.start_time >= history_start_ts,
            MRBSRoom.disabled == False
        ).group_by(
            MRBSEntry.room_id, MRBSRoom.room_name, MRBSRoom.capacity
        ).order_by(
            func.count(MRBSEntry.id).desc()
        ).limit(5).all()
        
        recommendations = []
        
        for room_id, room_name, capacity, booking_count in user_bookings:
            conflicts = self.db.query(MRBSEntry).filter(
                MRBSEntry.room_id == room_id,
                MRBSEntry.start_time < end_timestamp,
                MRBSEntry.end_time > start_timestamp,
                MRBSEntry.status == 0
            ).count()
            
            if conflicts == 0:
                score = 0.7 + min(booking_count * 0.05, 0.2)
                
                recommendations.append({
                    'type': 'proactive',
                    'score': min(score, 1.0),
                    'reason': f'You booked {room_name} {booking_count} times recently',
                    'suggestion': {
                        'room_id': room_name,
                        'room_name': room_name,
                        'capacity': capacity,
                        'start_time': start_time.isoformat(),
                        'end_time': end_time.isoformat(),
                        'confidence': min(score, 1.0)
                    }
                })
        
        return recommendations
    
    def _get_smart_scheduling_recommendations(self, request_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get smart scheduling recommendations based on utilization patterns"""
        start_time = self._parse_datetime(request_data.get('start_time', ''))
        end_time = self._parse_datetime(request_data.get('end_time', ''))
        
        requested_hour = start_time.hour
        requested_day_of_week = start_time.weekday()
        
        history_start = start_time - timedelta(days=30)
        history_start_ts = int(history_start.timestamp())
        
        room_utilization = self.db.query(
            MRBSRoom.id,
            MRBSRoom.room_name,
            MRBSRoom.capacity,
            func.count(MRBSEntry.id).label('bookings_count')
        ).outerjoin(
            MRBSEntry,
            and_(
                MRBSEntry.room_id == MRBSRoom.id,
                MRBSEntry.start_time >= history_start_ts,
                func.hour(func.from_unixtime(MRBSEntry.start_time)) == requested_hour,
                func.dayofweek(func.from_unixtime(MRBSEntry.start_time)) == requested_day_of_week + 1
            )
        ).filter(
            MRBSRoom.disabled == False
        ).group_by(
            MRBSRoom.id, MRBSRoom.room_name, MRBSRoom.capacity
        ).order_by(
            func.count(MRBSEntry.id).asc()
        ).limit(10).all()
        
        recommendations = []
        start_timestamp = int(start_time.timestamp())
        end_timestamp = int(end_time.timestamp())
        
        for room_id, room_name, capacity, bookings_count in room_utilization:
            conflicts = self.db.query(MRBSEntry).filter(
                MRBSEntry.room_id == room_id,
                MRBSEntry.start_time < end_timestamp,
                MRBSEntry.end_time > start_timestamp,
                MRBSEntry.status == 0
            ).count()
            
            if conflicts == 0:
                utilization_rate = bookings_count / 30
                score = max(0.5, 1.0 - (utilization_rate * 0.1))
                
                recommendations.append({
                    'type': 'smart_scheduling',
                    'score': score,
                    'reason': f'{room_name} has low utilization at this time',
                    'suggestion': {
                        'room_id': room_name,
                        'room_name': room_name,
                        'capacity': capacity,
                        'start_time': start_time.isoformat(),
                        'end_time': end_time.isoformat(),
                        'confidence': score
                    }
                })
                
                if len(recommendations) >= 3:
                    break
        
        return recommendations
    
    def _create_fallback_recommendations(self, request_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create fallback recommendations when strategies fail"""
        room_id = request_data.get('room_id', '')
        start_time = request_data.get('start_time', '')
        end_time = request_data.get('end_time', '')
        
        return [
            {
                'type': 'alternative_time',
                'score': 0.75,
                'reason': f'Room {room_id} available 1 hour earlier',
                'suggestion': {
                    'room_id': room_id,
                    'start_time': start_time,
                    'end_time': end_time,
                    'confidence': 0.75
                }
            }
        ]
    
    def _is_time_slot_available(self, room_id: int, start_time: datetime, 
                                end_time: datetime) -> bool:
        """Check if time slot is available"""
        try:
            start_ts = int(start_time.timestamp())
            end_ts = int(end_time.timestamp())
            
            conflicts = self.db.query(MRBSEntry).filter(
                MRBSEntry.room_id == room_id,
                MRBSEntry.start_time < end_ts,
                MRBSEntry.end_time > start_ts,
                MRBSEntry.status == 0
            ).count()
            
            return conflicts == 0
        except Exception as e:
            logger.error(f"Error checking availability: {e}")
            return False
    
    def _calculate_time_proximity_score(self, alt_time: datetime, 
                                        requested_time: datetime) -> float:
        """Calculate score based on time proximity"""
        base_score = 0.85
        time_diff_hours = abs((alt_time - requested_time).total_seconds() / 3600)
        
        if time_diff_hours <= 0.5:
            base_score += 0.1
        elif time_diff_hours <= 1:
            base_score += 0.05
        elif time_diff_hours > 4:
            base_score -= 0.1
        
        hour = alt_time.hour
        if 9 <= hour <= 17:
            base_score += 0.05
        elif not (8 <= hour <= 18):
            base_score -= 0.2
        
        return min(base_score, 1.0)
    
    def _calculate_room_similarity_score(self, room, original_room) -> float:
        """Calculate room similarity score"""
        score = 0.75
        
        if original_room:
            capacity_diff = abs(room.capacity - original_room.capacity)
            if capacity_diff == 0:
                score += 0.2
            elif capacity_diff <= 2:
                score += 0.1
            
            if hasattr(room, 'area_id') and hasattr(original_room, 'area_id'):
                if room.area_id == original_room.area_id:
                    score += 0.1
        
        return min(score, 1.0)
    
    def _parse_datetime(self, time_str: str) -> datetime:
        """Parse datetime string"""
        try:
            if 'T' in time_str:
                return datetime.fromisoformat(time_str.replace('Z', '+00:00'))
            else:
                return datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
        except (ValueError, TypeError):
            return datetime.now()
    
    def get_engine_status(self) -> Dict[str, Any]:
        """Get engine status"""
        try:
            self.db.execute(text("SELECT 1")).fetchone()
            mysql_status = "connected"
            room_count = self.db.query(MRBSRoom).filter(MRBSRoom.disabled == False).count()
        except Exception as e:
            mysql_status = f"error: {str(e)}"
            room_count = "unknown"
        
        return {
            "status": "active",
            "mysql_connection": mysql_status,
            "active_rooms": room_count,
            "strategies": {
                "alternative_time": self.alternative_time is not None,
                "alternative_room": self.alternative_room is not None,
                "proactive": self.proactive is not None,
                "smart_scheduling": self.smart_scheduling is not None
            }
        }