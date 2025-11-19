from typing import List, Dict, Any
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from utils.logger import get_logger
from config.app_config import settings
from models.booking import MRBSEntry
from models.room import MRBSRoom

logger = get_logger(__name__)

class BaseRecommendationEngine:
    """Base recommendation engine using rule-based logic"""
    
    def __init__(self, db: Session = None):
        self.db = db
        self.max_recommendations = settings.MAX_ALTERNATIVES
        logger.info("BaseRecommendationEngine initialized")
    
    def get_recommendations(self, request_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate base recommendations"""
        try:
            user_id = str(request_data.get('user_id', 'unknown'))
            room_name = request_data.get('room_id', '')
            start_time = request_data.get('start_time', '')
            end_time = request_data.get('end_time', '')
            
            logger.info(f"Generating base recommendations for user {user_id}")
            
            recommendations = []
            
            alt_time_recs = self._get_alternative_time_recommendations(
                room_name, start_time, end_time
            )
            recommendations.extend(alt_time_recs)
            
            alt_room_recs = self._get_alternative_room_recommendations(
                room_name, start_time, end_time
            )
            recommendations.extend(alt_room_recs)
            
            if not recommendations:
                recommendations = self._create_fallback_recommendations(request_data)
            
            recommendations = recommendations[:self.max_recommendations]
            logger.info(f"Generated {len(recommendations)} base recommendations")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating base recommendations: {e}")
            return self._create_fallback_recommendations(request_data)
    
    def _get_alternative_time_recommendations(
        self, room_name: str, start_time: str, end_time: str
    ) -> List[Dict[str, Any]]:
        """Find alternative time slots for the same room"""
        if not self.db:
            return []
        
        try:
            room = self.db.query(MRBSRoom).filter(
                MRBSRoom.room_name == room_name,
                MRBSRoom.disabled == False
            ).first()
            
            if not room:
                return []
            
            start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
            duration = end_dt - start_dt
            
            date_obj = start_dt.date()
            business_start = datetime.combine(date_obj, datetime.min.time().replace(
                hour=settings.BUSINESS_START_HOUR
            ))
            business_end = datetime.combine(date_obj, datetime.min.time().replace(
                hour=settings.BUSINESS_END_HOUR
            ))
            
            existing_bookings = self._get_day_bookings(room.id, start_dt)
            
            alternatives = []
            time_shifts = [-120, -60, -30, 30, 60, 120]
            
            for shift_minutes in time_shifts:
                new_start = start_dt + timedelta(minutes=shift_minutes)
                new_end = new_start + duration
                
                if new_start < business_start or new_end > business_end:
                    continue
                
                if self._is_time_available(room.id, new_start, new_end, existing_bookings):
                    score = 0.9 - abs(shift_minutes) / 120 * 0.3
                    alternatives.append({
                        'type': 'alternative_time',
                        'score': score,
                        'reason': f'{abs(shift_minutes)} minutes {"earlier" if shift_minutes < 0 else "later"}',
                        'suggestion': {
                            'room_name': room_name,
                            'start_time': new_start.isoformat(),
                            'end_time': new_end.isoformat(),
                            'confidence': score
                        },
                        'data_source': 'base_engine'
                    })
            
            return alternatives
            
        except Exception as e:
            logger.error(f"Error getting alternative time recommendations: {e}")
            return []
    
    def _get_alternative_room_recommendations(
        self, room_name: str, start_time: str, end_time: str
    ) -> List[Dict[str, Any]]:
        """Find alternative rooms for the same time"""
        if not self.db:
            return []
        
        try:
            target_room = self.db.query(MRBSRoom).filter(
                MRBSRoom.room_name == room_name
            ).first()
            
            if not target_room:
                return []
            
            start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
            
            start_ts = int(start_dt.timestamp())
            end_ts = int(end_dt.timestamp())
            
            alternative_rooms = self.db.query(MRBSRoom).filter(
                MRBSRoom.disabled == False,
                MRBSRoom.room_name != room_name,
                MRBSRoom.capacity >= target_room.capacity
            ).order_by(
                MRBSRoom.capacity
            ).limit(10).all()
            
            alternatives = []
            for room in alternative_rooms:
                conflicts = self.db.query(MRBSEntry).filter(
                    MRBSEntry.room_id == room.id,
                    MRBSEntry.start_time < end_ts,
                    MRBSEntry.end_time > start_ts,
                    MRBSEntry.status == 0
                ).count()
                
                if conflicts == 0:
                    capacity_diff = abs(room.capacity - target_room.capacity)
                    score = 0.85 if capacity_diff == 0 else 0.75 if capacity_diff <= 2 else 0.65
                    
                    alternatives.append({
                        'type': 'alternative_room',
                        'score': score,
                        'reason': f'Similar capacity room available',
                        'suggestion': {
                            'room_name': room.room_name,
                            'start_time': start_time,
                            'end_time': end_time,
                            'capacity': room.capacity,
                            'confidence': score
                        },
                        'data_source': 'base_engine'
                    })
                    
                    if len(alternatives) >= 5:
                        break
            
            return alternatives
            
        except Exception as e:
            logger.error(f"Error getting alternative room recommendations: {e}")
            return []
    
    def _get_day_bookings(self, room_id: int, date_obj: datetime) -> List[MRBSEntry]:
        """Get all bookings for a room on a specific day"""
        try:
            day_start = int(datetime.combine(date_obj.date(), datetime.min.time()).timestamp())
            day_end = int(datetime.combine(date_obj.date(), datetime.max.time()).timestamp())
            
            bookings = self.db.query(MRBSEntry).filter(
                MRBSEntry.room_id == room_id,
                MRBSEntry.start_time >= day_start,
                MRBSEntry.end_time <= day_end
            ).order_by(MRBSEntry.start_time).all()
            
            return bookings
        except Exception as e:
            logger.error(f"Error fetching day bookings: {e}")
            return []
    
    def _is_time_available(
        self, room_id: int, start_time: datetime, end_time: datetime, 
        existing_bookings: List[MRBSEntry]
    ) -> bool:
        """Check if time slot is available"""
        start_ts = int(start_time.timestamp())
        end_ts = int(end_time.timestamp())
        
        for booking in existing_bookings:
            if booking.start_time < end_ts and booking.end_time > start_ts:
                return False
        return True
    
    def _create_fallback_recommendations(self, request_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create fallback recommendations when no data available"""
        room_name = request_data.get('room_id', 'Room')
        start_time = request_data.get('start_time', '')
        end_time = request_data.get('end_time', '')
        
        return [
            {
                'type': 'alternative_time',
                'score': 0.7,
                'reason': 'Recommended time slot',
                'suggestion': {
                    'room_name': room_name,
                    'start_time': start_time,
                    'end_time': end_time,
                    'confidence': 0.7
                },
                'data_source': 'fallback'
            }
        ]
    
    def get_engine_status(self) -> Dict[str, Any]:
        """Get engine status"""
        return {
            'engine_type': 'base',
            'status': 'active',
            'database_connected': self.db is not None,
            'max_recommendations': self.max_recommendations
        }