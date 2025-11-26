from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

from ..models.embedding_model import EmbeddingModel
from ..data.analytics_processor import AnalyticsProcessor
from models.booking import MRBSEntry
from models.room import MRBSRoom
from ..utils.time_utils import TimeUtils

logger = logging.getLogger(__name__)

class AlternativeTimeStrategy:
    def __init__(self, db: Session):
        self.db = db
        self.analytics = AnalyticsProcessor(db)
        self.embedding_model = EmbeddingModel()
        self.time_utils = TimeUtils()
        self.business_hours = {'start': 7, 'end': 21, 'slot_duration': 30}
        logger.debug("Initialized AlternativeTimeStrategy")
    
    async def generate_alternatives(self, room_name: str, date: str, requested_start: str, requested_end: str, user_patterns: Dict[str, Any]) -> List[Dict[str, Any]]:
        logger.info(f"Generating time alternatives for {room_name} on {date}")
        
        try:
            validation = self._validate_inputs(room_name, date, requested_start, requested_end)
            if not validation['valid']:
                logger.warning(f"Invalid inputs: {validation['error']}")
                return []
            
            room = self.db.query(MRBSRoom).filter(MRBSRoom.room_name == room_name).first()
            if not room:
                logger.warning(f"Room not found: {room_name}")
                return []
            
            date_obj = datetime.strptime(date, "%Y-%m-%d")
            requested_start_dt = datetime.strptime(f"{date} {requested_start}", "%Y-%m-%d %H:%M")
            requested_end_dt = datetime.strptime(f"{date} {requested_end}", "%Y-%m-%d %H:%M")
            duration = requested_end_dt - requested_start_dt
            
            existing_bookings = self._get_day_bookings(room.id, date_obj)
            
            alternatives = []
            alternatives.extend(self._find_gap_alternatives(existing_bookings, duration, date_obj, user_patterns))
            alternatives.extend(self._find_pattern_based_alternatives(room_name, date, duration, user_patterns, existing_bookings))
            alternatives.extend(self._find_shifted_time_alternatives(requested_start_dt, requested_end_dt, existing_bookings, user_patterns))
            
            if len(alternatives) < 5:
                alternatives.extend(await self._find_adjacent_day_alternatives(room_name, date, requested_start, requested_end, user_patterns))
            
            alternatives = self._deduplicate_alternatives(alternatives)
            alternatives.sort(key=lambda x: x['confidence_score'], reverse=True)
            
            logger.info(f"Generated {len(alternatives)} alternative time slots")
            return alternatives[:10]
            
        except Exception as e:
            logger.error(f"Error generating time alternatives: {e}")
            return []
    
    def _validate_inputs(self, room_name: str, date: str, start_time: str, end_time: str) -> Dict[str, Any]:
        try:
            if not room_name: return {'valid': False, 'error': 'Room name is required'}
            
            datetime.strptime(date, "%Y-%m-%d")
            start_dt = datetime.strptime(start_time, "%H:%M")
            end_dt = datetime.strptime(end_time, "%H:%M")
            
            if end_dt <= start_dt: return {'valid': False, 'error': 'End time must be after start time'}
            return {'valid': True}
        except ValueError as e:
            return {'valid': False, 'error': f'Invalid date/time format: {e}'}
    
    def _get_day_bookings(self, room_id: int, date_obj: datetime) -> List[MRBSEntry]:
        try:
            day_start = int(datetime.combine(date_obj, datetime.min.time()).timestamp())
            day_end = int(datetime.combine(date_obj, datetime.max.time()).timestamp())
            
            bookings = self.db.query(MRBSEntry).filter(
                and_(MRBSEntry.room_id == room_id, MRBSEntry.start_time >= day_start, MRBSEntry.end_time <= day_end)
            ).order_by(MRBSEntry.start_time).all()
            
            logger.debug(f"Found {len(bookings)} existing bookings for room {room_id} on {date_obj.date()}")
            return bookings
        except Exception as e:
            logger.error(f"Error fetching day bookings: {e}")
            return []
    
    def _find_gap_alternatives(self, bookings: List[MRBSEntry], duration: timedelta, date_obj: datetime, user_patterns: Dict[str, Any]) -> List[Dict[str, Any]]:
        alternatives = []
        
        try:
            business_start = datetime.combine(date_obj, datetime.min.time().replace(hour=self.business_hours['start']))
            business_end = datetime.combine(date_obj, datetime.min.time().replace(hour=self.business_hours['end']))
            
            if not bookings:
                current = business_start
                while current + duration <= business_end:
                    confidence = self._calculate_time_confidence(current.strftime("%H:%M"), user_patterns)
                    alternatives.append({
                        'start_time': current.strftime("%H:%M"),
                        'end_time': (current + duration).strftime("%H:%M"),
                        'date': date_obj.strftime("%Y-%m-%d"),
                        'confidence_score': confidence,
                        'reason': 'Available slot - no conflicts',
                        'strategy': 'gap_analysis',
                        'day_type': 'same_day'
                    })
                    current += timedelta(minutes=self.business_hours['slot_duration'])
            else:
                gaps = self._identify_time_gaps(bookings, business_start, business_end)
                for gap_start, gap_end in gaps:
                    current = gap_start
                    while current + duration <= gap_end:
                        confidence = self._calculate_time_confidence(current.strftime("%H:%M"), user_patterns)
                        alternatives.append({
                            'start_time': current.strftime("%H:%M"),
                            'end_time': (current + duration).strftime("%H:%M"),
                            'date': date_obj.strftime("%Y-%m-%d"),
                            'confidence_score': confidence,
                            'reason': 'Available between existing bookings',
                            'strategy': 'gap_analysis',
                            'day_type': 'same_day'
                        })
                        current += timedelta(minutes=self.business_hours['slot_duration'])
            
            logger.debug(f"Found {len(alternatives)} gap-based alternatives")
        except Exception as e:
            logger.error(f"Error finding gap alternatives: {e}")
        
        return alternatives
    
    def _identify_time_gaps(self, bookings: List[MRBSEntry], business_start: datetime, business_end: datetime) -> List[Tuple[datetime, datetime]]:
        if not bookings: return [(business_start, business_end)]
        
        gaps = []
        sorted_bookings = sorted(bookings, key=lambda b: b.start_time)
        
        first_start = datetime.fromtimestamp(sorted_bookings[0].start_time)
        if business_start < first_start: gaps.append((business_start, first_start))
        
        for i in range(len(sorted_bookings) - 1):
            current_end = datetime.fromtimestamp(sorted_bookings[i].end_time)
            next_start = datetime.fromtimestamp(sorted_bookings[i + 1].start_time)
            if current_end < next_start: gaps.append((current_end, next_start))
        
        last_end = datetime.fromtimestamp(sorted_bookings[-1].end_time)
        if last_end < business_end: gaps.append((last_end, business_end))
        
        return gaps
    
    def _find_pattern_based_alternatives(self, room_name: str, date: str, duration: timedelta, user_patterns: Dict[str, Any], existing_bookings: List[MRBSEntry]) -> List[Dict[str, Any]]:
        alternatives = []
        
        try:
            preferred_times = user_patterns.get('preferred_times', [])
            
            try:
                room_popular_times = self.analytics.get_room_popular_times(room_name)
            except Exception as e:
                logger.warning(f"Could not get room popular times: {e}")
                room_popular_times = []
            
            suggested_times = [{'time': t, 'weight': 0.8, 'reason': 'Based on your booking history'} for t in preferred_times]
            suggested_times.extend([{'time': t, 'weight': 0.6, 'reason': 'Popular time for this room'} for t in room_popular_times if t not in [s['time'] for s in suggested_times]])
            
            for suggestion in suggested_times:
                try:
                    start_time = datetime.strptime(f"{date} {suggestion['time']}", "%Y-%m-%d %H:%M")
                    end_time = start_time + duration
                    
                    if not self._check_time_conflicts(start_time, end_time, existing_bookings):
                        alternatives.append({
                            'start_time': start_time.strftime("%H:%M"),
                            'end_time': end_time.strftime("%H:%M"),
                            'date': date,
                            'confidence_score': suggestion['weight'],
                            'reason': suggestion['reason'],
                            'strategy': 'pattern_based',
                            'day_type': 'same_day'
                        })
                except (ValueError, AttributeError):
                    logger.warning(f"Invalid time format in patterns: {suggestion['time']}")
                    continue
            
            logger.debug(f"Found {len(alternatives)} pattern-based alternatives")
        except Exception as e:
            logger.error(f"Error finding pattern-based alternatives: {e}")
        
        return alternatives
    
    def _find_shifted_time_alternatives(self, requested_start: datetime, requested_end: datetime, existing_bookings: List[MRBSEntry], user_patterns: Dict[str, Any]) -> List[Dict[str, Any]]:
        alternatives = []
        duration = requested_end - requested_start
        
        try:
            for shift_minutes in [-120, -60, -30, 30, 60, 120]:
                new_start = requested_start + timedelta(minutes=shift_minutes)
                new_end = new_start + duration
                
                if (new_start.hour < self.business_hours['start'] or new_end.hour >= self.business_hours['end']): continue
                if self._check_time_conflicts(new_start, new_end, existing_bookings): continue
                
                confidence = self._calculate_shifted_time_confidence(shift_minutes, user_patterns)
                shift_direction = "earlier" if shift_minutes < 0 else "later"
                shift_amount = abs(shift_minutes)
                
                alternatives.append({
                    'start_time': new_start.strftime("%H:%M"),
                    'end_time': new_end.strftime("%H:%M"),
                    'date': new_start.strftime("%Y-%m-%d"),
                    'confidence_score': confidence,
                    'reason': f'{shift_amount} minutes {shift_direction} than requested',
                    'strategy': 'time_shift',
                    'day_type': 'same_day',
                    'shift_minutes': shift_minutes
                })
            
            logger.debug(f"Found {len(alternatives)} shifted time alternatives")
        except Exception as e:
            logger.error(f"Error finding shifted time alternatives: {e}")
        
        return alternatives
    
    def _check_time_conflicts(self, start_time: datetime, end_time: datetime, existing_bookings: List[MRBSEntry]) -> bool:
        start_ts, end_ts = int(start_time.timestamp()), int(end_time.timestamp())
        return any(booking.start_time < end_ts and booking.end_time > start_ts for booking in existing_bookings)
    
    async def _find_adjacent_day_alternatives(self, room_name: str, date: str, requested_start: str, requested_end: str, user_patterns: Dict[str, Any]) -> List[Dict[str, Any]]:
        alternatives = []
        
        try:
            date_obj = datetime.strptime(date, "%Y-%m-%d")
            days_to_check = [-1, 1, 2, 3] if not user_patterns.get('prefers_weekdays', False) else [-1, 1]
            
            for day_offset in days_to_check:
                alt_date = date_obj + timedelta(days=day_offset)
                alt_date_str = alt_date.strftime("%Y-%m-%d")
                
                if (user_patterns.get('prefers_weekdays', False) and alt_date.weekday() >= 5) or alt_date < datetime.now().date():
                    continue
                
                availability = await self._check_time_availability(room_name, alt_date_str, requested_start, requested_end)
                
                if availability['available']:
                    confidence = self._calculate_adjacent_day_confidence(day_offset, alt_date, user_patterns)
                    day_name = alt_date.strftime("%A")
                    day_description = "tomorrow" if day_offset == 1 else "yesterday" if day_offset == -1 else f"in {day_offset} days"
                    
                    alternatives.append({
                        'start_time': requested_start,
                        'end_time': requested_end,
                        'date': alt_date_str,
                        'confidence_score': confidence,
                        'reason': f'Same time on {day_name} ({day_description})',
                        'strategy': 'adjacent_day',
                        'day_type': 'different_day',
                        'day_offset': day_offset
                    })
            
            logger.debug(f"Found {len(alternatives)} adjacent day alternatives")
        except Exception as e:
            logger.error(f"Error finding adjacent day alternatives: {e}")
        
        return alternatives
    
    def _calculate_time_confidence(self, time_str: str, user_patterns: Dict[str, Any]) -> float:
        try:
            base_confidence = 0.5
            preferred_times = user_patterns.get('preferred_times', [])
            time_minutes = self.time_utils.time_to_minutes(time_str)
            
            for pref_time in preferred_times:
                pref_minutes = self.time_utils.time_to_minutes(pref_time)
                if abs(time_minutes - pref_minutes) <= 30:
                    base_confidence += 0.3
                    break
            
            if 540 <= time_minutes <= 1080: base_confidence += 0.2
            elif 480 <= time_minutes <= 1140: base_confidence += 0.1
            
            if time_minutes < 420 or time_minutes > 1260: base_confidence -= 0.3
            elif time_minutes < 480 or time_minutes > 1200: base_confidence -= 0.1
            
            common_times = [600, 660, 720, 780, 840, 900, 960, 1020]
            if any(abs(time_minutes - ct) <= 15 for ct in common_times): base_confidence += 0.1
            
            return round(min(1.0, max(0.1, base_confidence)), 2)
        except Exception as e:
            logger.error(f"Error calculating time confidence: {e}")
            return 0.5
    
    def _calculate_shifted_time_confidence(self, shift_minutes: int, user_patterns: Dict[str, Any]) -> float:
        base_confidence = 0.6 - (abs(shift_minutes) / 120.0) * 0.3
        if shift_minutes > 0: base_confidence += 0.05
        return round(min(1.0, max(0.1, base_confidence)), 2)
    
    def _calculate_adjacent_day_confidence(self, day_offset: int, alt_date: datetime, user_patterns: Dict[str, Any]) -> float:
        base_confidence = {1: 0.6, -1: 0.6, 2: 0.4, -2: 0.4}.get(abs(day_offset), 0.3)
        if day_offset > 0: base_confidence += 0.05
        
        user_weekday = user_patterns.get('weekday_distribution', {})
        day_name = alt_date.strftime('%A').lower()
        if day_name in user_weekday:
            base_confidence += user_weekday[day_name] * 0.2
        
        return round(min(1.0, max(0.1, base_confidence)), 2)
    
    async def _check_time_availability(self, room_name: str, date: str, start_time: str, end_time: str) -> Dict[str, Any]:
        try:
            room = self.db.query(MRBSRoom).filter(MRBSRoom.room_name == room_name).first()
            if not room: return {'available': False, 'reason': 'Room not found'}
            
            start_dt = datetime.strptime(f"{date} {start_time}", "%Y-%m-%d %H:%M")
            end_dt = datetime.strptime(f"{date} {end_time}", "%Y-%m-%d %H:%M")
            start_ts, end_ts = int(start_dt.timestamp()), int(end_dt.timestamp())
            
            conflict = self.db.query(MRBSEntry).filter(
                and_(MRBSEntry.room_id == room.id, MRBSEntry.start_time < end_ts, MRBSEntry.end_time > start_ts)
            ).first()
            
            return {'available': conflict is None, 'reason': 'Time slot is available' if conflict is None else 'Time slot is booked'}
        except Exception as e:
            logger.error(f"Error checking time availability: {e}")
            return {'available': False, 'reason': f'Error checking availability: {str(e)}'}
    
    def _deduplicate_alternatives(self, alternatives: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        deduplicated = [alt for alt in alternatives if (alt['date'], alt['start_time'], alt['end_time']) not in seen and not seen.add((alt['date'], alt['start_time'], alt['end_time']))]
        logger.debug(f"Deduplicated {len(alternatives)} alternatives to {len(deduplicated)}")
        return deduplicated