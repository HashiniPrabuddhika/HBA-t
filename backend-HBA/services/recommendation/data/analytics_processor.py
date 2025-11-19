from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict, Counter
import logging
from models.room import MRBSRoom
from models.booking import MRBSEntry

logger = logging.getLogger(__name__)


class AnalyticsProcessor:
    def __init__(self, db: Session):
        self.db = db
        self._cache = {}  
        self._cache_ttl = {}
    
    async def get_user_booking_patterns(self, user_id: str) -> Dict[str, Any]:
        cache_key = f"user_patterns_{user_id}"
        if self._is_cached_valid(cache_key):
            return self._cache[cache_key]
        
        bookings = self.db.query(MRBSEntry).filter(MRBSEntry.create_by == user_id).order_by(MRBSEntry.start_time.desc()).limit(100).all()
        
        if not bookings:
            return self._cache_result(cache_key, {
                'total_bookings': 0, 'preferred_rooms': [], 'preferred_times': [],
                'booking_patterns': {}, 'seasonal_patterns': {}, 'recurring_patterns': []
            })
        
        patterns = {
            'total_bookings': len(bookings),
            'preferred_rooms': self._analyze_room_preferences(bookings),
            'preferred_times': self._analyze_time_preferences(bookings),
            'booking_patterns': self._analyze_booking_patterns(bookings),
            'seasonal_patterns': self._analyze_seasonal_patterns(bookings),
            'recurring_patterns': self._analyze_recurring_patterns(bookings),
            'recent_bookings': self._get_recent_bookings(bookings),
            'typically_available_times': self._infer_available_times(bookings),
            'prefers_weekdays': self._analyze_weekday_preference(bookings)
        }
        
        return self._cache_result(cache_key, patterns)
    
    def _analyze_room_preferences(self, bookings: List[MRBSEntry]) -> List[str]:
        room_counts = defaultdict(int)
        for booking in bookings:
            room = self.db.query(MRBSRoom).filter(MRBSRoom.id == booking.room_id).first()
            if room:
                room_counts[room.room_name] += 1
        return [r[0] for r in sorted(room_counts.items(), key=lambda x: x[1], reverse=True)[:5]]
    
    def _analyze_time_preferences(self, bookings: List[MRBSEntry]) -> List[str]:
        time_counts = defaultdict(int)
        for booking in bookings:
            time_counts[datetime.fromtimestamp(booking.start_time).strftime("%H:%M")] += 1
        return [t[0] for t in sorted(time_counts.items(), key=lambda x: x[1], reverse=True)[:5]]
    
    def _analyze_booking_patterns(self, bookings: List[MRBSEntry]) -> Dict[str, Any]:
        durations, weekdays, advance_bookings = [], [], []
        
        for booking in bookings:
            durations.append((booking.end_time - booking.start_time) / 3600)
            weekdays.append(datetime.fromtimestamp(booking.start_time).weekday())
            if hasattr(booking, 'timestamp') and booking.timestamp:
                advance_bookings.append((datetime.fromtimestamp(booking.start_time) - booking.timestamp).days)
        
        return {
            'avg_duration': np.mean(durations) if durations else 0,
            'preferred_weekdays': Counter(weekdays).most_common(3),
            'avg_advance_booking': np.mean(advance_bookings) if advance_bookings else 0,
            'frequency': self._calculate_booking_frequency(bookings)
        }
    
    def _analyze_seasonal_patterns(self, bookings: List[MRBSEntry]) -> Dict[str, Any]:
        monthly_patterns = defaultdict(lambda: {'count': 0, 'preferred_rooms': [], 'preferred_times': []})
        
        for booking in bookings:
            dt = datetime.fromtimestamp(booking.start_time)
            monthly_patterns[dt.month]['count'] += 1
            
            room = self.db.query(MRBSRoom).filter(MRBSRoom.id == booking.room_id).first()
            if room:
                monthly_patterns[dt.month]['preferred_rooms'].append(room.room_name)
            
            monthly_patterns[dt.month]['preferred_times'].append({
                'start_time': dt.strftime("%H:%M"),
                'end_time': datetime.fromtimestamp(booking.end_time).strftime("%H:%M")
            })
        
        for month, data in monthly_patterns.items():
            data['preferred_rooms'] = [item[0] for item in Counter(data['preferred_rooms']).most_common(3)]
            data['preferred_times'] = data['preferred_times'][:3]
        
        return dict(monthly_patterns)
    
    def _analyze_recurring_patterns(self, bookings: List[MRBSEntry]) -> List[Dict[str, Any]]:
        grouped_bookings = defaultdict(list)
        
        for booking in bookings:
            room = self.db.query(MRBSRoom).filter(MRBSRoom.id == booking.room_id).first()
            if not room: continue
            
            start, end = datetime.fromtimestamp(booking.start_time), datetime.fromtimestamp(booking.end_time)
            key = (room.room_name, start.strftime("%H:%M"), end.strftime("%H:%M"))
            grouped_bookings[key].append(start)
        
        recurring_patterns = []
        for (room_name, start_time, end_time), dates in grouped_bookings.items():
            if len(dates) >= 3:
                dates.sort()
                weekly_intervals = [1 for i in range(1, len(dates)) if 6 <= (dates[i] - dates[i-1]).days <= 8]
                
                if len(weekly_intervals) >= 2:
                    recurring_patterns.append({
                        'room_name': room_name, 'start_time': start_time, 'end_time': end_time,
                        'frequency': 'weekly', 'occurrences': len(dates),
                        'consistency_score': len(weekly_intervals) / (len(dates) - 1),
                        'last_booking_date': dates[-1].strftime("%Y-%m-%d")
                    })
        
        return recurring_patterns
    
    def _get_recent_bookings(self, bookings: List[MRBSEntry]) -> List[Dict[str, Any]]:
        recent = []
        for booking in bookings[:10]:
            room = self.db.query(MRBSRoom).filter(MRBSRoom.id == booking.room_id).first()
            if room:
                start = datetime.fromtimestamp(booking.start_time)
                recent.append({
                    'room_name': room.room_name, 'date': start.strftime("%Y-%m-%d"),
                    'start_time': start.strftime("%H:%M"),
                    'end_time': datetime.fromtimestamp(booking.end_time).strftime("%H:%M")
                })
        return recent
    
    def _infer_available_times(self, bookings: List[MRBSEntry]) -> List[str]:
        booking_times = set()
        for booking in bookings:
            current = datetime.fromtimestamp(booking.start_time)
            end = datetime.fromtimestamp(booking.end_time)
            while current < end:
                booking_times.add(current.strftime("%H:%M"))
                current += timedelta(minutes=30)
        
        all_slots = [f"{h:02d}:{m:02d}" for h in range(7, 21) for m in [0, 30]]
        return [slot for slot in all_slots if slot not in booking_times][:10]
    
    def _analyze_weekday_preference(self, bookings: List[MRBSEntry]) -> bool:
        weekday_count = sum(1 for b in bookings if datetime.fromtimestamp(b.start_time).weekday() < 5)
        weekend_count = len(bookings) - weekday_count
        return weekday_count > weekend_count * 2
    
    def _calculate_booking_frequency(self, bookings: List[MRBSEntry]) -> str:
        if len(bookings) < 2: return 'occasional'
        
        span = (datetime.fromtimestamp(bookings[0].start_time) - datetime.fromtimestamp(bookings[-1].start_time)).days or 1
        freq = len(bookings) / span
        
        return 'frequent' if freq > 0.5 else 'regular' if freq > 0.1 else 'occasional'
    
    async def get_room_features(self, room_name: str) -> Dict[str, Any]:
        cache_key = f"room_features_{room_name}"
        if self._is_cached_valid(cache_key): return self._cache[cache_key]
        
        room = self.db.query(MRBSRoom).filter(MRBSRoom.room_name == room_name).first()
        if not room: return self._cache_result(cache_key, {})
        
        recent_bookings = self.db.query(MRBSEntry).filter(
            MRBSEntry.room_id == room.id,
            MRBSEntry.start_time > int((datetime.now() - timedelta(days=30)).timestamp())
        ).all()
        
        return self._cache_result(cache_key, {
            'id': room.id, 'name': room.room_name, 'capacity': room.capacity,
            'description': room.description, 'area_id': room.area_id,
            'popularity_score': len(recent_bookings),
            'avg_booking_duration': self._calculate_avg_duration(recent_bookings),
            'peak_hours': self._get_peak_hours(recent_bookings),
            'utilization_rate': self._calculate_utilization_rate(recent_bookings)
        })
    
    def _calculate_avg_duration(self, bookings: List[MRBSEntry]) -> float:
        return np.mean([(b.end_time - b.start_time) / 3600 for b in bookings]) if bookings else 0
    
    def _get_peak_hours(self, bookings: List[MRBSEntry]) -> List[str]:
        hour_counts = defaultdict(int)
        for booking in bookings:
            hour_counts[datetime.fromtimestamp(booking.start_time).hour] += 1
        return [f"{h:02d}:00" for h, _ in sorted(hour_counts.items(), key=lambda x: x[1], reverse=True)[:3]]
    
    def _calculate_utilization_rate(self, bookings: List[MRBSEntry]) -> float:
        if not bookings: return 0
        total_hours = sum((b.end_time - b.start_time) / 3600 for b in bookings)
        return min(1.0, total_hours / (14 * 30))
    
    async def get_user_room_preferences(self, user_id: str) -> Dict[str, Any]:
        patterns = await self.get_user_booking_patterns(user_id)
        return {
            'preferred_rooms': patterns.get('preferred_rooms', []),
            'avoided_rooms': [],
            'capacity_preference': self._infer_capacity_preference(user_id),
            'location_preference': self._infer_location_preference(user_id)
        }
    
    def _infer_capacity_preference(self, user_id: str) -> Dict[str, Any]:
        capacities = []
        for booking in self.db.query(MRBSEntry).filter(MRBSEntry.create_by == user_id).limit(50).all():
            room = self.db.query(MRBSRoom).filter(MRBSRoom.id == booking.room_id).first()
            if room: capacities.append(room.capacity)
        
        return {
            'avg_preferred_capacity': np.mean(capacities) if capacities else 20,
            'min_capacity': min(capacities) if capacities else 10,
            'max_capacity': max(capacities) if capacities else 50
        }
    
    def _infer_location_preference(self, user_id: str) -> Dict[str, Any]:
        area_counts = defaultdict(int)
        for booking in self.db.query(MRBSEntry).filter(MRBSEntry.create_by == user_id).limit(50).all():
            room = self.db.query(MRBSRoom).filter(MRBSRoom.id == booking.room_id).first()
            if room: area_counts[room.area_id] += 1
        
        preferred_areas = sorted(area_counts.items(), key=lambda x: x[1], reverse=True)
        return {
            'preferred_areas': [area[0] for area in preferred_areas[:3]],
            'area_flexibility': len(area_counts) > 1
        }
    
    def get_room_popular_times(self, room_name: str) -> List[str]:
        room = self.db.query(MRBSRoom).filter(MRBSRoom.room_name == room_name).first()
        if not room: return []
        
        recent_bookings = self.db.query(MRBSEntry).filter(
            MRBSEntry.room_id == room.id,
            MRBSEntry.start_time > int((datetime.now() - timedelta(days=60)).timestamp())
        ).all()
        
        time_counts = defaultdict(int)
        for booking in recent_bookings:
            time_counts[datetime.fromtimestamp(booking.start_time).strftime("%H:%M")] += 1
        
        return [t[0] for t in sorted(time_counts.items(), key=lambda x: x[1], reverse=True)[:5]]
    
    async def get_optimal_booking_times(self, limit: int = 10) -> List[Dict[str, Any]]:
        cache_key = f"optimal_times_{limit}"
        if self._is_cached_valid(cache_key): return self._cache[cache_key]
        
        recent_bookings = self.db.query(MRBSEntry).filter(
            MRBSEntry.start_time > int((datetime.now() - timedelta(days=30)).timestamp())
        ).all()
        
        time_room_stats = defaultdict(lambda: {'bookings': 0, 'rooms': set(), 'avg_duration': []})
        
        for booking in recent_bookings:
            time_slot = datetime.fromtimestamp(booking.start_time).strftime("%H:%M")
            duration = (booking.end_time - booking.start_time) / 3600
            
            room = self.db.query(MRBSRoom).filter(MRBSRoom.id == booking.room_id).first()
            if room:
                time_room_stats[time_slot]['bookings'] += 1
                time_room_stats[time_slot]['rooms'].add(room.room_name)
                time_room_stats[time_slot]['avg_duration'].append(duration)
        
        optimal_times = []
        for time_slot, stats in time_room_stats.items():
            if stats['bookings'] > 0:
                popularity = stats['bookings'] / 30
                room_variety = len(stats['rooms'])
                avg_duration = np.mean(stats['avg_duration'])
                
                optimal_times.append({
                    'time_slot': time_slot, 'start_time': time_slot,
                    'end_time': self._add_hours(time_slot, avg_duration),
                    'room_name': list(stats['rooms'])[0] if stats['rooms'] else 'Various',
                    'popularity': popularity,
                    'optimality_score': popularity * 0.4 + room_variety * 0.3 + (2 / avg_duration) * 0.3,
                    'avg_duration': avg_duration
                })
        
        optimal_times.sort(key=lambda x: x['optimality_score'], reverse=True)
        return self._cache_result(cache_key, optimal_times[:limit])
    
    def _add_hours(self, time_str: str, hours: float) -> str:
        try:
            return (datetime.strptime(time_str, "%H:%M") + timedelta(hours=hours)).strftime("%H:%M")
        except:
            return time_str
    
    async def get_room_analytics(self, room_name: str) -> Dict[str, Any]:
        cache_key = f"room_analytics_{room_name}"
        if self._is_cached_valid(cache_key): return self._cache[cache_key]
        
        room_features = await self.get_room_features(room_name)
        popular_times = self.get_room_popular_times(room_name)
        
        room = self.db.query(MRBSRoom).filter(MRBSRoom.room_name == room_name).first()
        if not room: return self._cache_result(cache_key, {})
        
        recent_bookings = self.db.query(MRBSEntry).filter(
            MRBSEntry.room_id == room.id,
            MRBSEntry.start_time > int((datetime.now() - timedelta(days=30)).timestamp())
        ).all()
        
        analytics = {
            **room_features, 'popular_times': popular_times,
            'total_bookings_30_days': len(recent_bookings),
            'unique_users': len(set(b.create_by for b in recent_bookings)),
            'booking_trends': self._analyze_booking_trends(recent_bookings),
            'conflict_analysis': self._analyze_conflicts(room.id),
            'recommendations': self._generate_room_recommendations(room_features, recent_bookings)
        }
        
        return self._cache_result(cache_key, analytics)
    
    def _analyze_booking_trends(self, bookings: List[MRBSEntry]) -> Dict[str, Any]:
        if not bookings: return {'trend': 'stable', 'weekly_pattern': {}}
        
        weekly_bookings = defaultdict(int)
        for booking in bookings:
            weekly_bookings[datetime.fromtimestamp(booking.start_time).isocalendar()[1]] += 1
        
        weeks = sorted(weekly_bookings.keys())
        if len(weeks) >= 2:
            first_half = np.mean([weekly_bookings[w] for w in weeks[:len(weeks)//2]])
            second_half = np.mean([weekly_bookings[w] for w in weeks[len(weeks)//2:]])
            trend = 'increasing' if second_half > first_half * 1.2 else 'decreasing' if second_half < first_half * 0.8 else 'stable'
        else:
            trend = 'stable'
        
        return {'trend': trend, 'weekly_pattern': dict(weekly_bookings)}
    
    def _analyze_conflicts(self, room_id: int) -> Dict[str, Any]:
        overlapping = self.db.query(MRBSEntry).filter(MRBSEntry.room_id == room_id).all()
        
        conflicts = 0
        for i, b1 in enumerate(overlapping):
            for b2 in overlapping[i+1:]:
                if b1.start_time < b2.end_time and b1.end_time > b2.start_time:
                    conflicts += 1
        
        return {'total_conflicts': conflicts, 'conflict_rate': conflicts / len(overlapping) if overlapping else 0}
    
    def _generate_room_recommendations(self, room_features: Dict[str, Any], bookings: List[MRBSEntry]) -> List[str]:
        recommendations = []
        
        utilization = room_features.get('utilization_rate', 0)
        if utilization < 0.3:
            recommendations.append("Room is underutilized - consider promoting for smaller meetings")
        elif utilization > 0.8:
            recommendations.append("Room is highly utilized - consider capacity management")
        
        avg_duration = room_features.get('avg_booking_duration', 0)
        if avg_duration < 1:
            recommendations.append("Mostly short bookings - suitable for quick meetings")
        elif avg_duration > 3:
            recommendations.append("Long bookings typical - consider for workshops/training")
        
        return recommendations
    
    async def update_user_preferences(self, user_id: str, booking_data: Dict[str, Any]):
        cache_key = f"user_patterns_{user_id}"
        if cache_key in self._cache:
            del self._cache[cache_key]
            del self._cache_ttl[cache_key]
    
    def _is_cached_valid(self, cache_key: str) -> bool:
        return (cache_key in self._cache and 
                cache_key in self._cache_ttl and 
                datetime.now() < self._cache_ttl[cache_key])
    
    def _cache_result(self, cache_key: str, result: Any, ttl_minutes: int = 30) -> Any:
        self._cache[cache_key] = result
        self._cache_ttl[cache_key] = datetime.now() + timedelta(minutes=ttl_minutes)
        return result