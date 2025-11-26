import json
import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta, date
from pathlib import Path
import logging
from dataclasses import dataclass, asdict
from collections import defaultdict
import csv

logger = logging.getLogger(__name__)

@dataclass
class BookingEvent:
    event_id: str
    user_id: int
    room_id: int
    event_type: str
    timestamp: str
    booking_date: str
    start_time: str
    end_time: str
    duration_minutes: int
    metadata: Dict[str, Any]

@dataclass
class RecommendationEvent:
    event_id: str
    user_id: int
    recommendation_type: str
    recommended_items: List[Dict[str, Any]]
    timestamp: str
    accepted: bool
    accepted_item_id: Optional[int]
    response_time_ms: int
    context: Dict[str, Any]

class AnalyticsManager:
    def __init__(self, base_path: str = "data/analytics"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        for path in ["events", "aggregates", "reports", "exports"]:
            (self.base_path / path).mkdir(exist_ok=True)
        
        self._init_analytics_db()
    
    def _init_analytics_db(self):
        self.conn = sqlite3.connect(str(self.base_path / "analytics.db"), check_same_thread=False)
        
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS booking_events (
                event_id TEXT PRIMARY KEY, user_id INTEGER NOT NULL, room_id INTEGER NOT NULL,
                event_type TEXT NOT NULL, timestamp TIMESTAMP NOT NULL, booking_date DATE NOT NULL,
                start_time TIME NOT NULL, end_time TIME NOT NULL, duration_minutes INTEGER NOT NULL,
                metadata TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
            
            CREATE TABLE IF NOT EXISTS recommendation_events (
                event_id TEXT PRIMARY KEY, user_id INTEGER NOT NULL, recommendation_type TEXT NOT NULL,
                recommended_items TEXT NOT NULL, timestamp TIMESTAMP NOT NULL, accepted BOOLEAN DEFAULT FALSE,
                accepted_item_id INTEGER, response_time_ms INTEGER, context TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
            
            CREATE TABLE IF NOT EXISTS user_behavior_patterns (
                user_id INTEGER PRIMARY KEY, preferred_times TEXT, preferred_rooms TEXT,
                booking_frequency REAL, advance_booking_days REAL, cancellation_rate REAL,
                modification_rate REAL, recommendation_acceptance_rate REAL,
                last_updated TIMESTAMP, pattern_confidence REAL);
            
            CREATE TABLE IF NOT EXISTS room_utilization (
                room_id INTEGER, date DATE, hour INTEGER, utilization_rate REAL,
                booking_count INTEGER, total_duration_minutes INTEGER, avg_booking_duration REAL,
                PRIMARY KEY (room_id, date, hour));
            
            CREATE TABLE IF NOT EXISTS daily_aggregates (
                date DATE PRIMARY KEY, total_bookings INTEGER DEFAULT 0, total_cancellations INTEGER DEFAULT 0,
                total_modifications INTEGER DEFAULT 0, unique_users INTEGER DEFAULT 0, unique_rooms INTEGER DEFAULT 0,
                avg_booking_duration REAL DEFAULT 0, peak_hour INTEGER, recommendation_requests INTEGER DEFAULT 0,
                recommendation_acceptances INTEGER DEFAULT 0);
            
            CREATE TABLE IF NOT EXISTS recommendation_performance (
                recommendation_type TEXT, date DATE, total_requests INTEGER DEFAULT 0,
                total_acceptances INTEGER DEFAULT 0, acceptance_rate REAL DEFAULT 0,
                avg_response_time_ms REAL DEFAULT 0, PRIMARY KEY (recommendation_type, date));
            
            CREATE INDEX IF NOT EXISTS idx_booking_events_user_time ON booking_events(user_id, timestamp);
            CREATE INDEX IF NOT EXISTS idx_booking_events_room_time ON booking_events(room_id, timestamp);
            CREATE INDEX IF NOT EXISTS idx_booking_events_date ON booking_events(booking_date);
            CREATE INDEX IF NOT EXISTS idx_recommendation_events_user ON recommendation_events(user_id);
            CREATE INDEX IF NOT EXISTS idx_recommendation_events_type ON recommendation_events(recommendation_type);
        """)
        self.conn.commit()
    
    def log_booking_event(self, be: BookingEvent) -> bool:
        try:
            self.conn.execute("""
                INSERT OR REPLACE INTO booking_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (be.event_id, be.user_id, be.room_id, be.event_type, be.timestamp, 
                  be.booking_date, be.start_time, be.end_time, be.duration_minutes, json.dumps(be.metadata)))
            self.conn.commit()
            
            self._update_daily_aggregates(be)
            self._update_room_utilization(be)
            logger.debug(f"Logged booking event: {be.event_id}")
            return True
        except Exception as e:
            logger.error(f"Error logging booking event: {e}")
            return False
    
    def log_recommendation_event(self, re: RecommendationEvent) -> bool:
        try:
            self.conn.execute("""
                INSERT OR REPLACE INTO recommendation_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (re.event_id, re.user_id, re.recommendation_type, json.dumps(re.recommended_items),
                  re.timestamp, re.accepted, re.accepted_item_id, re.response_time_ms, json.dumps(re.context)))
            self.conn.commit()
            
            self._update_recommendation_performance(re)
            logger.debug(f"Logged recommendation event: {re.event_id}")
            return True
        except Exception as e:
            logger.error(f"Error logging recommendation event: {e}")
            return False
    
    def _update_daily_aggregates(self, be: BookingEvent):
        try:
            self.conn.execute("INSERT OR IGNORE INTO daily_aggregates (date) VALUES (?)", (be.booking_date,))
            
            if be.event_type == 'booking_created':
                self.conn.execute("""
                    UPDATE daily_aggregates SET total_bookings = total_bookings + 1,
                    unique_users = (SELECT COUNT(DISTINCT user_id) FROM booking_events 
                                   WHERE booking_date = ? AND event_type = 'booking_created'),
                    unique_rooms = (SELECT COUNT(DISTINCT room_id) FROM booking_events 
                                   WHERE booking_date = ? AND event_type = 'booking_created'),
                    avg_booking_duration = (SELECT AVG(duration_minutes) FROM booking_events 
                                           WHERE booking_date = ? AND event_type = 'booking_created')
                    WHERE date = ?
                """, (be.booking_date, be.booking_date, be.booking_date, be.booking_date))
            elif be.event_type == 'booking_cancelled':
                self.conn.execute("UPDATE daily_aggregates SET total_cancellations = total_cancellations + 1 WHERE date = ?", (be.booking_date,))
            elif be.event_type == 'booking_modified':
                self.conn.execute("UPDATE daily_aggregates SET total_modifications = total_modifications + 1 WHERE date = ?", (be.booking_date,))
            
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error updating daily aggregates: {e}")
    
    def _update_room_utilization(self, be: BookingEvent):
        try:
            if be.event_type != 'booking_created':
                return
            
            hour = int(be.start_time.split(':')[0])
            self.conn.execute("""
                INSERT OR IGNORE INTO room_utilization VALUES (?, ?, ?, 0, 0, 0, 0)
            """, (be.room_id, be.booking_date, hour))
            
            self.conn.execute("""
                UPDATE room_utilization SET booking_count = booking_count + 1,
                total_duration_minutes = total_duration_minutes + ?,
                avg_booking_duration = total_duration_minutes / booking_count,
                utilization_rate = CAST(total_duration_minutes AS REAL) / 60.0
                WHERE room_id = ? AND date = ? AND hour = ?
            """, (be.duration_minutes, be.room_id, be.booking_date, hour))
            
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error updating room utilization: {e}")
    
    def _update_recommendation_performance(self, re: RecommendationEvent):
        try:
            event_date = datetime.fromisoformat(re.timestamp).date()
            self.conn.execute("""
                INSERT OR IGNORE INTO recommendation_performance VALUES (?, ?, 0, 0, 0, 0)
            """, (re.recommendation_type, event_date))
            
            if re.accepted:
                self.conn.execute("""
                    UPDATE recommendation_performance SET total_requests = total_requests + 1,
                    total_acceptances = total_acceptances + 1,
                    acceptance_rate = CAST(total_acceptances AS REAL) / total_requests,
                    avg_response_time_ms = ((avg_response_time_ms * (total_requests - 1) + ?) / total_requests)
                    WHERE recommendation_type = ? AND date = ?
                """, (re.response_time_ms, re.recommendation_type, event_date))
            else:
                self.conn.execute("""
                    UPDATE recommendation_performance SET total_requests = total_requests + 1,
                    acceptance_rate = CAST(total_acceptances AS REAL) / total_requests,
                    avg_response_time_ms = ((avg_response_time_ms * (total_requests - 1) + ?) / total_requests)
                    WHERE recommendation_type = ? AND date = ?
                """, (re.response_time_ms, re.recommendation_type, event_date))
            
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error updating recommendation performance: {e}")
    
    def analyze_user_behavior(self, user_id: int) -> Dict[str, Any]:
        try:
            cursor = self.conn.execute("""
                SELECT * FROM booking_events WHERE user_id = ? AND timestamp > ? ORDER BY timestamp DESC
            """, (user_id, (datetime.now() - timedelta(days=90)).isoformat()))
            
            events = cursor.fetchall()
            if not events:
                return {}
            
            booking_times = []
            room_preferences = defaultdict(int)
            advance_days = []
            cancellation_count = modification_count = total_bookings = 0
            
            for event in events:
                event_type, timestamp, booking_date, start_time, room_id = event[3], datetime.fromisoformat(event[4]), datetime.fromisoformat(event[5]).date(), event[6], event[2]
                
                if event_type == 'booking_created':
                    total_bookings += 1
                    booking_times.append(start_time)
                    room_preferences[room_id] += 1
                    advance_days.append((booking_date - timestamp.date()).days)
                elif event_type == 'booking_cancelled':
                    cancellation_count += 1
                elif event_type == 'booking_modified':
                    modification_count += 1
            
            cursor = self.conn.execute("""
                SELECT COUNT(*), SUM(CASE WHEN accepted THEN 1 ELSE 0 END)
                FROM recommendation_events WHERE user_id = ? AND timestamp > ?
            """, (user_id, (datetime.now() - timedelta(days=90)).isoformat()))
            
            rec_total, rec_accepted = cursor.fetchone()
            
            patterns = {
                'user_id': user_id, 'total_bookings': total_bookings,
                'booking_frequency': total_bookings / 13 if total_bookings > 0 else 0,
                'preferred_times': self._find_preferred_times(booking_times),
                'preferred_rooms': dict(sorted(room_preferences.items(), key=lambda x: x[1], reverse=True)[:5]),
                'advance_booking_days': np.mean(advance_days) if advance_days else 0,
                'cancellation_rate': cancellation_count / total_bookings if total_bookings > 0 else 0,
                'modification_rate': modification_count / total_bookings if total_bookings > 0 else 0,
                'recommendation_acceptance_rate': rec_accepted / rec_total if rec_total > 0 else 0,
                'pattern_confidence': min(1.0, total_bookings / 10.0)
            }
            
            self._store_user_patterns(patterns)
            return patterns
        except Exception as e:
            logger.error(f"Error analyzing user behavior: {e}")
            return {}
    
    def _find_preferred_times(self, booking_times: List[str]) -> List[str]:
        if not booking_times:
            return []
        
        time_slots = defaultdict(int)
        for time_str in booking_times:
            hour = int(time_str.split(':')[0])
            slot = ('morning' if 6 <= hour < 9 else 'late_morning' if 9 <= hour < 12 else
                   'lunch' if 12 <= hour < 14 else 'afternoon' if 14 <= hour < 17 else
                   'evening' if 17 <= hour < 20 else 'other')
            time_slots[slot] += 1
        
        return [slot[0] for slot in sorted(time_slots.items(), key=lambda x: x[1], reverse=True)[:3]]
    
    def _store_user_patterns(self, patterns: Dict[str, Any]):
        try:
            self.conn.execute("""
                INSERT OR REPLACE INTO user_behavior_patterns VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (patterns['user_id'], json.dumps(patterns['preferred_times']), json.dumps(patterns['preferred_rooms']),
                  patterns['booking_frequency'], patterns['advance_booking_days'], patterns['cancellation_rate'],
                  patterns['modification_rate'], patterns['recommendation_acceptance_rate'], 
                  datetime.now().isoformat(), patterns['pattern_confidence']))
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error storing user patterns: {e}")
    
    def get_room_utilization_report(self, start_date: date, end_date: date) -> Dict[str, Any]:
        try:
            cursor = self.conn.execute("""
                SELECT room_id, AVG(utilization_rate) as avg_utilization, SUM(booking_count) as total_bookings,
                       AVG(avg_booking_duration) as avg_duration, MAX(utilization_rate) as peak_utilization
                FROM room_utilization WHERE date BETWEEN ? AND ? GROUP BY room_id ORDER BY avg_utilization DESC
            """, (start_date, end_date))
            
            room_data = [{'room_id': r[0], 'avg_utilization': r[1], 'total_bookings': r[2], 
                         'avg_duration_minutes': r[3], 'peak_utilization': r[4]} for r in cursor.fetchall()]
            
            cursor = self.conn.execute("""
                SELECT hour, AVG(utilization_rate) FROM room_utilization 
                WHERE date BETWEEN ? AND ? GROUP BY hour ORDER BY hour
            """, (start_date, end_date))
            
            hourly_pattern = {row[0]: row[1] for row in cursor.fetchall()}
            
            return {
                'date_range': {'start': str(start_date), 'end': str(end_date)},
                'room_utilization': room_data, 'hourly_pattern': hourly_pattern,
                'total_rooms': len(room_data),
                'avg_system_utilization': np.mean([r['avg_utilization'] for r in room_data]) if room_data else 0
            }
        except Exception as e:
            logger.error(f"Error generating utilization report: {e}")
            return {}
    
    def get_recommendation_performance_report(self, days: int = 30) -> Dict[str, Any]:
        try:
            start_date = (datetime.now() - timedelta(days=days)).date()
            
            cursor = self.conn.execute("""
                SELECT recommendation_type, SUM(total_requests), SUM(total_acceptances),
                       AVG(acceptance_rate), AVG(avg_response_time_ms)
                FROM recommendation_performance WHERE date >= ? GROUP BY recommendation_type ORDER BY AVG(acceptance_rate) DESC
            """, (start_date,))
            
            performance_data = [{'recommendation_type': r[0], 'total_requests': r[1], 'total_acceptances': r[2],
                               'acceptance_rate': r[3], 'avg_response_time_ms': r[4]} for r in cursor.fetchall()]
            
            cursor = self.conn.execute("""
                SELECT date, SUM(total_requests), SUM(total_acceptances)
                FROM recommendation_performance WHERE date >= ? GROUP BY date ORDER BY date
            """, (start_date,))
            
            daily_trend = [{'date': r[0], 'requests': r[1], 'acceptances': r[2], 
                           'acceptance_rate': r[2] / r[1] if r[1] > 0 else 0} for r in cursor.fetchall()]
            
            total_req = sum(p['total_requests'] for p in performance_data)
            total_acc = sum(p['total_acceptances'] for p in performance_data)
            
            return {
                'period_days': days, 'performance_by_type': performance_data, 'daily_trend': daily_trend,
                'overall_acceptance_rate': total_acc / total_req if total_req > 0 else 0,
                'total_requests': total_req, 'total_acceptances': total_acc
            }
        except Exception as e:
            logger.error(f"Error generating recommendation performance report: {e}")
            return {}
    
    def export_data(self, data_type: str, start_date: date, end_date: date, format: str = 'csv') -> Optional[str]:
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{data_type}_{start_date}_{end_date}_{timestamp}.{format}"
            
            queries = {
                'booking_events': "SELECT * FROM booking_events WHERE booking_date BETWEEN ? AND ? ORDER BY timestamp",
                'recommendation_events': "SELECT * FROM recommendation_events WHERE DATE(timestamp) BETWEEN ? AND ? ORDER BY timestamp",
                'user_patterns': "SELECT * FROM user_behavior_patterns ORDER BY user_id"
            }
            
            if data_type not in queries:
                logger.error(f"Unknown data type: {data_type}")
                return None
            
            cursor = self.conn.execute(queries[data_type], 
                                     (start_date, end_date) if data_type != 'user_patterns' else ())
            
            file_path = self.base_path / "exports" / filename
            
            if format == 'csv':
                with open(file_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([desc[0] for desc in cursor.description])
                    writer.writerows(cursor.fetchall())
            elif format == 'json':
                columns = [desc[0] for desc in cursor.description]
                data = [dict(zip(columns, row)) for row in cursor.fetchall()]
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, default=str)
            
            logger.info(f"Exported {data_type} data to {file_path}")
            return str(file_path)
        except Exception as e:
            logger.error(f"Error exporting data: {e}")
            return None
    
    def get_analytics_summary(self) -> Dict[str, Any]:
        try:
            thirty_days_ago = (datetime.now() - timedelta(days=30)).date()
            
            cursor = self.conn.execute("""
                SELECT SUM(total_bookings), SUM(total_cancellations), SUM(unique_users), AVG(avg_booking_duration)
                FROM daily_aggregates WHERE date >= ?
            """, (thirty_days_ago,))
            booking_stats = cursor.fetchone()
            
            total_events = self.conn.execute("SELECT COUNT(*) FROM booking_events WHERE timestamp >= ?", 
                                           ((datetime.now() - timedelta(days=30)).isoformat(),)).fetchone()[0]
            total_rec_events = self.conn.execute("SELECT COUNT(*) FROM recommendation_events WHERE timestamp >= ?", 
                                               ((datetime.now() - timedelta(days=30)).isoformat(),)).fetchone()[0]
            db_size = self.conn.execute("SELECT page_count * page_size FROM pragma_page_count(), pragma_page_size()").fetchone()[0]
            
            return {
                'last_30_days': {
                    'total_bookings': booking_stats[0] or 0, 'total_cancellations': booking_stats[1] or 0,
                    'unique_users': booking_stats[2] or 0, 'avg_booking_duration': booking_stats[8] or 0,
                    'total_events': total_events, 'total_recommendation_events': total_rec_events
                },
                'database_info': {'size_bytes': db_size, 'size_mb': db_size / (1024 * 1024)}
            }
        except Exception as e:
            logger.error(f"Error getting analytics summary: {e}")
            return {}
    
    def cleanup_old_data(self, days_to_keep: int = 365):
        try:
            cutoff_date = (datetime.now() - timedelta(days=days_to_keep)).isoformat()
            
            old_booking_count = self.conn.execute("SELECT COUNT(*) FROM booking_events WHERE timestamp < ?", (cutoff_date,)).fetchone()[0]
            old_rec_count = self.conn.execute("SELECT COUNT(*) FROM recommendation_events WHERE timestamp < ?", (cutoff_date,)).fetchone()[0]
            
            self.conn.execute("DELETE FROM booking_events WHERE timestamp < ?", (cutoff_date,))
            self.conn.execute("DELETE FROM recommendation_events WHERE timestamp < ?", (cutoff_date,))
            
            old_aggregate_date = (datetime.now() - timedelta(days=days_to_keep * 2)).date()
            for table in ['daily_aggregates', 'room_utilization', 'recommendation_performance']:
                self.conn.execute(f"DELETE FROM {table} WHERE date < ?", (old_aggregate_date,))
            
            self.conn.commit()
            self.conn.execute("VACUUM")
            
            logger.info(f"Cleaned up {old_booking_count} booking events and {old_rec_count} recommendation events")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def close(self):
        if hasattr(self, 'conn'):
            self.conn.close()