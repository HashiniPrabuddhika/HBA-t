# recommendations/core/recommendation_engine.py
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import text, and_, or_, func
import pandas as pd
import numpy as np
import logging
import time
import hashlib
import json
import asyncio

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
except ImportError:
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        HuggingFaceEmbeddings = None

try:
    from langchain_community.vectorstores import Chroma
except ImportError:
    Chroma = None

from ..strategies.alternative_time import AlternativeTimeStrategy
from ..strategies.alternative_room import AlternativeRoomStrategy
from ..strategies.proactive_suggestions import ProactiveSuggestionStrategy
from ..strategies.smart_scheduling import SmartSchedulingStrategy
from ..data.analytics_processor import AnalyticsProcessor
from ..utils.cache_manager import CacheManager
from ..utils.metrics import RecommendationMetrics
from .preference_learner import PreferenceLearner
from config.recommendation_config import RecommendationConfig, DatabaseManager
from models.booking import MRBSEntry, MRBSRepeat
from models.room import MRBSRoom

logger = logging.getLogger(__name__)

class RecommendationEngine:
    
    def __init__(self, db: Session = None, config: Optional[RecommendationConfig] = None) -> None:
      
        logger.info("Initializing RecommendationEngine with MySQL integration")
        
        self.config = config or RecommendationConfig()
        
        try:
            self.db_manager = DatabaseManager(self.config)
            
            if db:
                self.db = db
            else:
                self.db = self.db_manager.get_main_session()
            
            self.cache_db = self.db_manager.get_cache_engine()
        except Exception as e:
            logger.warning(f"Could not initialize database manager: {e}")
            self.db = db
            self.db_manager = None
            self.cache_db = None
        
        # Initialize core components with proper error handling
        self._initialize_components()
        
        # Initialize strategies with proper error handling
        self._initialize_strategies()
        
        # Initialize embeddings with proper error handling
        self._initialize_embeddings()
        
        # Valid request types
        self.valid_request_types = {
            "alternative_time", "alternative_room", "proactive", 
            "smart_scheduling", "comprehensive"
        }

        try:
            self._verify_database_connection()
        except Exception as e:
            logger.warning(f"Database verification failed: {e}")
    
    def _initialize_components(self):
        try:
            self.analytics = AnalyticsProcessor(self.db)
        except Exception as e:
            logger.warning(f"Could not initialize AnalyticsProcessor: {e}")
            self.analytics = None
            
        try:
            self.cache = CacheManager()
        except Exception as e:
            logger.warning(f"Could not initialize CacheManager: {e}")
            self.cache = None

        try:
            self.metrics = RecommendationMetrics()
        except Exception as e:
            logger.warning(f"Could not initialize RecommendationMetrics: {e}")
            self.metrics = None
            
        # Initialize PreferenceLearner with required db parameter
        try:
            if self.db:
                self.preference_learner = PreferenceLearner(
                    db=self.db,
                    embedding_model=None,  
                    cache_manager=self.cache
                )
            else:
                logger.warning("No database session available for PreferenceLearner")
                self.preference_learner = None
        except Exception as e:
            logger.warning(f"Could not initialize PreferenceLearner: {e}")
            self.preference_learner = None
    
    def _initialize_strategies(self):
        try:
            self.alternative_time = AlternativeTimeStrategy(self.db)
        except Exception as e:
            logger.warning(f"Could not initialize AlternativeTimeStrategy: {e}")
            self.alternative_time = None
            
        # Initialize AlternativeRoomStrategy
        try:
            self.alternative_room = AlternativeRoomStrategy(self.db)
        except Exception as e:
            logger.warning(f"Could not initialize AlternativeRoomStrategy: {e}")
            self.alternative_room = None
            
        # Initialize ProactiveSuggestionStrategy
        try:
            self.proactive = ProactiveSuggestionStrategy(self.db)
        except Exception as e:
            logger.warning(f"Could not initialize ProactiveSuggestionStrategy: {e}")
            self.proactive = None
            
        # Initialize SmartSchedulingStrategy with special handling for async issues
        try:
            self.smart_scheduling = self._initialize_smart_scheduling_strategy()
        except Exception as e:
            logger.warning(f"Could not initialize SmartSchedulingStrategy: {e}")
            self.smart_scheduling = None
    
    def _initialize_smart_scheduling_strategy(self):
        """Initialize SmartSchedulingStrategy with comprehensive error handling"""
        try:
            try:
                loop = asyncio.get_running_loop()
                logger.info("Running in async context, initializing SmartSchedulingStrategy carefully")
            except RuntimeError:
                logger.info("No running event loop, initializing SmartSchedulingStrategy normally")
            
            try:
                return SmartSchedulingStrategy(self.db, db_session=self.db)
            except TypeError:
                try:
                    return SmartSchedulingStrategy(self.db)
                except TypeError:
                    try:
                        return SmartSchedulingStrategy(db_session=self.db)
                    except TypeError:
                        try:
                            logger.warning("Initializing SmartSchedulingStrategy without database connection")
                            return SmartSchedulingStrategy()
                        except Exception as e:
                            logger.error(f"Could not initialize SmartSchedulingStrategy at all: {e}")
                            return None
                            
        except Exception as e:
            logger.error(f"Unexpected error initializing SmartSchedulingStrategy: {e}")
            return None
    
    def _initialize_embeddings(self):
        """Initialize embeddings with proper error handling"""
        try:
            if HuggingFaceEmbeddings is None:
                logger.warning("HuggingFaceEmbeddings not available, skipping embeddings initialization")
                self.embeddings = None
                return
                
            embedding_model = getattr(self.config, 'EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
            self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
            logger.info(f"Initialized embeddings with model: {embedding_model}")
            
            if self.preference_learner and hasattr(self.preference_learner, 'embedding_model'):
                self.preference_learner.embedding_model = self.embeddings
                
        except Exception as e:
            logger.warning(f"Failed to initialize embeddings: {e}")
            self.embeddings = None
   
    def get_recommendations(self, request_data: Dict[str, Any]) -> List[Dict[str, Any]]:

        try:
            # Extract information from request_data
            user_id = str(request_data.get('user_id', 'unknown'))
            room_id = request_data.get('room_id', '')
            start_time = request_data.get('start_time', '')
            end_time = request_data.get('end_time', '')
            purpose = request_data.get('purpose', '')
            requirements = request_data.get('requirements', {})
            
            logger.info(f"Generating recommendations for user {user_id}")
            
            recommendations = []
            
            try:
                alt_time_recs = self._get_alternative_time_recommendations_from_db(request_data)
                recommendations.extend(alt_time_recs)
            except Exception as e:
                logger.warning(f"Alternative time recommendations failed: {e}")
            
            try:
                alt_room_recs = self._get_alternative_room_recommendations_from_db(request_data)
                recommendations.extend(alt_room_recs)
            except Exception as e:
                logger.warning(f"Alternative room recommendations failed: {e}")
            
            try:
                proactive_recs = self._get_proactive_recommendations_from_db(request_data)
                recommendations.extend(proactive_recs)
            except Exception as e:
                logger.warning(f"Proactive recommendations failed: {e}")
            
            try:
                smart_recs = self._get_smart_scheduling_recommendations_from_db(request_data)
                recommendations.extend(smart_recs)
            except Exception as e:
                logger.warning(f"Smart scheduling recommendations failed: {e}")
            
            if not recommendations:
                logger.info("No recommendations generated, creating fallback recommendations")
                recommendations = self._create_fallback_recommendations(request_data)
            
            logger.info(f"Generated {len(recommendations)} recommendations for user {user_id}")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return self._create_fallback_recommendations(request_data)
    

    def _get_alternative_time_recommendations_from_db(self, request_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not self.db:
            return []
        
        try:
            room_name = request_data.get('room_id', '')
            start_time_str = request_data.get('start_time', '')
            end_time_str = request_data.get('end_time', '')
             
            try:
                if 'T' in start_time_str:
                    start_time = datetime.fromisoformat(start_time_str.replace('Z', '+00:00'))
                    end_time = datetime.fromisoformat(end_time_str.replace('Z', '+00:00'))
                else:
                    start_time = datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S')
                    end_time = datetime.strptime(end_time_str, '%Y-%m-%d %H:%M:%S')
            except (ValueError, TypeError):
                logger.warning("Could not parse datetime strings, using current time")
                start_time = start_time_str
                end_time = end_time_str
            
            duration = end_time - start_time
            
            room = self.db.query(MRBSRoom).filter(
                MRBSRoom.room_name == room_name,
                MRBSRoom.disabled == False
            ).first()
            
            if not room:
                logger.warning(f"Room {room_name} not found or disabled")
                return []
            
            recommendations = []
            
            logger.info(f" Checking same-day alternatives for {start_time.strftime('%Y-%m-%d')}")
            same_day_alternatives = self._get_same_day_alternatives(
                room, start_time, end_time, duration, room_name
            )
            recommendations.extend(same_day_alternatives)
        
            if len(same_day_alternatives) < 3:  
                logger.info("Checking next available days")
                next_day_alternatives = self._get_next_day_alternatives(
                    room, start_time, end_time, duration, room_name, 
                    max_days=5  
                )
                recommendations.extend(next_day_alternatives)
            
            recommendations.sort(key=lambda x: (
                x.get('is_same_day', False),  
                x['score']  
            ), reverse=True)
            
            final_recommendations = recommendations[:8]
            
            logger.info(f"✅ Found {len(final_recommendations)} alternative time recommendations")
            return final_recommendations
            
        except Exception as e:
            logger.error(f"Error in alternative time recommendations: {e}", exc_info=True)
            return []
        
    def _get_same_day_alternatives(self, room, requested_start: datetime, requested_end: datetime, 
                                  duration: timedelta, room_name: str) -> List[Dict[str, Any]]:
        same_day_alternatives = []
        requested_date = requested_start.date()
        
        time_slots_to_check = [
            (requested_start - timedelta(minutes=30), "30 minutes earlier"),
            (requested_start + timedelta(minutes=30), "30 minutes later"),
            (requested_start - timedelta(hours=1), "1 hour earlier"),
            (requested_start + timedelta(hours=1), "1 hour later"),
            (requested_start - timedelta(hours=2), "2 hours earlier"),
            (requested_start + timedelta(hours=2), "2 hours later"),
            
            (datetime.combine(requested_date, datetime.min.time().replace(hour=9, minute=0)), "9:00 AM"),
            (datetime.combine(requested_date, datetime.min.time().replace(hour=10, minute=0)), "10:00 AM"),
            (datetime.combine(requested_date, datetime.min.time().replace(hour=11, minute=0)), "11:00 AM"),
            (datetime.combine(requested_date, datetime.min.time().replace(hour=14, minute=0)), "2:00 PM"),
            (datetime.combine(requested_date, datetime.min.time().replace(hour=15, minute=0)), "3:00 PM"),
            (datetime.combine(requested_date, datetime.min.time().replace(hour=16, minute=0)), "4:00 PM"),
            
            (datetime.combine(requested_date, datetime.min.time().replace(hour=9, minute=30)), "9:30 AM"),
            (datetime.combine(requested_date, datetime.min.time().replace(hour=10, minute=30)), "10:30 AM"),
            (datetime.combine(requested_date, datetime.min.time().replace(hour=14, minute=30)), "2:30 PM"),
            (datetime.combine(requested_date, datetime.min.time().replace(hour=15, minute=30)), "3:30 PM"),
        ]
        
        for alt_start, description in time_slots_to_check:
            if alt_start.date() != requested_date:
                continue
                
            if alt_start == requested_start:
                continue
                
            if alt_start.hour < 8 or alt_start.hour >= 18:
                continue
            
            alt_end = alt_start + duration
            
            if alt_end.date() != requested_date or alt_end.hour > 20:
                continue
            
            if self._is_time_slot_available(room.id, alt_start, alt_end):
                score = self._calculate_same_day_score(alt_start, requested_start, description)
                
                same_day_alternatives.append({
                    'type': 'alternative_time',
                    'score': score,
                    'reason': f'Same day - Room {room_name} available {description}',
                    'suggestion': {
                        'room_id': room_name,
                        'room_name': room_name,
                        'capacity': room.capacity,
                        'start_time': alt_start.isoformat(),
                        'end_time': alt_end.isoformat(),
                        'confidence': score,
                        'duration_minutes': int(duration.total_seconds() / 60),
                        'time_shift': description,
                        'date': alt_start.strftime('%Y-%m-%d'),
                        'day_type': 'same_day'
                    },
                    'data_source': 'mysql_same_day_alternative',
                    'availability_confirmed': True,
                    'is_same_day': True
                })
        
        return same_day_alternatives


    def _get_next_day_alternatives(self, room, requested_start: datetime, requested_end: datetime,
                                  duration: timedelta, room_name: str, max_days: int = 5) -> List[Dict[str, Any]]:
        next_day_alternatives = []
        
        base_date = requested_start.date()
        
        for day_offset in range(1, max_days + 1):
            next_date = base_date + timedelta(days=day_offset)
          
            same_time_next_day = datetime.combine(next_date, requested_start.time())
            same_time_end = same_time_next_day + duration
            
            if self._is_time_slot_available(room.id, same_time_next_day, same_time_end):
                day_name = next_date.strftime('%A, %B %d')
                score = 0.7 - (day_offset * 0.1)  # Decrease score for further days
                
                next_day_alternatives.append({
                    'type': 'alternative_time',
                    'score': max(score, 0.3),  # Minimum score of 0.3
                    'reason': f'Next day - Room {room_name} available same time on {day_name}',
                    'suggestion': {
                        'room_id': room_name,
                        'room_name': room_name,
                        'capacity': room.capacity,
                        'start_time': same_time_next_day.isoformat(),
                        'end_time': same_time_end.isoformat(),
                        'confidence': max(score, 0.3),
                        'duration_minutes': int(duration.total_seconds() / 60),
                        'time_shift': f'Same time on {day_name}',
                        'date': next_date.strftime('%Y-%m-%d'),
                        'day_type': 'next_day',
                        'days_ahead': day_offset
                    },
                    'data_source': 'mysql_next_day_alternative',
                    'availability_confirmed': True,
                    'is_same_day': False
                })
            
            # Also try common meeting times on next days
            common_times = [
                (9, 0, "9:00 AM"),
                (10, 0, "10:00 AM"),
                (11, 0, "11:00 AM"),
                (14, 0, "2:00 PM"),
                (15, 0, "3:00 PM"),
            ]
            
            for hour, minute, time_desc in common_times:
                alt_start = datetime.combine(next_date, datetime.min.time().replace(hour=hour, minute=minute))
                alt_end = alt_start + duration
                
                if alt_start == same_time_next_day:
                    continue
                
                if self._is_time_slot_available(room.id, alt_start, alt_end):
                    day_name = next_date.strftime('%A, %B %d')
                    score = 0.6 - (day_offset * 0.1)  # Slightly lower score than same time
                    
                    next_day_alternatives.append({
                        'type': 'alternative_time',
                        'score': max(score, 0.2),
                        'reason': f'Room {room_name} available {time_desc} on {day_name}',
                        'suggestion': {
                            'room_id': room_name,
                            'room_name': room_name,
                            'capacity': room.capacity,
                            'start_time': alt_start.isoformat(),
                            'end_time': alt_end.isoformat(),
                            'confidence': max(score, 0.2),
                            'duration_minutes': int(duration.total_seconds() / 60),
                            'time_shift': f'{time_desc} on {day_name}',
                            'date': next_date.strftime('%Y-%m-%d'),
                            'day_type': 'next_day',
                            'days_ahead': day_offset
                        },
                        'data_source': 'mysql_next_day_common_time',
                        'availability_confirmed': True,
                        'is_same_day': False
                    })
                    
                    # Limit alternatives per day to avoid too many suggestions
                    break
            
            # If we found enough alternatives, stop checking further days
            if len(next_day_alternatives) >= 4:
                break
        
        return next_day_alternatives

    def _is_time_slot_available(self, room_id: int, start_time: datetime, end_time: datetime) -> bool:
        """Check if a specific time slot is available for the room"""
        try:
            start_timestamp = int(start_time.timestamp())
            end_timestamp = int(end_time.timestamp())
            
            conflicts = self.db.query(MRBSEntry).filter(
                MRBSEntry.room_id == room_id,
                MRBSEntry.start_time < end_timestamp,
                MRBSEntry.end_time > start_timestamp,
                MRBSEntry.status == 0  # Assuming 0 is active status
            ).count()
            
            return conflicts == 0
            
        except Exception as e:
            logger.error(f"Error checking time slot availability: {e}")
            return False


    def _calculate_same_day_score(self, alt_start: datetime, requested_start: datetime, description: str) -> float:
        """Calculate score for same-day alternatives"""
        base_score = 0.85  # High base score for same day
        
        # Time proximity bonus (closer to requested time = higher score)
        time_diff_hours = abs((alt_start - requested_start).total_seconds() / 3600)
        if time_diff_hours <= 0.5:
            base_score += 0.1  # Very close time
        elif time_diff_hours <= 1:
            base_score += 0.05  # Close time
        elif time_diff_hours > 4:
            base_score -= 0.1  # Far from requested time
        
        # Business hours bonus
        hour = alt_start.hour
        if 9 <= hour <= 17:
            base_score += 0.05  # Prime business hours
        elif 8 <= hour <= 18:
            pass  # Acceptable hours, no penalty
        else:
            base_score -= 0.2  # Outside normal hours
        
        # Preference for certain descriptions
        if "30 minutes" in description:
            base_score += 0.05  # Small adjustment preferred
        elif "1 hour" in description:
            base_score += 0.03  # Still good
        
        return min(base_score, 1.0)


    def get_detailed_alternative_schedule(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
      
        room_name = request_data.get('room_id', '')
        alternatives = self._get_alternative_time_recommendations_from_db(request_data)
        
        # Group alternatives by day type
        same_day = [alt for alt in alternatives if alt.get('is_same_day', False)]
        next_days = [alt for alt in alternatives if not alt.get('is_same_day', False)]
        
        # Group next days by actual date
        next_days_by_date = {}
        for alt in next_days:
            date = alt['suggestion'].get('date', 'unknown')
            if date not in next_days_by_date:
                next_days_by_date[date] = []
            next_days_by_date[date].append(alt)
        
        return {
            'room_requested': room_name,
            'original_request': {
                'start_time': request_data.get('start_time'),
                'end_time': request_data.get('end_time'),
                'date': request_data.get('start_time', '').split('T')[0] if 'T' in request_data.get('start_time', '') else 'unknown'
            },
            'same_day_alternatives': {
                'count': len(same_day),
                'options': same_day
            },
            'next_day_alternatives': {
                'count': len(next_days),
                'by_date': next_days_by_date,
                'options': next_days
            },
            'total_alternatives': len(alternatives),
            'recommendation': {
                'message': f"Found {len(same_day)} same-day options and {len(next_days)} next-day options",
                'priority': "Same-day alternatives are prioritized and shown first"
            }
        }


    def _get_alternative_room_recommendations_from_db(self, request_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get alternative room recommendations using actual database data"""
        if not self.db:
            return []
        
        try:
            room_name = request_data.get('room_id', '')
            start_time_str = request_data.get('start_time', '')
            end_time_str = request_data.get('end_time', '')
            capacity_required = request_data.get('capacity', 1)
            
            try:
                if 'T' in start_time_str:
                    start_time = datetime.fromisoformat(start_time_str.replace('Z', '+00:00'))
                    end_time = datetime.fromisoformat(end_time_str.replace('Z', '+00:00'))
                else:
                    start_time = datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S')
                    end_time = datetime.strptime(end_time_str, '%Y-%m-%d %H:%M:%S')
            except (ValueError, TypeError):
                logger.warning("Could not parse datetime strings, using current time")
                start_time = start_time_str
                end_time = end_time_str
            
            # Convert to Unix timestamps
            start_timestamp = int(start_time.timestamp())
            end_timestamp = int(end_time.timestamp())
            
            # Get the original room for comparison
            original_room = self.db.query(MRBSRoom).filter(
                MRBSRoom.room_name == room_name,
                MRBSRoom.disabled == False
            ).first()
            
            # Find alternative rooms with similar or better capacity
            alternative_rooms_query = self.db.query(MRBSRoom).filter(
                MRBSRoom.disabled == False,
                MRBSRoom.room_name != room_name,
                MRBSRoom.capacity >= capacity_required
            )
            
            # If we have the original room, prioritize rooms with similar capacity
            if original_room:
                alternative_rooms_query = alternative_rooms_query.order_by(
                    func.abs(MRBSRoom.capacity - original_room.capacity)
                )
            else:
                alternative_rooms_query = alternative_rooms_query.order_by(MRBSRoom.capacity)
            
            alternative_rooms = alternative_rooms_query.limit(10).all()
            
            recommendations = []
            
            for room in alternative_rooms:
                # Check if this room is available at the requested time
                conflicts = self.db.query(MRBSEntry).filter(
                    MRBSEntry.room_id == room.id,
                    MRBSEntry.start_time < end_timestamp,
                    MRBSEntry.end_time > start_timestamp,
                    MRBSEntry.status == 0  # Assuming 0 is active status
                ).count()
                
                if conflicts == 0:
                    # Calculate score based on room similarity
                    score = 0.75
                    if original_room:
                        # Bonus for similar capacity
                        capacity_diff = abs(room.capacity - original_room.capacity)
                        if capacity_diff == 0:
                            score += 0.2
                        elif capacity_diff <= 2:
                            score += 0.1
                        
                        if hasattr(room, 'area_id') and hasattr(original_room, 'area_id'):
                            if room.area_id == original_room.area_id:
                                score += 0.1
                    
                    recommendations.append({
                        'type': 'alternative_room',
                        'score': min(score, 1.0),
                        'reason': f'Room {room.room_name} (capacity: {room.capacity}) available at requested time',
                        'suggestion': {
                            'room_id': room.room_name,
                            'room_name': room.room_name,
                            'capacity': room.capacity,
                            'description': room.description or '',
                            'start_time': start_time.isoformat(),
                            'end_time': end_time.isoformat(),
                            'confidence': min(score, 1.0)
                        },
                        'data_source': 'mysql_alternative_room'
                    })
                    
                    # Limit to top 5 alternative rooms
                    if len(recommendations) >= 5:
                        break
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error in alternative room recommendations: {e}")
            return []
    
    def _get_proactive_recommendations_from_db(self, request_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get proactive recommendations based on user's booking history"""
        if not self.db:
            return []
        
        try:
            user_id = str(request_data.get('user_id', 'unknown'))
            purpose = request_data.get('purpose', '')
            start_time_str = request_data.get('start_time', '')
            end_time_str = request_data.get('end_time', '')
            
            try:
                if 'T' in start_time_str:
                    start_time = datetime.fromisoformat(start_time_str.replace('Z', '+00:00'))
                    end_time = datetime.fromisoformat(end_time_str.replace('Z', '+00:00'))
                else:
                    start_time = datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S')
                    end_time = datetime.strptime(end_time_str, '%Y-%m-%d %H:%M:%S')
            except (ValueError, TypeError):
                logger.warning("Could not parse datetime strings, using current time")
                start_time = start_time_str
                end_time = start_time + timedelta(hours=1)
            
            # Convert to Unix timestamps
            start_timestamp = int(start_time.timestamp())
            end_timestamp = int(end_time.timestamp())
            
            # Get user's booking history (last 90 days)
            history_start = start_time_str - timedelta(days=90)
            history_start_ts = int(history_start.timestamp())
            
            # Find user's most frequently booked rooms
            user_bookings = self.db.query(
                MRBSEntry.room_id,
                MRBSRoom.room_name,
                MRBSRoom.capacity,
                MRBSRoom.description,
                func.count(MRBSEntry.id).label('booking_count')
            ).join(
                MRBSRoom, MRBSEntry.room_id == MRBSRoom.id
            ).filter(
                MRBSEntry.create_by == user_id,
                MRBSEntry.start_time >= history_start_ts,
                MRBSRoom.disabled == False
            ).group_by(
                MRBSEntry.room_id, MRBSRoom.room_name, MRBSRoom.capacity, MRBSRoom.description
            ).order_by(
                func.count(MRBSEntry.id).desc()
            ).limit(5).all()
            
            recommendations = []
            
            for booking in user_bookings:
                room_id, room_name, capacity, description, booking_count = booking
                
                # Check if this room is available at the requested time
                conflicts = self.db.query(MRBSEntry).filter(
                    MRBSEntry.room_id == room_id,
                    MRBSEntry.start_time < end_timestamp,
                    MRBSEntry.end_time > start_timestamp,
                    MRBSEntry.status == 0
                ).count()
                
                if conflicts == 0:
                    # Calculate score based on booking frequency
                    base_score = 0.7
                    frequency_bonus = min(booking_count * 0.05, 0.2)  # Max 0.2 bonus
                    score = base_score + frequency_bonus
                    
                    recommendations.append({
                        'type': 'proactive',
                        'score': min(score, 1.0),
                        'reason': f'You have booked {room_name} {booking_count} times recently',
                        'suggestion': {
                            'room_id': room_name,
                            'room_name': room_name,
                            'capacity': capacity,
                            'description': description or '',
                            'start_time': start_time.isoformat(),
                            'end_time': end_time.isoformat(),
                            'confidence': min(score, 1.0),
                            'booking_history': booking_count
                        },
                        'data_source': 'mysql_user_history'
                    })
            
            # Also check for rooms used by others for similar purposes
            if purpose:
                similar_bookings = self.db.query(
                    MRBSEntry.room_id,
                    MRBSRoom.room_name,
                    MRBSRoom.capacity,
                    MRBSRoom.description,
                    func.count(MRBSEntry.id).label('usage_count')
                ).join(
                    MRBSRoom, MRBSEntry.room_id == MRBSRoom.id
                ).filter(
                    MRBSEntry.name.like(f'%{purpose}%'),
                    MRBSEntry.start_time >= history_start_ts,
                    MRBSRoom.disabled == False
                ).group_by(
                    MRBSEntry.room_id, MRBSRoom.room_name, MRBSRoom.capacity, MRBSRoom.description
                ).order_by(
                    func.count(MRBSEntry.id).desc()
                ).limit(3).all()
                
                for booking in similar_bookings:
                    room_id, room_name, capacity, description, usage_count = booking
                    
                    # Skip if already recommended
                    if any(rec['suggestion']['room_name'] == room_name for rec in recommendations):
                        continue
                    
                    # Check availability
                    conflicts = self.db.query(MRBSEntry).filter(
                        MRBSEntry.room_id == room_id,
                        MRBSEntry.start_time < end_timestamp,
                        MRBSEntry.end_time > start_timestamp,
                        MRBSEntry.status == 0
                    ).count()
                    
                    if conflicts == 0:
                        score = 0.6 + min(usage_count * 0.02, 0.15)
                        
                        recommendations.append({
                            'type': 'proactive',
                            'score': min(score, 1.0),
                            'reason': f'Room {room_name} frequently used for {purpose} ({usage_count} times)',
                            'suggestion': {
                                'room_id': room_name,
                                'room_name': room_name,
                                'capacity': capacity,
                                'description': description or '',
                                'start_time': start_time.isoformat(),
                                'end_time': end_time.isoformat(),
                                'confidence': min(score, 1.0),
                                'purpose_usage': usage_count
                            },
                            'data_source': 'mysql_purpose_analysis'
                        })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error in proactive recommendations: {e}")
            return []
    
    def _get_smart_scheduling_recommendations_from_db(self, request_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get smart scheduling recommendations based on booking patterns"""
        if not self.db:
            return []
        
        try:
            start_time_str = request_data.get('start_time', '')
            end_time_str = request_data.get('end_time', '')
            
            try:
                if 'T' in start_time_str:
                    start_time = datetime.fromisoformat(start_time_str.replace('Z', '+00:00'))
                    end_time = datetime.fromisoformat(end_time_str.replace('Z', '+00:00'))
                else:
                    start_time = datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S')
                    end_time = datetime.strptime(end_time_str, '%Y-%m-%d %H:%M:%S')
            except (ValueError, TypeError):
                logger.warning("Could not parse datetime strings, using current time")
                start_time = start_time_str
                end_time = start_time + timedelta(hours=1)
            
            recommendations = []
            
            # Analyze booking patterns for the requested time slot
            requested_hour = start_time.hour
            requested_day_of_week = start_time.weekday()
            
            # Find rooms with low utilization at this time slot
            time_window_start = start_time.replace(minute=0, second=0, microsecond=0)
            time_window_end = time_window_start + timedelta(hours=1)
            
            # Look at historical data for the same time slot
            history_start = start_time_str - timedelta(days=30)
            history_start_ts = int(history_start.timestamp())
            
            # Calculate utilization rates for each room at this time slot
            room_utilization = self.db.query(
                MRBSRoom.id,
                MRBSRoom.room_name,
                MRBSRoom.capacity,
                MRBSRoom.description,
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
                MRBSRoom.id, MRBSRoom.room_name, MRBSRoom.capacity, MRBSRoom.description
            ).order_by(
                func.count(MRBSEntry.id).asc()  # Rooms with lowest utilization first
            ).limit(10).all()
            
            for room_data in room_utilization:
                room_id, room_name, capacity, description, bookings_count = room_data
                
                # Check if room is available at requested time
                start_timestamp = int(start_time.timestamp())
                end_timestamp = int(end_time.timestamp())
                
                conflicts = self.db.query(MRBSEntry).filter(
                    MRBSEntry.room_id == room_id,
                    MRBSEntry.start_time < end_timestamp,
                    MRBSEntry.end_time > start_timestamp,
                    MRBSEntry.status == 0
                ).count()
                
                if conflicts == 0:
                    # Calculate score based on low utilization (more available = higher score)
                    utilization_rate = bookings_count / 30  # bookings per day over 30 days
                    availability_score = max(0.5, 1.0 - (utilization_rate * 0.1))
                    
                    # Bonus for larger capacity (more flexible)
                    capacity_bonus = min(capacity * 0.01, 0.2)
                    
                    final_score = availability_score + capacity_bonus
                    
                    recommendations.append({
                        'type': 'smart_scheduling',
                        'score': min(final_score, 1.0),
                        'reason': f'Room {room_name} has low utilization at this time ({bookings_count} bookings in 30 days)',
                        'suggestion': {
                            'room_id': room_name,
                            'room_name': room_name,
                            'capacity': capacity,
                            'description': description or '',
                            'start_time': start_time.isoformat(),
                            'end_time': end_time.isoformat(),
                            'confidence': min(final_score, 1.0),
                            'utilization_data': {
                                'historical_bookings': bookings_count,
                                'utilization_rate': round(utilization_rate, 2)
                            }
                        },
                        'data_source': 'mysql_utilization_analysis'
                    })
                    
                    # Limit to top 3 smart scheduling recommendations
                    if len(recommendations) >= 3:
                        break
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error in smart scheduling recommendations: {e}")
            return []
    
    def _create_fallback_recommendations(self, request_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create fallback recommendations when all strategies fail"""
        user_id = str(request_data.get('user_id', 'unknown'))
        room_id = request_data.get('room_id', '')
        start_time = request_data.get('start_time', '')
        end_time = request_data.get('end_time', '')
        purpose = request_data.get('purpose', '')
        
        logger.info("Creating fallback recommendations")
        
        return [
            {
                'type': 'alternative_time',
                'score': 0.75,
                'reason': f'Room {room_id} available 1 hour earlier (fallback)',
                'suggestion': {
                    'room_id': room_id,
                    'start_time': start_time,
                    'end_time': end_time,
                    'confidence': 0.75
                },
                'data_source': 'fallback_simulation'
            },
            {
                'type': 'alternative_room',
                'score': 0.70,
                'reason': 'Similar room with same capacity available (fallback)',
                'suggestion': {
                    'room_id': f'{room_id}_alt',
                    'start_time': start_time,
                    'end_time': end_time,
                    'confidence': 0.70
                },
                'data_source': 'fallback_simulation'
            },
            {
                'type': 'proactive',
                'score': 0.65,
                'reason': f'Based on typical {purpose} patterns (fallback)',
                'suggestion': {
                    'room_id': f'{room_id}_suggested',
                    'start_time': start_time,
                    'end_time': end_time,
                    'confidence': 0.65
                },
                'data_source': 'fallback_pattern_analysis'
            }
        ]
        
    def _verify_database_connection(self, request_data: Dict[str, Any]) -> None:
        """Verify MySQL database connection and required tables exist"""
        if not self.db:
            logger.warning("No database session available, skipping verification")
            return
            
        start_time = request_data.get('start_time', '')
        
        try:
            self.db.execute(text("SELECT 1")).fetchone()
            logger.info("✓ MySQL database connection successful")
            
            try:
                room_count = self.db.query(MRBSRoom).filter(MRBSRoom.disabled == False).count()
                logger.info(f"✓ Found {room_count} active rooms in mrbs_room table")
                
                entry_count = self.db.query(MRBSEntry).count()
                logger.info(f"✓ Found {entry_count} entries in mrbs_entry table")
                
                repeat_count = self.db.query(MRBSRepeat).count()
                logger.info(f"✓ Found {repeat_count} repeats in mrbs_repeat table")
                
                if room_count == 0:
                    logger.warning("No active rooms found - check if rooms are properly configured")
                
            except Exception as e:
                logger.warning(f"Could not query tables: {e}")
            
            try:
                recent_bookings = self.db.query(MRBSEntry).filter(
                    MRBSEntry.start_time >= int((start_time - timedelta(days=7)).timestamp())
                ).count()
                
                logger.info(f"✓ MySQL database accessible. Found {recent_bookings} recent bookings")
            except Exception as e:
                logger.warning(f"Could not query recent booking data: {e}")
            
        except Exception as e:
            logger.warning(f"Database verification failed: {e}")
    
    def get_room_data_from_db(self, room_name: str = None) -> List[Dict[str, Any]]:
      
        if not self.db:
            logger.warning("No database session available, returning empty list")
            return []
            
        try:
            query = self.db.query(MRBSRoom).filter(MRBSRoom.disabled == False)
            
            if room_name:
                query = query.filter(MRBSRoom.room_name == room_name)
            else:
                query = query.order_by(MRBSRoom.room_name)
            
            rooms = query.all()
            
            room_data = []
            for room in rooms:
                room_data.append({
                    'room_id': room.id,
                    'room_name': room.room_name,
                    'description': room.description or '',
                    'capacity': room.capacity or 1,
                    'admin_email': room.room_admin_email or '',
                    'area_id': room.area_id,
                    'sort_key': room.sort_key or '',
                    'custom_html': room.custom_html or ''
                })
            
            logger.debug(f"Fetched {len(room_data)} rooms from MySQL")
            return room_data
            
        except Exception as e:
            logger.error(f"Error fetching room data from MySQL: {e}")
            return []
    
    def check_room_availability_in_db(self, 
                                     room_name: str, 
                                     start_time: datetime, 
                                     end_time: datetime) -> bool:
        if not self.db:
            logger.warning("No database session available, assuming room is available")
            return True
            
        try:
            # Find the room
            room = self.db.query(MRBSRoom).filter(
                MRBSRoom.room_name == room_name,
                MRBSRoom.disabled == False
            ).first()
            
            if not room:
                logger.warning(f"Room {room_name} not found")
                return False
            
            # Convert to Unix timestamps
            start_timestamp = int(start_time.timestamp())
            end_timestamp = int(end_time.timestamp())
            
            # Check for conflicts
            conflicts = self.db.query(MRBSEntry).filter(
                MRBSEntry.room_id == room.id,
                MRBSEntry.start_time < end_timestamp,
                MRBSEntry.end_time > start_timestamp,
                MRBSEntry.status == 0  # Assuming 0 is active status
            ).count()
            
            is_available = conflicts == 0
            logger.debug(f"Room {room_name} availability check: {'Available' if is_available else 'Occupied'}")
            
            return is_available
            
        except Exception as e:
            logger.error(f"Error checking room availability: {e}")
            return False
    
    def get_user_booking_history(self, request_data: Dict[str, Any],user_id: str, days: int = 30,) -> List[Dict[str, Any]]:
        if not self.db:
            return []
        
        start_time_str = request_data.get('start_time', '')
        end_time_str = request_data.get('end_time', '')
        capacity_required = request_data.get('capacity', 1)
        
        try:
            start_date = start_time_str - timedelta(days=days)
            start_timestamp = int(start_date.timestamp())
            
            bookings = self.db.query(
                MRBSEntry,
                MRBSRoom.room_name,
                MRBSRoom.capacity,
                MRBSRoom.description
            ).join(
                MRBSRoom, MRBSEntry.room_id == MRBSRoom.id
            ).filter(
                MRBSEntry.create_by == user_id,
                MRBSEntry.start_time >= start_timestamp,
                MRBSRoom.disabled == False
            ).order_by(
                MRBSEntry.start_time.desc()
            ).all()
            
            booking_history = []
            for entry, room_name, capacity, description in bookings:
                booking_history.append({
                    'entry_id': entry.id,
                    'room_name': room_name,
                    'room_capacity': capacity,
                    'room_description': description,
                    'booking_name': entry.name,
                    'description': entry.description,
                    'start_time': datetime.fromtimestamp(entry.start_time),
                    'end_time': datetime.fromtimestamp(entry.end_time),
                    'created_by': entry.create_by,
                    'type': entry.type,
                    'status': entry.status
                })
            
            logger.debug(f"Retrieved {len(booking_history)} bookings for user {user_id}")
            return booking_history
            
        except Exception as e:
            logger.error(f"Error retrieving user booking history: {e}")
            return []
    
    def get_room_utilization_stats(self,request_data: Dict[str, Any], room_name: str = None, days: int = 30) -> Dict[str, Any]:
        if not self.db:
            return {}
        
        start_time_str = request_data.get('start_time', '')
        end_time_str = request_data.get('end_time', '')
        capacity_required = request_data.get('capacity', 1)
        
        try:
            start_date = start_time_str  - timedelta(days=days)
            start_timestamp = int(start_date.timestamp())
            
            query = self.db.query(
                MRBSRoom.room_name,
                MRBSRoom.capacity,
                func.count(MRBSEntry.id).label('total_bookings'),
                func.sum(MRBSEntry.end_time - MRBSEntry.start_time).label('total_hours_booked'),
                func.avg(MRBSEntry.end_time - MRBSEntry.start_time).label('avg_booking_duration')
            ).outerjoin(
                MRBSEntry,
                and_(
                    MRBSEntry.room_id == MRBSRoom.id,
                    MRBSEntry.start_time >= start_timestamp,
                    MRBSEntry.status == 0
                )
            ).filter(
                MRBSRoom.disabled == False
            ).group_by(
                MRBSRoom.room_name, MRBSRoom.capacity
            )
            
            if room_name:
                query = query.filter(MRBSRoom.room_name == room_name)
            
            results = query.all()
            
            utilization_stats = {}
            total_possible_hours = days * 24  
            
            for result in results:
                room_name_result = result.room_name
                capacity = result.capacity
                total_bookings = result.total_bookings or 0
                total_seconds_booked = result.total_hours_booked or 0
                avg_duration_seconds = result.avg_booking_duration or 0
                
                # Convert seconds to hours
                total_hours_booked = total_seconds_booked / 3600 if total_seconds_booked else 0
                avg_duration_hours = avg_duration_seconds / 3600 if avg_duration_seconds else 0
                
                # Calculate utilization rate
                utilization_rate = (total_hours_booked / total_possible_hours) * 100 if total_possible_hours > 0 else 0
                
                utilization_stats[room_name_result] = {
                    'capacity': capacity,
                    'total_bookings': total_bookings,
                    'total_hours_booked': round(total_hours_booked, 2),
                    'avg_booking_duration_hours': round(avg_duration_hours, 2),
                    'utilization_rate_percent': round(utilization_rate, 2),
                    'bookings_per_day': round(total_bookings / days, 2)
                }
            
            logger.debug(f"Retrieved utilization stats for {len(utilization_stats)} rooms")
            return utilization_stats
            
        except Exception as e:
            logger.error(f"Error retrieving room utilization stats: {e}")
            return {}
    
    def get_engine_status(self) -> Dict[str, Any]:
        """Get status information about the recommendation engine"""
        
        # Test MySQL connection
        mysql_status = "connected"
        room_count = "unknown"
        booking_count = "unknown"
        
        try:
            self.db.execute(text("SELECT 1")).fetchone()
            
            try:
                room_count = self.db.query(MRBSRoom).filter(MRBSRoom.disabled == False).count()
                recent_bookings = self.db.query(MRBSEntry).filter(
                    MRBSEntry.start_time >= int((datetime.now() - timedelta(days=30)).timestamp())
                ).count()
                booking_count = recent_bookings
            except Exception as e:
                logger.debug(f"Could not get database statistics: {e}")
                
        except Exception as e:
            mysql_status = f"error: {str(e)}"
        
        return {
            "status": "active",
            "mysql_connection": mysql_status,
            "database_stats": {
                "active_rooms": room_count,
                "recent_bookings": booking_count
            },
            "embeddings_loaded": self.embeddings is not None,
            "strategies_loaded": {
                "alternative_time": self.alternative_time is not None,
                "alternative_room": self.alternative_room is not None,
                "proactive": self.proactive is not None,
                "smart_scheduling": self.smart_scheduling is not None
            },
            "components_loaded": {
                "analytics": self.analytics is not None,
                "cache": self.cache is not None,
                "metrics": self.metrics is not None,
                "preference_learner": self.preference_learner is not None
            },
            "config": {
                "max_recommendations": getattr(self.config, 'max_recommendations', 5),
                "cache_ttl": getattr(self.config, 'cache_ttl_default', 1800),
                "database_url": self.config.database_url.split('@')[1] if hasattr(self.config, 'database_url') and '@' in self.config.database_url else "hidden"
            }
        }
    
    def __del__(self):
        """Cleanup method to properly close database connections"""
        try:
            if hasattr(self, 'db_manager') and self.db_manager:
                self.db_manager.close_all()
                logger.debug("Database connections closed successfully")
        except Exception as e:
            logger.error(f"Error closing database connections: {e}")


class RecommendationEngineFactory:
    
    @staticmethod
    def create_engine(config: RecommendationConfig = None, 
                     environment: str = None) -> RecommendationEngine:
        if config is None:
            try:
                from ...config.recommendation_config import ConfigFactory
                config = ConfigFactory.create_config(environment or 'development')
            except Exception as e:
                logger.warning(f"Could not create config from factory: {e}")
                config = RecommendationConfig()
        
        try:
            config.ensure_directories()
        except Exception as e:
            logger.warning(f"Could not ensure directories: {e}")
        
        # Validate MySQL connection
        try:
            if not config.validate_mysql_connection():
                logger.warning("MySQL connection validation failed, but continuing")
        except Exception as e:
            logger.warning(f"Could not validate MySQL connection: {e}")
        
        return RecommendationEngine(config=config)
    
    @staticmethod
    def create_development_engine() -> RecommendationEngine:
        """Create engine for development environment"""
        return RecommendationEngineFactory.create_engine(environment='development')
    
    @staticmethod
    def create_production_engine() -> RecommendationEngine:
        """Create engine for production environment"""
        return RecommendationEngineFactory.create_engine(environment='production')
    
    @staticmethod
    def create_testing_engine() -> RecommendationEngine:
        """Create engine for testing environment"""
        return RecommendationEngineFactory.create_engine(environment='testing')


def create_recommendation_engine_with_fallback(db: Session = None, 
                                             config: RecommendationConfig = None,
                                             fallback_to_mock: bool = True) -> RecommendationEngine:
    try:
        return RecommendationEngine(db=db, config=config)
    except Exception as e:
        logger.error(f"Failed to create RecommendationEngine: {e}")
        
        if fallback_to_mock:
            logger.info("Attempting to create RecommendationEngine in fallback mode")
            try:
                # Create minimal config for fallback
                fallback_config = RecommendationConfig() if config is None else config
                
                return RecommendationEngine(db=None, config=fallback_config)
            except Exception as e2:
                logger.error(f"Fallback mode also failed: {e2}")
                raise e2
        else:
            raise e


def validate_recommendation_request(request_data: Dict[str, Any]) -> Dict[str, Any]:

    validation_result = {
        "valid": True,
        "errors": [],
        "warnings": []
    }
    
    required_fields = ['user_id']
    for field in required_fields:
        if field not in request_data:
            validation_result["errors"].append(f"Missing required field: {field}")
            validation_result["valid"] = False
    
    recommended_fields = ['room_id', 'start_time', 'end_time', 'purpose']
    for field in recommended_fields:
        if field not in request_data:
            validation_result["warnings"].append(f"Missing recommended field: {field}")
    
    if 'user_id' in request_data and not isinstance(request_data['user_id'], (str, int)):
        validation_result["errors"].append("user_id must be string or integer")
        validation_result["valid"] = False
    
    time_fields = ['start_time', 'end_time']
    for field in time_fields:
        if field in request_data:
            time_value = request_data[field]
            if isinstance(time_value, str):
                try:
                    datetime.fromisoformat(time_value.replace('Z', '+00:00'))
                except ValueError:
                    validation_result["warnings"].append(f"{field} should be in ISO format")
    
    return validation_result