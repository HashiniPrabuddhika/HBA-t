from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from ..models.embedding_model import EmbeddingModel
from ..data.analytics_processor import AnalyticsProcessor
from src.models import MRBSRoom, MRBSEntry
from datetime import datetime
import numpy as np

class AlternativeRoomStrategy:
    def __init__(self, db: Session):
        self.db = db
        self.analytics = AnalyticsProcessor(db)
        self.embedding_model = EmbeddingModel()
    
    async def find_similar_rooms(self, target_room: str, date: str, start_time: str, end_time: str, room_features: Dict[str, Any], user_preferences: Dict[str, Any]) -> List[Dict[str, Any]]:
        target_room_obj = self.db.query(MRBSRoom).filter(MRBSRoom.room_name == target_room).first()
        if not target_room_obj: return []
        
        available_rooms = await self._get_available_rooms(date, start_time, end_time)
        
        alternatives = []
        alternatives.extend(await self._find_embedding_similar_rooms(target_room, available_rooms, room_features))
        alternatives.extend(self._find_capacity_similar_rooms(target_room_obj, available_rooms, room_features))
        alternatives.extend(await self._find_preference_based_rooms(available_rooms, user_preferences))
        alternatives.extend(self._find_location_similar_rooms(target_room_obj, available_rooms))
        
        # Remove duplicates and sort
        seen = set()
        unique = [alt for alt in alternatives if not (alt['room_name'] in seen or seen.add(alt['room_name']))]
        return sorted(unique, key=lambda x: x['confidence_score'], reverse=True)[:8]
    
    async def _get_available_rooms(self, date: str, start_time: str, end_time: str) -> List[MRBSRoom]:
        start_dt = datetime.strptime(f"{date} {start_time}", "%Y-%m-%d %H:%M")
        end_dt = datetime.strptime(f"{date} {end_time}", "%Y-%m-%d %H:%M")
        start_ts, end_ts = int(start_dt.timestamp()), int(end_dt.timestamp())
        
        booked_ids = [r[0] for r in self.db.query(MRBSEntry.room_id).filter(MRBSEntry.start_time < end_ts, MRBSEntry.end_time > start_ts).distinct().all()]
        return self.db.query(MRBSRoom).filter(~MRBSRoom.id.in_(booked_ids), MRBSRoom.disabled == False).all()
    
    async def _find_embedding_similar_rooms(self, target_room: str, available_rooms: List[MRBSRoom], room_features: Dict[str, Any]) -> List[Dict[str, Any]]:
        alternatives = []
        target_room_obj = next((r for r in available_rooms if r.room_name == target_room), None)
        
        if not target_room_obj:
            similar_rooms = self.embedding_model.find_similar_rooms(target_room, n_results=10)
            for similar in similar_rooms:
                room_obj = next((r for r in available_rooms if str(r.id) == similar['room_id']), None)
                if room_obj:
                    alternatives.append({
                        'room_name': room_obj.room_name,
                        'room_id': room_obj.id,
                        'confidence_score': similar['similarity_score'],
                        'reason': 'Similar features and characteristics',
                        'capacity': room_obj.capacity,
                        'description': room_obj.description or 'No description available'
                    })
        return alternatives
    
    def _find_capacity_similar_rooms(self, target_room: MRBSRoom, available_rooms: List[MRBSRoom], room_features: Dict[str, Any]) -> List[Dict[str, Any]]:
        alternatives = []
        target_capacity = target_room.capacity
        
        for room in available_rooms:
            if room.id == target_room.id: continue
            
            capacity_diff = abs(room.capacity - target_capacity)
            if capacity_diff == 0:
                confidence, reason = 0.9, 'Same capacity'
            elif capacity_diff <= target_capacity * 0.2:
                confidence, reason = 0.8, 'Similar capacity'
            elif capacity_diff <= target_capacity * 0.5:
                confidence, reason = 0.6, 'Comparable capacity'
            else:
                confidence, reason = 0.3, 'Different capacity but available'
            
            alternatives.append({
                'room_name': room.room_name, 'room_id': room.id,
                'confidence_score': confidence, 'reason': reason,
                'capacity': room.capacity, 'description': room.description or 'No description available'
            })
        return alternatives
    
    async def _find_preference_based_rooms(self, available_rooms: List[MRBSRoom], user_preferences: Dict[str, Any]) -> List[Dict[str, Any]]:
        alternatives = []
        preferred_rooms = user_preferences.get('preferred_rooms', [])
        
        for room in available_rooms:
            if room.room_name in preferred_rooms:
                preference_rank = preferred_rooms.index(room.room_name)
                confidence = max(0.5, 0.9 - (preference_rank * 0.1))
                
                alternatives.append({
                    'room_name': room.room_name, 'room_id': room.id,
                    'confidence_score': confidence, 'reason': 'You have booked this room before',
                    'capacity': room.capacity, 'description': room.description or 'No description available'
                })
        return alternatives
    
    def _find_location_similar_rooms(self, target_room: MRBSRoom, available_rooms: List[MRBSRoom]) -> List[Dict[str, Any]]:
        return [{
            'room_name': room.room_name, 'room_id': room.id,
            'confidence_score': 0.7, 'reason': 'Same building/area',
            'capacity': room.capacity, 'description': room.description or 'No description available'
        } for room in available_rooms if room.id != target_room.id and room.area_id == target_room.area_id]