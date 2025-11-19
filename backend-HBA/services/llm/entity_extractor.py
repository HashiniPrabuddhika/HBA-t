import spacy
import re
from dateutil.parser import parse as parse_date
from datetime import datetime
from typing import Dict, List, Set, Optional
from functools import lru_cache
from sqlalchemy.orm import Session

from models.room import MRBSRoom
from utils.logger import get_logger

logger = get_logger(__name__)

nlp = spacy.load("en_core_web_sm")

KNOWN_ROOMS = {"LT1", "LT2", "MainHall", "Lab1", "Auditorium"}

DELETE_KEYWORDS = {"delete", "remove", "cancel", "unbook", "clear", "drop", "eliminate"}

class EntityExtractor:
    """Service for extracting entities from natural language text"""
    
    _room_cache: Optional[Set[str]] = None
    _cache_timestamp: Optional[datetime] = None
    _cache_ttl_seconds: int = 300  # 5 minutes
    
    @classmethod
    def _load_rooms_from_db(cls, db: Session) -> Set[str]:
        """
        Load all active room names from database
        
        Args:
            db: Database session
            
        Returns:
            Set of room names
        """
        try:
            rooms = db.query(MRBSRoom).filter(MRBSRoom.disabled == False).all()
            room_names = {room.room_name for room in rooms}
            logger.info(f"Loaded {len(room_names)} rooms from database")
            return room_names
        except Exception as e:
            logger.error(f"Failed to load rooms from database: {e}")
            return {"LT1", "LT2", "MainHall", "Lab1", "Auditorium"}
    
    @classmethod
    def _get_known_rooms(cls, db: Optional[Session] = None) -> Set[str]:
        """
        Get known rooms with caching
        
        Args:
            db: Optional database session
            
        Returns:
            Set of room names
        """
        now = datetime.now()
        
        # Check if cache is valid
        if (cls._room_cache is not None and 
            cls._cache_timestamp is not None and 
            (now - cls._cache_timestamp).total_seconds() < cls._cache_ttl_seconds):
            return cls._room_cache
        
        # Refresh cache
        if db is not None:
            cls._room_cache = cls._load_rooms_from_db(db)
            cls._cache_timestamp = now
            return cls._room_cache
        
        # Fallback to default rooms if no DB session
        logger.warning("No database session provided, using default room list")
        return {"LT1", "LT2", "MainHall", "Lab1", "Auditorium"}
    
    @classmethod
    def refresh_room_cache(cls, db: Session) -> None:
        """
        Force refresh of room cache
        
        Args:
            db: Database session
        """
        cls._room_cache = cls._load_rooms_from_db(db)
        cls._cache_timestamp = datetime.now()
        logger.info("Room cache refreshed")
    
    @staticmethod
    def extract_time(text: str) -> List[str]:
       
        time_matches = re.findall(r'\b\d{1,2}[:.]?\d{0,2}\s?(?:am|pm|AM|PM)?\b', text)
        times = []
        
        for t in time_matches:
            try:
                if ':' in t or '.' in t:
                    parsed = datetime.strptime(t.replace(".", ":"), "%I:%M %p")
                else:
                    parsed = datetime.strptime(t, "%I %p")
                times.append(parsed.strftime("%H:%M"))
            except ValueError:
                try:
                    parsed = parse_date(t)
                    times.append(parsed.strftime("%H:%M"))
                except Exception as e:
                    logger.debug(f"Failed to parse time '{t}': {e}")
                    continue
        
        return times[:2]
    
    @classmethod
    def extract_entities(cls, text: str, db: Optional[Session] = None) -> Dict[str, str]:
       
        doc = nlp(text)
        entities = {}
        
        # Get known rooms (from DB or cache)
        known_rooms = cls._get_known_rooms(db)
        
        # Extract room name
        text_upper = text.upper()
        for room in known_rooms:
            # Case-insensitive exact match
            if room.upper() in text_upper:
                entities["room_name"] = room
                logger.debug(f"Extracted room name: {room}")
                break
        
        # Alternative: check tokens
        if "room_name" not in entities:
            for token in doc:
                if token.text in known_rooms:
                    entities["room_name"] = token.text
                    logger.debug(f"Extracted room name from token: {token.text}")
                    break
        
        # Extract date
        for ent in doc.ents:
            if ent.label_ == "DATE":
                try:
                    parsed_date = parse_date(ent.text)
                    entities["date"] = parsed_date.strftime("%Y-%m-%d")
                    logger.debug(f"Extracted date: {entities['date']}")
                    break
                except Exception as e:
                    logger.debug(f"Failed to parse date '{ent.text}': {e}")
                    continue
        
        # Extract start_time and end_time
        times = cls.extract_time(text)
        if len(times) >= 1:
            entities["start_time"] = times[0]
            logger.debug(f"Extracted start_time: {times[0]}")
        if len(times) >= 2:
            entities["end_time"] = times[1]
            logger.debug(f"Extracted end_time: {times[1]}")
        
        return entities


def extract_entities(text: str, db: Optional[Session] = None) -> Dict[str, str]:
 
    return EntityExtractor.extract_entities(text, db)


def refresh_room_cache(db: Session) -> None:
  
    EntityExtractor.refresh_room_cache(db)