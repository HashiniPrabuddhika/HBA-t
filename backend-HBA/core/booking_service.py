from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
from fastapi import HTTPException
import time

from models.booking import MRBSEntry
from models.room import MRBSRoom
from models.user import MRBSUser, MRBSModule
from core.validation_service import ValidationService
from utils.logger import get_logger
# from services.recommendation.hybrid_engine import HybridRecommendationEngine
from services.recommendations.core.hybridRecommendations import hybridRecommendationsEngine as HybridRecommendationEngine
from config.recommendation_config import RecommendationConfig 

logger = get_logger(__name__)

config = RecommendationConfig()

class BookingService:
    
    def __init__(self, db: Session, recommendation_engine=None):
        self.db = db
        self.recommendation_engine = recommendation_engine 
        self.validator = ValidationService()
        try:
            self.recommendation_engine = HybridRecommendationEngine(config=config)
            logger.info("Recommendation engine initialized successfully")
        except Exception as e:
            logger.warning(f"Recommendation engine initialization failed: {e}")
            self.recommendation_engine = None
    
    def check_availability(self, room_name: str, date: str, start_time: str, 
                      end_time: str) -> Dict[str, Any]:
        logger.info(f"Checking availability: {room_name} on {date} {start_time}-{end_time}")
        
        room = self.db.query(MRBSRoom).filter(MRBSRoom.room_name == room_name).first()
        
        if not room:
            recommendations = self._get_recommendations(room_name, date, start_time, end_time)
            return {
                "status": "room_not_found",
                "message": f"Room '{room_name}' not found.",
                "recommendations": recommendations
            }
        
        start_dt = datetime.strptime(f"{date} {start_time}", "%Y-%m-%d %H:%M")
        end_dt = datetime.strptime(f"{date} {end_time}", "%Y-%m-%d %H:%M")
        
        start_ts = int(time.mktime(start_dt.timetuple()))
        end_ts = int(time.mktime(end_dt.timetuple()))
        
        conflicting = self.db.query(MRBSEntry).filter(
            MRBSEntry.room_id == room.id,
            MRBSEntry.start_time < end_ts,
            MRBSEntry.end_time > start_ts,
        ).first()
        
        if conflicting:
            recommendations = self._get_recommendations(room_name, date, start_time, end_time)
            return {
                "status": "unavailable",
                "message": f"{room_name} is already booked for that time. Here are some available alternatives:",
                "recommendations": recommendations  
            }
        
        return {
            "status": "available",
            "message": f"{room_name} is available from {start_time} to {end_time} on {date}."
        }
    
    
    def add_booking(self, room_name: str, name: str, date: str, start_time: str, 
                   end_time: str, created_by: str) -> Dict[str, Any]:
        logger.info(f"Creating booking: {room_name} on {date} for {created_by}")
        
        try:
            self.validator.validate_future_datetime(date, start_time, "book")
        
            room = self.db.query(MRBSRoom).filter(MRBSRoom.room_name == room_name).first()
        
            if not room:
                recommendations = self._get_recommendations(room_name, date, start_time, end_time)
                raise HTTPException(
                    status_code=404,
                    detail={
                        "error": "Room not found",
                        "message": f"Room '{room_name}' not found.",
                        "recommendations": recommendations
                }
            )
        
            start_dt = datetime.strptime(f"{date} {start_time}", "%Y-%m-%d %H:%M")
            end_dt = datetime.strptime(f"{date} {end_time}", "%Y-%m-%d %H:%M")
        
            start_ts = int(time.mktime(start_dt.timetuple()))
            end_ts = int(time.mktime(end_dt.timetuple()))
        
            if end_ts <= start_ts:
                raise HTTPException(status_code=400, detail="End time must be after start time")
            
            conflict = self.db.query(MRBSEntry).filter(
                MRBSEntry.room_id == room.id,
                MRBSEntry.start_time < end_ts,
                MRBSEntry.end_time > start_ts,
            ).first()
            
            if conflict:
                recommendations = self._get_recommendations(room_name, date, start_time, end_time)
                return {
                    "status": "unavailable",
                    "message": f"Room '{room_name}' is already booked. Here are alternatives:",
                    "recommendations": recommendations
                }
            
            current_datetime = datetime.now()
            
            try:
                new_booking = MRBSEntry(
                    start_time=start_ts,
                    end_time=end_ts,
                    entry_type=0,
                    repeat_id=None,
                    room_id=room.id,
                    timestamp=current_datetime,
                    create_by=created_by,
                    modified_by=created_by,
                    name=name,
                    type='E',
                    description=f"Booked by {created_by}",
                    status=0,
                    reminded=None,
                    info_time=None,
                    info_user=None,
                    info_text=None,
                    ical_uid=f"{room_name}_{start_ts}_{end_ts}",
                    ical_sequence=0,
                    ical_recur_id=None
                )
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error creating booking object: {e}")
        
            
            try:
                self.db.add(new_booking)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error adding booking to session: {e}")
            
            try:
                self.db.commit()
            except Exception as e:
                self.db.rollback()
                raise HTTPException(status_code=500, detail=f"Database commit failed: {e}")
            
            try:
               self.db.refresh(new_booking)
            
            except Exception as e:
                pass
            logger.info(f"Booking created successfully: ID {new_booking.id}")
            
            return {
                "message": "Booking created successfully",
                "booking_id": new_booking.id,
                "room": room_name,
                "date": date,
                "start_time": start_time,
                "end_time": end_time,
                "created_by": created_by
            }
        
        except HTTPException:
            raise
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")
        
    
    def update_booking(original_room_name: str, original_date: str, original_start_time: str, 
                    original_end_time: str, new_room_name: str = None, new_date: str = None,
                    new_start_time: str = None, new_end_time: str = None, 
                    modified_by: str = "system", db: Session = None):
        try:
            room = db.query(MRBSRoom).filter(MRBSRoom.room_name == original_room_name).first()
            if not room:
                return {"status": "room_not_found", "message": f"Room '{original_room_name}' not found."}
            
            start_dt = datetime.strptime(f"{original_date} {original_start_time}", "%Y-%m-%d %H:%M")
            end_dt = datetime.strptime(f"{original_date} {original_end_time}", "%Y-%m-%d %H:%M")
            start_ts, end_ts = int(time.mktime(start_dt.timetuple())), int(time.mktime(end_dt.timetuple()))
            
            booking = db.query(MRBSEntry).filter(
                MRBSEntry.room_id == room.id,
                MRBSEntry.start_time == start_ts,
                MRBSEntry.end_time == end_ts
            ).first()
            
            if not booking:
                return {"status": "booking_not_found", 
                    "message": f"No booking found for {original_room_name} on {original_date} from {original_start_time} to {original_end_time}."}
            
            if booking.create_by != modified_by:
                raise HTTPException(
                    status_code=403, 
                    detail=f"Access denied. Only the booking creator ({booking.create_by}) can update this booking."
                )
            
            final_room_name = new_room_name or original_room_name
            final_date = new_date or original_date
            final_start_time = new_start_time or original_start_time
            final_end_time = new_end_time or original_end_time
            
            final_room_id = room.id
            if new_room_name and new_room_name != original_room_name:
                new_room = db.query(MRBSRoom).filter(MRBSRoom.room_name == new_room_name).first()
                if not new_room:
                    return {"status": "new_room_not_found", "message": f"New room '{new_room_name}' not found."}
                final_room_id = new_room.id
            
            final_start_dt = datetime.strptime(f"{final_date} {final_start_time}", "%Y-%m-%d %H:%M")
            final_end_dt = datetime.strptime(f"{final_date} {final_end_time}", "%Y-%m-%d %H:%M")
            final_start_ts, final_end_ts = int(time.mktime(final_start_dt.timetuple())), int(time.mktime(final_end_dt.timetuple()))
            
            if final_end_ts <= final_start_ts:
                return {"status": "invalid_time", "message": "End time must be after start time."}
            
            if (final_room_id != room.id or final_start_ts != start_ts or final_end_ts != end_ts):
                conflict = db.query(MRBSEntry).filter(
                    MRBSEntry.room_id == final_room_id,
                    MRBSEntry.start_time < final_end_ts,
                    MRBSEntry.end_time > final_start_ts,
                    MRBSEntry.id != booking.id
                ).first()
                
                if conflict:
                    return {"status": "unavailable", "message": "The new time slot is not available."}
            
            booking.room_id = final_room_id
            booking.start_time = final_start_ts
            booking.end_time = final_end_ts
            booking.modified_by = modified_by
            booking.timestamp = datetime.now()
            
            db.commit()
            
            return {
                "status": "success",
                "message": "Booking updated successfully",
                "booking_id": booking.id,
                "original": {"room": original_room_name, "date": original_date, "start_time": original_start_time, "end_time": original_end_time},
                "updated": {"room": final_room_name, "date": final_date, "start_time": final_start_time, "end_time": final_end_time},
                "modified_by": modified_by
            }
            
        except HTTPException:
            raise
        except Exception as e:
            db.rollback()
            raise HTTPException(status_code=500, detail=f"Error updating booking: {e}")

    
    def delete_booking(self, booking_id: int) -> Dict[str, Any]:
        try:
            logger.info(f"Deleting booking: ID {booking_id}")
            
            booking = self.db.query(MRBSEntry).filter(MRBSEntry.id == booking_id).first()
            if not booking:
                raise HTTPException(status_code=404, detail="Booking not found")
            
            self.db.delete(booking)
            self.db.commit()
            
            logger.info(f"Booking deleted successfully: ID {booking_id}")
            
            return {"status": "success", "message": "Booking deleted successfully"}
        
        except Exception as e:
            raise HTTPException(status_code=500, detail="Internal server error")
            
    def cancel_booking(self, room_name: str, date: str, start_time: str, 
                      end_time: str, user_email: str) -> Dict[str, Any]:
        """Cancel a booking with authorization check"""
        logger.info(f"Canceling booking: {room_name} on {date} by {user_email}")
        
        try:
            room = self.db.query(MRBSRoom).filter(MRBSRoom.room_name == room_name).first()
            
            if not room:
                return {
                    "status": "room_not_found",
                    "message": f"Room '{room_name}' not found."
                }
            
            start_dt = datetime.strptime(f"{date} {start_time}", "%Y-%m-%d %H:%M")
            end_dt = datetime.strptime(f"{date} {end_time}", "%Y-%m-%d %H:%M")
            start_ts = int(time.mktime(start_dt.timetuple()))
            end_ts = int(time.mktime(end_dt.timetuple()))
            
            booking = self.db.query(MRBSEntry).filter(
                MRBSEntry.room_id == room.id,
                MRBSEntry.start_time == start_ts,
                MRBSEntry.end_time == end_ts
            ).first()
            
            if not booking:
                return {
                    "status": "no_booking_found",
                    "message": f"No booking found for {room_name} on {date} from {start_time} to {end_time}."
                }
            
            if booking.create_by != user_email:
                raise HTTPException(
                    status_code=403,
                    detail=f"Access denied. Only the booking creator ({booking.create_by}) can cancel this booking."
                )
            
            self.db.delete(booking)
            self.db.commit()
            
            logger.info(f"Booking canceled successfully: ID {booking.id}")
            
            return {
                "status": "success",
                "message": f"Successfully cancelled booking for {room_name} on {date}.",
                "cancelled_booking_id": booking.id
            }
        except HTTPException:
            raise
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid date/time format: {e}")
        except Exception as e:
            self.db.rollback()
            raise HTTPException(status_code=500, detail=f"Error cancelling booking: {e}")

    
    def get_available_slots(self, room_name: str, date: str) -> Dict[str, Any]:
        """Get all available 30-minute time slots for a room on a given date"""
        logger.info(f"Getting available slots: {room_name} on {date}")
        
        self.validator.validate_future_datetime(date, "00:00", "check available slots")
        
        room = self.db.query(MRBSRoom).filter(MRBSRoom.room_name == room_name).first()
        
        if not room:
            raise HTTPException(status_code=404, detail=f"Room '{room_name}' not found.")
        
        date_obj = datetime.strptime(date, "%Y-%m-%d")
        start_time = datetime.combine(date_obj, datetime.min.time()) + timedelta(hours=7)
        end_time = datetime.combine(date_obj, datetime.min.time()) + timedelta(hours=21)
        
        all_slots = []
        current = start_time
        while current < end_time:
            slot_start = current
            slot_end = current + timedelta(minutes=30)
            all_slots.append((int(time.mktime(slot_start.timetuple())), int(time.mktime(slot_end.timetuple()))))
            current = slot_end
        
        day_start_ts = int(time.mktime(start_time.timetuple()))
        day_end_ts = int(time.mktime(end_time.timetuple()))
        
        bookings = self.db.query(MRBSEntry).filter(
            MRBSEntry.room_id == room.id,
            MRBSEntry.start_time < day_end_ts,
            MRBSEntry.end_time > day_start_ts
        ).all()
        
        available_slots = []
        current_time = datetime.now()
        
        for slot_start, slot_end in all_slots:
            slot_datetime = datetime.fromtimestamp(slot_start)
            
            if slot_datetime <= current_time:
                continue
            
            conflict = any(
                booking.start_time < slot_end and booking.end_time > slot_start
                for booking in bookings
            )
            
            if not conflict:
                available_slots.append({
                    "start_time": datetime.fromtimestamp(slot_start).strftime("%H:%M"),
                    "end_time": datetime.fromtimestamp(slot_end).strftime("%H:%M")
                })
        
        if not available_slots:
            recommendations = self._get_recommendations(room_name, date, "09:00", "17:00")
            return {
                "status": "no_slots_available",
                "message": f"No available time slots found for {room_name} on {date}.",
                "room": room_name,
                "date": date,
                "available_slots": [],
                "recommendations": recommendations
            }
        
        return {
            "room": room_name,
            "date": date,
            "available_slots": available_slots
        }
    
    def get_bookings_by_date_and_room(self, date: str, room_id: int) -> List[Dict[str, Any]]:
        logger.info(f"Fetching bookings for room {room_id} on {date}")
        
        try:
            day_start = datetime.strptime(date, "%Y-%m-%d")
            next_day = day_start + timedelta(days=1)
            
            start_ts = int(time.mktime(day_start.timetuple()))
            next_day_ts = int(time.mktime(next_day.timetuple()))
            
            bookings = (
                self.db.query(MRBSEntry)
                .filter(
                    MRBSEntry.start_time < next_day_ts,
                    MRBSEntry.end_time > start_ts,
                    MRBSEntry.room_id == room_id
                )
                .order_by(MRBSEntry.start_time.asc())
                .all()
            )
            
            return [
                {
                    "id": b.id,
                    "room_id": b.room_id,
                    "name": b.name,
                    "date": datetime.fromtimestamp(b.start_time).strftime("%Y-%m-%d"),
                    "start_time": datetime.fromtimestamp(b.start_time).strftime("%H:%M"),
                    "end_time": datetime.fromtimestamp(b.end_time).strftime("%H:%M"),
                    "created_by": b.create_by,
                    "status": b.status,
                }
                for b in bookings
            ]
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid date format (expected YYYY-MM-DD): {e}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error fetching bookings by date: {e}")

    
    def _get_recommendations(self, room_name: str, date: str, start_time: str, 
                        end_time: str) -> List[Dict[str, Any]]:
        
        if not self.recommendation_engine:
            logger.warning("Recommendation engine not available")
            return []
        
        try:
            start_dt = datetime.strptime(f"{date} {start_time}", "%Y-%m-%d %H:%M")
            end_dt = datetime.strptime(f"{date} {end_time}", "%Y-%m-%d %H:%M")
            
            request_data = {
                "user_id": "system",
                "room_id": room_name,
                "start_time": start_dt,  
                "end_time": end_dt,      
                "date": date,
                "purpose": "meeting",
                "capacity": 1,
                "requirements": {"original_room": room_name}
            }
            
            recommendations = self.recommendation_engine.get_recommendations(request_data)
            logger.info(f"Generated {len(recommendations)} recommendations")
            return recommendations
            
        except Exception as e:
            logger.error(f"Recommendation system error: {e}")
            return []
 

def fetch_user_profile_by_email(email: str, db: Session):
    user = db.query(MRBSUser).filter(MRBSUser.email == email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return {
        "id": user.id,
        "name": user.name,
        "email": user.email,
        "role": user.role
    }
    

def fetch_booking_by_id(booking_id: int, db: Session):
    try:
        booking = db.query(MRBSEntry).filter(MRBSEntry.id == booking_id).first()
        if not booking:
            raise HTTPException(status_code=404, detail="Booking not found")
        return booking
    except Exception as e:
        print(f"Error fetching booking by ID: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
    
def check_available_slotes(self, room_name: str, date: str, start_time: str, end_time: str, db: Session):
    
    self.validator.validate_future_datetime(date, "00:00", "check available slots")
    
    print(f"Checking availability for room: {room_name}")
    print(f"Date: {date}, Start time: {start_time}, End time: {end_time}")

    room = db.query(MRBSRoom).filter(MRBSRoom.room_name == room_name).first()
    print(f"Queried room from DB: {room}")

    # Convert to datetime objects first
    date_obj = datetime.strptime(date, "%Y-%m-%d")
    start_time = datetime.combine(date_obj, datetime.min.time()) + timedelta(hours=7)  # 7 AM
    end_time = datetime.combine(date_obj, datetime.min.time()) + timedelta(hours=21)  # 9 PM


    all_slots = []
    current = start_time
    while current < end_time:
        slot_start = current
        slot_end = current + timedelta(minutes=30)
        all_slots.append((int(time.mktime(slot_start.timetuple())), int(time.mktime(slot_end.timetuple()))))
        current = slot_end

    # Step 3: Get all bookings for that day and room
    day_start_ts = int(time.mktime(start_time.timetuple()))
    day_end_ts = int(time.mktime(end_time.timetuple()))

    bookings = db.query(MRBSEntry).filter(
        MRBSEntry.room_id == room.id,
        MRBSEntry.start_time < day_end_ts,
        MRBSEntry.end_time > day_start_ts
    ).all()

    available_slots = []
    current_time = datetime.now()
    
    for slot_start, slot_end in all_slots:
        
        slot_datetime = datetime.fromtimestamp(slot_start)
        
        # Skip past slots
        if slot_datetime <= current_time:
            continue
        
        conflict = any(
            booking.start_time < slot_end and booking.end_time > slot_start
            for booking in bookings
        )
        if not conflict:
            available_slots.append({
                "start_time": datetime.fromtimestamp(slot_start).strftime("%H:%M"),
                "end_time": datetime.fromtimestamp(slot_end).strftime("%H:%M")
            })
            
    if not available_slots:
        recommendations = self._get_recommendations(room_name, date, start_time, end_time, db)
        return {
            "status": "no_slots_available",
            "message": f"No available time slots found for {room_name} on {date}. Here are some available alternatives you might like:",
            "room": room_name,
            "date": date,
            "available_slots": [],
            "recommendations": recommendations
        } 
            
    if not room:
        print("Room not found!")
        recommendations = self._get_recommendations(room_name, date, start_time, end_time, db)
        raise HTTPException(
            status_code=404, 
            detail={
                "error": "Room not found",
                "message": f"Room '{room_name}' not found.",
                "recommendations": recommendations
            }
        )
    
    return {"room": room_name, "date": date, "available_slots": available_slots}

def fetch_moduleCodes_by_user_email(email: str, db: Session):
    user = db.query(MRBSUser).filter(MRBSUser.email == email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    modules = db.query(MRBSModule).filter(MRBSModule.lecture_id == user.id).all()
    return [module.module_code for module in modules]

def fetch_all_halls(db: Session):
    halls = db.query(MRBSRoom).all()
    return [hall.room_name for hall in halls]


def fetch_halls_by_module_code(module_code: str, db: Session):
    module = db.query(MRBSModule).filter(MRBSModule.module_code == module_code).first()
    if not module:
        return []  
    halls = db.query(MRBSRoom).filter(MRBSRoom.capacity >= module.number_of_students).all()
    
    return [hall.room_name for hall in halls]

