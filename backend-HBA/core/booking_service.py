from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
from fastapi import HTTPException
import time

from models.booking import MRBSEntry
from models.room import MRBSRoom
from core.validation_service import ValidationService
from utils.logger import get_logger
from services.recommendation.hybrid_engine import HybridRecommendationEngine

logger = get_logger(__name__)


class BookingService:
    
    def __init__(self, db: Session, recommendation_engine=None):
        self.db = db
        self.recommendation_engine = recommendation_engine
        self.validator = ValidationService()
        try:
            self.recommendation_engine = HybridRecommendationEngine(db)
            logger.info("Recommendation engine initialized successfully")
        except Exception as e:
            logger.warning(f"Recommendation engine initialization failed: {e}")
            self.recommendation_engine = None
    
    def check_availability(self, room_name: str, date: str, start_time: str, 
                          end_time: str) -> Dict[str, Any]:
        logger.info(f"Checking availability: {room_name} on {date} {start_time}-{end_time}")
        
        room = self.db.query(MRBSRoom).filter(MRBSRoom.room_name == room_name).first()
        
        if not room:
            recommendations = self._get_recommendations(
                room_name="any",  
                date=date,
                start_time=start_time,
                end_time=end_time
            )
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
        
    
    def update_booking(self, booking_id: int, room_id: int, name: str, date: str,
                      start_timestamp: int, end_timestamp: int, modified_by: str) -> Dict[str, Any]:
        """Update an existing booking"""
        logger.info(f"Updating booking: ID {booking_id}")
        
        start_datetime = datetime.fromtimestamp(start_timestamp)
        self.validator.validate_future_datetime(
            start_datetime.strftime("%Y-%m-%d"),
            start_datetime.strftime("%H:%M"),
            "update booking"
        )
        
        booking = self.db.query(MRBSEntry).filter(MRBSEntry.id == booking_id).first()
        if not booking:
            raise HTTPException(status_code=404, detail="Booking not found")
        
        booking.room_id = room_id
        booking.start_time = start_timestamp
        booking.end_time = end_timestamp
        booking.timestamp = date
        booking.modified_by = modified_by
        booking.name = name
        
        self.db.commit()
        self.db.refresh(booking)
        
        logger.info(f"Booking updated successfully: ID {booking_id}")
        
        return {
            "status": "success",
            "message": "Booking updated successfully",
            "modified_by": modified_by
        }
    
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
        """Get room recommendations from recommendation engine"""
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