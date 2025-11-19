import logging
from sqlalchemy.orm import Session
from datetime import datetime
import time
from fastapi import HTTPException
from . import models
from datetime import datetime, timedelta
from recommendtion.config.recommendation_config import RecommendationConfig
from recommendtion.recommendations.core.recommendation_engine import RecommendationEngine
from recommendtion.recommendations.core.hybridRecommendations import hybridRecommendationsEngine
from typing import Dict, Any
import json
import re

config = RecommendationConfig()
recommendation_engine = RecommendationEngine(config=config)
enhanced_engine = hybridRecommendationsEngine(config=config)  

def validate_future_datetime(date_str: str, time_str: str, context: str = "booking", llm=None) -> None:
    """
    Validates that the provided date and time are in the future.
    Uses LLM for natural language date parsing if needed.
    
    Args:
        date_str: Date in format YYYY-MM-DD or natural language
        time_str: Time in format HH:MM or natural language
        context: Context for error message (e.g., 'booking', 'checking availability')
        llm: Optional LLM instance for natural language parsing
    
    Raises:
        HTTPException: If the datetime is in the past
    """
    try:
        # Try standard format first
        booking_datetime = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
    except ValueError:
        # If standard format fails and LLM is provided, try natural language parsing
        if llm:
            try:
                normalized = _parse_natural_datetime_with_llm(date_str, time_str, llm)
                booking_datetime = datetime.strptime(
                    f"{normalized['date']} {normalized['time']}", 
                    "%Y-%m-%d %H:%M"
                )
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid date/time format. Please use YYYY-MM-DD and HH:MM, or clear natural language: {e}"
                )
        else:
            raise HTTPException(
                status_code=400,
                detail="Invalid date/time format. Please use YYYY-MM-DD for date and HH:MM for time."
            )
    
    current_datetime = datetime.now()
    
    if booking_datetime <= current_datetime:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Past datetime not allowed",
                "message": f"Cannot {context} for past date/time. Please select a future date and time.",
                "requested_datetime": booking_datetime.strftime("%Y-%m-%d %H:%M"),
                "current_datetime": current_datetime.strftime("%Y-%m-%d %H:%M")
            }
        )


def _parse_natural_datetime_with_llm(date_str: str, time_str: str, llm) -> Dict[str, str]:
    """
    Uses LLM to parse natural language date/time into standard format.
    
    Args:
        date_str: Natural language date (e.g., "tomorrow", "next Monday")
        time_str: Natural language time (e.g., "morning", "3pm")
        llm: LLM instance for parsing
    
    Returns:
        Dict with 'date' (YYYY-MM-DD) and 'time' (HH:MM)
    """
    current_date = datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.now().strftime("%H:%M")
    
    prompt = f"""
You are a datetime parser. Convert natural language date and time to standard format.

Current date: {current_date}
Current time: {current_time}

User provided:
- Date: "{date_str}"
- Time: "{time_str}"

Convert to standard format. Respond ONLY in JSON:
{{
  "date": "YYYY-MM-DD",
  "time": "HH:MM",
  "is_valid": true/false
}}

Examples:
- "tomorrow" at "morning" → {{"date": "2025-11-16", "time": "09:00", "is_valid": true}}
- "next Monday" at "3pm" → {{"date": "2025-11-18", "time": "15:00", "is_valid": true}}
- "yesterday" at "10am" → {{"date": "2025-11-14", "time": "10:00", "is_valid": false}}

Rules:
- Parse relative dates (tomorrow, next week, etc.) based on current date
- Convert time expressions (morning=09:00, afternoon=14:00, evening=18:00)
- Set is_valid=false for past dates/times
- If cannot parse, use current date/time and set is_valid=false

Respond in JSON only:
"""
    
    try:
        raw_response = llm._call(prompt)
        cleaned = re.sub(r"^```json|```$", "", raw_response.strip(), flags=re.MULTILINE).strip()
        parsed = json.loads(cleaned)
        
        if not parsed.get("is_valid", False):
            raise ValueError("LLM detected invalid or past date/time")
        
        return {
            "date": parsed["date"],
            "time": parsed["time"]
        }
    except Exception as e:
        raise ValueError(f"Failed to parse natural language datetime: {e}")


def validate_datetime_logic_with_llm(date_str: str, time_str: str, llm) -> Dict[str, Any]:
    """
    Uses LLM to validate booking logic and provide intelligent feedback.
    
    Args:
        date_str: Date string (YYYY-MM-DD or natural language)
        time_str: Time string (HH:MM or natural language)
        llm: LLM instance
    
    Returns:
        Dict with validation results and suggestions
    """
    current_datetime = datetime.now()
    
    # Try to parse the datetime
    try:
        if re.match(r'^\d{4}-\d{2}-\d{2}$', date_str) and re.match(r'^\d{2}:\d{2}$', time_str):
            # Standard format
            booking_datetime = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
            parsed_date = date_str
            parsed_time = time_str
        else:
            # Natural language - use LLM
            normalized = _parse_natural_datetime_with_llm(date_str, time_str, llm)
            parsed_date = normalized["date"]
            parsed_time = normalized["time"]
            booking_datetime = datetime.strptime(f"{parsed_date} {parsed_time}", "%Y-%m-%d %H:%M")
    except Exception as e:
        return {
            "is_valid": False,
            "error": "invalid_format",
            "message": f"Could not parse date/time: {e}",
            "parsed_date": None,
            "parsed_time": None
        }
    
    # Check if it's in the future
    if booking_datetime <= current_datetime:
        # Use LLM to generate helpful suggestion
        prompt = f"""
You are a helpful booking assistant. A user tried to book for a past date/time.

Current date/time: {current_datetime.strftime("%Y-%m-%d %H:%M")}
User requested: {booking_datetime.strftime("%Y-%m-%d %H:%M")}

Generate a helpful, concise message suggesting future alternatives. Respond ONLY in JSON:
{{
  "suggestion_message": "friendly suggestion with specific future date/time options",
  "alternative_dates": ["YYYY-MM-DD", "YYYY-MM-DD"],
  "alternative_times": ["HH:MM", "HH:MM"]
}}

Keep the message professional, helpful, and concise (max 2 sentences).
"""
        
        try:
            raw_response = llm._call(prompt)
            cleaned = re.sub(r"^```json|```$", "", raw_response.strip(), flags=re.MULTILINE).strip()
            llm_suggestion = json.loads(cleaned)
        except:
            llm_suggestion = {
                "suggestion_message": "Please select a future date and time for your booking.",
                "alternative_dates": [
                    (current_datetime + timedelta(days=1)).strftime("%Y-%m-%d"),
                    (current_datetime + timedelta(days=7)).strftime("%Y-%m-%d")
                ],
                "alternative_times": ["09:00", "14:00"]
            }
        
        return {
            "is_valid": False,
            "error": "past_datetime",
            "message": llm_suggestion["suggestion_message"],
            "parsed_date": parsed_date,
            "parsed_time": parsed_time,
            "requested_datetime": booking_datetime.strftime("%Y-%m-%d %H:%M"),
            "current_datetime": current_datetime.strftime("%Y-%m-%d %H:%M"),
            "suggestions": {
                "dates": llm_suggestion.get("alternative_dates", []),
                "times": llm_suggestion.get("alternative_times", [])
            }
        }
    
    return {
        "is_valid": True,
        "parsed_date": parsed_date,
        "parsed_time": parsed_time,
        "message": f"Valid future booking: {booking_datetime.strftime('%Y-%m-%d %H:%M')}"
    }
    
def get_room_recommendations(room_name: str, date: str, start_time: str, end_time: str, db: Session):
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
        
        recommendations = enhanced_engine.get_recommendations(request_data)
        return recommendations
    except Exception as e:
        print(f"Recommendation system error: {e}")
        return []


def check_availability(room_name: str, date: str, start_time: str, end_time: str, db: Session):
    print(f"Checking availability for room: {room_name}")
    print(f"Date: {date}, Start time: {start_time}, End time: {end_time}")

    room = db.query(models.MRBSRoom).filter(models.MRBSRoom.room_name == room_name).first()
    print(f"Queried room from DB: {room}")

    # Convert to datetime objects first
    start_dt = datetime.strptime(f"{date} {start_time}", "%Y-%m-%d %H:%M")
    end_dt = datetime.strptime(f"{date} {end_time}", "%Y-%m-%d %H:%M")

    # Convert datetime to Unix timestamps (int)
    start_ts = int(time.mktime(start_dt.timetuple()))
    end_ts = int(time.mktime(end_dt.timetuple()))
    print(f"Converted start datetime to Unix timestamp: {start_ts}")
    print(f"Converted end datetime to Unix timestamp: {end_ts}")

    # Query for conflicting bookings using Unix timestamps
    conflicting = db.query(models.MRBSEntry).filter(
        models.MRBSEntry.room_id == room.id,
        models.MRBSEntry.start_time < end_ts,
        models.MRBSEntry.end_time > start_ts,
    ).first()
    print(f"Conflicting booking found: {conflicting}")

    if conflicting:
        message = f"{room_name} is already booked for that time. Here are some available alternatives you might like:"
        print(message)
        recommendations = get_room_recommendations(room_name, date, start_time, end_time, db)
        return {"status": "unavailable",
            "message": message,
            "recommendations": recommendations
            }
    
    if not room:
        print("Room not found!")
        recommendations = get_room_recommendations(room_name, date, start_time, end_time, db)
    
        return {
                "status": "room_not_found",
                "message": f"Room '{room_name}' not found.",
                "recommendations": recommendations
        }
        
    message = f"{room_name} is available from {start_time} to {end_time} on {date}."
    print(message)
    return {"status": "available", "message": message}

def fetch_booking_by_id(booking_id: int, db: Session):
    try:
        booking = db.query(models.MRBSEntry).filter(models.MRBSEntry.id == booking_id).first()
        if not booking:
            raise HTTPException(status_code=404, detail="Booking not found")
        return booking
    except Exception as e:
        print(f"Error fetching booking by ID: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


def update_booking_general(booking_id: int, room_id: int, name: str, date: str, start_timestamp: int, end_timestamp: int, modified_by: str, db: Session):
    try:
        start_datetime = datetime.fromtimestamp(start_timestamp)
        validate_future_datetime(
            start_datetime.strftime("%Y-%m-%d"),
            start_datetime.strftime("%H:%M"),
            "update booking"
        )
        
        booking = db.query(models.MRBSEntry).filter(models.MRBSEntry.id == booking_id).first()
        if not booking:
            raise HTTPException(status_code=404, detail="Booking not found")
        logging.info(f"Updating booking: {booking_id}, {booking.name}, {date}, {start_timestamp}, {end_timestamp} by {modified_by}")
        
        # Update with valid fields
        booking.room_id = room_id
        booking.start_time = start_timestamp
        booking.end_time = end_timestamp
        booking.timestamp = date
        booking.modified_by = modified_by  
        booking.name = name

        db.commit()
        db.refresh(booking)

        return {
            "status": "success", 
            "message": "Booking updated successfully",
            "modified_by": modified_by
        }
    except Exception as e:
        print(f"Error updating booking: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

    
def delete_booking(booking_id: int, db: Session):
    try:
        booking = db.query(models.MRBSEntry).filter(models.MRBSEntry.id == booking_id).first()
        if not booking:
            raise HTTPException(status_code=404, detail="Booking not found")

        db.delete(booking)
        db.commit()
        return {"status": "success", "message": "Booking deleted successfully"}
    except Exception as e:
        print(f"Error deleting booking: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

def add_booking(room_name: str,name: str, date: str, start_time: str, end_time: str, created_by: str, db: Session):
    
    try:
        validate_future_datetime(date, start_time, "book")
        room = db.query(models.MRBSRoom).filter(models.MRBSRoom.room_name == room_name).first()
        
        if not room:
            recommendations = get_room_recommendations(room_name, date, start_time, end_time, db)
            raise HTTPException(
                status_code=404, 
                detail={
                    "error": "Room not found",
                    "message": f"Room '{room_name}' not found.",
                    "recommendations": recommendations
                }
            )
        
        try:
            start_dt = datetime.strptime(f"{date} {start_time}", "%Y-%m-%d %H:%M")
            end_dt = datetime.strptime(f"{date} {end_time}", "%Y-%m-%d %H:%M")

            
            start_ts = int(time.mktime(start_dt.timetuple()))
            end_ts = int(time.mktime(end_dt.timetuple()))

            
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid date/time format: {e}")
        
        if end_ts <= start_ts:
            raise HTTPException(status_code=400, detail="End time must be after start time")
        
        
        conflict = db.query(models.MRBSEntry).filter(
            models.MRBSEntry.room_id == room.id,
            models.MRBSEntry.start_time < end_ts,
            models.MRBSEntry.end_time > start_ts,
        ).first()
        
        if conflict:
            recommendations = get_room_recommendations(room_name, date, start_time, end_time, db)
            return {
                "status": "unavailable",
                "message": f"Room '{room_name}' is already booked for that time. Here are some available alternatives you might like:",
                "recommendations": recommendations
            }
        
        current_datetime = datetime.now()
        
        try:
            new_booking = models.MRBSEntry(
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
            db.add(new_booking)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error adding booking to session: {e}")
        
        try:
            db.commit()
        except Exception as e:
            db.rollback()
            raise HTTPException(status_code=500, detail=f"Database commit failed: {e}")
        
        try:
            db.refresh(new_booking)
        except Exception as e:
            pass
        
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
    
    
    
def check_available_slotes(room_name: str, date: str, start_time: str, end_time: str, db: Session):
    # alternative slotes
    
    validate_future_datetime(date, "00:00", "check available slots")
    
    print(f"Checking availability for room: {room_name}")
    print(f"Date: {date}, Start time: {start_time}, End time: {end_time}")

    room = db.query(models.MRBSRoom).filter(models.MRBSRoom.room_name == room_name).first()
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

    bookings = db.query(models.MRBSEntry).filter(
        models.MRBSEntry.room_id == room.id,
        models.MRBSEntry.start_time < day_end_ts,
        models.MRBSEntry.end_time > day_start_ts
    ).all()

    # Step 4: Filter available slots
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
        recommendations = get_room_recommendations(room_name, date, start_time, end_time, db)
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
        recommendations = get_room_recommendations(room_name, date, start_time, end_time, db)
        raise HTTPException(
            status_code=404, 
            detail={
                "error": "Room not found",
                "message": f"Room '{room_name}' not found.",
                "recommendations": recommendations
            }
        )
    
    return {"room": room_name, "date": date, "available_slots": available_slots}


def book_recommendation_directly(recommendation: Dict[str, Any], created_by: str, db: Session):
   
    try:
        suggestion = recommendation.get('suggestion', {})
        if not suggestion:
            raise HTTPException(status_code=400, detail="Invalid recommendation: missing suggestion data")
        
        room_name = suggestion.get('room_name')
        start_time_str = suggestion.get('start_time')
        end_time_str = suggestion.get('end_time')
        
        if not all([room_name, start_time_str, end_time_str]):
            raise HTTPException(
                status_code=400, 
                detail="Invalid recommendation: missing room_name, start_time, or end_time"
            )
        
        # Parse the datetime strings
        try:
            start_dt = datetime.fromisoformat(start_time_str.replace('Z', '+00:00'))
            end_dt = datetime.fromisoformat(end_time_str.replace('Z', '+00:00'))
        except ValueError:
            # Try alternative parsing if ISO format fails
            try:
                start_dt = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S")
                end_dt = datetime.strptime(end_time_str, "%Y-%m-%d %H:%M:%S")
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"Invalid datetime format in recommendation: {e}")
        
        date = start_dt.strftime("%Y-%m-%d")
        start_time = start_dt.strftime("%H:%M") 
        end_time = end_dt.strftime("%H:%M")
        
        print(f"Booking recommendation: {room_name} on {date} from {start_time} to {end_time} by {created_by}")
        
        result = add_booking(
            room_name=room_name,
            name="Recommendation Booking", 
            date=date,
            start_time=start_time,
            end_time=end_time,
            created_by=created_by,
            db=db,
            validate_availability=False  
        )
        
        result['recommendation_type'] = recommendation.get('type', 'unknown')
        result['recommendation_score'] = recommendation.get('score', 0)
        result['recommendation_reason'] = recommendation.get('reason', '')
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error booking recommendation: {e}")

def fetch_moduleCodes_by_user_email(email: str, db: Session):
    user = db.query(models.MRBSUser).filter(models.MRBSUser.email == email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    modules = db.query(models.MRBSModule).filter(models.MRBSModule.lecture_id == user.id).all()
    return [module.module_code for module in modules]

def fetch_all_halls(db: Session):
    halls = db.query(models.MRBSRoom).all()
    return [hall.room_name for hall in halls]

def fetch_halls_by_module_code(module_code: str, db: Session):
    # Fetch the module first
    module = db.query(models.MRBSModule).filter(models.MRBSModule.module_code == module_code).first()
    if not module:
        return []  # Module not found

    # Fetch halls that can accommodate the number of students
    halls = db.query(models.MRBSRoom).filter(models.MRBSRoom.capacity >= module.number_of_students).all()
    
    return [hall.room_name for hall in halls]



def update_booking(original_room_name: str, original_date: str, original_start_time: str, 
                   original_end_time: str, new_room_name: str = None, new_date: str = None,
                   new_start_time: str = None, new_end_time: str = None, 
                   modified_by: str = "system", db: Session = None):
    try:
        room = db.query(models.MRBSRoom).filter(models.MRBSRoom.room_name == original_room_name).first()
        if not room:
            return {"status": "room_not_found", "message": f"Room '{original_room_name}' not found."}
        
        start_dt = datetime.strptime(f"{original_date} {original_start_time}", "%Y-%m-%d %H:%M")
        end_dt = datetime.strptime(f"{original_date} {original_end_time}", "%Y-%m-%d %H:%M")
        start_ts, end_ts = int(time.mktime(start_dt.timetuple())), int(time.mktime(end_dt.timetuple()))
        
        booking = db.query(models.MRBSEntry).filter(
            models.MRBSEntry.room_id == room.id,
            models.MRBSEntry.start_time == start_ts,
            models.MRBSEntry.end_time == end_ts
        ).first()
        
        if not booking:
            return {"status": "booking_not_found", 
                   "message": f"No booking found for {original_room_name} on {original_date} from {original_start_time} to {original_end_time}."}
        
        # AUTHORIZATION CHECK: Only creator can update
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
            new_room = db.query(models.MRBSRoom).filter(models.MRBSRoom.room_name == new_room_name).first()
            if not new_room:
                return {"status": "new_room_not_found", "message": f"New room '{new_room_name}' not found."}
            final_room_id = new_room.id
        
        final_start_dt = datetime.strptime(f"{final_date} {final_start_time}", "%Y-%m-%d %H:%M")
        final_end_dt = datetime.strptime(f"{final_date} {final_end_time}", "%Y-%m-%d %H:%M")
        final_start_ts, final_end_ts = int(time.mktime(final_start_dt.timetuple())), int(time.mktime(final_end_dt.timetuple()))
        
        if final_end_ts <= final_start_ts:
            return {"status": "invalid_time", "message": "End time must be after start time."}
        
        if (final_room_id != room.id or final_start_ts != start_ts or final_end_ts != end_ts):
            conflict = db.query(models.MRBSEntry).filter(
                models.MRBSEntry.room_id == final_room_id,
                models.MRBSEntry.start_time < final_end_ts,
                models.MRBSEntry.end_time > final_start_ts,
                models.MRBSEntry.id != booking.id
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

def cancel_booking(room_name: str, date: str, start_time: str, end_time: str, user_email: str, db: Session):
    try:
        room = db.query(models.MRBSRoom).filter(models.MRBSRoom.room_name == room_name).first()
        
        if not room:
            return {
                "status": "room_not_found",
                "message": f"Room '{room_name}' not found."
            }
        
        start_dt = datetime.strptime(f"{date} {start_time}", "%Y-%m-%d %H:%M")
        end_dt = datetime.strptime(f"{date} {end_time}", "%Y-%m-%d %H:%M")
        start_ts = int(time.mktime(start_dt.timetuple()))
        end_ts = int(time.mktime(end_dt.timetuple()))
        
        booking = db.query(models.MRBSEntry).filter(
            models.MRBSEntry.room_id == room.id,
            models.MRBSEntry.start_time == start_ts,
            models.MRBSEntry.end_time == end_ts
        ).first()
        
        if not booking:
            return {
                "status": "no_booking_found",
                "message": f"No booking found for {room_name} on {date} from {start_time} to {end_time}."
            }
        
        # AUTHORIZATION CHECK: Only creator can cancel
        if booking.create_by != user_email:
            raise HTTPException(
                status_code=403, 
                detail=f"Access denied. Only the booking creator ({booking.create_by}) can cancel this booking."
            )
        
        db.delete(booking)
        db.commit()
        
        return {
            "status": "success",
            "message": f"Successfully cancelled booking for {room_name} on {date} from {start_time} to {end_time}.",
            "cancelled_booking_id": booking.id
        }
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date/time format: {e}")
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error cancelling booking: {e}")

def get_bookings_by_date_and_room(date: str, room_id: int, db: Session):
    try:
        day_start = datetime.strptime(date, "%Y-%m-%d")
        next_day = day_start + timedelta(days=1)

        start_ts = int(time.mktime(day_start.timetuple()))
        next_day_ts = int(time.mktime(next_day.timetuple()))

        bookings = (
            db.query(models.MRBSEntry)
            .filter(
                models.MRBSEntry.start_time < next_day_ts,
                models.MRBSEntry.end_time > start_ts,
                models.MRBSEntry.room_id == room_id
            )
            .order_by(models.MRBSEntry.start_time.asc())
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

def fetch_user_profile_by_email(email: str, db: Session):
    user = db.query(models.MRBSUser).filter(models.MRBSUser.email == email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return {
        "id": user.id,
        "name": user.name,
        "email": user.email,
        "role": user.role
    }
    
