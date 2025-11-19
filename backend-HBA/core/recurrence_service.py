from datetime import datetime, timedelta
from dateutil.rrule import rrulestr
from typing import Dict, List, Any
from sqlalchemy.orm import Session
from fastapi import HTTPException
import json
import re

from core.booking_service import BookingService
from utils.logger import get_logger

logger = get_logger(__name__)


class RecurrenceService:
    
    RECURRENCE_PROMPT = """
You are an intelligent assistant that extracts recurrence patterns from booking requests.

From the following user request:
"{user_input}"

Detect if it contains a recurring booking pattern.
If yes, output the rule in strict JSON:
{{
  "is_recurring": true,
  "frequency": "daily" | "weekly" | "monthly",
  "days_of_week": ["Monday", "Wednesday"],
  "start_time": "HH:MM",
  "room_name": "<room_name>",
  "module_code": "<module_code>",
  "end_time": "HH:MM",
  "start_date": "YYYY-MM-DD",
  "end_date": "YYYY-MM-DD"
}}

If no recurrence is found, return:
{{
  "is_recurring": false
}}

Respond in JSON only.
"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def extract_recurrence(self, user_input: str) -> Dict[str, Any]:
       
        logger.debug(f"Extracting recurrence from: {user_input}")
        
        small_talk = {"hi", "hello", "hey", "thanks", "thank you"}
        if user_input.lower().strip() in small_talk:
            logger.debug("Detected small talk, skipping LLM call")
            return {"is_recurring": False, "reason": "small_talk"}
        
        prompt = self.RECURRENCE_PROMPT.format(user_input=user_input)
        
        try:
            raw = self.llm._call(prompt)
            cleaned = re.sub(r"^```json|```$", "", raw.strip(), flags=re.MULTILINE).strip()
            parsed = json.loads(cleaned)
            logger.debug(f"Parsed recurrence data: {parsed}")
            return parsed
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON from LLM response")
            return {"is_recurring": False, "reason": "json_error"}
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return {"is_recurring": False, "reason": "llm_unavailable"}
    
    def build_rrule_from_extracted(self, data: Dict[str, Any]) -> str:
        
        freq_map = {"daily": "DAILY", "weekly": "WEEKLY", "monthly": "MONTHLY"}
        freq = freq_map.get(data.get("frequency"))
        
        if not freq:
            raise ValueError("Invalid frequency")
        
        byday = ""
        if freq == "WEEKLY" and "days_of_week" in data:
            day_map = {
                "Monday": "MO",
                "Tuesday": "TU",
                "Wednesday": "WE",
                "Thursday": "TH",
                "Friday": "FR",
                "Saturday": "SA",
                "Sunday": "SU",
            }
            days = [day_map[d] for d in data["days_of_week"] if d in day_map]
            if days:
                byday = ";BYDAY=" + ",".join(days)
        
        rrule = f"FREQ={freq}{byday}"
        logger.info(f"Built RRULE: {rrule}")
        return rrule
    
    async def handle_recurring_booking(self, params: Dict[str, Any], db: Session) -> Dict[str, Any]:
        
        room_name = params.get("room_name")
        module_code = params.get("module_code")
        start_date = params.get("start_date")
        end_date = params.get("end_date")
        start_time = params.get("start_time")
        end_time = params.get("end_time")
        recurrence_rule = params.get("recurrence_rule")
        created_by = params.get("created_by", "system")
        
        if not all([room_name, start_date, end_date, start_time, end_time, recurrence_rule]):
            raise HTTPException(status_code=400, detail="Missing parameters for recurring booking")
        
        try:
            start_date_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid date format: {str(e)}")
        
        current_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        if start_date_dt < current_date:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Past date not allowed",
                    "message": "Cannot create recurring booking starting from a past date.",
                    "requested_start_date": start_date,
                    "current_date": current_date.strftime("%Y-%m-%d")
                }
            )
        
        try:
            rule = rrulestr(recurrence_rule, dtstart=start_date_dt)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid recurrence rule: {str(e)}")
        
        bookings_created = []
        skipped_past_dates = []
        current_datetime = datetime.now()
        
        booking_service = BookingService(db)
        
        for occurrence in rule.between(start_date_dt, end_date_dt, inc=True):
            date_str = occurrence.strftime("%Y-%m-%d")
            
            booking_datetime = datetime.strptime(f"{date_str} {start_time}", "%Y-%m-%d %H:%M")
            if booking_datetime <= current_datetime:
                skipped_past_dates.append(date_str)
                continue
            
            availability = booking_service.check_availability(
                room_name=room_name,
                date=date_str,
                start_time=start_time,
                end_time=end_time,
                db=db,
            )
            
            if availability["status"] != "available":
                logger.warning(f"Room unavailable for {date_str}")
                return {
                    "status": "unavailable",
                    "message": f"{room_name} is NOT available on {date_str} from {start_time} to {end_time}."
                }
            
            booking = booking_service.add_booking(
                room_name=room_name,
                name=module_code,
                date=date_str,
                start_time=start_time,
                end_time=end_time,
                created_by=params.get("created_by", "system"),
                db=db,
            )
            bookings_created.append(booking)
        
        response = {
            "status": "success",
            "message": f"Created {len(bookings_created)} recurring bookings.",
            "bookings": bookings_created,
        }
        
        if skipped_past_dates:
            response["skipped_dates"] = skipped_past_dates
            response["skipped_count"] = len(skipped_past_dates)
        
        return response