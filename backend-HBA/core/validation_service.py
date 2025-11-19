from datetime import datetime
from typing import Dict, Any
from fastapi import HTTPException
from utils.logger import get_logger
import re
import json

logger = get_logger(__name__)


class ValidationService:
    
    @staticmethod
    def validate_time_format(time_str: str) -> bool:
        """Validate time string format (HH:MM)"""
        try:
            datetime.strptime(time_str, "%H:%M")
            return True
        except ValueError:
            return False
    
    @staticmethod
    def validate_date_format(date_str: str) -> bool:
        """Validate date string format (YYYY-MM-DD)"""
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
            return True
        except ValueError:
            return False
    
    @staticmethod
    def validate_future_datetime(date_str: str, time_str: str, context: str = "booking") -> None:
        
        try:
            booking_datetime = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
        except ValueError:
            if llm:
                try:
                    normalized = ValidationService._parse_natural_datetime_with_llm(date_str, time_str, llm)
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
    
    @staticmethod
    def validate_datetime_logic_with_llm(date_str: str, time_str: str, llm) -> Dict[str, Any]:
       
        current_datetime = datetime.now()
        
        try:
            if re.match(r'^\d{4}-\d{2}-\d{2}$', date_str) and re.match(r'^\d{2}:\d{2}$', time_str):
                booking_datetime = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
                parsed_date = date_str
                parsed_time = time_str
            else:
                normalized = ValidationService._parse_natural_datetime_with_llm(date_str, time_str, llm)
                parsed_date = normalized["date"]
                parsed_time = normalized["time"]
                booking_datetime = datetime.strptime(f"{parsed_date} {parsed_time}", "%Y-%m-%d %H:%M")
        except Exception as e:
            logger.error(f"Datetime parsing failed: {e}")
            return {
                "is_valid": False,
                "error": "invalid_format",
                "message": f"Could not parse date/time: {str(e)}",
                "parsed_date": None,
                "parsed_time": None
            }
        
        if booking_datetime <= current_datetime:
            llm_suggestion = ValidationService._get_llm_suggestions(booking_datetime, current_datetime, llm)
            
            return {
                "is_valid": False,
                "error": "past_datetime",
                "message": llm_suggestion["suggestion_message"],
                "parsed_date": parsed_date,
                "parsed_time": parsed_time,
                "requested_datetime": booking_datetime.strftime("%Y-%m-%d %H:%M"),
                "current_datetime": current_datetime.strftime("%Y-%m-%d %H:%M"),
                "suggestions": llm_suggestion.get("suggestions", {})
            }
        
        return {
            "is_valid": True,
            "parsed_date": parsed_date,
            "parsed_time": parsed_time,
            "message": f"Valid future booking: {booking_datetime.strftime('%Y-%m-%d %H:%M')}"
        }
    
    @staticmethod
    def _parse_natural_datetime_with_llm(date_str: str, time_str: str, llm) -> Dict[str, str]:
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

Rules:
- Parse relative dates (tomorrow, next week, etc.) based on current date
- Convert time expressions (morning=09:00, afternoon=14:00, evening=18:00)
- Set is_valid=false for past dates/times
- If cannot parse, use current date/time and set is_valid=false

Respond in JSON only.
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
            logger.error(f"LLM datetime parsing failed: {e}")
            raise ValueError(f"Failed to parse natural language datetime: {e}")
    
    @staticmethod
    def _get_llm_suggestions(booking_datetime: datetime, current_datetime: datetime, llm) -> Dict[str, Any]:
        """Get helpful suggestions from LLM for past datetime"""
        from datetime import timedelta
        
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
            return json.loads(cleaned)
        except Exception as e:
            logger.warning(f"LLM suggestion generation failed: {e}")
            return {
                "suggestion_message": "Please select a future date and time for your booking.",
                "alternative_dates": [
                    (current_datetime + timedelta(days=1)).strftime("%Y-%m-%d"),
                    (current_datetime + timedelta(days=7)).strftime("%Y-%m-%d")
                ],
                "alternative_times": ["09:00", "14:00"]
            }