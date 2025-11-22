from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Dict, Any
import json
import re
from datetime import datetime
from config.database_config import get_db
from services.llm.deepseek_llm import DeepSeekLLM
from services.llm.entity_extractor import extract_entities, refresh_room_cache
from core.booking_service import BookingService
from core.validation_service import ValidationService
from core.recurrence_service import RecurrenceService
from middleware.auth import get_current_user_email
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


class QuestionRequest(BaseModel):
    session_id: str
    question: str


class RecommendationBookingRequest(BaseModel):
    session_id: str
    recommendation: Dict[str, Any]
    created_by: str


REQUIRED_FIELDS = {
     "check_availability": ["room_name", "date", "start_time", "end_time"],
    "add_booking": ["room_name", "date", "module_code", "start_time", "end_time"],
    "add_recurring_booking": ["room_name", "start_date", "end_date", "start_time", "end_time", "module_code", "recurrence_rule"],
    "alternatives": ["date", "start_time", "end_time"],
    "cancel_booking": ["room_name", "date", "start_time", "end_time"],
    "update_booking": ["original_room_name", "original_date", "original_start_time", "original_end_time"] 
}

FALLBACK_QUESTIONS = {
   "room_name": "Which room would you like to book?",
    "module_code": "What is the module code?",
    "date": "What date would you like to book it for? Please use YYYY-MM-DD format.",
    "start_date": "What is the start date? Please use YYYY-MM-DD format.",
    "end_date": "What is the end date? Please use YYYY-MM-DD format?",
    "start_time": "What start time do you want? Please use HH:MM format.",
    "end_time": "What end time do you want? Please use HH:MM format.",
    "recurrence_rule": "Please specify the recurrence (e.g. every Monday).",
    "booking_id": "Please provide the booking ID to cancel.",
    "original_room_name": "Which room's booking do you want to update?",
    "original_date": "What date was the original booking for?",
    "original_start_time": "What was the original start time?",
    "original_end_time": "What was the original end time?",
}

session_store = {}


def get_missing_params(params: dict, required_fields: list) -> list:
    return [f for f in required_fields if f not in params or not params[f]]

def validate_time_format(time_str: str) -> bool:
    try:
        datetime.strptime(time_str, "%H:%M")
        return True
    except ValueError:
        return False

@router.post("/ask_llm/")
async def ask_llm(
    request: QuestionRequest,
    db: Session = Depends(get_db),
    user_email: str = Depends(get_current_user_email)
):
   
    session_id = request.session_id
    question = request.question.strip()
    
    session = session_store.get(session_id, {
        "action": None,
        "params": {},
        "last_asked": None,
        "missing_fields": [],
        "user_email": user_email
    })
    
    session["user_email"] = user_email
    
    llm = DeepSeekLLM()
    validator = ValidationService()
    
    if session["last_asked"]:
        last_param = session["last_asked"]
        session["params"][last_param] = question
        session["missing_fields"] = [f for f in session["missing_fields"] if f != last_param]
        session["last_asked"] = None
        session_store[session_id] = session
    else:
        recurrence_service = RecurrenceService(llm)
        recurrence_data = recurrence_service.extract_recurrence(question)
        
        if recurrence_data.get("is_recurring"):
            try:
                recurrence_rule = recurrence_service.build_rrule_from_extracted(recurrence_data)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Error building recurrence rule: {str(e)}")
            
            params = {
                "action": "add_recurring_booking",
                "parameters": {
                    "room_name": recurrence_data.get("room_name"),
                    "module_code": recurrence_data.get("module_code"),
                    "start_date": recurrence_data.get("start_date"),
                    "end_date": recurrence_data.get("end_date"),
                    "start_time": recurrence_data.get("start_time"),
                    "end_time": recurrence_data.get("end_time"),
                    "recurrence_rule": recurrence_rule,
                    "created_by": user_email,
                }
            }
            
            # Pass db session to extract_entities
            extracted = extract_entities(question)
            if "room_name" in extracted:
                params["parameters"]["room_name"] = extracted["room_name"]
            
            session["action"] = params["action"]
            session["params"] = params["parameters"]
            session["last_asked"] = None
            session_store[session_id] = session
        else:
            llm = DeepSeekLLM()
            prompt = f"""
You are an intelligent assistant that helps manage room bookings.

From the following user request:
\"{question}\"

Extract the **action** and its corresponding **parameters** in **strict JSON format**.

Supported actions:
- "check_availability"
- "add_booking"
- "cancel_booking"
- "alternatives"
- "update_booking"

If the request is not related to any of these actions, return:
{{ "action": "unsupported", "parameters": {{}} }}


Required JSON structure:
{{
  "action":  "check_availability" | "add_booking" | "cancel_booking" | "alternatives",
  "parameters": {{
    "room_name": "...",
    "module_code": "...",
    "date": "yyyy-mm-dd",
    "start_time": "HH:MM",
    "end_time": "HH:MM"
  }}
}}

For update_booking:
{{
  "action": "update_booking",
  "parameters": {{
    "original_room_name": "...",
    "original_date": "yyyy-mm-dd",
    "original_start_time": "HH:MM",
    "original_end_time": "HH:MM",
    "new_room_name": "..." (optional),
    "new_date": "yyyy-mm-dd" (optional),
    "new_start_time": "HH:MM" (optional),
    "new_end_time": "HH:MM" (optional)
  }}
}}

Respond in **only JSON format**, without explanations.
"""
            try:
                llm_response = llm._call(prompt)
                cleaned_response = re.sub(r"^```json|```$", "", llm_response.strip(), flags=re.MULTILINE).strip()
                parsed = json.loads(cleaned_response)
            except Exception as e:
                logger.error(f"LLM parsing failed: {e}")
                return {
                    "status": "unsupported_action",
                    "message": "I can only help with room bookings. Please provide booking details.",
                }
            
            if "action" not in parsed or "parameters" not in parsed:
                return {
                    "status": "unsupported_action",
                    "message": "I can only help with room bookings. Please provide booking details.",
                }
            
            action = parsed["action"]
            params = parsed["parameters"]
            
            if action == "unsupported":
                return {
                    "status": "unsupported_action",
                    "message": "I'm here to help with room bookings only."
                }
            
            # Pass db session to extract_entities
            extracted = extract_entities(question)
            for key, value in extracted.items():
                if key not in params or not params[key]:
                    params[key] = value
            
            params["created_by"] = user_email
            
            session["action"] = action
            session["params"] = params
            session["last_asked"] = None
            session_store[session_id] = session
    
    required_fields = REQUIRED_FIELDS.get(session["action"], [])
    missing_fields = get_missing_params(session["params"], required_fields)
    session["missing_fields"] = missing_fields
    
    for time_field in ["start_time", "end_time"]:
        if time_field in session["params"] and session["params"][time_field]:
            if not validator.validate_time_format(session["params"][time_field]):
                return {
                    "status": "invalid_time_format",
                    "message": f"{time_field} must be in HH:MM format.",
                }
    
    if missing_fields:
        next_missing = missing_fields[0]
        session["last_asked"] = next_missing
        session_store[session_id] = session
        return {
            "status": "missing_parameters",
            "missing_parameter": next_missing,
            "message": FALLBACK_QUESTIONS.get(next_missing, f"Please provide {next_missing}."),
        }
    
    action = session["action"]
    params = session["params"]
    
    if action in ["add_booking", "check_availability", "add_recurring_booking"]:
        date_field = "date" if action != "add_recurring_booking" else "start_date"
        if date_field in params and "start_time" in params:
            validation_result = validator.validate_datetime_logic_with_llm(
                params[date_field],
                params["start_time"],
                llm
            )
            
            if not validation_result["is_valid"]:
                return {
                    "status": "invalid_datetime",
                    "error": validation_result["error"],
                    "message": validation_result["message"],
                    "suggestions": validation_result.get("suggestions", {}),
                    "requested_datetime": validation_result.get("requested_datetime"),
                    "current_datetime": validation_result.get("current_datetime")
                }
            
            params[date_field] = validation_result["parsed_date"]
            params["start_time"] = validation_result["parsed_time"]
    
    booking_service = BookingService(db)
    
    if action == "check_availability":
        return booking_service.check_availability(
            room_name=params["room_name"],
            date=params["date"],
            start_time=params["start_time"],
            end_time=params["end_time"],
            
        )
    elif action == "add_booking":
        return booking_service.add_booking(
            room_name=params["room_name"],
            date=params["date"],
            name=params["module_code"],
            start_time=params["start_time"],
            end_time=params["end_time"],
            created_by=user_email, 
        )
    elif action == "add_recurring_booking":
        recurrence_service = RecurrenceService(llm)
        params["created_by"] = user_email
        return await recurrence_service.handle_recurring_booking(params, db)
    elif action == "alternatives":
        return booking_service.check_available_slotes(
            date=params["date"],
            start_time=params["start_time"],
            end_time=params["end_time"],
            
        )
    elif action == "cancel_booking":
        return booking_service.cancel_booking(
           room_name=params["room_name"],
            date=params["date"],
            start_time=params["start_time"],
            end_time=params["end_time"],
            user_email=user_email,  
           
        )
    elif action == "update_booking":
        if params.get("new_date") and params.get("new_start_time"):
            validation_result = validator.validate_datetime_logic_with_llm(
                params["new_date"],
                params["new_start_time"],
                llm
            )
            
            if not validation_result["is_valid"]:
                return {
                    "status": "invalid_datetime",
                    "error": validation_result["error"],
                    "message": validation_result["message"],
                    "suggestions": validation_result.get("suggestions", {}),
                }
            
            params["new_date"] = validation_result["parsed_date"]
            params["new_start_time"] = validation_result["parsed_time"]
        
        return booking_service.update_booking(
            original_room_name=params["original_room_name"],
            original_date=params["original_date"],
            original_start_time=params["original_start_time"],
            original_end_time=params["original_end_time"],
            new_room_name=params.get("new_room_name"),
            new_date=params.get("new_date"),
            new_start_time=params.get("new_start_time"),
            new_end_time=params.get("new_end_time"),
            modified_by=user_email, 
            
        )
    return {"status": "error", "message": "Unhandled action."}


@router.post("/book-recommendation/")
async def book_recommendation(
    request: RecommendationBookingRequest,
    db: Session = Depends(get_db),
    user_email: str = Depends(get_current_user_email)
):
    
    try:
         
        return BookingService.book_recommendation_directly(
            recommendation=request.recommendation,
            created_by=user_email, 
            db=db
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to book recommendation: {e}")
