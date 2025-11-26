from datetime import datetime, date, time
from fastapi import FastAPI, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from models.booking import MRBSEntry
from models.room import MRBSRoom
from utils.database import get_db
import logging
import os
from langchain_core.language_models import BaseLLM
from langchain_core.outputs import LLMResult, Generation
import requests
from pydantic import BaseModel
from config.app_config import settings

from typing import Optional, List, Any
from api.routes.chat_routes import router 
from fastapi.middleware.cors import CORSMiddleware
from core.booking_service import fetch_user_profile_by_email as fetch_profile_logic
from api.routes.swap_routes import router as swap_router
from middleware.auth import get_current_user_email
app = FastAPI()

# 👇 Allow frontend on localhost:3000
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["x-access-token"],
)

app.include_router(router)
app.include_router(swap_router)
# Configure logging
# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
# logger = logging.getLogger(__name__)

# from bs4 import BeautifulSoup
import requests

# dieerect end apis for testing
# remove when project comes to the end
@app.get("/fetch_bookings")
def fetch_bookings(room_name: str, db: Session = Depends(get_db)):
    print(f"🔵 Received Input -> room_name: {room_name}")

    room = db.query(MRBSRoom).filter(MRBSRoom.room_name == room_name).first()
    if not room:
        print(f"🔴 Room '{room_name}' not found")
        raise HTTPException(status_code=404, detail="Room not found")

    print(f"✅ Room Found -> room_id: {room.id}")

    existing_bookings = (
        db.query(MRBSEntry)
        .filter(MRBSEntry.room_id == room.id)
        .all()
    )

    if existing_bookings:
        print(f"✅ Room '{room_name}' has bookings")
        return existing_bookings
    
    print(f"ℹ️ Room '{room_name}' isn't booked at this time")
    return {"message": f"{room_name} isn't booked at this time"}

@app.get("/check_availability/")
def check_availability(room_name: str, date: date, start_time: str, end_time: str, db: Session = Depends(get_db)):
    print(f"🔵 Received Input -> room_name: {room_name}, date: {date}, start_time: {start_time}, end_time: {end_time}")

    # Strip newline characters and spaces (if any)
    start_time = start_time.strip()
    end_time = end_time.strip()
    print(f"🟢 Cleaned Input -> start_time: {start_time}, end_time: {end_time}")

    # Convert start_time and end_time to `time` objects
    try:
        start_time_obj = datetime.strptime(start_time, "%H:%M").time()
        end_time_obj = datetime.strptime(end_time, "%H:%M").time()
        print(f"🟡 Parsed Time -> start_time_obj: {start_time_obj}, end_time_obj: {end_time_obj}")
    except ValueError as e:
        print(f"🔴 Time Parsing Error: {e}")
        raise HTTPException(status_code=400, detail="Invalid time format. Use HH:MM.")

    # Convert date and time to timestamps
    start_timestamp = int(datetime.combine(date, start_time_obj).timestamp())
    end_timestamp = int(datetime.combine(date, end_time_obj).timestamp())
    print(f"🟣 Converted Timestamps -> start_timestamp: {start_timestamp}, end_timestamp: {end_timestamp}")

    # Check if the room exists
    room = db.query(MRBSRoom).filter(MRBSRoom.room_name == room_name).first()
    if not room:
        print(f"🔴 Room '{room_name}' not found")
        raise HTTPException(status_code=404, detail="Room not found")

    print(f"✅ Room Found -> room_id: {room.id}")

    # Check for overlapping bookings
    existing_booking = (
        db.query(MRBSEntry)
        .filter(
            MRBSEntry.room_id == room.id,
            MRBSEntry.start_time < end_timestamp,
            MRBSEntry.end_time > start_timestamp
        )
        .first()
    )

    if existing_booking:
        print(f"❌ Room '{room_name}' is NOT available at this time")
        return {"message": f"{room_name} is NOT available at this time"}
    
    print(f"✅ Room '{room_name}' is available for booking")
    return {"message": f"{room_name} is available. You can book it."}



# -----------------------------
# Add a booking
# -----------------------------
class BookingRequest(BaseModel):
    room_name: str
    name: str
    date: date
    start_time: str
    end_time: str

@app.post("/booking/add")
def add_booking_endpoint(request: BookingRequest, db: Session = Depends(get_db)):
    from core.booking_service import add_booking
    return add_booking(
        request.room_name,
        request.name,
        request.date,
        request.start_time,
        request.end_time,
        "system",
        db
    )

@app.get("/booking/fetch_booking_by_id")
def fetch_booking_by_id(booking_id: int, db: Session = Depends(get_db)):
    from core.booking_service import fetch_booking_by_id as fetch_booking_logic

    booking = fetch_booking_logic(booking_id, db)
    if not booking:
        raise HTTPException(status_code=404, detail="Booking not found")

    # Get room name
    room = db.query(MRBSRoom).filter(MRBSRoom.id == booking.room_id).first()
    room_name = room.room_name if room else None

    # Convert Unix timestamps -> HH:MM
    start_time_str = datetime.fromtimestamp(booking.start_time).strftime("%H:%M")
    end_time_str = datetime.fromtimestamp(booking.end_time).strftime("%H:%M")

    return {
        "id": booking.id,
        "name": booking.name,
        "description": booking.description,
        "created_by": booking.create_by,
        "modified_by": booking.modified_by,
        "room_id": booking.room_id,
        "room_name": room_name,
        "start_time": start_time_str,
        "end_time": end_time_str,
        "timestamp": booking.timestamp,
        "type": booking.type,
    }
# -----------------------------
# Get available time slots
# -----------------------------
@app.get("/booking/available_slots")
def available_slots_endpoint(
    room_name: str,
    date: str,
    db: Session = Depends(get_db)
):
    from core.booking_service import check_available_slotes
    return check_available_slotes(room_name, date, "00:00", "23:59", db)

@app.delete("/booking/delete")
def delete_booking(booking_id: int, db: Session = Depends(get_db),user_email: str = Depends(get_current_user_email)):
    from core.booking_service import delete_booking
    booking = db.query(MRBSEntry).filter(MRBSEntry.id == booking_id).first()
   
    if booking.create_by != user_email:
        raise HTTPException(status_code=403, detail="Access denied")
    return delete_booking(booking_id, db)


class UpdateBookingRequest(BaseModel):
    booking_id: int
    room_name: str
    name:str
    date: date
    start_time: str
    end_time: str


@app.put("/booking/update_booking")
def update_booking(request: UpdateBookingRequest, db: Session = Depends(get_db)):
    from core.booking_service import update_booking as update_booking_logic

    # Convert times -> UNIX timestamps
    try:
        start_time_obj = datetime.strptime(request.start_time, "%H:%M").time()
        end_time_obj = datetime.strptime(request.end_time, "%H:%M").time()
        
        start_timestamp = int(datetime.combine(request.date, start_time_obj).timestamp())
        end_timestamp = int(datetime.combine(request.date, end_time_obj).timestamp())
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid time format. Use HH:MM")

    # 🔎 Find the room by name
    room = db.query(MRBSRoom).filter(MRBSRoom.room_name == request.room_name).first()
    if not room:
        raise HTTPException(status_code=404, detail=f"Room '{request.room_name}' not found")

    # ✅ Pass room_id instead of room_name
    return update_booking_logic(
        request.booking_id,
        room.id,
        request.name,
        request.date,
        start_timestamp,
        end_timestamp,
        db
    )
    
@app.get("/booking/fetch_moduleCodes_by_user_email")
def fetch_moduleCodes_by_user_email(email: str, db: Session = Depends(get_db)):
    from core.booking_service import fetch_moduleCodes_by_user_email as fetch_modules_logic

    return fetch_modules_logic(email, db)

@app.get("/booking/all_halls")
def fetch_all_halls(db: Session = Depends(get_db)):
    from core.booking_service import fetch_all_halls as fetch_halls_logic

    return fetch_halls_logic(db)

@app.get("/booking/fetch_halls_by_moduleCode")
def fetch_halls_by_moduleCode(module_code: str, db: Session = Depends(get_db)):
    from core.booking_service import fetch_halls_by_module_code as fetch_halls_logic

    return fetch_halls_logic(module_code, db)

@app.get("/bookings/by-date/{date}/{room_id}")
def get_bookings_by_date_endpoint(date: str, room_id: int, db: Session = Depends(get_db)):
    
    from core.booking_service import get_bookings_by_date_and_room
    return get_bookings_by_date_and_room(date, room_id, db)


@app.get("/fetch_user_profile_by_email/{email}")
def fetch_user_profile_by_email_route(email: str, db: Session = Depends(get_db)):
    return fetch_profile_logic(email, db)