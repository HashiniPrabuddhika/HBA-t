from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from datetime import datetime, date, time
from typing import Dict, Any
from pydantic import BaseModel

from config.database_config import get_db
from models.booking import MRBSEntry
from models.room import MRBSRoom
from models.user import MRBSUser, MRBSModule
from core.booking_service import BookingService
from middleware.auth import get_current_user_email
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


class BookingRequest(BaseModel):
    room_name: str
    name: str
    date: str
    start_time: str
    end_time: str


class UpdateBookingRequest(BaseModel):
    booking_id: int
    room_name: str
    name: str
    date: str
    start_time: str
    end_time: str


@router.get("/fetch_bookings")
async def fetch_bookings(
    room_name: str,
    db: Session = Depends(get_db),
    user_email: str = Depends(get_current_user_email)
):
    """Fetch all bookings for a specific room"""
    logger.info(f"Fetching bookings for room: {room_name}")
    
    room = db.query(MRBSRoom).filter(MRBSRoom.room_name == room_name).first()
    if not room:
        raise HTTPException(status_code=404, detail="Room not found")
    
    bookings = db.query(MRBSEntry).filter(MRBSEntry.room_id == room.id).all()
    
    if not bookings:
        return {"message": f"{room_name} has no bookings"}
    
    return bookings


@router.get("/check-availability")
async def check_availability(
    room_name: str,
    date: date,
    start_time: str,
    end_time: str,
    db: Session = Depends(get_db),
    user_email: str = Depends(get_current_user_email)
):
   
    # service = BookingService(db)
    # return service.check_availability(room_name, date, start_time, end_time)
    
    start_time = start_time.strip()
    end_time = end_time.strip()
    try:
        start_time_obj = datetime.strptime(start_time, "%H:%M").time()
        end_time_obj = datetime.strptime(end_time, "%H:%M").time()
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid time format. Use HH:MM.")

    start_timestamp = int(datetime.combine(date, start_time_obj).timestamp())
    end_timestamp = int(datetime.combine(date, end_time_obj).timestamp())
   
    room = db.query(MRBSRoom).filter(MRBSRoom.room_name == room_name).first()
    if not room:
        raise HTTPException(status_code=404, detail="Room not found")

    
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
        return {"message": f"{room_name} is NOT available at this time"}
    
    return {"message": f"{room_name} is available. You can book it."}




@router.post("/booking/add")
async def add_booking_endpoint(
    request: BookingRequest,
    db: Session = Depends(get_db),
    user_email: str = Depends(get_current_user_email)
):
    service = BookingService(db)
    return service.add_booking(
        request.room_name,
        request.name,
        request.date,
        request.start_time,
        request.end_time,
        user_email,
        db
    )


@router.get("/booking/fetch_booking_by_id")
async def fetch_booking_by_id(
    booking_id: int,
    db: Session = Depends(get_db),
    user_email: str = Depends(get_current_user_email)
):
    """Fetch a specific booking by ID"""
    logger.info(f"Fetching booking: {booking_id}")
    
    booking = db.query(MRBSEntry).filter(MRBSEntry.id == booking_id).first()
    if not booking:
        raise HTTPException(status_code=404, detail="Booking not found")
    
    room = db.query(MRBSRoom).filter(MRBSRoom.id == booking.room_id).first()
    room_name = room.room_name if room else None
    
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


@router.get("/bookings/available-slots")
async def available_slots(
    room_name: str,
    date: str,
    db: Session = Depends(get_db),
    user_email: str = Depends(get_current_user_email)
):
    """Get all available time slots for a room on a given date"""
    service = BookingService(db)
    return service.get_available_slots(room_name, date, "00:00", "23:59", db)


@router.delete("/booking/delete")
async def delete_booking(
    booking_id: int,
    db: Session = Depends(get_db),
    user_email: str = Depends(get_current_user_email)
):
    service = BookingService(db)
    return service.delete_booking(booking_id)


@router.put("/booking/update_booking")
async def update_booking(
    request: UpdateBookingRequest,
    db: Session = Depends(get_db),
    user_email: str = Depends(get_current_user_email)
):
    start_dt = datetime.strptime(f"{request.date} {request.start_time}", "%Y-%m-%d %H:%M")
    end_dt = datetime.strptime(f"{request.date} {request.end_time}", "%Y-%m-%d %H:%M")
    
    start_timestamp = int(start_dt.timestamp())
    end_timestamp = int(end_dt.timestamp())
    
    room = db.query(MRBSRoom).filter(MRBSRoom.room_name == request.room_name).first()
    if not room:
        raise HTTPException(status_code=404, detail=f"Room '{request.room_name}' not found")
    
    service = BookingService(db)
    return service.update_booking(
        request.booking_id,
        room.id,
        request.name,
        request.date,
        start_timestamp,
        end_timestamp,
        user_email
    )


@router.get("/booking/fetch_moduleCodes_by_user_email")
async def fetch_user_modules(
    email: str,
    db: Session = Depends(get_db),
    user_email: str = Depends(get_current_user_email)
):
    user = db.query(MRBSUser).filter(MRBSUser.email == email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    modules = db.query(MRBSModule).filter(MRBSModule.lecture_id == user.id).all()
    return [module.module_code for module in modules]


@router.get("/booking/all_halls")
async def fetch_all_halls(
    db: Session = Depends(get_db),
    user_email: str = Depends(get_current_user_email)
):
    halls = db.query(MRBSRoom).all()
    return [hall.room_name for hall in halls]


@router.get("/booking/fetch_halls_by_moduleCode")
async def fetch_halls_by_module(
    module_code: str,
    db: Session = Depends(get_db),
    user_email: str = Depends(get_current_user_email)
):
    module = db.query(MRBSModule).filter(MRBSModule.module_code == module_code).first()
    if not module:
        return []
    
    halls = db.query(MRBSRoom).filter(MRBSRoom.capacity >= module.number_of_students).all()
    return [hall.room_name for hall in halls]


@router.get("/bookings/by-date/{date}/{room_id}")
async def get_bookings_by_date(
    date: str,
    room_id: int,
    db: Session = Depends(get_db),
    user_email: str = Depends(get_current_user_email)
):
    service = BookingService(db)
    return service.get_bookings_by_date_and_room(date, room_id)


@router.get("/fetch_user_profile_by_email/{email}")
async def fetch_user_profile(
    email: str,
    db: Session = Depends(get_db),
    user_email: str = Depends(get_current_user_email)
):
    user = db.query(MRBSUser).filter(MRBSUser.email == email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return {
        "id": user.id,
        "name": user.name,
        "email": user.email,
        "role": getattr(user, 'role', 'user')
    }