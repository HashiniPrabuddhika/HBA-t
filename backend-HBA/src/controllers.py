# from fastapi import APIRouter, Depends, HTTPException
# from sqlalchemy.orm import Session
# from typing import Dict
# from datetime import datetime
# from availability_logic import services  # assuming your service code is here
# from .database import get_db  # function to get a DB session

# router = APIRouter(prefix="/booking", tags=["Booking"])

# # -----------------------------
# # Check availability for a room
# # -----------------------------
# @router.get("/check_availability")
# def api_check_availability(
#     room_name: str,
#     date: str,
#     start_time: str,
#     end_time: str,
#     db: Session = Depends(get_db)
# ):
#     return services.check_availability(room_name, date, start_time, end_time, db)


# # -----------------------------
# # Add a booking
# # -----------------------------
# @router.post("/add")
# def api_add_booking(
#     room_name: str,
#     date: str,
#     start_time: str,
#     end_time: str,
#     created_by: str,
#     db: Session = Depends(get_db)
# ):
#     return services.add_booking(room_name, date, start_time, end_time, created_by, db)


# # -----------------------------
# # Get available time slots
# # -----------------------------
# @router.get("/available_slots")
# def api_available_slots(
#     room_name: str,
#     date: str,
#     db: Session = Depends(get_db)
# ):
#     # For slot generation, you can pass dummy start/end times (ignored in service)
#     return services.check_available_slotes(room_name, date, "00:00", "23:59", db)


# # -----------------------------
# # Book a recommendation directly
# # -----------------------------
# @router.post("/book_recommendation")
# def api_book_recommendation(
#     recommendation: Dict,
#     created_by: str,
#     db: Session = Depends(get_db)
# ):
#     return services.book_recommendation_directly(recommendation, created_by, db)


# # -----------------------------
# # Get room recommendations
# # -----------------------------
# @router.get("/recommendations")
# def api_get_recommendations(
#     room_name: str,
#     date: str,
#     start_time: str,
#     end_time: str,
#     db: Session = Depends(get_db)
# ):
#     return services.get_room_recommendations(room_name, date, start_time, end_time, db)
