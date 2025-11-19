from pydantic import BaseModel, Field, validator
from typing import Optional
from datetime import date


class BookingCreate(BaseModel):
    """Schema for creating a booking"""
    
    room_name: str = Field(..., min_length=1, max_length=100)
    name: str = Field(..., min_length=1, max_length=100)
    date: str = Field(..., pattern=r'^\d{4}-\d{2}-\d{2}$')
    start_time: str = Field(..., pattern=r'^\d{2}:\d{2}$')
    end_time: str = Field(..., pattern=r'^\d{2}:\d{2}$')
    
    @validator('date')
    def validate_date_format(cls, v):
        """Validate date format"""
        try:
            from datetime import datetime
            datetime.strptime(v, '%Y-%m-%d')
            return v
        except ValueError:
            raise ValueError('Date must be in YYYY-MM-DD format')
    
    @validator('start_time', 'end_time')
    def validate_time_format(cls, v):
        """Validate time format"""
        try:
            from datetime import datetime
            datetime.strptime(v, '%H:%M')
            return v
        except ValueError:
            raise ValueError('Time must be in HH:MM format')


class BookingUpdate(BaseModel):
    """Schema for updating a booking"""
    
    room_id: int
    name: str = Field(..., min_length=1, max_length=100)
    date: str = Field(..., pattern=r'^\d{4}-\d{2}-\d{2}$')
    start_time: str = Field(..., pattern=r'^\d{2}:\d{2}$')
    end_time: str = Field(..., pattern=r'^\d{2}:\d{2}$')


class BookingResponse(BaseModel):
    """Schema for booking response"""
    
    status: str
    message: str
    booking_id: Optional[int] = None
    room: Optional[str] = None
    date: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    created_by: Optional[str] = None


class AvailabilityCheck(BaseModel):
    """Schema for availability check"""
    
    room_name: str
    date: str
    start_time: str
    end_time: str


class TimeSlot(BaseModel):
    """Schema for time slot"""
    
    start_time: str
    end_time: str


class AvailableSlotsResponse(BaseModel):
    """Schema for available slots response"""
    
    room: str
    date: str
    available_slots: list[TimeSlot]