from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class SwapRequestCreate(BaseModel):
    """Schema for creating a swap request"""
    
    requested_booking_id: int = Field(..., gt=0)
    offered_booking_id: Optional[int] = Field(None, gt=0)


class SwapResponse(BaseModel):
    """Schema for swap request response"""
    
    message: str
    swap_id: int
    requested_by: int
    requested_booking_id: int
    offered_booking_id: Optional[int]
    offered_by: Optional[int]
    status: str


class SwapResponseFull(BaseModel):
    """Schema for detailed swap request"""
    
    id: int
    status: str
    created_at: datetime
    requested_by: int
    offered_by: Optional[int]
    requester_name: Optional[str]
    offerer_name: Optional[str]
    requester_email: Optional[str]
    offerer_email: Optional[str]
    requested_module_code: Optional[str]
    offered_module_code: Optional[str]
    requested_time_slot: Optional[str]
    offered_time_slot: Optional[str]
    requested_room_name: Optional[str]
    offered_room_name: Optional[str]
    
    class Config:
        from_attributes = True