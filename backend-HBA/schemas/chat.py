from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List


class ChatRequest(BaseModel):
    """Schema for chat request"""
    
    session_id: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)


class ChatResponse(BaseModel):
    """Schema for chat response"""
    
    status: str
    message: str
    missing_parameter: Optional[str] = None
    error: Optional[str] = None
    suggestions: Optional[Dict[str, List[str]]] = None
    booking_id: Optional[int] = None
    room: Optional[str] = None
    date: Optional[str] = None
    available_slots: Optional[List[Dict[str, str]]] = None
    recommendations: Optional[List[Dict[str, Any]]] = None