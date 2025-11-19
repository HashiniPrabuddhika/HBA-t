# recommendtion/recommendations/api/recommendation_routes.py
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from ..core.recommendation_engine import RecommendationEngine
from src.database import get_db
import json

# Create the router instance - this is what the test is looking for
router = APIRouter(prefix="/recommendations", tags=["recommendations"])

class RecommendationRequest(BaseModel):
    user_id: str
    request_type: str  # 'alternative_time', 'alternative_room', 'proactive', 'smart_scheduling', 'comprehensive'
    context: Dict[str, Any]

class FeedbackRequest(BaseModel):
    user_id: str
    recommendation_id: str
    action: str  # 'accepted', 'rejected', 'ignored'
    booking_data: Optional[Dict[str, Any]] = None

async def get_engine(db: Session = Depends(get_db)) -> RecommendationEngine:
    """Helper to get recommendation engine"""
    return RecommendationEngine(db)

async def handle_request(func, success_data: dict):
    """Generic request handler with error handling"""
    try:
        result = await func()
        return {"status": "success", **success_data, **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Request failed: {str(e)}")

@router.post("/get_recommendations/")
async def get_recommendations(
    request: RecommendationRequest,
    engine: RecommendationEngine = Depends(get_engine)
):
    """Get recommendations based on request type and context"""
    return await handle_request(
        lambda: engine.get_recommendations(
            user_id=request.user_id, request_type=request.request_type, context=request.context
        ).then(lambda recommendations: {"recommendations": recommendations}),
        {"user_id": request.user_id, "request_type": request.request_type}
    )

@router.post("/feedback/")
async def provide_feedback(
    request: FeedbackRequest,
    engine: RecommendationEngine = Depends(get_engine)
):
    """Provide feedback on recommendations to improve future suggestions"""
    return await handle_request(
        lambda: engine.learn_from_booking(
            user_id=request.user_id,
            booking_data=request.booking_data or {},
            outcome="success" if request.action == "accepted" else "rejected"
        ).then(lambda _: {}),
        {"message": "Feedback recorded successfully"}
    )

@router.get("/user_patterns/{user_id}")
async def get_user_patterns(
    user_id: str,
    engine: RecommendationEngine = Depends(get_engine)
):
    """Get user booking patterns and preferences"""
    return await handle_request(
        lambda: engine.analytics.get_user_booking_patterns(user_id).then(lambda patterns: {"patterns": patterns}),
        {"user_id": user_id}
    )

@router.get("/room_analytics/{room_name}")
async def get_room_analytics(
    room_name: str,
    engine: RecommendationEngine = Depends(get_engine)
):
    """Get analytics for a specific room"""
    return await handle_request(
        lambda: engine.analytics.get_room_analytics(room_name).then(lambda analytics: {"analytics": analytics}),
        {"room_name": room_name}
    )

@router.get("/optimal_times/")
async def get_optimal_booking_times(
    limit: int = Query(10, description="Number of optimal times to return"),
    engine: RecommendationEngine = Depends(get_engine)
):
    """Get optimal booking times across all rooms"""
    return await handle_request(
        lambda: engine.analytics.get_optimal_booking_times(limit=limit).then(
            lambda optimal_times: {"optimal_times": optimal_times}
        ),
        {}
    )

recommendation_router = router