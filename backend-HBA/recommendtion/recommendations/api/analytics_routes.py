# recommendtion/recommendations/api/analytics_routes.py
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime, date
from ..core.recommendation_engine import RecommendationEngine
from src.database import get_db
import json

router = APIRouter(prefix="/analytics", tags=["analytics"])

class DateRange(BaseModel):
    start_date: date
    end_date: date

async def get_engine(db: Session = Depends(get_db)) -> RecommendationEngine:
    """Helper to get recommendation engine"""
    return RecommendationEngine(db)

async def analytics_handler(func, success_data: dict):
    """Generic analytics request handler with error handling"""
    try:
        result = await func()
        return {"status": "success", **success_data, **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analytics request failed: {str(e)}")

@router.get("/system_stats/")
async def get_system_analytics(
    days: int = Query(30, description="Number of days to analyze"),
    engine: RecommendationEngine = Depends(get_engine)
):
    """Get overall system analytics and performance metrics"""
    return await analytics_handler(
        lambda: engine.analytics.get_system_analytics(days=days).then(lambda analytics: {"analytics": analytics}),
        {"period_days": days}
    )

@router.get("/room_utilization/")
async def get_room_utilization(
    days: int = Query(30, description="Number of days to analyze"),
    engine: RecommendationEngine = Depends(get_engine)
):
    """Get room utilization rates and patterns"""
    return await analytics_handler(
        lambda: engine.analytics.get_room_utilization(days=days).then(lambda utilization: {"utilization": utilization}),
        {"period_days": days}
    )

@router.get("/peak_hours/")
async def get_peak_hours(
    room_name: Optional[str] = Query(None, description="Specific room to analyze"),
    engine: RecommendationEngine = Depends(get_engine)
):
    """Get peak booking hours analysis"""
    return await analytics_handler(
        lambda: engine.analytics.get_peak_hours(room_name=room_name).then(lambda peak_hours: {"peak_hours": peak_hours}),
        {"room_name": room_name or "all_rooms"}
    )

@router.get("/booking_trends/")
async def get_booking_trends(
    period: str = Query("weekly", description="Trend period: daily, weekly, monthly"),
    engine: RecommendationEngine = Depends(get_engine)
):
    """Get booking trends over time"""
    return await analytics_handler(
        lambda: engine.analytics.get_booking_trends(period=period).then(lambda trends: {"trends": trends}),
        {"period": period}
    )

@router.get("/user_activity/{user_id}")
async def get_user_activity(
    user_id: str,
    days: int = Query(90, description="Number of days to analyze"),
    engine: RecommendationEngine = Depends(get_engine)
):
    """Get detailed user activity and booking patterns"""
    return await analytics_handler(
        lambda: engine.analytics.get_user_activity(user_id=user_id, days=days).then(lambda activity: {"activity": activity}),
        {"user_id": user_id, "period_days": days}
    )

@router.get("/recommendation_effectiveness/")
async def get_recommendation_effectiveness(
    days: int = Query(30, description="Number of days to analyze"),
    engine: RecommendationEngine = Depends(get_engine)
):
    """Get recommendation system effectiveness metrics"""
    return await analytics_handler(
        lambda: engine.analytics.get_recommendation_effectiveness(days=days).then(lambda effectiveness: {"effectiveness": effectiveness}),
        {"period_days": days}
    )

@router.get("/popular_times/")
async def get_popular_booking_times(
    room_name: Optional[str] = Query(None, description="Specific room to analyze"),
    limit: int = Query(20, description="Number of popular times to return"),
    engine: RecommendationEngine = Depends(get_engine)
):
    """Get most popular booking times and durations"""
    return await analytics_handler(
        lambda: engine.analytics.get_popular_booking_times(room_name=room_name, limit=limit).then(
            lambda popular_times: {"popular_times": popular_times}
        ),
        {"room_name": room_name or "all_rooms", "limit": limit}
    )

@router.get("/cancellation_patterns/")
async def get_cancellation_patterns(
    days: int = Query(60, description="Number of days to analyze"),
    engine: RecommendationEngine = Depends(get_engine)
):
    """Get booking cancellation patterns and reasons"""
    return await analytics_handler(
        lambda: engine.analytics.get_cancellation_patterns(days=days).then(
            lambda patterns: {"cancellation_patterns": patterns}
        ),
        {"period_days": days}
    )

@router.get("/room_conflicts/")
async def get_room_conflicts(
    days: int = Query(7, description="Number of days to analyze"),
    engine: RecommendationEngine = Depends(get_engine)
):
    """Get room booking conflicts and overlaps"""
    return await analytics_handler(
        lambda: engine.analytics.get_room_conflicts(days=days).then(lambda conflicts: {"conflicts": conflicts}),
        {"period_days": days}
    )

@router.get("/efficiency_metrics/")
async def get_efficiency_metrics(engine: RecommendationEngine = Depends(get_engine)):
    """Get overall system efficiency and performance metrics"""
    return await analytics_handler(
        lambda: engine.analytics.get_efficiency_metrics().then(lambda metrics: {"efficiency_metrics": metrics}),
        {}
    )