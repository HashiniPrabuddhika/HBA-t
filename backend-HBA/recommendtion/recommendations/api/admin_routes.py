# recommendtion/recommendations/api/admin_routes.py
from fastapi import APIRouter, Depends, HTTPException, Query, Body
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime, date
from ..core.recommendation_engine import RecommendationEngine
from src.database import get_db
import json

router = APIRouter(prefix="/admin", tags=["admin"])

class SystemConfig(BaseModel):
    recommendation_weight: float = 1.0
    popularity_weight: float = 0.5
    recency_weight: float = 0.3
    user_preference_weight: float = 0.8
    time_penalty_factor: float = 0.1
    max_recommendations: int = 10
    learning_rate: float = 0.01

class BulkUserData(BaseModel):
    user_ids: List[str]
    action: str  # 'export', 'delete', 'reset_preferences'

async def get_engine(db: Session = Depends(get_db)) -> RecommendationEngine:
    """Helper to get recommendation engine"""
    return RecommendationEngine(db)

async def handle_request(func, *args, **kwargs):
    """Generic request handler with error handling"""
    try:
        return {"status": "success", **await func(*args, **kwargs)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Request failed: {str(e)}")

@router.get("/system_health/")
async def get_system_health(engine: RecommendationEngine = Depends(get_engine)):
    """Get comprehensive system health and status"""
    return await handle_request(
        lambda: engine.get_system_health().then(
            lambda health: {"system_health": health, "timestamp": datetime.now().isoformat()}
        )
    )

@router.post("/config/update/")
async def update_system_config(config: SystemConfig, engine: RecommendationEngine = Depends(get_engine)):
    """Update system configuration parameters"""
    return await handle_request(
        lambda: engine.update_system_config(config.dict()).then(
            lambda _: {"message": "System configuration updated successfully", "new_config": config.dict()}
        )
    )

@router.get("/config/current/")
async def get_current_config(engine: RecommendationEngine = Depends(get_engine)):
    """Get current system configuration"""
    return await handle_request(lambda: engine.get_system_config().then(lambda config: {"config": config}))

@router.post("/data/cleanup/")
async def cleanup_old_data(
    days_old: int = Query(90, description="Delete data older than this many days"),
    data_types: List[str] = Query(["bookings", "recommendations", "analytics"], description="Types of data to clean"),
    engine: RecommendationEngine = Depends(get_engine)
):
    """Clean up old system data"""
    return await handle_request(
        lambda: engine.cleanup_old_data(days_old=days_old, data_types=data_types).then(
            lambda result: {"cleanup_result": result, "days_old": days_old, "data_types": data_types}
        )
    )

@router.post("/users/bulk_action/")
async def bulk_user_action(request: BulkUserData, engine: RecommendationEngine = Depends(get_engine)):
    """Perform bulk actions on user data"""
    actions = {
        "export": engine.export_user_data,
        "delete": engine.delete_user_data,
        "reset_preferences": engine.reset_user_preferences
    }
    
    if request.action not in actions:
        raise HTTPException(status_code=400, detail=f"Invalid action: {request.action}")
    
    return await handle_request(
        lambda: actions[request.action](request.user_ids).then(
            lambda result: {"action": request.action, "affected_users": len(request.user_ids), "result": result}
        )
    )

@router.get("/performance/metrics/")
async def get_performance_metrics(
    hours: int = Query(24, description="Number of hours to analyze"),
    engine: RecommendationEngine = Depends(get_engine)
):
    """Get detailed performance metrics"""
    return await handle_request(
        lambda: engine.get_performance_metrics(hours=hours).then(
            lambda metrics: {"performance_metrics": metrics, "analysis_hours": hours}
        )
    )

@router.post("/recommendations/retrain/")
async def retrain_recommendation_model(
    force: bool = Query(False, description="Force retrain even if recent training exists"),
    engine: RecommendationEngine = Depends(get_engine)
):
    """Trigger recommendation model retraining"""
    return await handle_request(
        lambda: engine.retrain_model(force=force).then(
            lambda result: {"retrain_result": result, "forced": force}
        )
    )

@router.get("/audit/logs/")
async def get_audit_logs(
    limit: int = Query(100, description="Number of logs to return"),
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    action_type: Optional[str] = Query(None, description="Filter by action type"),
    engine: RecommendationEngine = Depends(get_engine)
):
    """Get system audit logs"""
    return await handle_request(
        lambda: engine.get_audit_logs(limit=limit, user_id=user_id, action_type=action_type).then(
            lambda logs: {"audit_logs": logs, "filters": {"limit": limit, "user_id": user_id, "action_type": action_type}}
        )
    )

@router.get("/rooms/management/")
async def get_room_management_data(engine: RecommendationEngine = Depends(get_engine)):
    """Get comprehensive room management data"""
    return await handle_request(
        lambda: engine.get_room_management_data().then(lambda data: {"room_management_data": data})
    )

@router.post("/alerts/configure/")
async def configure_system_alerts(
    alert_config: Dict[str, Any] = Body(...),
    engine: RecommendationEngine = Depends(get_engine)
):
    """Configure system alerts and notifications"""
    return await handle_request(
        lambda: engine.configure_alerts(alert_config).then(
            lambda _: {"message": "Alert configuration updated successfully", "alert_config": alert_config}
        )
    )

@router.get("/backup/create/")
async def create_system_backup(
    include_user_data: bool = Query(True, description="Include user data in backup"),
    engine: RecommendationEngine = Depends(get_engine)
):
    """Create system backup"""
    return await handle_request(
        lambda: engine.create_system_backup(include_user_data=include_user_data).then(
            lambda result: {"backup_result": result, "included_user_data": include_user_data}
        )
    )

@router.post("/maintenance/mode/")
async def toggle_maintenance_mode(
    enable: bool = Query(..., description="Enable or disable maintenance mode"),
    message: str = Query("System maintenance in progress", description="Maintenance message"),
    engine: RecommendationEngine = Depends(get_engine)
):
    """Toggle system maintenance mode"""
    return await handle_request(
        lambda: engine.set_maintenance_mode(enable=enable, message=message).then(
            lambda _: {"maintenance_mode": enable, "message": message}
        )
    )