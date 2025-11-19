from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Dict, List, Any
from datetime import datetime
from ..core.enhanced_recommendation_engine import EnhancedRecommendationEngine
from ..core.recommendation_engine import RecommendationEngine
from src.database import get_db

router = APIRouter(prefix="/api/v2/recommendations", tags=["Enhanced Recommendations"])

@router.post("/intelligent")
async def get_intelligent_recommendations(request_data: Dict[str, Any], db: Session = Depends(get_db)):
    """Get intelligent recommendations using ML and LLM with graceful fallback"""
    try:
        engine = EnhancedRecommendationEngine(db=db)
        recs = engine.get_enhanced_recommendations(request_data)
        return {
            "success": True, "recommendations": recs, "total_count": len(recs),
            "enhancement_available": any(r.get('enhancement_type') == 'ml_llm_enhanced' for r in recs),
            "timestamp": datetime.now().isoformat()
        }
    except Exception:
        try:
            std_engine = RecommendationEngine(db=db)
            recs = std_engine.get_recommendations(request_data)
            return {
                "success": True, "recommendations": recs, "total_count": len(recs),
                "enhancement_available": False, "fallback_used": True,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Both engines failed: {str(e)}")

@router.get("/engine-status")
async def get_engine_status(db: Session = Depends(get_db)):
    """Get status of both standard and enhanced engines"""
    try:
        std_engine = RecommendationEngine(db=db)
        enh_engine = EnhancedRecommendationEngine(db=db)
        
        return {
            "standard_engine": std_engine.get_engine_status(),
            "enhanced_engine": {
                "ml_embeddings_available": enh_engine.enhanced_embeddings is not None,
                "llm_processor_available": enh_engine.llm_processor is not None,
                "enhancement_ready": (enh_engine.enhanced_embeddings is not None or enh_engine.llm_processor is not None)
            },
            "recommendation_modes": {
                "standard": True,
                "ml_enhanced": enh_engine.enhanced_embeddings is not None,
                "llm_enhanced": enh_engine.llm_processor is not None,
                "hybrid": (enh_engine.enhanced_embeddings is not None and enh_engine.llm_processor is not None)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

enhanced_recommendation_router = router