from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session, joinedload
from datetime import datetime
from pydantic import BaseModel
from typing import Optional

from config.database_config import get_db
from models.booking import MRBSEntry, MRBSSwapRequest
from models.user import MRBSUser, MRBSModule
from middleware.auth import get_current_user_email
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


class SwapRequestCreate(BaseModel):
    requested_by_email: str
    requested_booking_id: int
    offered_booking_id: Optional[int] = None


@router.post("/swap/request")
def create_swap(
    payload: SwapRequestCreate,
    db: Session = Depends(get_db),
    user_email: str = Depends(get_current_user_email)
):
    logger.info(f"Creating swap request from {payload.requested_by_email}")
    
    if payload.requested_by_email != user_email:
        raise HTTPException(
            status_code=403, 
            detail="You can only create swap requests for yourself"
        )
    user = db.query(MRBSUser).filter(MRBSUser.email == payload.requested_by_email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    offered_user = None
    if payload.offered_booking_id:
        offered_user = (
            db.query(MRBSUser)
            .join(MRBSModule, MRBSModule.lecture_id == MRBSUser.id)
            .join(MRBSEntry, MRBSEntry.name == MRBSModule.module_code)
            .filter(MRBSEntry.id == payload.offered_booking_id)
            .first()
        )
        
        if not offered_user:
            raise HTTPException(status_code=404, detail="Offered user not found")
    
    swap = MRBSSwapRequest(
        requested_by=user.id,
        requested_booking_id=payload.requested_booking_id,
        offered_booking_id=payload.offered_booking_id,
        offered_by=offered_user.id if offered_user else None,
        status="pending"
    )
    
    db.add(swap)
    db.commit()
    db.refresh(swap)
    
    logger.info(f"Swap request created: ID {swap.id}")
    
    return {
        "message": "Swap request created successfully",
        "swap_id": swap.id,
        "requested_by": user.id,
        "requested_booking_id": payload.requested_booking_id,
        "offered_booking_id": payload.offered_booking_id,
        "offered_by": offered_user.id if offered_user else None,
        "status": swap.status
    }


@router.post("/swap/respond")
def respond_swap(
    swap_id: int,
    response: str,
    db: Session = Depends(get_db),
    user_email: str = Depends(get_current_user_email)
):
    logger.info(f"Responding to swap {swap_id}: {response}")
    
    swap = db.query(MRBSSwapRequest).filter(MRBSSwapRequest.id == swap_id).first()
    if not swap:
        raise HTTPException(status_code=404, detail="Swap request not found")
    
    user = db.query(MRBSUser).filter(MRBSUser.email == user_email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    if user.id not in [swap.requested_by, swap.offered_by]:
        raise HTTPException(
            status_code=403, 
            detail="You are not authorized to respond to this swap request"
        )
        
    requested_booking = db.query(MRBSEntry).filter(MRBSEntry.id == swap.requested_booking_id).first()
    offered_booking = db.query(MRBSEntry).filter(MRBSEntry.id == swap.offered_booking_id).first()
    
    if not requested_booking or not offered_booking:
        raise HTTPException(status_code=404, detail="One or both bookings not found")
    
    if response == "approved":
        temp_start = requested_booking.start_time
        temp_end = requested_booking.end_time
        
        requested_booking.start_time = offered_booking.start_time
        requested_booking.end_time = offered_booking.end_time
        
        offered_booking.start_time = temp_start
        offered_booking.end_time = temp_end
        
        swap.status = "approved"
        
        db.commit()
        
        logger.info(f"Swap approved: ID {swap_id}")
        return {"message": "Swap approved and booking times exchanged successfully"}
    
    elif response == "rejected":
        swap.status = "rejected"
        db.commit()
        
        logger.info(f"Swap rejected: ID {swap_id}")
        return {"message": "Swap rejected"}
    
    else:
        raise HTTPException(status_code=400, detail="Invalid response value. Use 'approved' or 'rejected'.")


@router.get("/swap/pending/{user_id}")
def pending_swaps(
    user_id: int,
    db: Session = Depends(get_db),
    user_email: str = Depends(get_current_user_email)
):
    """Get all pending swap requests for a user"""
    logger.info(f"Fetching pending swaps for user {user_id}")
    
    swaps = (
        db.query(MRBSSwapRequest)
        .join(MRBSEntry, MRBSSwapRequest.requested_booking_id == MRBSEntry.id)
        .filter(
            MRBSEntry.create_by == user_email,
            MRBSSwapRequest.status == "pending"
        )
        .all()
    )
    
    return {
        "pending_requests": [
            {
                "swap_id": s.id,
                "requested_by": s.requested_by,
                "requested_booking_id": s.requested_booking_id
            }
            for s in swaps
        ]
    }


@router.get("/swap/get_all_requests")
def get_all_swaps(
    db: Session = Depends(get_db),
    user_email: str = Depends(get_current_user_email)
):
    """Get all swap requests for a user"""
    logger.info(f"Fetching all swaps for user {user_email}")
    
    try:
        swaps = (
            db.query(MRBSSwapRequest)
            .options(
                joinedload(MRBSSwapRequest.requester),
                joinedload(MRBSSwapRequest.offerer),
                joinedload(MRBSSwapRequest.requested_booking).joinedload(MRBSEntry.room),
                joinedload(MRBSSwapRequest.offered_booking).joinedload(MRBSEntry.room)
            )
            .filter(MRBSSwapRequest.status == "pending")
            .all()
        )


        def format_time(ts: int):
                try:
                    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
                except Exception:
                    return None

        result = []
        for swap in swaps:
                # 🧩 Requested booking’s module code
                requested_module = (
                    db.query(MRBSModule)
                    .filter(MRBSModule.module_code == swap.requested_booking.name)
                    .first()
                    if swap.requested_booking else None
                )

                # 🧩 Offered booking’s module code
                offered_module = (
                    db.query(MRBSModule)
                    .filter(MRBSModule.module_code == swap.offered_booking.name)
                    .first()
                    if swap.offered_booking else None
                )

                # 🕒 Convert Unix timestamps
                requested_start = format_time(swap.requested_booking.start_time) if swap.requested_booking else None
                requested_end = format_time(swap.requested_booking.end_time) if swap.requested_booking else None
                offered_start = format_time(swap.offered_booking.start_time) if swap.offered_booking else None
                offered_end = format_time(swap.offered_booking.end_time) if swap.offered_booking else None

                result.append({
                    "id": swap.id,
                    "status": swap.status,
                    "created_at": swap.timestamp,

                    # 👥 Requester & Offerer Info
                    "requested_by": swap.requested_by,
                    "offered_by": swap.offered_by,
                    "requester_name": swap.requester.name if swap.requester else None,
                    "offerer_name": swap.offerer.name if swap.offerer else None,
                    "requester_email": swap.requester.email if swap.requester else None,
                    "offerer_email": swap.offerer.email if swap.offerer else None,

                    # 📘 Module Info
                    "requested_module_code": requested_module.module_code if requested_module else None,
                    "offered_module_code": offered_module.module_code if offered_module else None,

                    # 🕒 Time Slot Info
                    "requested_time_slot": f"{requested_start} - {requested_end}" if requested_start and requested_end else None,
                    "offered_time_slot": f"{offered_start} - {offered_end}" if offered_start and offered_end else None,

                    # 🏫 Room Info
                    "requested_room_name": swap.requested_booking.room.room_name if swap.requested_booking and swap.requested_booking.room else None,
                    "offered_room_name": swap.offered_booking.room.room_name if swap.offered_booking and swap.offered_booking.room else None,

                    # 📨 Readable Summary Message
                    "message": (
                        f"Swap request from {swap.requester.name if swap.requester else 'Unknown'} "
                        f"({swap.requester.email if swap.requester else 'N/A'}) "
                        f"for booking {swap.requested_booking_id} "
                        f"({requested_module.module_code if requested_module else 'N/A'}) "
                        f"in {swap.requested_booking.room.room_name if swap.requested_booking and swap.requested_booking.room else 'Unknown room'} "
                        f"({requested_start} - {requested_end})"
                        + (
                            f" ↔ offered booking {swap.offered_booking_id} "
                            f"({offered_module.module_code if offered_module else 'N/A'}) "
                            f"in {swap.offered_booking.room.room_name if swap.offered_booking and swap.offered_booking.room else 'Unknown room'} "
                            f"({offered_start} - {offered_end}) "
                            f"by {swap.offerer.name if swap.offerer else 'Unknown'} "
                            f"({swap.offerer.email if swap.offerer else 'N/A'})"
                            if swap.offered_booking else ""
                        )
                    )
                })

        return result   
    except Exception as e:
        logger.error(f"Error fetching swap requests: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching swap requests: {str(e)}")
        

@router.get("/swap/requests")
def get_all_requests(
    db: Session = Depends(get_db),
    user_email: str = Depends(get_current_user_email)
):
    """Get all pending swap requests with detailed information"""
    logger.info("Fetching all pending swap requests")
    
    swaps = (
        db.query(MRBSSwapRequest)
        .options(
            joinedload(MRBSSwapRequest.requester),
            joinedload(MRBSSwapRequest.offerer),
            joinedload(MRBSSwapRequest.requested_booking).joinedload(MRBSEntry.room),
            joinedload(MRBSSwapRequest.offered_booking).joinedload(MRBSEntry.room)
        )
        .filter(MRBSSwapRequest.status == "pending")
        .all()
    )
    
    def format_time(ts: int) -> Optional[str]:
        try:
            return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
        except Exception:
            return None
    
    result = []
    for swap in swaps:
        requested_module = (
            db.query(MRBSModule)
            .filter(MRBSModule.module_code == swap.requested_booking.name)
            .first()
            if swap.requested_booking else None
        )
        
        offered_module = (
            db.query(MRBSModule)
            .filter(MRBSModule.module_code == swap.offered_booking.name)
            .first()
            if swap.offered_booking else None
        )
        
        requested_start = format_time(swap.requested_booking.start_time) if swap.requested_booking else None
        requested_end = format_time(swap.requested_booking.end_time) if swap.requested_booking else None
        offered_start = format_time(swap.offered_booking.start_time) if swap.offered_booking else None
        offered_end = format_time(swap.offered_booking.end_time) if swap.offered_booking else None
        
        result.append({
            "id": swap.id,
            "status": swap.status,
            "created_at": swap.timestamp,
            "requested_by": swap.requested_by,
            "offered_by": swap.offered_by,
            "requester_name": swap.requester.name if swap.requester else None,
            "offerer_name": swap.offerer.name if swap.offerer else None,
            "requester_email": swap.requester.email if swap.requester else None,
            "offerer_email": swap.offerer.email if swap.offerer else None,
            "requested_module_code": requested_module.module_code if requested_module else None,
            "offered_module_code": offered_module.module_code if offered_module else None,
            "requested_time_slot": f"{requested_start} - {requested_end}" if requested_start and requested_end else None,
            "offered_time_slot": f"{offered_start} - {offered_end}" if offered_start and offered_end else None,
            "requested_room_name": swap.requested_booking.room.room_name if swap.requested_booking and swap.requested_booking.room else None,
            "offered_room_name": swap.offered_booking.room.room_name if swap.offered_booking and swap.offered_booking.room else None,
            "message": (
                f"Swap request from {swap.requester.name if swap.requester else 'Unknown'} "
                f"for booking {swap.requested_booking_id} "
                f"({requested_module.module_code if requested_module else 'N/A'}) "
                f"in {swap.requested_booking.room.room_name if swap.requested_booking and swap.requested_booking.room else 'Unknown room'} "
                f"({requested_start} - {requested_end})"
                + (
                    f" ↔ offered booking {swap.offered_booking_id} "
                    f"({offered_module.module_code if offered_module else 'N/A'}) "
                    f"in {swap.offered_booking.room.room_name if swap.offered_booking and swap.offered_booking.room else 'Unknown room'} "
                    f"({offered_start} - {offered_end}) "
                    f"by {swap.offerer.name if swap.offerer else 'Unknown'}"
                    if swap.offered_booking else ""
                )
            )
        })
    
    return result