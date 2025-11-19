from sqlalchemy import Column, Integer, String, ForeignKey, TIMESTAMP, func
from sqlalchemy.orm import relationship
from datetime import datetime
from utils.database import Base


class MRBSSwapRequest(Base):
    """Booking swap request model"""
    
    __tablename__ = "swap_requests"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    requested_by = Column(
        Integer,
        ForeignKey("mrbs_users.id", onupdate="CASCADE", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    requested_booking_id = Column(
        Integer,
        ForeignKey("mrbs_entry.id", onupdate="CASCADE", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    offered_booking_id = Column(
        Integer,
        ForeignKey("mrbs_entry.id", onupdate="CASCADE", ondelete="SET NULL"),
        nullable=True,
        index=True
    )
    status = Column(
        String(20),
        nullable=False,
        default="pending",
        index=True
    )
    timestamp = Column(
        "created_at",
        TIMESTAMP,
        nullable=False,
        server_default=func.now()
    )
    offered_by = Column(
        Integer,
        ForeignKey("mrbs_users.id", onupdate="CASCADE", ondelete="SET NULL"),
        nullable=True,
        index=True
    )
    response_timestamp = Column(TIMESTAMP, nullable=True)
    notes = Column(String(500), nullable=True)
    
    requester = relationship(
        "MRBSUser",
        foreign_keys=[requested_by],
        back_populates="requested_swaps"
    )
    offerer = relationship(
        "MRBSUser",
        foreign_keys=[offered_by],
        back_populates="offered_swaps"
    )
    requested_booking = relationship(
        "MRBSEntry",
        foreign_keys=[requested_booking_id]
    )
    offered_booking = relationship(
        "MRBSEntry",
        foreign_keys=[offered_booking_id]
    )
    
    def __repr__(self):
        return f"<MRBSSwapRequest(id={self.id}, status='{self.status}', requested_by={self.requested_by})>"
    
    @property
    def is_pending(self) -> bool:
        """Check if swap request is pending"""
        return self.status == "pending"
    
    @property
    def is_approved(self) -> bool:
        """Check if swap request is approved"""
        return self.status == "approved"
    
    @property
    def is_rejected(self) -> bool:
        """Check if swap request is rejected"""
        return self.status == "rejected"
    
    @property
    def is_expired(self) -> bool:
        """Check if swap request is expired (older than 7 days and still pending)"""
        if not self.is_pending:
            return False
        
        days_old = (datetime.now() - self.timestamp).days
        return days_old > 7
    
    @property
    def age_hours(self) -> int:
        """Get age of swap request in hours"""
        return int((datetime.now() - self.timestamp).total_seconds() / 3600)
    
    def approve(self):
        """Approve the swap request"""
        self.status = "approved"
        self.response_timestamp = datetime.now()
    
    def reject(self, notes: str = None):
        """Reject the swap request"""
        self.status = "rejected"
        self.response_timestamp = datetime.now()
        if notes:
            self.notes = notes
    
    def can_be_responded_by(self, user_id: int) -> bool:
        """Check if user can respond to this swap request"""
        return self.offered_by == user_id or self.requested_by == user_id