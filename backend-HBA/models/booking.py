from sqlalchemy import Column, Integer, String, Text, ForeignKey, TIMESTAMP, SmallInteger, func
from sqlalchemy.orm import relationship

from config.database_config import Base


class MRBSRepeat(Base):
    """Recurring booking template"""
    __tablename__ = "mrbs_repeat"

    id = Column(Integer, primary_key=True, autoincrement=True)
    start_time = Column(Integer, nullable=False, default=0)
    end_time = Column(Integer, nullable=False, default=0)
    entry_type = Column(Integer, nullable=False, default=0)
    timestamp = Column(TIMESTAMP, nullable=False)
    create_by = Column(String(80), nullable=False, default="")
    modified_by = Column(String(80), nullable=False, default="")
    name = Column(String(80), nullable=False, default="")
    type = Column(String(1), nullable=False, default="E")
    description = Column(Text, nullable=True)
    status = Column(SmallInteger, nullable=False, default=0)
    reminded = Column(Integer, nullable=True)
    info_time = Column(Integer, nullable=True)
    info_user = Column(String(80), nullable=True)
    info_text = Column(Text, nullable=True)
    ical_uid = Column(String(255), nullable=False, default="")
    ical_sequence = Column(SmallInteger, nullable=False, default=0)
    ical_recur_id = Column(String(16), nullable=True)

    entries = relationship("MRBSEntry", back_populates="repeat")


class MRBSEntry(Base):
    """Individual booking entry"""
    __tablename__ = "mrbs_entry"

    id = Column(Integer, primary_key=True, autoincrement=True)
    start_time = Column(Integer, nullable=False, default=0)
    end_time = Column(Integer, nullable=False, default=0)
    entry_type = Column(Integer, nullable=False, default=0)
    repeat_id = Column(Integer, ForeignKey("mrbs_repeat.id", ondelete="CASCADE", onupdate="CASCADE"), nullable=True)
    room_id = Column(Integer, ForeignKey("mrbs_room.id", onupdate="CASCADE"), nullable=False, default=1)
    timestamp = Column(TIMESTAMP, nullable=False, server_default=func.now())
    create_by = Column(String(80), nullable=False, default="")
    modified_by = Column(String(80), nullable=False, default="")
    name = Column(String(80), nullable=False, default="")
    type = Column(String(1), nullable=False, default="E")
    description = Column(Text, nullable=True)
    status = Column(SmallInteger, nullable=False, default=0)
    reminded = Column(Integer, nullable=True)
    info_time = Column(Integer, nullable=True)
    info_user = Column(String(80), nullable=True)
    info_text = Column(Text, nullable=True)
    ical_uid = Column(String(255), nullable=False, default="")
    ical_sequence = Column(SmallInteger, nullable=False, default=0)
    ical_recur_id = Column(String(16), nullable=True)

    room = relationship("MRBSRoom", back_populates="bookings")
    repeat = relationship("MRBSRepeat", back_populates="entries")


class MRBSSwapRequest(Base):
    """Booking swap request"""
    __tablename__ = "swap_requests"

    id = Column(Integer, primary_key=True, autoincrement=True)
    requested_by = Column(Integer, ForeignKey("mrbs_users.id", onupdate="CASCADE", ondelete="CASCADE"), nullable=False)
    requested_booking_id = Column(Integer, ForeignKey("mrbs_entry.id", onupdate="CASCADE", ondelete="CASCADE"), nullable=False)
    offered_booking_id = Column(Integer, ForeignKey("mrbs_entry.id", onupdate="CASCADE", ondelete="SET NULL"), nullable=True)
    status = Column(String(20), nullable=False, default="pending")
    timestamp = Column("created_at", TIMESTAMP, nullable=False, server_default=func.now())
    offered_by = Column(Integer, ForeignKey("mrbs_users.id", onupdate="CASCADE", ondelete="SET NULL"), nullable=True)

    requester = relationship("MRBSUser", foreign_keys=[requested_by])
    offerer = relationship("MRBSUser", foreign_keys=[offered_by])
    requested_booking = relationship("MRBSEntry", foreign_keys=[requested_booking_id])
    offered_booking = relationship("MRBSEntry", foreign_keys=[offered_booking_id])