from sqlalchemy import Column, Integer, String, Text, ForeignKey, Boolean
from sqlalchemy.orm import relationship

from config.database_config import Base


class MRBSArea(Base):
    """Building area or floor"""
    __tablename__ = "mrbs_area"

    id = Column(Integer, primary_key=True, autoincrement=True)
    area_name = Column(String(30), nullable=False, unique=True)
    disabled = Column(Boolean, nullable=False, default=False)
    morningstarts = Column(Integer, nullable=False, default=7)
    eveningends = Column(Integer, nullable=False, default=19)

    rooms = relationship("MRBSRoom", back_populates="area")


class MRBSRoom(Base):
    """Room or hall for booking"""
    __tablename__ = "mrbs_room"

    id = Column(Integer, primary_key=True, autoincrement=True)
    disabled = Column(Boolean, nullable=False, default=False)
    area_id = Column(Integer, ForeignKey("mrbs_area.id", onupdate="CASCADE"), nullable=False, default=0)
    room_name = Column(String(25), nullable=False, unique=True)
    sort_key = Column(String(25), nullable=False, default="")
    description = Column(String(60), nullable=True)
    capacity = Column(Integer, nullable=False, default=0)
    room_admin_email = Column(Text, nullable=True)
    custom_html = Column(Text, nullable=True)

    area = relationship("MRBSArea", back_populates="rooms")
    bookings = relationship("MRBSEntry", back_populates="room")