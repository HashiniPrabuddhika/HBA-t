from sqlalchemy import Column, Integer, String, Text, ForeignKey, TIMESTAMP, SmallInteger, Boolean, func
from sqlalchemy.orm import relationship
from src.database import Base

class MRBSArea(Base):
    __tablename__ = "mrbs_area"

    id = Column(Integer, primary_key=True, autoincrement=True)
    area_name = Column(String(30), nullable=False, unique=True)
    disabled = Column(Boolean, nullable=False, default=False)
    
    morningstarts = Column(Integer, nullable=False, default=7)   # Opens at 7 AM
    eveningends = Column(Integer, nullable=False, default=19)    # Closes at 7 PM

    # Relationship to rooms
    rooms = relationship("MRBSRoom", back_populates="area")
  
class MRBSRoom(Base):
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

    # Relationship to bookings
    area = relationship("MRBSArea", back_populates="rooms")
    bookings = relationship("MRBSEntry", back_populates="room")


class MRBSRepeat(Base):
    __tablename__ = "mrbs_repeat"

    id = Column(Integer, primary_key=True, autoincrement=True)
    start_time = Column(Integer, nullable=False, default=0)  # Unix timestamp
    end_time = Column(Integer, nullable=False, default=0)  # Unix timestamp
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

    # Relationship to entries
    entries = relationship("MRBSEntry", back_populates="repeat")


class MRBSEntry(Base):
    __tablename__ = "mrbs_entry"

    id = Column(Integer, primary_key=True, autoincrement=True)
    start_time = Column(Integer, nullable=False, default=0)  # Unix timestamp
    end_time = Column(Integer, nullable=False, default=0)  # Unix timestamp
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

    # Relationships
    room = relationship("MRBSRoom", back_populates="bookings")
    repeat = relationship("MRBSRepeat", back_populates="entries")

class MRBSModule(Base):
    __tablename__ = "mrbs_module"

    id = Column(Integer, primary_key=True, autoincrement=True)
    module_code = Column(String(50), nullable=False, unique=True)
    number_of_students = Column(Integer, nullable=False)
    lecture_id = Column(Integer, ForeignKey("mrbs_users.id", onupdate="CASCADE", ondelete="CASCADE"), nullable=False)

    # Relationship to Lecturer (assuming MRBSUser is your users table)
    lecturer = relationship("MRBSUser", back_populates="modules")
    
class MRBSUser(Base):
    __tablename__ = "mrbs_users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    email = Column(String(100), nullable=False, unique=True)
    name = Column(String(100), nullable=False)

    # Add this:
    modules = relationship("MRBSModule", back_populates="lecturer", cascade="all, delete-orphan")

class MRBSSwapRequest(Base):
    __tablename__ = "swap_requests"

    id = Column(Integer, primary_key=True, autoincrement=True)
    requested_by = Column(Integer, ForeignKey("mrbs_users.id", onupdate="CASCADE", ondelete="CASCADE"), nullable=False)
    requested_booking_id = Column(Integer, ForeignKey("mrbs_entry.id", onupdate="CASCADE", ondelete="CASCADE"), nullable=False)
    offered_booking_id = Column(Integer, ForeignKey("mrbs_entry.id", onupdate="CASCADE", ondelete="SET NULL"), nullable=True)
    status = Column(String(20), nullable=False, default="pending")  # 'pending', 'approved', 'rejected'
    timestamp = Column("created_at", TIMESTAMP, nullable=False, server_default=func.now())
    offered_by = Column(Integer, ForeignKey("mrbs_users.id", onupdate="CASCADE", ondelete="SET NULL"), nullable=True)

    # Relationships
    requester = relationship("MRBSUser", foreign_keys=[requested_by])
    offerer = relationship("MRBSUser", foreign_keys=[offered_by])
    requested_booking = relationship("MRBSEntry", foreign_keys=[requested_booking_id])
    offered_booking = relationship("MRBSEntry", foreign_keys=[offered_booking_id])

