from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship

from config.database_config import Base


class MRBSUser(Base):
    """System user"""
    __tablename__ = "mrbs_users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    email = Column(String(100), nullable=False, unique=True)
    name = Column(String(100), nullable=False)

    modules = relationship("MRBSModule", back_populates="lecturer", cascade="all, delete-orphan")


class MRBSModule(Base):
    """Module or course"""
    __tablename__ = "mrbs_module"

    id = Column(Integer, primary_key=True, autoincrement=True)
    module_code = Column(String(50), nullable=False, unique=True)
    number_of_students = Column(Integer, nullable=False)
    lecture_id = Column(Integer, ForeignKey("mrbs_users.id", onupdate="CASCADE", ondelete="CASCADE"), nullable=False)

    lecturer = relationship("MRBSUser", back_populates="modules")