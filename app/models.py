from sqlalchemy import Column, String, Integer, DateTime, Text, JSON, func, Enum
from sqlalchemy.dialects.postgresql import UUID
import uuid
import enum
from .db import Base

class UserRole(enum.Enum):
    admin = "admin"
    researcher = "researcher"
    user = "user"

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String(80), unique=True, nullable=False)
    password_hash = Column(String(200), nullable=False)
    role = Column(Enum(UserRole, name="user_roles"), nullable=False, default=UserRole.user)

class Job(Base):
    __tablename__ = "jobs"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    case_id = Column(String(64), nullable=True)
    file_path = Column(Text, nullable=False)
    sha256 = Column(String(64), nullable=False)
    status = Column(String(16), nullable=False, default="queued") # queued|running|done|error
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    result_json = Column(JSON, nullable=True)