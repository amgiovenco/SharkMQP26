from sqlalchemy import Column, ForeignKey, String, Integer, DateTime, Text, JSON, func, Enum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.inspection import inspect as sa_inspect
import uuid
import enum
from typing import Any, Dict
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
    role = Column(Enum(UserRole, name="user_roles"), nullable=False, default=UserRole.user) # admin|researcher|user
    first_name = Column(String(80), nullable=True)
    last_name = Column(String(80), nullable=True)
    job_title = Column(String(120), nullable=True)

    def __repr__(self) -> str:
        role_val = self.role.value if isinstance(self.role, enum.Enum) else self.role
        return f"<User id={self.id} username={self.username!r} role={role_val!r}>"

    @property
    def full_name(self) -> str:
        parts = [p for p in (self.first_name, self.last_name) if p]
        return " ".join(parts) if parts else ""

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {}
        for col in sa_inspect(self.__class__).columns:
            val = getattr(self, col.name)
            # SQLAlchemy Enum -> python enum
            if isinstance(val, enum.Enum):
                data[col.name] = val.value
            # datetime -> string
            elif hasattr(val, "isoformat"):
                try:
                    data[col.name] = val.isoformat()
                except Exception:
                    data[col.name] = str(val)
            # UUID -> string
            elif isinstance(val, uuid.UUID):
                data[col.name] = str(val)
            else:
                data[col.name] = val
        return data


class Job(Base):
    __tablename__ = "jobs"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    case_id = Column(String(64), nullable=True)
    file_path = Column(Text, nullable=False)
    sha256 = Column(String(64), nullable=False)
    status = Column(String(16), nullable=False, default="queued") # queued|running|done|error
    created_at = Column(DateTime(timezone=True), server_default=func.now()) # time when job was created
    started_at = Column(DateTime(timezone=True), nullable=True) # time when job started processing
    finished_at = Column(DateTime(timezone=True), nullable=True) # time when job finished processing
    result_json = Column(JSON, nullable=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    def __repr__(self) -> str:
        return f"<Job id={self.id} sha256={self.sha256!r} status={self.status!r}>"

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {}
        for col in sa_inspect(self.__class__).columns:
            val = getattr(self, col.name)
            # SQLAlchemy Enum -> python enum
            if isinstance(val, enum.Enum):
                data[col.name] = val.value
            # datetime -> string
            elif hasattr(val, "isoformat"):
                try:
                    data[col.name] = val.isoformat()
                except Exception:
                    data[col.name] = str(val)
            # UUID -> string
            elif isinstance(val, uuid.UUID):
                data[col.name] = str(val)
            else:
                data[col.name] = val
        return data