from sqlalchemy import Column, ForeignKey, String, Integer, DateTime, Text, JSON, func, Enum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
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

    # relationships
    jobs = relationship("Job", back_populates="user", lazy="select")
    research_cases = relationship("Case", back_populates="researcher", lazy="select", foreign_keys="Case.researcher_id")

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


class Case(Base):
    """
    A Case groups multiple jobs. It represents a dataset / submission brought in by a person
    and handled by a researcher. Uses UUID primary key.
    """
    __tablename__ = "cases"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(120), nullable=True)  # brief title for the case
    description = Column(Text, nullable=True)
    person_name = Column(String(120), nullable=True)      # person who brought in the data

    # researcher relationship (replaces researcher_name string)
    researcher_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    researcher = relationship("User", back_populates="research_cases", lazy="joined", foreign_keys=[researcher_id])

    data_created = Column(DateTime(timezone=True), nullable=True)  # when the underlying data was created
    created_at = Column(DateTime(timezone=True), server_default=func.now())  # when case record was created
    case_metadata = Column("metadata", JSON, nullable=True)

    # relationship to jobs
    jobs = relationship("Job", back_populates="case", cascade="all, delete-orphan", lazy="select")

    def __repr__(self) -> str:
        return f"<Case id={self.id} title={self.title!r}>"

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {}
        for col in sa_inspect(self.__class__).columns:
            val = getattr(self, col.name)
            if isinstance(val, enum.Enum):
                data[col.name] = val.value
            elif hasattr(val, "isoformat"):
                try:
                    data[col.name] = val.isoformat()
                except Exception:
                    data[col.name] = str(val)
            elif isinstance(val, uuid.UUID):
                data[col.name] = str(val)
            else:
                data[col.name] = val
        # include job ids for convenience (don't eager-load full jobs here)
        data["job_ids"] = [str(j.id) for j in self.jobs] if self.jobs is not None else []
        # include brief researcher info if available
        if self.researcher is not None:
            data["researcher"] = {
                "id": self.researcher.id,
                "username": self.researcher.username,
                "full_name": self.researcher.full_name,
            }
        else:
            data["researcher"] = None
        return data


class Job(Base):
    __tablename__ = "jobs"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    case_id = Column(UUID(as_uuid=True), ForeignKey("cases.id"), nullable=True)
    file_path = Column(Text, nullable=False)
    sha256 = Column(String(64), nullable=False)
    status = Column(String(16), nullable=False, default="queued") # queued|running|done|error
    created_at = Column(DateTime(timezone=True), server_default=func.now()) # time when job was created
    started_at = Column(DateTime(timezone=True), nullable=True) # time when job started processing
    finished_at = Column(DateTime(timezone=True), nullable=True) # time when job finished processing
    result_json = Column(JSON, nullable=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    # relationships
    case = relationship("Case", back_populates="jobs", lazy="joined")
    user = relationship("User", back_populates="jobs", lazy="joined")

    # new: one-to-many results for a job (additional structured outputs)
    results = relationship(
        "JobResult",
        back_populates="job",
        cascade="all, delete-orphan",
        lazy="select",
    )

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
        # include brief case info if available
        if self.case is not None:
            data["case"] = {"id": str(self.case.id), "title": self.case.title}
        else:
            data["case"] = None
        # include brief user info if available
        if self.user is not None:
            data["user"] = {"id": self.user.id, "username": self.user.username, "full_name": self.user.full_name}
        else:
            data["user"] = None
        # include results (list of result dicts)
        data["results"] = [r.to_dict() for r in self.results] if self.results is not None else []
        return data


class JobResult(Base):
    """
    One result entry for a job. Stores structured result payloads (JSON) and timestamp.
    """
    __tablename__ = "job_results"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id = Column(UUID(as_uuid=True), ForeignKey("jobs.id"), nullable=False)
    result = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # relationship back to Job
    job = relationship("Job", back_populates="results", lazy="joined")

    def __repr__(self) -> str:
        return f"<JobResult id={self.id} job_id={self.job_id}>"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "job_id": str(self.job_id) if self.job_id is not None else None,
            "result": self.result,
            "created_at": self.created_at.isoformat() if hasattr(self.created_at, "isoformat") else self.created_at,
        }