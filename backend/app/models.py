from sqlalchemy import Column, ForeignKey, String, Integer, DateTime, Text, JSON, func, Enum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.inspection import inspect as sa_inspect
import uuid
import enum
from typing import Any, Dict, Optional
from .db import Base

def _iso(dt) -> Optional[str]:
    return dt.isoformat() if getattr(dt, "isoformat", None) else None

def _uuid(v) -> Optional[str]:
    return str(v) if isinstance(v, uuid.UUID) else (str(v) if v is not None else None)


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
        return {
            "id": self.id,
            "username": self.username,
            "role": self.role.value if isinstance(self.role, enum.Enum) else self.role,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "full_name": self.full_name,
            "job_title": self.job_title,
        }


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
        return {
            "id": _uuid(self.id),
            "title": self.title,
            "description": self.description,
            "person_name": self.person_name,
            "researcher_id": self.researcher_id,
            "data_created": _iso(self.data_created),
            "created_at": _iso(self.created_at),
            "metadata": self.case_metadata,
            "researcher": (
                {
                    "id": self.researcher.id,
                    "username": self.researcher.username,
                    "full_name": self.researcher.full_name,
                }
                if self.researcher is not None
                else None
            ),
            "job_ids": [ _uuid(j.id) for j in (self.jobs or []) ],
        }


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
        return {
            "id": _uuid(self.id),
            "case_id": _uuid(self.case_id),
            "file_path": self.file_path,
            "sha256": self.sha256,
            "status": self.status,
            "created_at": _iso(self.created_at),
            "started_at": _iso(self.started_at),
            "finished_at": _iso(self.finished_at),
            "result_json": self.result_json,
            "case": (
                {"id": _uuid(self.case.id), "title": self.case.title}
                if self.case is not None
                else None
            ),
            "user": (
                {"id": self.user.id, "username": self.user.username, "full_name": self.user.full_name}
                if self.user is not None
                else None
            ),
            "results": [r.to_dict() for r in (self.results or [])],
        }


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
            "id": _uuid(self.id),
            "job_id": _uuid(self.job_id),
            "result": self.result,
            "created_at": _iso(self.created_at),
        }