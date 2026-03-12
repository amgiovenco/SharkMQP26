from sqlalchemy import Column, ForeignKey, String, Integer, DateTime, Text, JSON, func, Enum, Boolean, Index
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

def _lttb(x: list, y: list, target: int) -> tuple[list, list]:
    """Largest Triangle Three Buckets downsampling. Preserves visual shape."""
    n = len(x)
    if n <= target:
        return x, y

    result_x, result_y = [x[0]], [y[0]]
    bucket_size = (n - 2) / (target - 2)

    for i in range(target - 2):
        avg_start = int((i + 1) * bucket_size) + 1
        avg_end = min(int((i + 2) * bucket_size) + 1, n)
        avg_x = sum(x[avg_start:avg_end]) / (avg_end - avg_start)
        avg_y = sum(y[avg_start:avg_end]) / (avg_end - avg_start)

        range_start = int(i * bucket_size) + 1
        range_end = int((i + 1) * bucket_size) + 1
        prev_x, prev_y = result_x[-1], result_y[-1]

        max_area, max_idx = -1, range_start
        for j in range(range_start, range_end):
            area = abs((prev_x - avg_x) * (y[j] - prev_y) - (prev_x - x[j]) * (avg_y - prev_y))
            if area > max_area:
                max_area, max_idx = area, j

        result_x.append(x[max_idx])
        result_y.append(y[max_idx])

    result_x.append(x[-1])
    result_y.append(y[-1])
    return result_x, result_y


class UserRole(enum.Enum):
    admin = "admin"
    researcher = "researcher"
    user = "user"


class OrganizationRole(enum.Enum):
    owner = "owner"
    admin = "admin"
    researcher = "researcher"
    member = "member"


class Organization(Base):
    """
    Organization table for multi-tenancy support.
    Each organization represents a separate environmental group, research institution, etc.
    """
    __tablename__ = "organizations"

    id = Column(Integer, primary_key=True)
    name = Column(String(200), nullable=False)
    slug = Column(String(100), unique=True, nullable=False, index=True)
    description = Column(Text, nullable=True)
    status = Column(String(20), default="active", nullable=False)  # active|suspended
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    memberships = relationship("OrganizationMembership", back_populates="organization", cascade="all, delete-orphan")
    cases = relationship("Case", back_populates="organization")
    jobs = relationship("Job", back_populates="organization")
    job_results = relationship("JobResult", back_populates="organization")
    registration_codes = relationship("RegistrationCode", back_populates="organization", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Organization id={self.id} name={self.name!r} slug={self.slug!r}>"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "slug": self.slug,
            "description": self.description,
            "status": self.status,
            "created_at": _iso(self.created_at),
        }


class OrganizationMembership(Base):
    """
    Links users to organizations with roles.
    A user can belong to multiple organizations with different roles.
    """
    __tablename__ = "organization_memberships"
    __table_args__ = (
        Index('idx_org_user', 'organization_id', 'user_id', unique=True),
        Index('idx_org_status', 'organization_id', 'status'),
    )

    id = Column(Integer, primary_key=True)
    organization_id = Column(Integer, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    role = Column(Enum(OrganizationRole, name="organization_roles"), nullable=False)
    joined_at = Column(DateTime(timezone=True), server_default=func.now())
    status = Column(String(20), default="active", nullable=False)  # active|suspended

    # Relationships
    organization = relationship("Organization", back_populates="memberships")
    user = relationship("User", back_populates="memberships")

    def __repr__(self) -> str:
        role_val = self.role.value if isinstance(self.role, enum.Enum) else self.role
        return f"<OrganizationMembership org_id={self.organization_id} user_id={self.user_id} role={role_val!r}>"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "organization_id": self.organization_id,
            "user_id": self.user_id,
            "role": self.role.value if isinstance(self.role, enum.Enum) else self.role,
            "joined_at": _iso(self.joined_at),
            "status": self.status,
        }


class RegistrationCode(Base):
    """
    Registration codes for inviting new users to organizations.
    Codes can be single-use or multi-use, with optional expiration.
    """
    __tablename__ = "registration_codes"
    __table_args__ = (
        Index('idx_code_org', 'organization_id'),
        Index('idx_code_status', 'status'),
    )

    id = Column(Integer, primary_key=True)
    organization_id = Column(Integer, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False)
    code = Column(String(20), unique=True, nullable=False, index=True)
    role = Column(Enum(OrganizationRole, name="registration_code_roles"), nullable=False)
    created_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    expires_at = Column(DateTime(timezone=True), nullable=True)
    uses_remaining = Column(Integer, nullable=True)  # NULL = unlimited
    times_used = Column(Integer, default=0, nullable=False)
    status = Column(String(20), default="active", nullable=False)  # active|expired|disabled

    # Relationships
    organization = relationship("Organization", back_populates="registration_codes")
    creator = relationship("User", foreign_keys=[created_by])

    def __repr__(self) -> str:
        return f"<RegistrationCode code={self.code!r} org_id={self.organization_id} role={self.role}>"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "organization_id": self.organization_id,
            "code": self.code,
            "role": self.role.value if isinstance(self.role, enum.Enum) else self.role,
            "created_by": self.created_by,
            "created_at": _iso(self.created_at),
            "expires_at": _iso(self.expires_at),
            "uses_remaining": self.uses_remaining,
            "times_used": self.times_used,
            "status": self.status,
        }


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    email = Column(String(80), unique=True, nullable=False)
    password_hash = Column(String(200), nullable=False)
    role = Column(Enum(UserRole, name="user_roles"), nullable=False, default=UserRole.user) # admin|researcher|user (legacy - kept for backward compatibility)
    first_name = Column(String(80), nullable=True)
    last_name = Column(String(80), nullable=True)
    job_title = Column(String(120), nullable=True)
    is_system_admin = Column(Boolean, default=False, nullable=False)  # System admin can create organizations

    # relationships
    jobs = relationship("Job", back_populates="user", lazy="select")
    research_cases = relationship("Case", back_populates="researcher", lazy="select", foreign_keys="Case.researcher_id")
    memberships = relationship("OrganizationMembership", back_populates="user", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        role_val = self.role.value if isinstance(self.role, enum.Enum) else self.role
        return f"<User id={self.id} email={self.email!r} role={role_val!r}>"

    @property
    def full_name(self) -> str:
        parts = [p for p in (self.first_name, self.last_name) if p]
        return " ".join(parts) if parts else ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "email": self.email,
            "role": self.role.value if isinstance(self.role, enum.Enum) else self.role,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "full_name": self.full_name,
            "job_title": self.job_title,
            "is_system_admin": self.is_system_admin,
        }


class Case(Base):
    """
    A Case groups multiple jobs. It represents a dataset / submission brought in by a person
    and handled by a researcher. Uses UUID primary key.
    """
    __tablename__ = "cases"
    __table_args__ = (
        Index('idx_cases_organization', 'organization_id'),
        Index('idx_cases_org_researcher', 'organization_id', 'researcher_id'),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(120), nullable=True)  # brief title for the case
    description = Column(Text, nullable=True)
    person_name = Column(String(120), nullable=True) # person who brought in the data

    # Organization (multi-tenancy)
    organization_id = Column(Integer, ForeignKey("organizations.id"), nullable=True)  # Nullable initially for migration
    organization = relationship("Organization", back_populates="cases", lazy="joined")

    # researcher relationship (replaces researcher_name string)
    researcher_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    researcher = relationship("User", back_populates="research_cases", lazy="joined", foreign_keys=[researcher_id])

    data_created = Column(DateTime(timezone=True), nullable=True) # when the underlying data was created
    created_at = Column(DateTime(timezone=True), server_default=func.now()) # when case record was created
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
                    "email": self.researcher.email,
                    "full_name": self.researcher.full_name,
                }
                if self.researcher is not None
                else None
            ),
            "job_ids": [ _uuid(j.id) for j in (self.jobs or []) ],
        }


class Job(Base):
    __tablename__ = "jobs"
    __table_args__ = (
        Index('idx_jobs_organization', 'organization_id'),
        Index('idx_jobs_org_status', 'organization_id', 'status'),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    batch_id = Column(UUID(as_uuid=True), nullable=False) # groups multiple samples from same upload
    sample_index = Column(Integer, nullable=False, default=0) # which sample in batch (0-indexed)
    case_id = Column(UUID(as_uuid=True), ForeignKey("cases.id"), nullable=True)
    file_path = Column(Text, nullable=False)
    original_filename = Column(String(255), nullable=True) # stores the original uploaded filename
    sha256 = Column(String(64), nullable=False) # same for all samples in batch
    status = Column(String(16), nullable=False, default="queued") # queued|running|done|error
    created_at = Column(DateTime(timezone=True), server_default=func.now()) # time when job was created
    started_at = Column(DateTime(timezone=True), nullable=True) # time when job started processing
    finished_at = Column(DateTime(timezone=True), nullable=True) # time when job finished processing
    result_json = Column(JSON, nullable=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    # Organization (multi-tenancy)
    organization_id = Column(Integer, ForeignKey("organizations.id"), nullable=True)  # Nullable initially for migration
    organization = relationship("Organization", back_populates="jobs", lazy="joined")

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

    def to_dict(self, slim: bool = False) -> Dict[str, Any]:
        if slim and self.result_json and isinstance(self.result_json, dict):
            result_json = {k: v for k, v in self.result_json.items() if k != "curve_data"}
        elif self.result_json and isinstance(self.result_json, dict):
            cd = self.result_json.get("curve_data")
            if cd and isinstance(cd, dict):
                freqs, sig = _lttb(cd.get("frequencies", []), cd.get("signal", []), 300)
                result_json = {
                    **self.result_json,
                    "curve_data": {"frequencies": freqs, "signal": sig},
                }
            else:
                result_json = self.result_json
        else:
            result_json = self.result_json

        return {
            "id": _uuid(self.id),
            "batch_id": _uuid(self.batch_id),
            "sample_index": self.sample_index,
            "case_id": _uuid(self.case_id),
            "file_path": self.file_path,
            "original_filename": self.original_filename,
            "sha256": self.sha256,
            "status": self.status,
            "created_at": _iso(self.created_at),
            "started_at": _iso(self.started_at),
            "finished_at": _iso(self.finished_at),
            "result_json": result_json,
            "case": (
                {"id": _uuid(self.case.id), "title": self.case.title}
                if self.case is not None
                else None
            ),
            "user": (
                {"id": self.user.id, "email": self.user.email, "full_name": self.user.full_name}
                if self.user is not None
                else None
            ),
            "results": [] if slim else [r.to_dict() for r in (self.results or [])],
        }


class JobResult(Base):
    """
    One result entry for a job. Stores structured result payloads (JSON) and timestamp.
    """
    __tablename__ = "job_results"
    __table_args__ = (
        Index('idx_job_results_organization', 'organization_id'),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id = Column(UUID(as_uuid=True), ForeignKey("jobs.id"), nullable=False)
    result = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Organization (multi-tenancy)
    organization_id = Column(Integer, ForeignKey("organizations.id"), nullable=True)  # Nullable initially for migration
    organization = relationship("Organization", back_populates="job_results", lazy="joined")

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