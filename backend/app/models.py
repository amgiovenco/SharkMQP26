from sqlalchemy import Column, ForeignKey, String, Integer, DateTime, Text, JSON, func, Index
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import uuid
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


class Job(Base):
    __tablename__ = "jobs"
    __table_args__ = (
        Index('idx_jobs_batch', 'batch_id'),
        Index('idx_jobs_status', 'status'),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    batch_id = Column(UUID(as_uuid=True), nullable=False)
    sample_index = Column(Integer, nullable=False, default=0)
    file_path = Column(Text, nullable=False)
    original_filename = Column(String(255), nullable=True)
    sha256 = Column(String(64), nullable=False)
    status = Column(String(16), nullable=False, default="queued")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    started_at = Column(DateTime(timezone=True), nullable=True)
    finished_at = Column(DateTime(timezone=True), nullable=True)
    result_json = Column(JSON, nullable=True)

    # Keep these columns nullable for backward compat with existing DB / worker
    user_id = Column(Integer, nullable=True)
    organization_id = Column(Integer, nullable=True)
    case_id = Column(UUID(as_uuid=True), nullable=True)

    results = relationship(
        "JobResult",
        back_populates="job",
        cascade="all, delete-orphan",
        lazy="select",
    )

    def __repr__(self) -> str:
        return f"<Job id={self.id} status={self.status!r}>"

    def to_dict(self) -> Dict[str, Any]:
        result_json = self.result_json
        if result_json and isinstance(result_json, dict):
            cd = result_json.get("curve_data")
            if cd and isinstance(cd, dict):
                freqs, sig = _lttb(cd.get("frequencies", []), cd.get("signal", []), 200)
                result_json = {
                    **result_json,
                    "curve_data": {"frequencies": freqs, "signal": sig},
                }

        return {
            "id": _uuid(self.id),
            "batch_id": _uuid(self.batch_id),
            "sample_index": self.sample_index,
            "file_path": self.file_path,
            "original_filename": self.original_filename,
            "sha256": self.sha256,
            "status": self.status,
            "created_at": _iso(self.created_at),
            "started_at": _iso(self.started_at),
            "finished_at": _iso(self.finished_at),
            "result_json": result_json,
        }


class JobResult(Base):
    __tablename__ = "job_results"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id = Column(UUID(as_uuid=True), ForeignKey("jobs.id"), nullable=False)
    result = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    organization_id = Column(Integer, nullable=True)

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
