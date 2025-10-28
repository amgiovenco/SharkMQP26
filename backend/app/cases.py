from typing import Optional
from uuid import UUID as UUID_t

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func

from .db import SessionLocal
from .models import Case, User, UserRole, Job
from .auth import get_current_user
from .logger import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["cases"])

# Helper to get DB connection
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Get the current user based on their jwt
def get_current_user_obj(username: str = Depends(get_current_user), db: Session = Depends(get_db)) -> User:
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=401, detail="Invalid token")
    return user

# Helper to require researcher or admin role
def _require_researcher_or_admin(user: User):
    if user.role.value not in ("admin", "researcher"):
        raise HTTPException(status_code=403, detail="Forbidden")

@router.post("")
def create_case(
    title: Optional[str] = None,
    description: Optional[str] = None,
    person_name: Optional[str] = None,
    researcher_id: Optional[int] = None,  # if omitted and caller is researcher, we'll assign them
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user_obj),
):
    # validate perms
    _require_researcher_or_admin(current_user)

    # validate user 
    assigned_researcher_id = researcher_id
    if assigned_researcher_id is None and current_user.role == UserRole.researcher:
        assigned_researcher_id = current_user.id

    # make case object
    case = Case(
        title=title,
        description=description,
        person_name=person_name,
        researcher_id=assigned_researcher_id,
    )

    db.add(case)
    db.commit()
    db.refresh(case)

    return case.to_dict()

@router.get("")
def list_cases(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    q: Optional[str] = Query(default=None, description="Search in title/description/person_name"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user_obj),
):
    query = db.query(Case)

    # optional simple search
    if q:
        like = f"%{q}%"
        query = query.filter(
            (Case.title.ilike(like)) |
            (Case.description.ilike(like)) |
            (Case.person_name.ilike(like))
        )

    # count rows and return them
    total = query.with_entities(func.count()).scalar() or 0
    rows = (
        query.order_by(Case.created_at.desc())
        .offset((page - 1) * per_page)
        .limit(per_page)
        .all()
    )
    return {"page": page, "per_page": per_page, "total": total, "cases": [c.to_dict() for c in rows]}

@router.get("/{case_id}")
def get_case(
    case_id: UUID_t,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user_obj),
):
    case = db.get(Case, case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    return case.to_dict()

@router.get("/{case_id}/jobs")
def list_case_jobs(
    case_id: UUID_t,
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user_obj),
):
    q = db.query(Job).filter(Job.case_id == case_id)

    # non-admin/researcher users only see their own jobs
    if current_user.role.value not in ("admin", "researcher"):
        q = q.filter(Job.user_id == current_user.id)

    # count jobs and return them
    total = q.with_entities(func.count()).scalar() or 0
    rows = (
        q.order_by(Job.created_at.desc())
        .offset((page - 1) * per_page)
        .limit(per_page)
        .all()
    )
    return {"page": page, "per_page": per_page, "total": total, "jobs": [j.to_dict() for j in rows]}

@router.delete("/{case_id}")
def delete_case(
    case_id: UUID_t,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user_obj),
):
    # Only admins can delete cases
    if current_user.role.value != "admin":
        raise HTTPException(status_code=403, detail="Only admins can delete cases")

    case = db.get(Case, case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    # Delete all jobs associated with this case first
    db.query(Job).filter(Job.case_id == case_id).delete()

    # Delete the case
    db.delete(case)
    db.commit()

    return {"detail": "Case deleted successfully"}

@router.delete("")
def delete_all_cases(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user_obj),
):
    # Only admins can delete all cases
    if current_user.role.value != "admin":
        raise HTTPException(status_code=403, detail="Only admins can delete cases")

    # Delete all jobs first
    db.query(Job).delete()

    # Delete all cases
    db.query(Case).delete()
    db.commit()

    return {"detail": "All cases and associated jobs deleted successfully"}
