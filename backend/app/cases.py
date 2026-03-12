from typing import Optional
from uuid import UUID as UUID_t

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func

from .db import SessionLocal
from .models import Case, User, UserRole, Job, Organization, OrganizationMembership
from .auth import get_current_user, get_current_user_obj, get_current_organization, get_current_membership
from .logger import get_logger
from .schemas import CreateCaseRequest

logger = get_logger(__name__)
router = APIRouter(tags=["cases"])

# Helper to get DB connection
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Helper to check if user has permission to create/edit cases
def _can_manage_cases(membership: OrganizationMembership) -> bool:
    """Check if user has owner, admin, or researcher role"""
    role_value = membership.role.value if hasattr(membership.role, 'value') else membership.role
    return role_value in ("owner", "admin", "researcher")

# Helper to check if user has permission to delete cases
def _can_delete_cases(membership: OrganizationMembership) -> bool:
    """Check if user has owner or admin role"""
    role_value = membership.role.value if hasattr(membership.role, 'value') else membership.role
    return role_value in ("owner", "admin")

@router.post("")
def create_case(
    payload: CreateCaseRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user_obj),
    current_org: Organization = Depends(get_current_organization),
    membership: OrganizationMembership = Depends(get_current_membership),
):
    # Check permissions (owner, admin, or researcher can create cases)
    if not _can_manage_cases(membership):
        raise HTTPException(status_code=403, detail="Insufficient permissions to create cases")

    # Assign researcher
    assigned_researcher_id = payload.researcher_id
    if assigned_researcher_id is None:
        assigned_researcher_id = current_user.id

    # Verify assigned researcher belongs to the organization
    if assigned_researcher_id:
        researcher_membership = db.query(OrganizationMembership).filter(
            OrganizationMembership.organization_id == current_org.id,
            OrganizationMembership.user_id == assigned_researcher_id,
            OrganizationMembership.status == "active"
        ).first()

        if not researcher_membership:
            raise HTTPException(status_code=400, detail="Assigned researcher not in organization")

    # Create case object with organization
    case = Case(
        title=payload.title,
        description=payload.description,
        person_name=payload.person_name,
        researcher_id=assigned_researcher_id,
        organization_id=current_org.id,
    )

    db.add(case)
    db.commit()
    db.refresh(case)

    logger.info(
        "Case created id=%s title=%s researcher_id=%s org_id=%s",
        case.id,
        case.title,
        case.researcher_id,
        current_org.id
    )
    return case.to_dict()

@router.get("")
def list_cases(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    q: Optional[str] = Query(default=None, description="Search in title/description/person_name"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user_obj),
    current_org: Organization = Depends(get_current_organization),
):
    # Filter by organization
    query = db.query(Case).filter(Case.organization_id == current_org.id)

    # Optional simple search
    if q:
        like = f"%{q}%"
        query = query.filter(
            (Case.title.ilike(like)) |
            (Case.description.ilike(like)) |
            (Case.person_name.ilike(like))
        )

    # Count total matching rows before pagination
    total = query.count()

    # Get paginated rows
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
    current_org: Organization = Depends(get_current_organization),
):
    # Verify case belongs to current organization
    case = db.query(Case).filter(
        Case.id == case_id,
        Case.organization_id == current_org.id
    ).first()

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
    current_org: Organization = Depends(get_current_organization),
    membership: OrganizationMembership = Depends(get_current_membership),
):
    # Verify case belongs to organization
    case = db.query(Case).filter(
        Case.id == case_id,
        Case.organization_id == current_org.id
    ).first()

    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    # Filter jobs by case and organization
    q = db.query(Job).filter(
        Job.case_id == case_id,
        Job.organization_id == current_org.id
    )

    # Members only see their own jobs
    role_value = membership.role.value if hasattr(membership.role, 'value') else membership.role
    if role_value == "member":
        q = q.filter(Job.user_id == current_user.id)

    # Count jobs and return them
    total = q.with_entities(func.count()).scalar() or 0
    rows = (
        q.order_by(Job.created_at.desc())
        .offset((page - 1) * per_page)
        .limit(per_page)
        .all()
    )
    return {"page": page, "per_page": per_page, "total": total, "jobs": [j.to_dict(slim=True) for j in rows]}

@router.delete("/{case_id}")
def delete_case(
    case_id: UUID_t,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user_obj),
    current_org: Organization = Depends(get_current_organization),
    membership: OrganizationMembership = Depends(get_current_membership),
):
    # Only owners and admins can delete cases
    if not _can_delete_cases(membership):
        raise HTTPException(status_code=403, detail="Only owners and admins can delete cases")

    # Verify case belongs to organization
    case = db.query(Case).filter(
        Case.id == case_id,
        Case.organization_id == current_org.id
    ).first()

    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    # Delete all jobs associated with this case (within same org)
    db.query(Job).filter(
        Job.case_id == case_id,
        Job.organization_id == current_org.id
    ).delete()

    # Delete the case
    db.delete(case)
    db.commit()

    logger.info("Case deleted id=%s by user=%s org=%s", case_id, current_user.id, current_org.id)
    return {"detail": "Case deleted successfully"}

@router.delete("")
def delete_all_cases(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user_obj),
    current_org: Organization = Depends(get_current_organization),
    membership: OrganizationMembership = Depends(get_current_membership),
):
    # Only owners and admins can delete all cases
    if not _can_delete_cases(membership):
        raise HTTPException(status_code=403, detail="Only owners and admins can delete cases")

    # Delete all jobs in this organization first
    deleted_jobs = db.query(Job).filter(
        Job.organization_id == current_org.id
    ).delete()

    # Delete all cases in this organization
    deleted_cases = db.query(Case).filter(
        Case.organization_id == current_org.id
    ).delete()

    db.commit()

    logger.warning(
        "All cases deleted: org=%s user=%s deleted_cases=%d deleted_jobs=%d",
        current_org.id,
        current_user.id,
        deleted_cases,
        deleted_jobs
    )
    return {"detail": f"All cases and associated jobs deleted successfully (cases: {deleted_cases}, jobs: {deleted_jobs})"}
