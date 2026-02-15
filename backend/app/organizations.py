"""
Organization management endpoints for multi-tenancy support.
"""

import secrets
from typing import Optional
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from pydantic import BaseModel

from .db import SessionLocal
from .models import Organization, OrganizationMembership, RegistrationCode, User, OrganizationRole
from .auth import get_current_user_obj, get_current_organization, get_current_membership
from .logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/organizations", tags=["Organizations"])


# Helper to get DB connection
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Request/Response Schemas
class CreateOrganizationRequest(BaseModel):
    name: str
    description: Optional[str] = None
    owner_email: str


class UpdateMemberRoleRequest(BaseModel):
    new_role: str


class CreateRegistrationCodeRequest(BaseModel):
    role: str
    uses_remaining: Optional[int] = None
    expires_at: Optional[datetime] = None


# Organization CRUD
@router.post("")
def create_organization(
    payload: CreateOrganizationRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user_obj),
):
    """
    Create a new organization. System admin only.
    """
    if not current_user.is_system_admin:
        logger.warning("Non-admin user %s attempted to create organization", current_user.id)
        raise HTTPException(status_code=403, detail="System admin only")

    # Generate slug from name
    slug = payload.name.lower().replace(" ", "-").replace("_", "-")

    # Check if slug already exists
    existing = db.query(Organization).filter(Organization.slug == slug).first()
    if existing:
        raise HTTPException(status_code=409, detail="Organization with this name already exists")

    # Find owner user
    owner = db.query(User).filter(User.email == payload.owner_email).first()
    if not owner:
        raise HTTPException(status_code=404, detail="Owner user not found")

    # Create organization
    org = Organization(
        name=payload.name,
        slug=slug,
        description=payload.description,
        status="active",
        created_at=datetime.now(timezone.utc)
    )
    db.add(org)
    db.flush()

    # Add owner membership
    membership = OrganizationMembership(
        organization_id=org.id,
        user_id=owner.id,
        role=OrganizationRole.owner,
        status="active",
        joined_at=datetime.now(timezone.utc)
    )
    db.add(membership)
    db.commit()
    db.refresh(org)

    logger.info(
        "Organization created: id=%s name=%s owner=%s by admin=%s",
        org.id,
        org.name,
        owner.email,
        current_user.email
    )

    return {
        "id": org.id,
        "name": org.name,
        "slug": org.slug,
        "description": org.description,
        "status": org.status,
    }


@router.get("/{org_id}")
def get_organization(
    org_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user_obj),
    current_org: Organization = Depends(get_current_organization),
):
    """
    Get organization details. Users can only view their own organization.
    """
    # Verify user is requesting their own organization (unless system admin)
    if not current_user.is_system_admin and current_org.id != org_id:
        raise HTTPException(status_code=403, detail="Access denied")

    org = db.get(Organization, org_id)
    if not org:
        raise HTTPException(status_code=404, detail="Organization not found")

    return {
        "id": org.id,
        "name": org.name,
        "slug": org.slug,
        "description": org.description,
        "status": org.status,
        "created_at": org.created_at.isoformat() if org.created_at else None,
    }


# Member Management
@router.get("/{org_id}/members")
def list_members(
    org_id: int,
    page: int = Query(1, ge=1),
    per_page: int = Query(50, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user_obj),
    current_org: Organization = Depends(get_current_organization),
):
    """
    List organization members.
    """
    if current_org.id != org_id:
        raise HTTPException(status_code=403, detail="Access denied")

    query = db.query(OrganizationMembership).join(User).filter(
        OrganizationMembership.organization_id == org_id,
        OrganizationMembership.status == "active"
    )

    total = query.count()
    memberships = (
        query.order_by(OrganizationMembership.joined_at.desc())
        .offset((page - 1) * per_page)
        .limit(per_page)
        .all()
    )

    return {
        "page": page,
        "per_page": per_page,
        "total": total,
        "members": [{
            "id": m.id,
            "user_id": m.user.id,
            "email": m.user.email,
            "first_name": m.user.first_name,
            "last_name": m.user.last_name,
            "full_name": m.user.full_name,
            "role": m.role.value if hasattr(m.role, 'value') else m.role,
            "joined_at": m.joined_at.isoformat() if m.joined_at else None,
        } for m in memberships]
    }


@router.patch("/{org_id}/members/{user_id}")
def update_member_role(
    org_id: int,
    user_id: int,
    payload: UpdateMemberRoleRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user_obj),
    current_org: Organization = Depends(get_current_organization),
    membership: OrganizationMembership = Depends(get_current_membership),
):
    """
    Change a member's role. Only owners and admins can change roles.
    """
    if current_org.id != org_id:
        raise HTTPException(status_code=403, detail="Access denied")

    # Check permissions
    role_value = membership.role.value if hasattr(membership.role, 'value') else membership.role
    if role_value not in ("owner", "admin"):
        raise HTTPException(status_code=403, detail="Only owners and admins can change roles")

    # Validate new role
    try:
        new_role = OrganizationRole(payload.new_role)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid role")

    # Can't change own role
    if user_id == current_user.id:
        raise HTTPException(status_code=400, detail="Cannot change your own role")

    # Get target membership
    target_membership = db.query(OrganizationMembership).filter(
        OrganizationMembership.organization_id == org_id,
        OrganizationMembership.user_id == user_id
    ).first()

    if not target_membership:
        raise HTTPException(status_code=404, detail="Member not found")

    # Update role
    target_membership.role = new_role
    db.commit()

    logger.info(
        "Member role updated: org=%s user=%s new_role=%s by=%s",
        org_id,
        user_id,
        new_role.value,
        current_user.id
    )

    return {"detail": "Role updated successfully"}


@router.delete("/{org_id}/members/{user_id}")
def remove_member(
    org_id: int,
    user_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user_obj),
    current_org: Organization = Depends(get_current_organization),
    membership: OrganizationMembership = Depends(get_current_membership),
):
    """
    Remove a member from the organization. Only owners and admins can remove members.
    """
    if current_org.id != org_id:
        raise HTTPException(status_code=403, detail="Access denied")

    # Check permissions
    role_value = membership.role.value if hasattr(membership.role, 'value') else membership.role
    if role_value not in ("owner", "admin"):
        raise HTTPException(status_code=403, detail="Only owners and admins can remove members")

    # Can't remove self
    if user_id == current_user.id:
        raise HTTPException(status_code=400, detail="Cannot remove yourself")

    # Get target membership
    target_membership = db.query(OrganizationMembership).filter(
        OrganizationMembership.organization_id == org_id,
        OrganizationMembership.user_id == user_id
    ).first()

    if not target_membership:
        raise HTTPException(status_code=404, detail="Member not found")

    # Delete membership
    db.delete(target_membership)
    db.commit()

    logger.info(
        "Member removed: org=%s user=%s by=%s",
        org_id,
        user_id,
        current_user.id
    )

    return {"detail": "Member removed successfully"}


# Registration Code Management
@router.get("/{org_id}/codes")
def list_registration_codes(
    org_id: int,
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user_obj),
    current_org: Organization = Depends(get_current_organization),
    membership: OrganizationMembership = Depends(get_current_membership),
):
    """
    List registration codes for the organization. Only owners and admins can view codes.
    """
    if current_org.id != org_id:
        raise HTTPException(status_code=403, detail="Access denied")

    # Check permissions
    role_value = membership.role.value if hasattr(membership.role, 'value') else membership.role
    if role_value not in ("owner", "admin"):
        raise HTTPException(status_code=403, detail="Only owners and admins can view registration codes")

    query = db.query(RegistrationCode).filter(
        RegistrationCode.organization_id == org_id
    )

    total = query.count()
    codes = (
        query.order_by(RegistrationCode.created_at.desc())
        .offset((page - 1) * per_page)
        .limit(per_page)
        .all()
    )

    return {
        "page": page,
        "per_page": per_page,
        "total": total,
        "codes": [{
            "id": c.id,
            "code": c.code,
            "role": c.role.value if hasattr(c.role, 'value') else c.role,
            "created_by": c.created_by,
            "created_at": c.created_at.isoformat() if c.created_at else None,
            "expires_at": c.expires_at.isoformat() if c.expires_at else None,
            "uses_remaining": c.uses_remaining,
            "times_used": c.times_used,
            "status": c.status,
        } for c in codes]
    }


@router.post("/{org_id}/codes")
def create_registration_code(
    org_id: int,
    payload: CreateRegistrationCodeRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user_obj),
    current_org: Organization = Depends(get_current_organization),
    membership: OrganizationMembership = Depends(get_current_membership),
):
    """
    Create a new registration code. Only owners and admins can create codes.
    """
    if current_org.id != org_id:
        raise HTTPException(status_code=403, detail="Access denied")

    # Check permissions
    role_value = membership.role.value if hasattr(membership.role, 'value') else membership.role
    if role_value not in ("owner", "admin"):
        raise HTTPException(status_code=403, detail="Only owners and admins can create registration codes")

    # Validate role (can't create owner codes)
    try:
        code_role = OrganizationRole(payload.role)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid role")

    if code_role == OrganizationRole.owner:
        raise HTTPException(status_code=400, detail="Cannot create owner registration codes")

    # Generate unique code
    code_str = f"SHARK-{secrets.token_hex(3).upper()}"

    # Ensure uniqueness
    while db.query(RegistrationCode).filter(RegistrationCode.code == code_str).first():
        code_str = f"SHARK-{secrets.token_hex(3).upper()}"

    # Create registration code
    reg_code = RegistrationCode(
        organization_id=org_id,
        code=code_str,
        role=code_role,
        created_by=current_user.id,
        created_at=datetime.now(timezone.utc),
        expires_at=payload.expires_at,
        uses_remaining=payload.uses_remaining,
        times_used=0,
        status="active"
    )
    db.add(reg_code)
    db.commit()
    db.refresh(reg_code)

    logger.info(
        "Registration code created: org=%s code=%s role=%s by=%s",
        org_id,
        code_str,
        code_role.value,
        current_user.id
    )

    return {
        "id": reg_code.id,
        "code": reg_code.code,
        "role": code_role.value,
        "uses_remaining": reg_code.uses_remaining,
        "expires_at": reg_code.expires_at.isoformat() if reg_code.expires_at else None,
    }


@router.delete("/{org_id}/codes/{code_id}")
def disable_registration_code(
    org_id: int,
    code_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user_obj),
    current_org: Organization = Depends(get_current_organization),
    membership: OrganizationMembership = Depends(get_current_membership),
):
    """
    Disable a registration code. Only owners and admins can disable codes.
    """
    if current_org.id != org_id:
        raise HTTPException(status_code=403, detail="Access denied")

    # Check permissions
    role_value = membership.role.value if hasattr(membership.role, 'value') else membership.role
    if role_value not in ("owner", "admin"):
        raise HTTPException(status_code=403, detail="Only owners and admins can disable registration codes")

    # Get code
    code = db.query(RegistrationCode).filter(
        RegistrationCode.id == code_id,
        RegistrationCode.organization_id == org_id
    ).first()

    if not code:
        raise HTTPException(status_code=404, detail="Registration code not found")

    # Disable code
    code.status = "disabled"
    db.commit()

    logger.info(
        "Registration code disabled: org=%s code=%s by=%s",
        org_id,
        code.code,
        current_user.id
    )

    return {"detail": "Registration code disabled successfully"}
