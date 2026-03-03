import secrets
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from .db import get_db
from .models import User, UserRole, Organization, OrganizationMembership, OrganizationRole, RegistrationCode
from .schemas import SetupStatusResponse, SetupCompleteRequest, TokenResponse
from .auth import hash_password, create_access_token
from .logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/setup", tags=["Setup"])


def _check_needs_setup(db: Session) -> bool:
    return db.query(User).count() == 0


@router.get("/status", response_model=SetupStatusResponse)
def get_setup_status(db: Session = Depends(get_db)):
    """
    Returns whether first-run setup is required.
    Always 200, safe for unauthenticated access.
    """
    needs_setup = _check_needs_setup(db)
    logger.info("Setup status check: needs_setup=%s", needs_setup)
    return SetupStatusResponse(needs_setup=needs_setup)


@router.post("/complete", response_model=TokenResponse)
def complete_setup(payload: SetupCompleteRequest, db: Session = Depends(get_db)):
    """
    Complete the first-run setup by creating the initial organization and admin user.
    Returns 403 if any users already exist (setup already done).
    """
    logger.info("Setup complete requested for org=%s admin=%s", payload.org_name, payload.admin_email)

    if not _check_needs_setup(db):
        logger.warning("Setup complete rejected: users already exist")
        raise HTTPException(status_code=403, detail="Setup already completed")

    # Validate inputs
    if not payload.org_name.strip():
        raise HTTPException(status_code=400, detail="Organization name is required")
    if not payload.admin_email.strip():
        raise HTTPException(status_code=400, detail="Admin email is required")
    if len(payload.admin_password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")
    if not payload.admin_first_name.strip():
        raise HTTPException(status_code=400, detail="Admin first name is required")
    if not payload.admin_last_name.strip():
        raise HTTPException(status_code=400, detail="Admin last name is required")

    try:
        # Create organization
        slug = payload.org_name.lower().replace(" ", "-").replace("_", "-")
        org = Organization(
            name=payload.org_name.strip(),
            slug=slug,
            description=payload.org_description,
            status="active",
        )
        db.add(org)
        db.flush()
        logger.info("Created organization id=%s slug=%s", org.id, org.slug)

        # Create admin user
        user = User(
            email=payload.admin_email.strip(),
            password_hash=hash_password(payload.admin_password),
            role=UserRole.admin,
            first_name=payload.admin_first_name.strip(),
            last_name=payload.admin_last_name.strip(),
            is_system_admin=True,
        )
        db.add(user)
        db.flush()
        logger.info("Created admin user id=%s email=%s", user.id, user.email)

        # Create org membership (owner)
        membership = OrganizationMembership(
            organization_id=org.id,
            user_id=user.id,
            role=OrganizationRole.owner,
            status="active",
            joined_at=datetime.now(timezone.utc),
        )
        db.add(membership)
        db.flush()

        # Create 3 registration codes (admin, researcher, member roles)
        for role in (OrganizationRole.admin, OrganizationRole.researcher, OrganizationRole.member):
            code_str = f"SHARK-{secrets.token_hex(3).upper()}"
            while db.query(RegistrationCode).filter(RegistrationCode.code == code_str).first():
                code_str = f"SHARK-{secrets.token_hex(3).upper()}"

            reg_code = RegistrationCode(
                organization_id=org.id,
                code=code_str,
                role=role,
                created_by=user.id,
                created_at=datetime.now(timezone.utc),
                expires_at=None,
                uses_remaining=None,
                times_used=0,
                status="active",
            )
            db.add(reg_code)

        db.commit()
        logger.info("Setup completed successfully for org_id=%s user_id=%s", org.id, user.id)

    except Exception as e:
        db.rollback()
        logger.error("Setup failed, rolling back: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Setup failed. Please try again.")

    # Return token so frontend can auto-login
    access_token = create_access_token(user.email)

    return TokenResponse(
        access_token=access_token,
        user={
            "id": user.id,
            "email": user.email,
            "role": user.role.value,
            "first_name": user.first_name,
            "last_name": user.last_name,
            "job_title": user.job_title,
            "is_system_admin": user.is_system_admin,
            "organizations": [{
                "id": org.id,
                "name": org.name,
                "slug": org.slug,
                "role": OrganizationRole.owner.value,
            }],
        },
    )
