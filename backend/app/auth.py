from datetime import datetime, timedelta, timezone
import jwt
from fastapi import Depends, HTTPException, APIRouter
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from .schemas import LoginRequest, TokenResponse, UploadResponse, JobResult, RegisterRequest, RegisterResponse, UpdateProfileRequest, ChangePasswordRequest, PublicRegisterRequest, PublicRegisterResponse
from .db import get_db
from passlib.context import CryptContext
from sqlalchemy.orm import Session
from .settings import settings
from .models import User, UserRole, Organization, OrganizationMembership, OrganizationRole, RegistrationCode
from .logger import get_logger

logger = get_logger(__name__)

pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")
auth_scheme = HTTPBearer(auto_error=False)

router = APIRouter()

# Helper to hash a password
def hash_password(p: str) -> str:
    logger.debug("Hashing password")
    return pwd.hash(p)

# Helper to verify a password against a hash
def verify_password(p: str, h: str) -> bool:
    logger.debug("Verifying password hash")
    return pwd.verify(p, h)

# Helper to create a jwt
def create_access_token(sub: str) -> str:
    payload = {
        "sub": sub,
        "exp": datetime.now(timezone.utc) + timedelta(seconds=settings.jwt_expires_seconds),
        "iat": datetime.now(timezone.utc),
    }
    token = jwt.encode(payload, settings.jwt_secret, algorithm="HS256")
    logger.info("Created access token for user=%s", sub)
    return token

# Helper to get the current user based on their jwt
def get_current_user(creds: HTTPAuthorizationCredentials = Depends(auth_scheme), db: Session = Depends(lambda: None)):
    # verify credentials
    if not creds:
        logger.warning("Missing credentials on request")
        raise HTTPException(status_code=401, detail="Missing credentials")
    
    # attempt to decode token
    token = creds.credentials
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=["HS256"])
        username = payload.get("sub")
        logger.debug("Token decoded for user=%s", username)
    except Exception as e:
        logger.warning("Invalid token provided: %s", str(e))
        raise HTTPException(status_code=401, detail="Invalid token")
    
    return username

def get_current_user_obj(username: str = Depends(get_current_user), db: Session = Depends(get_db)) -> User:
    """
    Get the full User object for the current authenticated user.
    """
    user = db.query(User).filter(User.username == username).first()
    if not user:
        logger.warning("User not found: %s", username)
        raise HTTPException(status_code=404, detail="User not found")
    return user


def get_current_organization(
    current_user: User = Depends(get_current_user_obj),
    db: Session = Depends(get_db)
) -> Organization:
    """
    Get user's current organization context.
    Returns their first active membership's organization.
    """
    membership = db.query(OrganizationMembership).filter(
        OrganizationMembership.user_id == current_user.id,
        OrganizationMembership.status == "active"
    ).first()

    if not membership:
        logger.warning("User %s has no organization access", current_user.username)
        raise HTTPException(status_code=403, detail="No organization access")

    organization = db.get(Organization, membership.organization_id)
    if not organization or organization.status != "active":
        logger.warning("User %s's organization is inactive", current_user.username)
        raise HTTPException(status_code=403, detail="Organization inactive")

    return organization


def get_current_membership(
    current_user: User = Depends(get_current_user_obj),
    current_org: Organization = Depends(get_current_organization),
    db: Session = Depends(get_db)
) -> OrganizationMembership:
    """
    Get user's membership in their current organization.
    """
    membership = db.query(OrganizationMembership).filter(
        OrganizationMembership.organization_id == current_org.id,
        OrganizationMembership.user_id == current_user.id,
        OrganizationMembership.status == "active"
    ).first()

    if not membership:
        logger.warning("User %s has no membership in org %s", current_user.username, current_org.id)
        raise HTTPException(status_code=403, detail="No organization membership")

    return membership


def require_org_role(*allowed_roles: str):
    """
    Decorator factory to require specific organization roles.
    Usage: @require_org_role("owner", "admin")
    """
    from functools import wraps

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, membership: OrganizationMembership = Depends(get_current_membership), **kwargs):
            role_value = membership.role.value if hasattr(membership.role, 'value') else membership.role
            if role_value not in allowed_roles:
                logger.warning(
                    "User org_id=%s role=%s attempted action requiring %s",
                    membership.organization_id,
                    role_value,
                    allowed_roles
                )
                raise HTTPException(status_code=403, detail="Insufficient permissions")
            return await func(*args, membership=membership, **kwargs)
        return wrapper
    return decorator


def require_role(required: UserRole):
    """
    Legacy role checker (kept for backward compatibility).
    New endpoints should use require_org_role instead.
    """
    def _inner(username: str = Depends(get_current_user), db: Session = Depends(get_db)):
        logger.debug("Checking role for user=%s required=%s", username, required)
        user = db.query(User).filter(User.username == username).first()

        # verify user exists
        if not user:
            logger.warning("User not found during role check: %s", username)
            raise HTTPException(status_code=403, detail="Forbidden")

        # verify role
        if user.role != required:
            logger.warning("User %s has insufficient role: %s (required=%s)", username, user.role, required)
            raise HTTPException(status_code=403, detail="Forbidden")

        logger.info("User %s authorized with role=%s", username, user.role)
        return user  # return full user

    return _inner

@router.post("/login", response_model=TokenResponse)
async def login(payload: LoginRequest, db: Session = Depends(get_db)):
    """
    Login endpoint. Returns JWT token + user object with organizations.
    """
    logger.info("Login attempt for username=%s", payload.username)

    # Validate input
    if not payload.username or not payload.password:
        logger.warning("Login attempt with empty credentials")
        raise HTTPException(status_code=400, detail="Username and password required")

    # Fetch user
    user = db.query(User).filter(User.username == payload.username).one_or_none()

    # Verify user exists and password matches
    if not user or not verify_password(payload.password, user.password_hash):
        logger.warning("Invalid login for username=%s", payload.username)
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # Get user's organizations
    memberships = db.query(OrganizationMembership).join(
        Organization
    ).filter(
        OrganizationMembership.user_id == user.id,
        OrganizationMembership.status == "active",
        Organization.status == "active"
    ).all()

    organizations = [{
        "id": m.organization.id,
        "name": m.organization.name,
        "slug": m.organization.slug,
        "role": m.role.value if hasattr(m.role, 'value') else m.role,
    } for m in memberships]

    # Generate token
    access_token = create_access_token(payload.username)

    logger.info("Login successful for username=%s id=%s orgs=%d", payload.username, user.id, len(organizations))

    # Return token + user data + organizations
    return TokenResponse(
        access_token=access_token,
        user={
            "id": user.id,
            "username": user.username,
            "role": user.role.value,
            "first_name": user.first_name,
            "last_name": user.last_name,
            "job_title": user.job_title,
            "is_system_admin": user.is_system_admin,
            "organizations": organizations,
        }
    )

@router.post("/signup", response_model=PublicRegisterResponse)
async def public_register(
    payload: PublicRegisterRequest,
    db: Session = Depends(get_db),
):
    """
    Public registration endpoint using registration codes.
    Anyone can register if they have a valid registration code.
    """
    logger.info("Public registration attempt for username=%s with code=%s", payload.username, payload.registration_code)

    # Validate input
    if not payload.username or not payload.password:
        logger.warning("Registration attempt with empty credentials")
        raise HTTPException(status_code=400, detail="Username and password required")

    if len(payload.password) < 8:
        logger.warning("Registration attempt with weak password for user=%s", payload.username)
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")

    # Validate registration code
    code = db.query(RegistrationCode).join(
        Organization
    ).filter(
        RegistrationCode.code == payload.registration_code.upper().strip(),
        RegistrationCode.status == "active",
        Organization.status == "active"
    ).first()

    if not code:
        logger.warning("Invalid registration code: %s", payload.registration_code)
        raise HTTPException(status_code=400, detail="Invalid registration code")

    # Check expiration
    if code.expires_at and code.expires_at < datetime.now(timezone.utc):
        logger.warning("Expired registration code: %s", code.code)
        raise HTTPException(status_code=400, detail="Registration code expired")

    # Check usage limit
    if code.uses_remaining is not None and code.uses_remaining <= 0:
        logger.warning("Registration code exhausted: %s", code.code)
        raise HTTPException(status_code=400, detail="Registration code exhausted")

    # Check if username taken
    existing = db.query(User).filter(User.username == payload.username).first()
    if existing:
        logger.warning("Attempt to register existing username=%s", payload.username)
        raise HTTPException(status_code=409, detail="Username already exists")

    # Create user (use legacy role 'user' for backward compatibility)
    hashed = hash_password(payload.password)
    user = User(
        username=payload.username,
        password_hash=hashed,
        role=UserRole.user,
        first_name=payload.first_name,
        last_name=payload.last_name,
        is_system_admin=False,
    )
    db.add(user)
    db.flush()

    # Add to organization with role from registration code
    membership = OrganizationMembership(
        organization_id=code.organization_id,
        user_id=user.id,
        role=code.role,
        status="active",
        joined_at=datetime.now(timezone.utc)
    )
    db.add(membership)

    # Update code usage
    code.times_used += 1
    if code.uses_remaining is not None:
        code.uses_remaining -= 1

    db.commit()

    logger.info(
        "User created via public registration: id=%s username=%s org_id=%s role=%s",
        user.id,
        user.username,
        code.organization_id,
        code.role.value if hasattr(code.role, 'value') else code.role
    )

    return PublicRegisterResponse(
        message="User created successfully",
        username=user.username
    )


@router.post("/register", response_model=RegisterResponse)
async def register_user(
    payload: RegisterRequest,
    _admin=Depends(require_role(UserRole.admin)),
    db: Session = Depends(get_db),
):
    """
    Register a new user. Admin only (legacy endpoint).
    """
    logger.info("Admin requested creation of user=%s with role=%s", payload.username, payload.role)

    # Validate input
    if not payload.username or not payload.password:
        logger.warning("Registration attempt with empty credentials")
        raise HTTPException(status_code=400, detail="Username and password required")
    
    if len(payload.password) < 8:
        logger.warning("Registration attempt with weak password for user=%s", payload.username)
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")

    # Check if username taken
    existing = db.query(User).filter(User.username == payload.username).first()
    if existing:
        logger.warning("Attempt to register existing username=%s", payload.username)
        raise HTTPException(status_code=409, detail="Username already exists")

    # Validate role
    try:
        role = UserRole(payload.role)
    except ValueError:
        logger.warning("Invalid role provided during registration: %s", payload.role)
        raise HTTPException(status_code=400, detail="Invalid role")

    # Hash password and create user
    hashed = hash_password(payload.password)
    user = User(
        username=payload.username,
        password_hash=hashed,
        role=role,
        first_name=payload.first_name if hasattr(payload, 'first_name') else None,
        last_name=payload.last_name if hasattr(payload, 'last_name') else None,
        job_title=payload.job_title if hasattr(payload, 'job_title') else None,
    )
    
    db.add(user)
    db.commit()
    db.refresh(user)
    
    logger.info("User created id=%s username=%s role=%s", user.id, user.username, user.role)
    return RegisterResponse(
        id=user.id,
        username=user.username,
        role=user.role.value,
        first_name=user.first_name,
        last_name=user.last_name,
    )

@router.put("/users/profile")
async def update_profile(
    payload: UpdateProfileRequest,
    current_username: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Update user profile information (name, job title).
    """
    logger.info("User %s requesting profile update", current_username)
    
    # Validate new password length if provided
    if not payload.first_name and not payload.last_name and not payload.job_title:
        raise HTTPException(status_code=400, detail="At least one field must be provided")
    
    # Get user
    user = db.query(User).filter(User.username == current_username).first()
    if not user:
        logger.warning("User not found during profile update: %s", current_username)
        raise HTTPException(status_code=404, detail="User not found")
    
    # Update fields
    if payload.first_name is not None:
        user.first_name = payload.first_name
    if payload.last_name is not None:
        user.last_name = payload.last_name
    if payload.job_title is not None:
        user.job_title = payload.job_title
    
    db.commit()
    db.refresh(user)
    
    logger.info("Profile updated for user=%s", current_username)
    return {
        "first_name": user.first_name,
        "last_name": user.last_name,
        "job_title": user.job_title,
        "full_name": user.full_name,
    }

@router.put("/users/password")
async def change_password(
    payload: ChangePasswordRequest,
    current_username: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Change user password.
    """
    logger.info("User %s requesting password change", current_username)
    
    # Validate password length
    if len(payload.new_password) < 8:
        logger.warning("Password change rejected for %s: too short", current_username)
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")
    
    # Get user
    user = db.query(User).filter(User.username == current_username).first()
    if not user:
        logger.warning("User not found during password change: %s", current_username)
        raise HTTPException(status_code=404, detail="User not found")
    
    # Hash and update password
    user.password_hash = hash_password(payload.new_password)
    db.commit()
    
    logger.info("Password changed for user=%s", current_username)
    return {"message": "Password changed successfully"}

@router.get("/me")
async def get_current_user_profile(
    current_username: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Get the current user's profile information.
    Validates JWT token and returns fresh user data including organizations.
    """
    logger.info("User %s requesting their profile", current_username)

    # Get user
    user = db.query(User).filter(User.username == current_username).first()
    if not user:
        logger.warning("User not found: %s", current_username)
        raise HTTPException(status_code=404, detail="User not found")

    # Get user's organizations
    memberships = db.query(OrganizationMembership).join(
        Organization
    ).filter(
        OrganizationMembership.user_id == user.id,
        OrganizationMembership.status == "active",
        Organization.status == "active"
    ).all()

    organizations = [{
        "id": m.organization.id,
        "name": m.organization.name,
        "slug": m.organization.slug,
        "role": m.role.value if hasattr(m.role, 'value') else m.role,
    } for m in memberships]

    logger.info("Profile fetched for user=%s", current_username)
    return {
        "id": user.id,
        "username": user.username,
        "role": user.role.value,
        "first_name": user.first_name,
        "last_name": user.last_name,
        "job_title": user.job_title,
        "is_system_admin": user.is_system_admin,
        "organizations": organizations,
    }