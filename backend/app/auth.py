from datetime import datetime, timedelta
import jwt
from fastapi import Depends, HTTPException, APIRouter
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from .schemas import LoginRequest, TokenResponse, UploadResponse, JobResult, RegisterRequest, RegisterResponse
from .db import get_db
from passlib.context import CryptContext
from sqlalchemy.orm import Session
from .settings import settings
from .models import User, UserRole
from .logger import get_logger
from datetime import datetime, timedelta, timezone

logger = get_logger(__name__)

pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")
auth_scheme = HTTPBearer(auto_error=False)

router = APIRouter()

def hash_password(p: str) -> str:
    logger.debug("Hashing password")
    return pwd.hash(p)

def verify_password(p: str, h: str) -> bool:
    logger.debug("Verifying password hash")
    return pwd.verify(p, h)

def create_access_token(sub: str) -> str:
    payload = {
        "sub": sub,
        "exp": datetime.now(timezone.utc) + timedelta(seconds=settings.jwt_expires_seconds),
        "iat": datetime.now(timezone.utc),
    }
    token = jwt.encode(payload, settings.jwt_secret, algorithm="HS256")
    logger.info("Created access token for user=%s", sub)
    return token


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

def require_role(required: UserRole):

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
    logger.info("Login attempt for username=%s", payload.username)
    user = db.query(User).filter(User.username == payload.username).one_or_none()

    # verify user exists and password matches
    if not user or not verify_password(payload.password, user.password_hash):
        logger.warning("Invalid login for username=%s", payload.username)
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    logger.info("Login successful for username=%s", payload.username)
    return TokenResponse(access_token=create_access_token(payload.username))

@router.post("/register", response_model=RegisterResponse)
async def register_user(
    payload: RegisterRequest,
    _admin=Depends(require_role(UserRole.admin)),
    db: Session = Depends(get_db),
):
    logger.info("Admin requested creation of user=%s with role=%s", payload.username, payload.role)

    # check if username taken
    existing = db.query(User).filter(User.username == payload.username).first()
    if existing:
        logger.warning("Attempt to register existing username=%s", payload.username)
        raise HTTPException(status_code=409, detail="Username already exists")

    # hash password and store role
    hashed = hash_password(payload.password)
    try:
        role = UserRole(payload.role) # validate against enum
    except ValueError:
        logger.warning("Invalid role provided during registration: %s", payload.role)
        raise HTTPException(status_code=400, detail="Invalid role")

    # add user to db
    user = User(username=payload.username, password_hash=hashed, role=role)
    db.add(user)
    db.commit()
    db.refresh(user)
    
    logger.info("User created id=%s username=%s role=%s", user.id, user.username, user.role)
    return RegisterResponse(id=user.id, username=user.username, role=user.role.value)