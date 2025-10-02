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

pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")
auth_scheme = HTTPBearer(auto_error=False)

router = APIRouter()

def hash_password(p: str) -> str:
    return pwd.hash(p)

def verify_password(p: str, h: str) -> bool:
    return pwd.verify(p, h)

def create_access_token(sub: str) -> str:
    payload = {
        "sub": sub,
        "exp": datetime.utcnow() + timedelta(seconds=settings.jwt_expires_seconds),
        "iat": datetime.utcnow(),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm="HS256")


def get_current_user(creds: HTTPAuthorizationCredentials = Depends(auth_scheme), db: Session = Depends(lambda: None)):
    if not creds:
        raise HTTPException(status_code=401, detail="Missing credentials")
    token = creds.credentials
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=["HS256"])
        username = payload.get("sub")
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")
    return username

def require_role(required: UserRole):
    def _inner(username: str = Depends(get_current_user), db: Session = Depends(get_db)):
        user = db.query(User).filter(User.username == username).first()
        if not user or user.role != required:
            raise HTTPException(status_code=403, detail="Forbidden")
        return user  # return full user
    return _inner

@router.post("/login", response_model=TokenResponse)
async def login(payload: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == payload.username).one_or_none()
    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return TokenResponse(access_token=create_access_token(payload.username))

@router.post("/register", response_model=RegisterResponse)
async def register_user(
    payload: RegisterRequest,
    _admin=Depends(require_role(UserRole.admin)),
    db: Session = Depends(get_db),
):
    # check if username taken
    existing = db.query(User).filter(User.username == payload.username).first()
    if existing:
        raise HTTPException(status_code=409, detail="Username already exists")

    # hash password and coerce role
    hashed = hash_password(payload.password)
    try:
        role = UserRole(payload.role) # validate against enum
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid role")

    user = User(username=payload.username, password_hash=hashed, role=role)
    db.add(user)
    db.commit()
    db.refresh(user)
    return RegisterResponse(id=user.id, username=user.username, role=user.role.value)