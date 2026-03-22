from datetime import datetime, timedelta, timezone
import jwt
from fastapi import Depends, HTTPException, APIRouter
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from .schemas import AccessKeyRequest
from .settings import settings
from .logger import get_logger

logger = get_logger(__name__)

auth_scheme = HTTPBearer(auto_error=False)

router = APIRouter()


def create_access_token(sub: str) -> str:
    payload = {
        "sub": sub,
        "exp": datetime.now(timezone.utc) + timedelta(seconds=settings.jwt_expires_seconds),
        "iat": datetime.now(timezone.utc),
    }
    token = jwt.encode(payload, settings.jwt_secret, algorithm="HS256")
    logger.info("Created access token for sub=%s", sub)
    return token


def get_current_user(creds: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    """Validate JWT token. Returns the subject (always 'anonymous' in stateless mode)."""
    if not creds:
        logger.warning("Missing credentials on request")
        raise HTTPException(status_code=401, detail="Missing credentials")

    token = creds.credentials
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=["HS256"])
        sub = payload.get("sub")
        logger.debug("Token decoded for sub=%s", sub)
    except Exception as e:
        logger.warning("Invalid token provided: %s", str(e))
        raise HTTPException(status_code=401, detail="Invalid token")

    return sub


@router.post("/verify-key")
async def verify_key(payload: AccessKeyRequest):
    """Validate access key and return a short-lived JWT."""
    if not settings.access_key:
        logger.error("ACCESS_KEY not configured on server")
        raise HTTPException(status_code=500, detail="Access key not configured")

    if payload.key != settings.access_key:
        logger.warning("Invalid access key attempt")
        raise HTTPException(status_code=401, detail="Invalid access key")

    token = create_access_token(sub="anonymous")
    logger.info("Access key verified, token issued")
    return {"access_token": token}
