from pydantic import BaseModel, constr
from typing import Optional, List, Literal, Any
from uuid import UUID

class LoginRequest(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    user: dict[str, Any]

class UploadResponse(BaseModel):
    job_id: UUID

class JobResult(BaseModel):
    job_id: UUID
    status: str
    result: Optional[Any] = None

class RegisterRequest(BaseModel):
    username: constr(strip_whitespace=True, min_length=3, max_length=80)
    password: constr(min_length=3, max_length=200)
    role: Literal["admin", "researcher", "user"] = "user"

class RegisterResponse(BaseModel):
    id: int
    username: str
    role: str

class UpdateProfileRequest(BaseModel):
    first_name: str | None = None
    last_name: str | None = None
    job_title: str | None = None

class ChangePasswordRequest(BaseModel):
    new_password: str

class CreateCaseRequest(BaseModel):
    title: str | None = None
    description: str | None = None
    person_name: str | None = None
    researcher_id: int | None = None