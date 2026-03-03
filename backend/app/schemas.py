from pydantic import BaseModel, constr
from typing import Optional, List, Literal, Any
from uuid import UUID

class LoginRequest(BaseModel):
    email: str
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
    email: constr(strip_whitespace=True, min_length=3, max_length=80)
    password: constr(min_length=3, max_length=200)
    role: Literal["admin", "researcher", "user"] = "user"

class RegisterResponse(BaseModel):
    id: int
    email: str
    role: str

class PublicRegisterRequest(BaseModel):
    email: constr(strip_whitespace=True, min_length=3, max_length=80)
    password: constr(min_length=8, max_length=200)
    registration_code: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None

class PublicRegisterResponse(BaseModel):
    message: str
    email: str

class UpdateProfileRequest(BaseModel):
    first_name: str | None = None
    last_name: str | None = None
    job_title: str | None = None

class ChangePasswordRequest(BaseModel):
    new_password: str

class SetupStatusResponse(BaseModel):
    needs_setup: bool

class SetupCompleteRequest(BaseModel):
    org_name: str
    org_description: Optional[str] = None
    admin_email: str
    admin_password: str
    admin_first_name: str
    admin_last_name: str

class CreateCaseRequest(BaseModel):
    title: str | None = None
    description: str | None = None
    person_name: str | None = None
    researcher_id: int | None = None