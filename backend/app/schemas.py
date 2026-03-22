from pydantic import BaseModel


class AccessKeyRequest(BaseModel):
    key: str
