from pydantic import BaseModel
from typing import Optional


class AccountCreateRequest(BaseModel):
    name: str
    email: str
    balance: float


class AccountUpdateRequest(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    balance: Optional[float] = None


class AccountResponse(BaseModel):
    id: int
    name: str
    email: str
    balance: float


class AccountCreateOrUpdateResponse(BaseModel):
    id: int
    name: str
    email: str
    balance: float

    class Config:
        from_attributes = True
