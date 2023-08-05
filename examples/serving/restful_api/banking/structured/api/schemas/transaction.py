from pydantic import BaseModel
from typing import List


class TransactionBase(BaseModel):
    amount: float


class TransactionCreate(TransactionBase):
    pass


class Transaction(TransactionBase):
    id: int
    account_id: int
    amount: float
    type: str
    timestamp: str

    class Config:
        from_attributes = True
