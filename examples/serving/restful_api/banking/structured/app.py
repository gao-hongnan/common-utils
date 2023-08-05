import os
import random
from pathlib import Path
from typing import List

from api.models import Base
from api.models.transaction import Transaction
from api.schemas import account, transaction
from api.models.account import Account
from faker import Faker
from fastapi import Depends, FastAPI, HTTPException
from rich.pretty import pprint
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

# Get the directory of the current script
CURRENT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

# Construct the path to the database file
DATABASE_URL = "/Users/gaohn/gaohn/common-utils/examples/serving/restful_api/banking/app/data/database.db"
# Set up database URL as an environment variable for better flexibility

SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DATABASE_URL}")

engine = create_engine(SQLALCHEMY_DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
session = SessionLocal()

app = FastAPI()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/accounts", response_model=List[account.Account])
def get_accounts(db: Session = Depends(get_db)) -> List[account.Account]:
    db_accounts = db.query(Account).all()
    return db_accounts


@app.get("/account/{account_id}", response_model=account.Account)
def get_account(account_id: int, db: Session = Depends(get_db)):
    db_account = db.query(Account).filter(Account.id == account_id).first()
    if db_account is None:
        raise HTTPException(status_code=404, detail="Account not found")
    return db_account


@app.post("/account/", response_model=account.Account)
def create_account(account: account.AccountCreate, db: Session = Depends(get_db)):
    db_account = Account(**account.model_dump(mode="python"), balance=0)
    db.add(db_account)
    db.commit()
    db.refresh(db_account)
    return db_account


# @app.put("/account/{account_id}", response_model=Account)
# def update_account(
#     account_id: int,
#     account: schemas.account.AccountCreate,
#     db: Session = Depends(get_db),
# ):
#     db_account = db.query(Account).filter(Account.id == account_id).first()
#     if db_account is None:
#         raise HTTPException(status_code=404, detail="Account not found")
#     for key, value in account.dict().items():
#         setattr(db_account, key, value)
#     db.commit()
#     return db_account


# @app.delete("/account/{account_id}")
# def delete_account(account_id: int, db: Session = Depends(get_db)):
#     db_account = db.query(Account).filter(Account.id == account_id).first()
#     if db_account is None:
#         raise HTTPException(status_code=404, detail="Account not found")
#     db.delete(db_account)
#     db.commit()
#     return {"message": "Account has been deleted successfully!"}
