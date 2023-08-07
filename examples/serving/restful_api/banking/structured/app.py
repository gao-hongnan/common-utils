from http import HTTPStatus
from typing import Any, Dict, List, TypeVar, Callable

from api.database.base import Base
from api.database.models.account import Account
from api.database.models.transaction import Transaction
from api.database.session import SessionLocal, engine, get_db
from api.schemas import account, transaction
from faker import Faker
from fastapi import Depends, FastAPI, HTTPException, Request
from rich.pretty import pprint
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker
import functools
import os
import threading
import time
from datetime import datetime
from typing import Any, Callable, Dict, TypeVar


session = SessionLocal()

app = FastAPI()


@app.get("/", tags=["General"])
def _index(request: Request) -> Dict[str, Any]:
    """
    Perform a health check on the server.

    This function is a simple health check endpoint that can be used to
    verify if the server is running correctly. It returns a dictionary
    with a message indicating the status of the server, the HTTP status
    code, and an empty data dictionary.

    Parameters
    ----------
    request : Request
        The request object that contains all the HTTP request
        information.

    Returns
    -------
    response : Dict[str, Any]
        A dictionary containing:
        - message: A string indicating the status of the server. If the
          server is running correctly, this will be "OK".
        - status-code: An integer representing the HTTP status code. If
          the server is running correctly, this will be 200.
        - data: An empty dictionary. This can be used to include any
          additional data if needed in the future.
    """
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK.value,
        "data": {},
    }
    return response


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
