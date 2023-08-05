from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from account import Account
from transaction import Transaction

# Create the engine (the source of database connectivity)
engine = create_engine("sqlite:///example.db")

# Create a configured "Session" class
Session = sessionmaker(bind=engine)

# Create a Session
session = Session()

account = session.query(Account).filter(Account.id == some_account_id).one()
transactions = (
    session.query(Transaction).filter(Transaction.account_id == account.id).all()
)
