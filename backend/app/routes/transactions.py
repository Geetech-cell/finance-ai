from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from ..database import SessionLocal
from ..models import Transaction

router = APIRouter()

class TransactionCreate(BaseModel):
    description: Optional[str] = None
    amount: float
    user_id: int
    category: Optional[str] = None

class TransactionResponse(BaseModel):
    id: int
    description: Optional[str]
    amount: float
    date: str
    category: Optional[str]
    user_id: int

@router.post("/", response_model=TransactionResponse, summary="Create transaction")
def create_transaction(transaction: TransactionCreate):
    """Create a new transaction"""
    db = SessionLocal()
    try:
        db_transaction = Transaction(
            description=transaction.description,
            amount=transaction.amount,
            user_id=transaction.user_id,
            category=transaction.category
        )
        db.add(db_transaction)
        db.commit()
        db.refresh(db_transaction)
        
        return TransactionResponse(
            id=db_transaction.id,
            description=db_transaction.description,
            amount=db_transaction.amount,
            date=str(db_transaction.date),
            category=db_transaction.category,
            user_id=db_transaction.user_id
        )
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        db.close()

@router.get("/", response_model=List[TransactionResponse], summary="List transactions")
def list_transactions(user_id: Optional[int] = None):
    """Get all transactions, optionally filtered by user"""
    db = SessionLocal()
    try:
        query = db.query(Transaction)
        if user_id:
            query = query.filter(Transaction.user_id == user_id)
        transactions = query.all()
        
        return [
            TransactionResponse(
                id=t.id,
                description=t.description,
                amount=t.amount,
                date=str(t.date),
                category=t.category,
                user_id=t.user_id
            ) for t in transactions
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        db.close()

@router.get("/{transaction_id}", response_model=TransactionResponse, summary="Get transaction")
def get_transaction(transaction_id: int):
    """Get a specific transaction"""
    db = SessionLocal()
    try:
        transaction = db.query(Transaction).filter(Transaction.id == transaction_id).first()
        if not transaction:
            raise HTTPException(status_code=404, detail="Transaction not found")
        
        return TransactionResponse(
            id=transaction.id,
            description=transaction.description,
            amount=transaction.amount,
            date=str(transaction.date),
            category=transaction.category,
            user_id=transaction.user_id
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        db.close()

@router.put("/{transaction_id}", response_model=TransactionResponse, summary="Update transaction")
def update_transaction(transaction_id: int, transaction: TransactionCreate):
    """Update a transaction"""
    db = SessionLocal()
    try:
        db_transaction = db.query(Transaction).filter(Transaction.id == transaction_id).first()
        if not db_transaction:
            raise HTTPException(status_code=404, detail="Transaction not found")
        
        # Update fields
        if transaction.description is not None:
            db_transaction.description = transaction.description
        if transaction.category is not None:
            db_transaction.category = transaction.category
        db_transaction.amount = transaction.amount
        
        db.commit()
        db.refresh(db_transaction)
        
        return TransactionResponse(
            id=db_transaction.id,
            description=db_transaction.description,
            amount=db_transaction.amount,
            date=str(db_transaction.date),
            category=db_transaction.category,
            user_id=db_transaction.user_id
        )
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        db.close()

@router.delete("/{transaction_id}", summary="Delete transaction")
def delete_transaction(transaction_id: int):
    """Delete a transaction"""
    db = SessionLocal()
    try:
        transaction = db.query(Transaction).filter(Transaction.id == transaction_id).first()
        if not transaction:
            raise HTTPException(status_code=404, detail="Transaction not found")
        
        db.delete(transaction)
        db.commit()
        
        return {"message": "Transaction deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        db.close()
