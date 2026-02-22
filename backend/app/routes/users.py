from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, EmailStr
from typing import List, Optional
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from ..database import SessionLocal
from ..models import User
from ..auth import hash_password

router = APIRouter()

class UserCreate(BaseModel):
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    id: int
    email: str

@router.post("/", response_model=UserResponse, summary="Create user")
def create_user(user: UserCreate):
    """Create a new user"""
    db = SessionLocal()
    try:
        # Check if user already exists
        existing_user = db.query(User).filter(User.email == user.email).first()
        if existing_user:
            raise HTTPException(status_code=400, detail="Email already registered")
        
        # Create new user
        hashed_password = hash_password(user.password)
        db_user = User(email=user.email, hashed_password=hashed_password)
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
        
        return UserResponse(id=db_user.id, email=db_user.email)
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        db.rollback()
        print(f"Database error: {str(e)}")  # Debug print
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        db.close()

@router.get("/", response_model=List[UserResponse], summary="List users")
def list_users():
    """Get all users"""
    db = SessionLocal()
    try:
        users = db.query(User).all()
        return [{"id": user.id, "email": user.email} for user in users]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()
