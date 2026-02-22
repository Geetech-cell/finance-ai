from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from ..database import SessionLocal
from ..models import Transaction

router = APIRouter()

class ForecastResponse(BaseModel):
    forecast_data: dict
    summary: dict

@router.get("/", response_model=ForecastResponse, summary="Get forecast")
def get_forecast(user_id: Optional[int] = Query(None), days: int = Query(30)):
    """Get financial forecast for a user"""
    db = SessionLocal()
    try:
        # For now, return mock forecast data
        # In a real implementation, this would use the trained ML models
        forecast_data = {
            "user_id": user_id,
            "days": days,
            "predicted_spending": 2500.50,
            "confidence": 0.85,
            "categories": {
                "Food & Dining": 800.00,
                "Transportation": 400.00,
                "Shopping": 600.00,
                "Bills & Utilities": 700.50
            }
        }
        
        summary = {
            "total_predicted": 2500.50,
            "daily_average": 83.35,
            "highest_category": "Food & Dining",
            "recommendation": "Consider reducing dining expenses by 15%"
        }
        
        return ForecastResponse(forecast_data=forecast_data, summary=summary)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecast error: {str(e)}")
    finally:
        db.close()
