"""
API Client for connecting Streamlit to FastAPI backend
"""
import requests
from typing import Dict, List, Optional, Any
import streamlit as st

class FinanceAPIClient:
    def __init__(self, base_url: str = "http://api:8000"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make HTTP request to API"""
        url = f"{self.base_url}{endpoint}"
        try:
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"API Error: {str(e)}")
            return {"error": str(e)}
    
    # Health check
    def health_check(self) -> Dict[str, Any]:
        """Check if API is running"""
        return self._make_request("GET", "/")
    
    # Users
    def create_user(self, email: str, password: str) -> Dict[str, Any]:
        """Create a new user"""
        return self._make_request("POST", "/users/", json={"email": email, "password": password})
    
    def get_users(self) -> List[Dict[str, Any]]:
        """Get all users"""
        response = self._make_request("GET", "/users/")
        # Handle both response formats: {"users": [...]} or [{"id": 1, "email": "..."}]
        if "users" in response:
            return response["users"]
        elif isinstance(response, list):
            return response
        else:
            return []
    
    # Transactions
    def create_transaction(self, user_id: int, description: str, amount: float, category: Optional[str] = None) -> Dict[str, Any]:
        """Create a new transaction"""
        data = {
            "description": description,
            "amount": amount,
            "user_id": user_id
        }
        if category:
            data["category"] = category
        return self._make_request("POST", "/transactions/", json=data)
    
    def get_transactions(self, user_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get transactions, optionally filtered by user"""
        params = {}
        if user_id:
            params["user_id"] = user_id
        response = self._make_request("GET", "/transactions/", params=params)
        # Handle both response formats: {"transactions": [...]} or [{"id": 1, ...}]
        if isinstance(response, list):
            return response
        elif "transactions" in response:
            return response["transactions"]
        else:
            return []
    
    def get_transaction(self, transaction_id: int) -> Dict[str, Any]:
        """Get a specific transaction"""
        return self._make_request("GET", f"/transactions/{transaction_id}")
    
    def update_transaction(self, transaction_id: int, **kwargs) -> Dict[str, Any]:
        """Update a transaction"""
        return self._make_request("PUT", f"/transactions/{transaction_id}", json=kwargs)
    
    def delete_transaction(self, transaction_id: int) -> Dict[str, Any]:
        """Delete a transaction"""
        return self._make_request("DELETE", f"/transactions/{transaction_id}")
    
    # Forecast
    def get_forecast(self, user_id: int, days: int = 30) -> Dict[str, Any]:
        """Get financial forecast for a user"""
        params = {"user_id": user_id, "days": days}
        return self._make_request("GET", "/forecast/", params=params)

# Initialize API client
def get_api_client():
    """Get cached API client instance"""
    # Default to localhost for local development
    base_url = "http://localhost:8000"
    return FinanceAPIClient(base_url)
