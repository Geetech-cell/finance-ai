from fastapi import FastAPI
import logging
from sqlalchemy.exc import OperationalError

from .routes import users, transactions, forecast
from .database import Base, engine
from . import models

app = FastAPI(title="AI Finance SaaS API")

logger = logging.getLogger(__name__)


@app.on_event("startup")
def _create_tables():
    try:
        Base.metadata.create_all(bind=engine)
    except OperationalError as e:
        logger.error("Database connection failed during startup table creation")
        logger.error(str(e))

@app.get("/")
def root():
    return {"message": "AI Finance SaaS API. See /docs"}

app.include_router(users.router, prefix="/users", tags=["Users"])
app.include_router(transactions.router, prefix="/transactions", tags=["Transactions"])
app.include_router(forecast.router, prefix="/forecast", tags=["Forecast"])
