from datetime import datetime, date, time
from fastapi import FastAPI, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from src.models import MRBSEntry, MRBSRoom
from src.database import get_db
import logging
import os
from langchain_core.language_models import BaseLLM
from langchain_core.outputs import LLMResult, Generation
import requests
from pydantic import BaseModel

from typing import Optional, List, Any
from src.api import router 
from src.deepseek_llm import DeepSeekLLM
from fastapi.middleware.cors import CORSMiddleware
from src.availability_logic import fetch_user_profile_by_email as fetch_profile_logic
from src.swap.swapMain import router as swap_router

from config.app_config import get_settings
from config.database_config import engine, Base
from api.routes import booking_routes, chat_routes, swap_routes
from utils.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)
app = FastAPI()

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_routes.router, prefix="/api/v1",tags=["Chat"])
app.include_router(swap_router)
app.include_router(booking_routes.router, prefix="/api/v1",tags=["Bookings"])
# app.include_router(swap_routes.router,prefix="/api/v1",tags=["Swaps"])

@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status": "healthy",
        "database": "connected",
        "timestamp": datetime.now().isoformat()
    }

