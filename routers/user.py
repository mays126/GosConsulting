import os
from contextlib import asynccontextmanager
from typing import List

from fastapi import APIRouter, Cookie, Depends, FastAPI
from sqlalchemy.orm import Session

from controllers.auth import (
    authenticate_user, refresh_access_token, logout_user, register_user
)

from controllers.user import *

from models.core import init_models
from models.schemas import UserCreate, Token, LLMQuestion, HistoryModel
from models.database import get_session
from utils import rag_pipeline
from utils.jwt_utils import get_current_user

router = APIRouter()



@router.post("/register", response_model=Token)
async def register(user_schema: UserCreate, db: Session = Depends(get_session)):
    return await register_user(user_schema, db)


@router.post("/login", response_model=Token)
async def login(user_schema: UserCreate, db: Session = Depends(get_session)):
    return await authenticate_user(user_schema, db)


@router.post("/refresh", response_model=Token)
async def refresh(refresh_token: str = Cookie(None), db: Session = Depends(get_session)):
    return await refresh_access_token(refresh_token, db)


@router.post("/logout")
async def logout(refresh_token: str = Cookie(None), db: Session = Depends(get_session)):
    return await logout_user(refresh_token, db)

@router.post("/send_request")
async def send_request(question: LLMQuestion, current_user = Depends(get_current_user),session: Session = Depends(get_session)):
    return await send_question(int(current_user.get("sub")),question,session)

@router.get("/get_history", response_model=List[HistoryItem])
async def get_history(current_user = Depends(get_current_user),db: Session = Depends(get_session)):
    return await get_requests_history(int(current_user.get("sub")),db)