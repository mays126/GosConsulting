from fastapi import HTTPException
from sqlalchemy.orm import Session
from datetime import timedelta
from fastapi.responses import JSONResponse


from utils.auth_services import (
    create_user, validate_user_auth, create_token, validate_refresh_token, delete_refresh_token
)
from models.schemas import UserCreate


async def register_user(user_schema: UserCreate, db: Session):
    user = await create_user(db, user_schema)

    access_token = await create_token(user.id, {"sub": str(user.id)}, timedelta(minutes=60),
                                      "access", db)
    refresh_token = await create_token(user.id, {"sub": str(user.id)}, timedelta(days=30), "refresh", db)

    return _create_auth_response(access_token, refresh_token)


async def authenticate_user(user_schema: UserCreate, db: Session):
    user = await validate_user_auth(db, user_schema.login, user_schema.password)

    access_token = await create_token(user.id, {"sub": str(user.id)}, timedelta(minutes=60),"access", db)
    refresh_token = await create_token(user.id, {"sub": str(user.id)}, timedelta(days=30), "refresh", db)

    return _create_auth_response(access_token, refresh_token)


async def refresh_access_token(refresh_token: str, db: Session):
    if not refresh_token:
        raise HTTPException(status_code=401, detail="Refresh token missing")

    user = await validate_refresh_token(db, refresh_token)
    access_token = await create_token(user.id, {"sub": str(user.id)}, timedelta(minutes=10),
                                      "access", db)

    return JSONResponse({"access_token": access_token, "token_type": "Bearer"})


async def logout_user(refresh_token: str, db: Session):
    if not refresh_token:
        raise HTTPException(status_code=401, detail="Refresh token missing")

    await delete_refresh_token(db, refresh_token)

    response = JSONResponse({"message": "Logged out"})
    response.delete_cookie("refresh_token")
    return response


def _create_auth_response(access_token: str, refresh_token: str):
    response = JSONResponse({"access_token": access_token, "token_type": "Bearer", "refresh_token": refresh_token})
    response.set_cookie(key="refresh_token", value=refresh_token, httponly=True )
    return response