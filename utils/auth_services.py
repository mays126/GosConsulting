from datetime import datetime, timedelta, timezone
from sqlalchemy import select, delete
from sqlalchemy.orm import Session
from fastapi import HTTPException

from models.core import User, Token
from utils.jwt_utils import hash_password, validate_password, encode_jwt, decode_jwt


async def create_user(db: Session, user_data):
    existing_user = db.execute(select(User).where(User.login == user_data.login))
    if existing_user.scalar():
        raise HTTPException(status_code=400, detail="User already exists")

    user = User(
        login=user_data.login,
        hashed_password=hash_password(user_data.password)
    )
    db.add(user)
    db.commit()
    return user


async def validate_user_auth(db: Session, user_login: str, user_password: str):
    result = db.execute(select(User).where(User.login == user_login))
    user: User = result.scalar()

    if not user or not validate_password(user_password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Incorrect username or password")

    return user


async def create_token(user_id: int, data: dict, expires_delta: timedelta, token_type: str, db: Session):
    expire = (datetime.now(timezone.utc) + expires_delta).replace(tzinfo=None)
    data["exp"] = expire
    data["token_type"] = token_type

    token_str = encode_jwt(data)

    if token_type == "refresh":
        token = Token(
            refresh_token=token_str,
            user_id=user_id,
            expires_at=expire,
            token_type=token_type,
            is_revoked=False,
        )
        db.add(token)
        db.commit()

    return token_str


async def validate_refresh_token(db: Session, refresh_token: str) -> User:
    result = db.execute(
        select(Token).where(
            Token.refresh_token == refresh_token,
            Token.is_revoked == False
        )
    )
    db_token: Token = result.scalar()

    if not db_token:
        raise HTTPException(status_code=401, detail="Refresh token not found")

    expires_at = db_token.expires_at

    if isinstance(expires_at, str):
        try:
            expires_at = datetime.fromisoformat(expires_at)
        except ValueError:
            raise HTTPException(status_code=500, detail="Invalid expires_at format in DB")

    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)

    if expires_at < datetime.now(timezone.utc):
        raise HTTPException(status_code=401, detail="Refresh token expired")

    user_result = db.execute(select(User).where(User.id == db_token.user_id))
    user = user_result.scalar()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return user


async def delete_refresh_token(db: Session, refresh_token: str):
    db.execute(delete(Token).where(Token.refresh_token == refresh_token))
    db.commit()

