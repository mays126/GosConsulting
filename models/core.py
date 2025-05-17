from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, Table, BigInteger, DateTime
from sqlalchemy.dialects.sqlite import BLOB
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.orm import DeclarativeBase
from .database import engine
from sqlalchemy.ext.declarative import declarative_base


Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column("id", Integer, primary_key=True, nullable=False, autoincrement=True)
    login = Column("login", String, nullable=False)
    hashed_password = Column("password",BLOB,nullable=False)

    user_requests = relationship("Request", back_populates="user", cascade="all, delete-orphan")

    tokens = relationship("Token", back_populates="user")


class Token(Base):
    __tablename__ = "tokens"

    id = Column("id", Integer, primary_key=True, nullable=False, autoincrement=True)
    refresh_token = Column("access_token", String, unique=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    expires_at = Column("expires_at", DateTime, nullable=False)
    is_revoked = Column("is_revoked", Boolean, nullable=False, default=False)
    token_type = Column("token_type", String, nullable=False)
    user = relationship("User", back_populates="tokens")

class Request(Base):
    __tablename__ = "requests"

    id = Column("id", Integer, primary_key=True, nullable=False, autoincrement=True)
    question = Column("question", String, nullable=False)
    answer = Column("answer", String, nullable=False)

    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    user = relationship("User", back_populates="user_requests")


def init_models():
    Base.metadata.create_all(engine)