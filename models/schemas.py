from pydantic import BaseModel
from typing import List, Optional, Union, Dict


class UserBase(BaseModel):
    login: str


class UserCreate(UserBase):
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str


class LLMQuestion(BaseModel):
    question: str

class LLMAnswer(BaseModel):
    answer: str