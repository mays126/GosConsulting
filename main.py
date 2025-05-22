from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers.user import router as user_router
from contextlib import asynccontextmanager
from models.core import init_models
from controllers.user import lifespan







app = FastAPI(lifespan=lifespan)

app.include_router(
    router=user_router,
    prefix=""
)


origins = [
    "http://localhost:8000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
