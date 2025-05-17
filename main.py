from fastapi import FastAPI
from routers.user import router as user_router
from contextlib import asynccontextmanager
from models.core import init_models
from controllers.user import lifespan







app = FastAPI(lifespan=lifespan)

app.include_router(
    router=user_router,
    prefix=""
)

