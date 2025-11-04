from fastapi import APIRouter
from app.api import upload, configuration, analysis

api_router = APIRouter(prefix="/api")

api_router.include_router(upload.router)
api_router.include_router(configuration.router)
api_router.include_router(analysis.router)