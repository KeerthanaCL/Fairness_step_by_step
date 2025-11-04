from fastapi import APIRouter
from app.api import upload, configuration, analysis, bias, mitigation

api_router = APIRouter(prefix="/api")

api_router.include_router(upload.router)
api_router.include_router(configuration.router)
api_router.include_router(analysis.router)
api_router.include_router(bias.router)
api_router.include_router(mitigation.router)