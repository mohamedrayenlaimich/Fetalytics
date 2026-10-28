from fastapi import APIRouter
from Backend.app.api.v1.endpoints.patients import router as patients_router
from Backend.app.api.v1.endpoints.scans import router as scans_router
from Backend.app.api.v1.endpoints.measurements import router as measurements_router
from Backend.app.api.v1.endpoints.analysis import router as analysis_router
from Backend.app.api.v1.endpoints.reports import router as reports_router

api_v1_router = APIRouter()

api_v1_router.include_router(patients_router)
api_v1_router.include_router(scans_router)
api_v1_router.include_router(measurements_router)
api_v1_router.include_router(analysis_router)
api_v1_router.include_router(reports_router)
