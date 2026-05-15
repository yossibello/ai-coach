from fastapi import APIRouter

from app.api.endpoints import (
    auth, activities, strava, garmin, recommendations, profile, fitness, nutrition,
    tracking, health, oura, fitbit,
)

api_router = APIRouter()

api_router.include_router(auth.router,            prefix="/auth",            tags=["auth"])
api_router.include_router(profile.router,         prefix="/profile",         tags=["profile"])
api_router.include_router(activities.router,      prefix="/activities",      tags=["activities"])
api_router.include_router(strava.router,          prefix="/strava",          tags=["strava"])
api_router.include_router(garmin.router,          prefix="/garmin",          tags=["garmin"])
api_router.include_router(fitness.router,         prefix="/fitness",         tags=["fitness"])
api_router.include_router(recommendations.router, prefix="/coach",           tags=["coach"])
api_router.include_router(nutrition.router,       prefix="/nutrition",       tags=["nutrition"])
api_router.include_router(tracking.router,        prefix="/tracking",        tags=["tracking"])
api_router.include_router(health.router,          prefix="/health",          tags=["health"])
api_router.include_router(oura.router,            prefix="/oura",            tags=["oura"])
api_router.include_router(fitbit.router,          prefix="/fitbit",          tags=["fitbit"])
