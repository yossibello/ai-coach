from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # App
    ENVIRONMENT: str = "development"
    SECRET_KEY: str = "changeme"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24 hours

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://aicoach:aicoach_secret@localhost:5432/aicoach"

    # Redis
    REDIS_URL: str = "redis://localhost:6379"

    # Strava
    STRAVA_CLIENT_ID: str = ""
    STRAVA_CLIENT_SECRET: str = ""
    STRAVA_WEBHOOK_VERIFY_TOKEN: str = "aicoach_webhook"
    STRAVA_REDIRECT_URI: str = "http://localhost:3000/api/strava/callback"

    # Fitbit OAuth
    FITBIT_CLIENT_ID: str = ""
    FITBIT_CLIENT_SECRET: str = ""
    FITBIT_REDIRECT_URI: str = "http://localhost:8000/api/v1/fitbit/callback"
    FRONTEND_URL: str = "http://localhost:3000"

    # OpenAI (optional text explanations)
    OPENAI_API_KEY: str = ""

    # ML
    ML_MODEL_PATH: str = "./models/cycling_coach.pt"

    # Storage
    UPLOAD_DIR: str = "./uploads"

    # CORS / Hosts
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:3001"]
    ALLOWED_HOSTS: List[str] = ["localhost", "127.0.0.1"]


settings = Settings()
