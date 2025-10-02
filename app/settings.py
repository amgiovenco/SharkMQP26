from pydantic import BaseModel
from pydantic_settings import BaseSettings
from typing import List
import dotenv
import os

dotenv.load_dotenv()

class Settings(BaseSettings):
    app_env: str = os.getenv("APP_ENV", "dev")
    allowed_origins: str = os.getenv("ALLOWED_ORIGINS", "*")

    jwt_secret: str = os.getenv("JWT_SECRET", "devsecret")
    jwt_expires_seconds: int = int(os.getenv("JWT_EXPIRES_SECONDS", "3600"))

    files_dir: str = os.getenv("FILES_DIR", "/tmp/sharkid_files")

    database_url: str = os.getenv("DATABASE_URL", "postgresql+psycopg2://localhost/postgres")
    redis_url: str = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")

settings = Settings()