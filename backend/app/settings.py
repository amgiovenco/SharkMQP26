from pydantic import BaseModel
from pydantic_settings import BaseSettings
from typing import List
import dotenv
import os

dotenv.load_dotenv()

class Settings(BaseSettings):
    # app
    app_env: str = os.getenv("APP_ENV", "dev")
    allowed_origins: str = os.getenv("ALLOWED_ORIGINS", "*")

    # auth
    jwt_secret: str = os.getenv("JWT_SECRET", "secret")
    jwt_expires_seconds: int = int(os.getenv("JWT_EXPIRES_SECONDS", "3600"))
    access_key: str = os.getenv("ACCESS_KEY", "")

    # redis
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    redis_queue: str = os.getenv("REDIS_QUEUE_NAME", "inference_queue")

    # file storage
    storage_dir: str = os.getenv("STORAGE_DIR", "/tmp/sharkid_files")

    # database
    database_url: str = os.getenv("DATABASE_URL", "postgresql+psycopg2://localhost/postgres")

settings = Settings()