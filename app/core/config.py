import os
from typing import List, Optional
from pydantic import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "NSFW Scanner API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Security
    SECRET_KEY: str = "your-secret-key-here"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Model paths
    NSFW_MODEL_PATH: str = "models/nsfw_model.h5"
    FEATURE_EXTRACTOR_PATH: str = "models/feature_extractor"
    
    # Cache
    REDIS_URL: str = "redis://localhost:6379"
    CACHE_TTL: int = 3600
    
    # Rate limiting
    RATE_LIMIT_PER_MINUTE: int = 100
    
    # File upload
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB
    ALLOWED_EXTENSIONS: List[str] = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"]
    
    class Config:
        case_sensitive = True

settings = Settings()
