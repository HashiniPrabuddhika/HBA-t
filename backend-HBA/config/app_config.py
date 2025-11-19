from pydantic_settings import BaseSettings
from pydantic import field_validator, Field
from functools import lru_cache
from typing import List


class Settings(BaseSettings):
    APP_NAME: str = "HBA Booking System"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    DATABASE_URL: str
    
    JWT_SECRET: str
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRY_HOURS: int = 1
    
    OPENAI_API_KEY: str
    OPENAI_API_KEY2: str
    LLM_MODEL: str = "z-ai/glm-4.5-air:free"
    LLM_BASE_URL: str = "https://openrouter.ai/api/v1/chat/completions"
    LLM_TEMPERATURE: float = 0.2
    
    CORS_ORIGINS: List[str] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://127.0.0.1:3000"]
    )

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            if v.startswith('[') and v.endswith(']'):
                import json
                try:
                    return json.loads(v)
                except json.JSONDecodeError:
                    v = v.strip('[]').replace('"', '').replace("'", "")
                    return [item.strip() for item in v.split(",") if item.strip()]
            return [item.strip() for item in v.split(",") if item.strip()]
        return v
    
    DATA_BASE_PATH: str = "./data"
    CACHE_BASE_PATH: str = "./data/cache"
    BACKUP_PATH: str = "./data/backups"
    
    CACHE_DB_PATH: str = "./data/cache/recommendations_cache.db"
    VECTOR_DB_PATH: str = "./data/embeddings"
    ANALYTICS_DB_PATH: str = "./data/analytics/analytics.db"
    
    CACHE_TTL_HOURS: int = 24
    CACHE_TTL: int = 300  
    
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    MAX_ALTERNATIVES: int = 10
    SIMILARITY_THRESHOLD: float = 0.7
    CONFIDENCE_THRESHOLD: float = 0.5
    ANALYTICS_WINDOW_DAYS: int = 30
    MIN_BOOKINGS_FOR_PATTERN: int = 3
    
    BUSINESS_START_HOUR: int = 7
    BUSINESS_END_HOUR: int = 21
    TIME_SLOT_MINUTES: int = 30
    
    CLUSTERING_MODEL_PATH: str = "./data/models/clustering_model.pkl"
    USER_EMBEDDING_PATH: str = "./data/embeddings/users"
    ROOM_EMBEDDING_PATH: str = "./data/embeddings/rooms"
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "allow"


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    Uses lru_cache to ensure only one Settings instance is created.
    """
    return Settings()


settings = get_settings()