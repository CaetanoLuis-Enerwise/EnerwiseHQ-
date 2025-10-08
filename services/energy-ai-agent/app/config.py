"""
Configuration for Enerwise Grid AI Agent
Production settings and environment variables
"""

from pydantic import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # API Settings
    APP_NAME: str = "Enerwise Grid AI Agent"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    
    # Model Settings
    USE_GPU: bool = False
    MODEL_CACHE_DIR: str = "./model_cache"
    ENSEMBLE_WEIGHTS: dict = {
        "nbeats": 0.35,
        "tft": 0.30, 
        "patchtst": 0.35
    }
    
    # External APIs
    WEATHER_API_URL: str = "https://api.open-meteo.com/v1/forecast"
    WEATHER_API_KEY: Optional[str] = None
    GRID_DATA_API: Optional[str] = None
    
    # Performance
    MAX_FORECAST_HORIZON: int = 168  # 1 week
    CACHE_TTL: int = 300  # 5 minutes
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Global settings instance
settings = Settings()
