# backend/core/config.py (Configuration management)
from pydantic_settings import BaseSettings
from typing import List
import os

class Settings(BaseSettings):
    # App settings
    APP_NAME: str = "AI Resume Analyzer"
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    PORT: int = 8000
    
    # Security
    SECRET_KEY: str = "your-secret-key-change-in-production"
    
    # CORS
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:3001", 
        "https://your-frontend-domain.com"
    ]
    
    # File upload settings
    MAX_FILE_SIZE: int = 16 * 1024 * 1024  # 16MB
    UPLOAD_FOLDER: str = "uploads"
    ALLOWED_EXTENSIONS: set = {"txt", "pdf", "docx", "html", "htm"}
    
    # AI/ML settings
    DEFAULT_INDUSTRY: str = "software_engineering"
    
    # Database (for future use)
    DATABASE_URL: str = "sqlite:///./resume_analyzer.db"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()

# Ensure upload directory exists
os.makedirs(settings.UPLOAD_FOLDER, exist_ok=True)