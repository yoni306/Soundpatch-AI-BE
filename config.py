from pydantic_settings import BaseSettings
from typing import List
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    PROJECT_NAME: str = "FastAPI App"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8000"]
    SUPABASE_URL: str
    SUPABASE_KEY: str
    STORAGE_BUCKET: str = "videos"
    UPLOAD_DIR: str = "temp_files"
    
    class Config:
        case_sensitive = True

settings = Settings() 