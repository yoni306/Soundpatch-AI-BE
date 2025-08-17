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
    SUPABASE_ORIGINAL_BUCKET: str = "recordings-original"
    SUPABASE_PROCESSED_BUCKET: str = "recordings-processed"
    STORAGE_BUCKET: str = "videos"
    UPLOAD_DIR: str = "temp_files"
    ASSEMBLYAI_API_KEY: str
    GEMINI_API_KEY: str
    # Model weights paths
    DETECT_NOISE_MODEL_WEIGHTS: str = "final_model.h5"
    TEXT_TO_MEL_MODEL_WEIGHTS: str = "text_to_mel_weights.h5"
    
    class Config:
        case_sensitive = True

settings = Settings() 