from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from config import settings
from routes.process import router as process_router

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="A basic FastAPI application"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(process_router)

@app.get("/")
async def root():
    return {"message": "Welcome to FastAPI application"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"} 