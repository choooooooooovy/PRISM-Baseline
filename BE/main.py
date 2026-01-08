# Load environment variables FIRST before any other imports
from dotenv import load_dotenv
from pathlib import Path
import os

env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

# Now import other modules
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import llm, report
from utils.logger import setup_logging

# Setup logging
setup_logging()

# Create FastAPI app
app = FastAPI(
    title="CASVE Decision Support API",
    description="Backend API for CASVE Worksheet with LLM integration",
    version="1.0.0"
)

# CORS configuration
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(llm.router, prefix="/api", tags=["llm"])
app.include_router(report.router, prefix="/api", tags=["report"])

@app.get("/")
async def root():
    return {
        "message": "CASVE Decision Support API",
        "status": "running",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
