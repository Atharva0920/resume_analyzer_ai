from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn
import os
from pathlib import Path

from api.routes import router as api_router
from core.config import settings

# Create FastAPI app
app = FastAPI(
    title="AI Resume Analyzer API",
    description="Advanced AI-powered resume analysis with NLP and ML insights",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix="/api")

# Serve React build files in production
if settings.ENVIRONMENT == "production":
    # Mount static files
    static_path = Path("static")
    if static_path.exists():
        app.mount("/static", StaticFiles(directory="static"), name="static")
        
        @app.get("/{full_path:path}", response_class=HTMLResponse)
        async def serve_react_app(full_path: str):
            """Serve React app for all non-API routes"""
            if full_path.startswith("api/"):
                raise HTTPException(status_code=404, detail="API endpoint not found")
            
            index_file = static_path / "index.html"
            if index_file.exists():
                return HTMLResponse(content=index_file.read_text(), status_code=200)
            raise HTTPException(status_code=404, detail="Frontend not found")

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "AI Resume Analyzer API",
        "version": "1.0.0",
        "docs": "/api/docs",
        "health": "/api/health"
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.PORT,
        reload=settings.ENVIRONMENT == "development"
    )