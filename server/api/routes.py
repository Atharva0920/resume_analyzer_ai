# backend/api/routes.py (API endpoints)
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
from fastapi.responses import JSONResponse
import time
from typing import Optional, Dict, Any
import tempfile
import os
from pathlib import Path

from models.schemas import (
    AnalyzeTextRequest, AnalysisResponse, ErrorResponse, 
    HealthResponse, IndustryEnum
)
from services.analysis_service import AnalysisService
from core.config import settings
from utils.validators import validate_file_type, validate_file_size

router = APIRouter()
analysis_service = AnalysisService()

@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint with system information"""
    return HealthResponse(
        message="AI Resume Analyzer API is running smoothly",
        version=settings.VERSION,
        environment=settings.ENVIRONMENT,
        supported_formats=list(settings.ALLOWED_EXTENSIONS)
    )

@router.post("/analyze-text", response_model=AnalysisResponse, tags=["Analysis"])
async def analyze_text(request: AnalyzeTextRequest):
    """Analyze resume text directly"""
    try:
        result = await analysis_service.analyze_resume_text(
            text=request.text,
            industry=request.industry.value
        )
        
        return AnalysisResponse(
            success=True,
            analysis=result["analysis"],
            text_length=result["text_length"],
            processing_time=result["processing_time"],
            message="Analysis completed successfully"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )

@router.post("/analyze-file", response_model=AnalysisResponse, tags=["Analysis"])
async def analyze_file(
    file: UploadFile = File(..., description="Resume file (PDF, DOCX, HTML, TXT)"),
    industry: IndustryEnum = Form(default=IndustryEnum.SOFTWARE_ENGINEERING)
):
    """Analyze uploaded resume file"""
    try:
        # Validate file
        validate_file_type(file.filename, settings.ALLOWED_EXTENSIONS)
        
        # Read file content
        content = await file.read()
        validate_file_size(content, settings.MAX_FILE_SIZE)
        
        # Analyze file
        result = await analysis_service.analyze_file(
            file_content=content,
            filename=file.filename,
            industry=industry.value
        )
        
        return AnalysisResponse(
            success=True,
            analysis=result["analysis"],
            text_length=result.get("extracted_text_length"),
            processing_time=result["processing_time"],
            message=f"File '{file.filename}' analyzed successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"File analysis failed: {str(e)}"
        )

@router.get("/industries", tags=["Utilities"])
async def get_supported_industries():
    """Get list of supported industries"""
    return {
        "industries": [
            {"value": "software_engineering", "label": "Software Engineering", "icon": "💻"},
            {"value": "data_science", "label": "Data Science", "icon": "📊"},
            {"value": "product_management", "label": "Product Management", "icon": "🚀"},
            {"value": "marketing", "label": "Marketing", "icon": "📢"},
            {"value": "finance", "label": "Finance", "icon": "💰"}
        ]
    }

@router.post("/optimize-resume", tags=["Optimization"])
async def optimize_resume_for_job(
    resume_text: str = Form(..., description="Resume text content"),
    job_description: str = Form(..., description="Job description to optimize for"),
    industry: IndustryEnum = Form(default=IndustryEnum.SOFTWARE_ENGINEERING)
):
    """Optimize resume for specific job description"""
    try:
        result = await analysis_service.optimize_for_job(
            resume_text=resume_text,
            job_description=job_description,
            industry=industry.value
        )
        
        return {
            "success": True,
            "optimization_result": result,
            "message": "Resume optimization completed successfully"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Optimization failed: {str(e)}"
        )

@router.post("/compare-resumes", tags=["Comparison"])
async def compare_resumes(
    resumes: Dict[str, str] = Form(..., description="Dictionary of resume texts to compare"),
    industry: IndustryEnum = Form(default=IndustryEnum.SOFTWARE_ENGINEERING)
):
    """Compare multiple resumes"""
    try:
        if len(resumes) < 2:
            raise HTTPException(
                status_code=400,
                detail="At least 2 resumes required for comparison"
            )
        
        result = await analysis_service.compare_resumes(
            resume_texts=list(resumes.values()),
            industry=industry.value
        )
        
        return {
            "success": True,
            "comparison_result": result,
            "message": f"Compared {len(resumes)} resumes successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Comparison failed: {str(e)}"
        )

@router.post("/export-report", tags=["Export"])
async def export_report(analysis_data: dict):
    """Generate and return detailed report"""
    try:
        # Generate a formatted report
        report = f"""
AI RESUME ANALYSIS REPORT
========================
Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}

OVERALL SCORES
--------------
ATS Score: {analysis_data.get('ats_score', 0)}/100
Readability Score: {analysis_data.get('readability_score', 0)}/100
Industry Alignment: {analysis_data.get('industry_alignment', {}).get('overall', 0):.1f}%
Experience: {analysis_data.get('experience_years', 0)} years

CONTACT INFORMATION
-------------------
Email: {analysis_data.get('contact_info', {}).get('email', 'Not provided')}
Phone: {analysis_data.get('contact_info', {}).get('phone', 'Not provided')}
LinkedIn: {analysis_data.get('contact_info', {}).get('linkedin', 'Not provided')}
GitHub: {analysis_data.get('contact_info', {}).get('github', 'Not provided')}

SKILLS ANALYSIS
---------------
Programming: {', '.join(analysis_data.get('skills', {}).get('programming', []))}
Frameworks: {', '.join(analysis_data.get('skills', {}).get('frameworks', []))}
Cloud Technologies: {', '.join(analysis_data.get('skills', {}).get('cloud', []))}

RECOMMENDATIONS
---------------
{chr(10).join(f"• {rec}" for rec in analysis_data.get('recommendations', []))}

MISSING SECTIONS
----------------
{', '.join(analysis_data.get('missing_sections', [])) if analysis_data.get('missing_sections') else 'None - all major sections present'}
        """
        
        return {
            "success": True, 
            "report": report.strip(),
            "report_type": "text",
            "generated_at": time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Report generation failed: {str(e)}"
        )

@router.get("/stats", tags=["Statistics"])
async def get_analysis_stats():
    """Get system statistics (placeholder for future implementation)"""
    return {
        "total_analyses": 0,  # Would be tracked in database
        "supported_formats": list(settings.ALLOWED_EXTENSIONS),
        "supported_industries": 5,
        "avg_processing_time": "2.5 seconds",
        "system_status": "operational"
    }

# Error handlers
# @router.exception_handler(HTTPException)
# async def http_exception_handler(request, exc):
#     """Handle HTTP exceptions"""
#     return JSONResponse(
#         status_code=exc.status_code,
#         content={
#             "success": False,
#             "error": exc.detail,
#             "error_code": f"HTTP_{exc.status_code}"
#         }
#     )

# @router.exception_handler(ValueError)
# async def value_error_handler(request, exc):
#     """Handle value errors"""
#     return JSONResponse(
#         status_code=400,
#         content={
#             "success": False,
#             "error": str(exc),
#             "error_code": "VALIDATION_ERROR"
#         }
#     )

# @router.exception_handler(Exception)
# async def general_exception_handler(request, exc):
#     """Handle general exceptions"""
#     return JSONResponse(
#         status_code=500,
#         content={
#             "success": False,
#             "error": "Internal server error",
#             "error_code": "INTERNAL_ERROR"
#         }
#     )