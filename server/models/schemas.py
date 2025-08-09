# backend/models/schemas.py (Pydantic models)
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional
from enum import Enum

class IndustryEnum(str, Enum):
    SOFTWARE_ENGINEERING = "software_engineering"
    DATA_SCIENCE = "data_science"
    PRODUCT_MANAGEMENT = "product_management"
    MARKETING = "marketing"
    FINANCE = "finance"

class AnalyzeTextRequest(BaseModel):
    text: str = Field(..., min_length=50, max_length=50000, description="Resume text content")
    industry: IndustryEnum = Field(default=IndustryEnum.SOFTWARE_ENGINEERING, description="Target industry")
    
    @validator('text')
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError('Text cannot be empty')
        return v.strip()

class ContactInfo(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    linkedin: Optional[str] = None
    github: Optional[str] = None
    website: Optional[str] = None
    portfolio: Optional[str] = None

class Skills(BaseModel):
    programming: List[str] = []
    frameworks: List[str] = []
    cloud: List[str] = []
    ai_ml: List[str] = []
    soft_skills: List[str] = []
    other_tech: List[str] = []

class AnalysisResult(BaseModel):
    contact_info: ContactInfo
    skills: Skills
    experience_years: int = Field(..., ge=0, le=50)
    education_level: str
    ats_score: float = Field(..., ge=0, le=100)
    keyword_density: Dict[str, float]
    sentiment_score: float = Field(..., ge=0, le=1)
    readability_score: float = Field(..., ge=0, le=100)
    recommendations: List[str]
    missing_sections: List[str]
    industry_alignment: Dict[str, float]

class AnalysisResponse(BaseModel):
    success: bool = True
    analysis: Optional[AnalysisResult] = None
    text_length: Optional[int] = None
    processing_time: Optional[float] = None
    message: Optional[str] = None

class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    error_code: Optional[str] = None

class HealthResponse(BaseModel):
    status: str = "healthy"
    message: str
    version: str
    environment: str
    supported_formats: List[str]

class OptimizationRequest(BaseModel):
    resume_text: str = Field(..., min_length=50, description="Resume text to optimize")
    job_description: str = Field(..., min_length=50, description="Job description to optimize for")
    industry: IndustryEnum = Field(default=IndustryEnum.SOFTWARE_ENGINEERING)

class ComparisonRequest(BaseModel):
    resumes: Dict[str, str] = Field(..., min_items=2, description="Dictionary of resume texts")
    industry: IndustryEnum = Field(default=IndustryEnum.SOFTWARE_ENGINEERING)