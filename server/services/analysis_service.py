# backend/services/analysis_service.py (Business logic)
import time
import tempfile
import os
from typing import Dict, Any
from dataclasses import asdict
from pathlib import Path

from core.resume_analyzer import AIResumeAnalyzer, ResumeOptimizer, ResumeComparer
from models.schemas import AnalysisResult, ContactInfo, Skills

class AnalysisService:
    def __init__(self):
        self.analyzer = AIResumeAnalyzer()
        self.optimizer = ResumeOptimizer()
        self.comparer = ResumeComparer()
    
    async def analyze_resume_text(self, text: str, industry: str) -> Dict[str, Any]:
        """Analyze resume text and return structured results"""
        start_time = time.time()
        
        try:
            # Perform analysis using the analyzer
            analysis = await self.analyzer.analyze_resume(text, industry)
            
            processing_time = time.time() - start_time
            
            return {
                "analysis": asdict(analysis),
                "text_length": len(text),
                "processing_time": round(processing_time, 2)
            }
            
        except Exception as e:
            raise Exception(f"Analysis failed: {str(e)}")
    
    async def analyze_file(self, file_content: bytes, filename: str, industry: str) -> Dict[str, Any]:
        """Analyze uploaded file"""
        # Save file temporarily
        file_extension = Path(filename).suffix.lower()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            tmp_file.write(file_content)
            tmp_path = tmp_file.name
        
        try:
            # Extract text from file
            resume_text = self.analyzer.file_processor.extract_text_from_file(tmp_path)
            
            # Analyze the extracted text
            result = await self.analyze_resume_text(resume_text, industry)
            result["extracted_text_length"] = len(resume_text)
            result["file_type"] = file_extension
            
            return result
            
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    async def optimize_for_job(self, resume_text: str, job_description: str, industry: str) -> Dict[str, Any]:
        """Optimize resume for specific job description"""
        try:
            result = await self.optimizer.optimize_for_job(resume_text, job_description, industry)
            return result
        except Exception as e:
            raise Exception(f"Optimization failed: {str(e)}")
    
    async def compare_resumes(self, resume_texts: list, industry: str) -> Dict[str, Any]:
        """Compare multiple resumes"""
        try:
            result = await self.comparer.compare_resumes(resume_texts, industry)
            return result
        except Exception as e:
            raise Exception(f"Comparison failed: {str(e)}")