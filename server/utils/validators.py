# backend/utils/validators.py (Input validation)
from fastapi import HTTPException
from pathlib import Path
from typing import Set

def validate_file_type(filename: str, allowed_extensions: Set[str]) -> None:
    """Validate file extension"""
    if not filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    file_extension = Path(filename).suffix.lower().lstrip('.')
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"File type '.{file_extension}' not supported. Allowed: {', '.join(allowed_extensions)}"
        )

def validate_file_size(content: bytes, max_size: int) -> None:
    """Validate file size"""
    if len(content) > max_size:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {max_size // (1024*1024)}MB"
        )
    
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Empty file provided")

def validate_text_content(text: str, min_length: int = 50, max_length: int = 50000) -> None:
    """Validate text content"""
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Text content cannot be empty")
    
    text_length = len(text.strip())
    
    if text_length < min_length:
        raise HTTPException(
            status_code=400, 
            detail=f"Text too short. Minimum length: {min_length} characters"
        )
    
    if text_length > max_length:
        raise HTTPException(
            status_code=400,
            detail=f"Text too long. Maximum length: {max_length} characters"
        )