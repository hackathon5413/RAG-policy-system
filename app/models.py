#!/usr/bin/env python3

from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, HttpUrl, field_validator
from typing import List, Optional
import re
from config import config  # Use unified config

# Security
security = HTTPBearer()

class HackRXRequest(BaseModel):
    """Request model for HackRX API endpoint"""
    documents: HttpUrl
    questions: List[str]
    
    @field_validator('questions')
    @classmethod
    def validate_questions(cls, v):
        if not v:
            raise ValueError('At least one question is required')
        if len(v) > 500:
            raise ValueError('Maximum 500 questions allowed')
        for question in v:
            if not question.strip():
                raise ValueError('Questions cannot be empty')
            if len(question) > 10000:
                raise ValueError('Question too long (max 10000 characters)')
        return v
    
    @field_validator('documents')
    @classmethod
    def validate_document_url(cls, v):
        url_str = str(v)
        # Check if it's a valid blob URL
        if not any(pattern in url_str.lower() for pattern in ['blob.core.windows.net', '.pdf']):
            raise ValueError('Document URL must be a valid PDF blob URL')
        return v

class HackRXResponse(BaseModel):
    """Response model for HackRX API endpoint"""
    answers: List[str]

class ErrorResponse(BaseModel):
    """Error response model"""
    detail: str
    error_code: Optional[str] = None
    timestamp: Optional[str] = None

# Authentication dependency
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Verify bearer token authentication"""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if credentials.credentials != config.bearer_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return credentials.credentials

