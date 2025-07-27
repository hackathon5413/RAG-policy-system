#!/usr/bin/env python3

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Dict, Any
import uvicorn
import logging
from datetime import datetime
import asyncio

# Import our models and dependencies
from .models import HackRXRequest, HackRXResponse, ErrorResponse, verify_token, validate_request_size
from .config import settings
from .document_processor import process_document_and_answer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app initialization
app = FastAPI(
    title=settings.app_name,
    description="Process large documents and make contextual decisions for insurance, legal, HR, and compliance domains",
    version=settings.version,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            detail=exc.detail,
            error_code=f"HTTP_{exc.status_code}",
            timestamp=datetime.utcnow().isoformat()
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            detail="Internal server error",
            error_code="INTERNAL_ERROR",
            timestamp=datetime.utcnow().isoformat()
        ).dict()
    )

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "message": "LLM Query-Retrieval System is running",
        "timestamp": datetime.utcnow().isoformat()
    }

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": settings.app_name,
        "version": settings.version,
        "docs": "/docs",
        "health": "/health",
        "api_endpoint": "/hackrx/run"
    }

# Main API endpoint - exact path from problem statement
@app.post("/hackrx/run", response_model=HackRXResponse)
async def run_hackrx(
    request: HackRXRequest,
    token: str = Depends(verify_token)
) -> HackRXResponse:
    """
    Run document analysis and answer questions
    
    This endpoint processes PDF documents from blob URLs and answers questions about them.
    
    - **documents**: PDF blob URL to process
    - **questions**: List of questions to answer about the document
    
    Returns structured answers based on document content analysis.
    """
    try:
        # Validate request
        request = validate_request_size(request)
        
        logger.info(f"Processing document: {request.documents}")
        logger.info(f"Number of questions: {len(request.questions)}")
        
        # Process document and answer questions
        result = await process_document_and_answer(
            str(request.documents), 
            request.questions
        )
        
        if result["success"]:
            logger.info(f"Successfully generated {len(result['answers'])} answers")
            return HackRXResponse(answers=result["answers"])
        else:
            logger.error(f"Document processing failed: {result['error']}")
            # Return error answers but with 200 status to match expected format
            return HackRXResponse(answers=result["answers"])
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing request: {str(e)}"
        )

# Development server
if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )
