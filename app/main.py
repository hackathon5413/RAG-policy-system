from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import logging
from datetime import datetime, timezone
from dotenv import load_dotenv

from .models import HackRXRequest, HackRXResponse, LocalTestRequest, ErrorResponse, verify_token
from config import config 
from .document_processor import process_document_and_answer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress verbose logs from various components
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("app.embeddings").setLevel(logging.WARNING) 
logging.getLogger("google").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

load_dotenv()

# FastAPI app initialization
app = FastAPI(
    title=config.app_name,
    description="Process large documents and make contextual decisions for insurance, legal, HR, and compliance domains",
    version=config.version,
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
            timestamp=datetime.now(timezone.utc).isoformat()
        ).model_dump()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            detail="Internal server error",
            error_code="INTERNAL_ERROR",
            timestamp=datetime.now(timezone.utc).isoformat()
        ).model_dump()
    )


@app.post("/api/v1/hackrx/run", response_model=HackRXResponse)
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

        logger.info(f"Incoming request - Document: {request.documents}")
        logger.info(f"Incoming request - Questions: {request.questions}")
        logger.info(f"Incoming request - Number of questions: {len(request.questions)}")
        
        result = await process_document_and_answer(
            str(request.documents), 
            request.questions
        )
        logger.info(f"Processing result: {result}")
        
        
        if result["success"]:
            logger.info(f"Successfully generated {len(result['answers'])} answers")
            response = HackRXResponse(answers=result["answers"])
            logger.info(f"Response - Success: True, Answers count: {len(response.answers)}")
            return response
        else:
            logger.error(f"Document processing failed: {result['error']}")
            response = HackRXResponse(answers=result["answers"])
            logger.info(f"Response - Success: False, Error answers count: {len(response.answers)}")
            return response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing request: {str(e)}"
        )

@app.post("/api/v1/test/local", response_model=HackRXResponse)
async def test_local_file(
    request: LocalTestRequest,
    token: str = Depends(verify_token)
) -> HackRXResponse:
    """Test endpoint for local file processing"""
    try:
        from .document_processor import process_local_document, answer_questions
        import os
        import hashlib
        
        if not os.path.exists(request.file_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        file_type = 'pdf' if request.file_path.lower().endswith('.pdf') else 'docx'
        url_hash = hashlib.md5(request.file_path.encode()).hexdigest()
        
        # Process document
        result = await process_local_document(request.file_path, file_type, url_hash)
        
        if not result["success"]:
            return HackRXResponse(answers=[f"Error: {result['error']}" for _ in request.questions])
        
        # Answer questions
        answers = await answer_questions(request.questions)
        
        return HackRXResponse(answers=answers)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing local file: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing local file: {str(e)}"
        )

# Development server
if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=config.host,
        port=config.port,
        reload=config.debug
    )
