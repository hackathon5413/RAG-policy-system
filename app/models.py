
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, HttpUrl, field_validator

from config import config

security = HTTPBearer()


class HackRXRequest(BaseModel):
    documents: HttpUrl
    questions: list[str]

    @field_validator("questions")
    @classmethod
    def validate_questions(cls, v):
        if not v:
            raise ValueError("At least one question is required")
        if len(v) > 500:
            raise ValueError("Maximum 500 questions allowed")
        for question in v:
            if not question.strip():
                raise ValueError("Questions cannot be empty")
            if len(question) > 10000:
                raise ValueError("Question too long (max 10000 characters)")
        return v

    @field_validator("documents")
    @classmethod
    def validate_document_url(cls, v):
        url_str = str(v)
        # Accept any valid HTTP/HTTPS URL
        if not url_str.lower().startswith("https://"):
            raise ValueError("Document URL must be a valid HTTPS URL")

        # Check that URL doesn't point to a .bin file
        if url_str.lower().endswith(".bin"):
            raise ValueError("Document URL cannot point to a .bin file")

        return v


class HackRXResponse(BaseModel):
    answers: list[str]


class LocalTestRequest(BaseModel):
    file_path: str
    questions: list[str]

    @field_validator("questions")
    @classmethod
    def validate_questions(cls, v):
        if not v:
            raise ValueError("At least one question is required")
        if len(v) > 500:
            raise ValueError("Maximum 500 questions allowed")
        for question in v:
            if not question.strip():
                raise ValueError("Questions cannot be empty")
            if len(question) > 10000:
                raise ValueError("Question too long (max 10000 characters)")
        return v


class ErrorResponse(BaseModel):
    detail: str
    error_code: str | None = None
    timestamp: str | None = None


async def verify_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> str:
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
