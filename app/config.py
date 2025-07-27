#!/usr/bin/env python3

import os
from typing import Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings"""
    
    # API Configuration
    app_name: str = "LLM-Powered Intelligent Query-Retrieval System"
    version: str = "1.0.0"
    debug: bool = False
    
    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Authentication
    bearer_token: str = "43e704a77310d35ab207cbb456481b2657cbf41a97bd1d2a3800e648acacb5c1"
    
    # RAG Configuration - optimized for better retrieval
    chunk_size: int = 1000  # Increased for more context
    chunk_overlap: int = 200  # Increased for better continuity
    embedding_model: str = "gemini-embedding-001"
    embedding_dimensions: int = 768
    vector_db_path: str = "./data/chroma_db"
    gemini_model: str = "gemini-2.0-flash-exp"
    top_k: int = 10  # Increased for better search
    
    # API Keys
    gemini_api_key: Optional[str] = None
    
    # File Processing
    max_file_size: int = 50 * 1024 * 1024  # 50MB
    allowed_file_types: list = ["pdf", "docx"]
    
    # Performance
    max_concurrent_requests: int = 10
    request_timeout: int = 300  # 5 minutes
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Global settings instance
settings = Settings()
