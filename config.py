
from pydantic_settings import BaseSettings


class AppConfig(BaseSettings):
    """Unified application configuration"""

    # API Configuration
    app_name: str = "LLM-Powered Intelligent Query-Retrieval System"
    version: str = "1.0.0"
    debug: bool = False

    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8080

    # Authentication
    bearer_token: str = (
        "43e704a77310d35ab207cbb456481b2657cbf41a97bd1d2a3800e648acacb5c1"
    )

    chunk_size: int = 1800
    chunk_overlap: int = 400
    embedding_model: str = "gemini-embedding-001"
    embedding_dimensions: int = 3072
    vector_db_path: str = "./data/chroma_db"
    gemini_model: str = "gemini-2.5-flash"
    gemini_url: str = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
    top_k: int = 20

    # Hybrid Search Configuration
    hybrid_search_enabled: bool = False
    hybrid_weight_semantic: float = 0.7
    hybrid_weight_keyword: float = 0.3
    bm25_top_k: int = 50

    # Query Expansion Configuration
    query_expansion_enabled: bool = False
    query_expansion_count: int = 3
    query_expansion_strategy: str = (
        "comprehensive"  # Options: "simple", "comprehensive", "domain_specific"
    )
    # Question batching
    question_batch_size: int = 5  # Max number of questions per LLM batch call

    # API Keys
    gemini_api_key: str | None = None

    # File Processing
    max_file_size: int = 500 * 1024 * 1024  # 500MB
    allowed_file_types: list = ["pdf", "docx", "doc"]

    # Performance
    max_concurrent_requests: int = 10
    request_timeout: int = 300  # 5 minutes
    agentic_urls: list = [
        "https://hackrx.blob.core.windows.net/hackrx/rounds/FinalRound4SubmissionPDF.pdf"
    ]

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "allow"

    def to_dict(self) -> dict:
        """Convert to dictionary for backward compatibility"""
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "embedding_model": self.embedding_model,
            "embedding_dimensions": self.embedding_dimensions,
            "vector_db_path": self.vector_db_path,
            "gemini_model": self.gemini_model,
            "gemini_url": self.gemini_url,
            "top_k": self.top_k,
        }

    def is_agentic_url(self, document_url: str) -> bool:
        return any(pattern in document_url for pattern in self.agentic_urls)


config = AppConfig()
