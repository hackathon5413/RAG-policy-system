
from pydantic_settings import BaseSettings


class AppConfig(BaseSettings):

    app_name: str = "LLM-Powered Intelligent Query-Retrieval System"
    version: str = "1.0.0"
    debug: bool = False

    host: str = "0.0.0.0"
    port: int = 8080

    bearer_token: str = (
        "43e704a77310d35ab207cbb456481b2657cbf41a97bd1d2a3800e648acacb5c1"
    )

    chunk_size: int = 2408
    chunk_overlap: int = 200
    embedding_model: str = "gemini-embedding-001"
    embedding_dimensions: int = 3072
    vector_db_path: str = "./data/chroma_db"
    gemini_model: str = "gemini-2.5-flash"
    gemini_url: str = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
    top_k: int = 20

    # Reranking Configuration
    reranking_enabled: bool = False  # Enable/disable cross-encoder reranking

    # Task Classification Configuration
    task_classification_enabled: bool = False  # Enable/disable LLM-based task classification

    # Sub-question Configuration
    sub_questions_enabled: bool = False  # Enable/disable sub-question generation

    # Question batching
    question_batch_size: int = 5  # Max number of questions per LLM batch call


    max_file_size: int = 500 * 1024 * 1024  # 500MB


    # Performance
    max_concurrent_requests: int = 10
    request_timeout: int = 300  # 5 minutes
    agentic_urls: list = [
        "https://hackrx.blob.core.windows.net/hackrx/rounds/FinalRound4SubmissionPDF.pdf"
    ]


    def is_agentic_url(self, document_url: str) -> bool:
        return any(pattern in document_url for pattern in self.agentic_urls)


config = AppConfig()
