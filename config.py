import os
from pathlib import Path

BASE_DIR = Path(__file__).parent
ASSETS_DIR = BASE_DIR / "assets"
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
BATCH_SIZE = 32

VECTOR_DB_PATH = str(EMBEDDINGS_DIR / "chroma_db")
METADATA_DB_PATH = str(DATA_DIR / "metadata.db")

INSURANCE_KEYWORDS = [
    "cover", "coverage", "benefit", "claim", "policy", "premium", "insured", 
    "deductible", "exclusion", "waiting period", "sum insured", "hospitalization",
    "treatment", "medical", "surgery", "ambulance", "maternity", "baby"
]
