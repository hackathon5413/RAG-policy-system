

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from .embeddings import GeminiEmbeddings
from config import config
import os

# Initialize embeddings
embeddings = GeminiEmbeddings()

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=config.chunk_size,
    chunk_overlap=config.chunk_overlap,
    separators=["\n\n", "\n", ". ", " "]
)

# Initialize vector store
os.makedirs(os.path.dirname(config.vector_db_path), exist_ok=True)
vectorstore = Chroma(
    persist_directory=config.vector_db_path,
    embedding_function=embeddings
)
