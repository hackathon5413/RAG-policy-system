from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from .embeddings import GeminiEmbeddings, OpenAIEmbeddings
from config import config
import os

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=config.chunk_size,
    chunk_overlap=config.chunk_overlap,
    separators=["\n\n", "\n", ". ", " "]
)

os.makedirs(os.path.dirname(config.vector_db_path), exist_ok=True)
os.makedirs(os.path.dirname(f"{config.vector_db_path}_openai"), exist_ok=True)

def get_embeddings():
    return GeminiEmbeddings()

def get_vectorstore():
    return Chroma(
        persist_directory=config.vector_db_path,
        embedding_function=get_embeddings()
    )

vectorstore = None
openai_vectorstore = None

def get_openai_embeddings():
    return OpenAIEmbeddings()

def get_openai_vectorstore():
    return Chroma(
        persist_directory=f"{config.vector_db_path}_openai",
        embedding_function=get_openai_embeddings()
    )

def init_vectorstore():
    global vectorstore
    if vectorstore is None:
        vectorstore = get_vectorstore()
    return vectorstore

def init_openai_vectorstore():
    global openai_vectorstore
    if openai_vectorstore is None:
        openai_vectorstore = get_openai_vectorstore()
    return openai_vectorstore
