import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from config import config

from .embeddings import GeminiEmbeddings

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=config.chunk_size,
    chunk_overlap=config.chunk_overlap,
    separators=["\n\n", "\n", ". ", " "],
)

os.makedirs(os.path.dirname(config.vector_db_path), exist_ok=True)


def get_embeddings():
    return GeminiEmbeddings()


def get_vectorstore():
    return Chroma(
        persist_directory=config.vector_db_path, embedding_function=get_embeddings()
    )


vectorstore = None


def init_vectorstore():
    global vectorstore
    if vectorstore is None:
        vectorstore = get_vectorstore()
    return vectorstore
