#!/usr/bin/env python3

import os
import json
import requests
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from jinja2 import Environment, FileSystemLoader
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.embeddings.base import Embeddings
from google import genai
from google.genai import types

load_dotenv()

# Configuration
CONFIG = {
    "chunk_size": 600,
    "chunk_overlap": 100,
    "embedding_model": "gemini-embedding-001",
    "embedding_dimensions": 768,  # Using 768 for cost/performance balance
    "vector_db_path": "./data/chroma_db",
    "gemini_model": "gemini-1.5-flash",
    "gemini_url": "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent", 
    "top_k": 5
}

# Load keywords once at import time
def load_keywords():
    try:
        with open('./config/section_keywords.json', 'r') as f:
            section_keywords = json.load(f)
        with open('./config/cleanup_keywords.json', 'r') as f:
            cleanup_keywords = json.load(f)
    except FileNotFoundError:
        section_keywords = {
            "exclusion": ["exclusion", "not cover", "excluded"],
            "coverage": ["cover", "benefit", "covered"],
            "claims": ["claim", "procedure", "documentation"]
        }
        cleanup_keywords = {
            "company_headers": ["bajaj allianz", "page ", "www.", "uin-"]
        }
    return section_keywords, cleanup_keywords


section_keywords, cleanup_keywords = load_keywords()

# Custom Gemini Embeddings Class
class GeminiEmbeddings(Embeddings):
    """Custom Gemini embeddings with task-specific optimization"""
    
    def __init__(self, model_name: str = "gemini-embedding-001", dimensions: int = 768):
        self.model_name = model_name
        self.dimensions = dimensions
        
        # Initialize Gemini client
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        
        self.client = genai.Client(api_key=api_key)
        
    def _normalize_embedding(self, embedding: List[float]) -> List[float]:
        """Normalize embedding vector for accurate similarity comparison"""
        embedding_np = np.array(embedding)
        norm = np.linalg.norm(embedding_np)
        if norm == 0:
            return embedding
        return (embedding_np / norm).tolist()
    
    def _get_embedding(self, text: str, task_type: str) -> List[float]:
        """Get embedding for a single text with specific task type"""
        try:
            config = types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=self.dimensions
            )
            
            result = self.client.models.embed_content(
                model=self.model_name,
                contents=text,
                config=config
            )
            
            # Extract embedding values from the response
            if result and result.embeddings:
                [embedding_obj] = result.embeddings
                embedding_values = embedding_obj.values
                
                # Ensure we have a list of floats
                if embedding_values is not None:
                    if isinstance(embedding_values, list):
                        embedding = embedding_values
                    elif hasattr(embedding_values, 'tolist'):
                        embedding = embedding_values.tolist()
                    else:
                        embedding = list(embedding_values)
                    
                    # Normalize if not using full 3072 dimensions
                    if self.dimensions != 3072:
                        embedding = self._normalize_embedding(embedding)
                        
                    return embedding
                else:
                    raise ValueError("Embedding values are None")
            else:
                raise ValueError("No embeddings returned from API")
            
        except Exception as e:
            print(f"Error generating embedding: {e}")
            # Return zero vector as fallback
            return [0.0] * self.dimensions
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents using RETRIEVAL_DOCUMENT task type"""
        embeddings = []
        for text in texts:
            embedding = self._get_embedding(text, "RETRIEVAL_DOCUMENT")
            embeddings.append(embedding)
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed query using RETRIEVAL_QUERY task type"""
        return self._get_embedding(text, "RETRIEVAL_QUERY")

embeddings = GeminiEmbeddings(
    model_name=CONFIG["embedding_model"],
    dimensions=CONFIG["embedding_dimensions"]
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CONFIG["chunk_size"],
    chunk_overlap=CONFIG["chunk_overlap"],
    separators=["\n\n", "\n", ". ", " "]
)

os.makedirs(os.path.dirname(CONFIG["vector_db_path"]), exist_ok=True)
vectorstore = Chroma(
    persist_directory=CONFIG["vector_db_path"],
    embedding_function=embeddings
)

# Initialize Jinja2 template environment
jinja_env = Environment(loader=FileSystemLoader('templates'))

def classify_section(text: str) -> str:
    """Classify text into section type"""
    text_lower = text.lower()
    
    for section_type, keywords in section_keywords.items():
        if any(keyword in text_lower for keyword in keywords):
            return section_type
    
    return "general"

def clean_text(text: str) -> str:
    """Clean text by removing headers, footers, irrelevant content"""
    lines = text.split('\n')
    cleaned_lines = []
    
    # Get all skip patterns
    all_skip_patterns = []
    for category in cleanup_keywords.values():
        all_skip_patterns.extend(category)
    
    for line in lines:
        line = line.strip()
        if not line or len(line) < 10:
            continue
        if any(skip in line.lower() for skip in all_skip_patterns):
            continue
        cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)

def process_pdf(pdf_path: str) -> Dict[str, Any]:
    """Process a single PDF file"""
    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        filename = Path(pdf_path).stem
        processed_docs = []
        
        for i, doc in enumerate(documents):
            cleaned_content = clean_text(doc.page_content)
            if len(cleaned_content) < 100:
                continue
            
            doc.page_content = cleaned_content
            doc.metadata.update({
                "filename": filename,
                "page": i + 1,
                "section_type": classify_section(cleaned_content)
            })
            processed_docs.append(doc)
        
        chunks = text_splitter.split_documents(processed_docs)
        
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = f"{filename}_chunk_{i}"
        
        vectorstore.add_documents(chunks)
        
        return {
            "success": True,
            "filename": filename,
            "chunks_created": len(chunks),
            "pages_processed": len(processed_docs)
        }
    
    except Exception as e:
        return {"success": False, "error": str(e)}

def process_directory(directory_path: str) -> Dict[str, Any]:
    """Process all PDFs in a directory"""
    directory = Path(directory_path)
    pdf_files = list(directory.glob("*.pdf"))
    
    if not pdf_files:
        return {"success": False, "error": "No PDF files found"}
    
    results = {"processed_files": [], "total_chunks": 0, "errors": []}
    
    for pdf_file in pdf_files:
        result = process_pdf(str(pdf_file))
        if result["success"]:
            results["processed_files"].append(result)
            results["total_chunks"] += result["chunks_created"]
            print(f"✅ {result['filename']}: {result['chunks_created']} chunks")
        else:
            results["errors"].append(f"{pdf_file.name}: {result['error']}")
            print(f"❌ {pdf_file.name}: {result['error']}")
    
    return results

def search_documents(query: str, section_type: Optional[str] = None) -> List[Dict[str, Any]]:
    """Search documents using similarity"""
    try:
        filter_dict = {"section_type": section_type} if section_type else None
        results = vectorstore.similarity_search_with_score(
            query, k=CONFIG["top_k"], filter=filter_dict
        )
        
        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "similarity": 1 - score,
                "source": f"{doc.metadata.get('filename', 'Unknown')} (Page {doc.metadata.get('page', 'N/A')})"
            })
        
        return formatted_results
    
    except Exception as e:
        print(f"Search error: {e}")
        return []

def format_search_results(query: str, results: List[Dict[str, Any]]) -> str:
    """Format search results using Jinja2 template"""
    template = jinja_env.get_template('search_results.j2')
    return template.render(query=query, results=results)

def call_gemini(prompt: str) -> str:
    """Call Gemini API with environment variable for API key"""
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return "Error: GEMINI_API_KEY environment variable not set"
        
        payload = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }],
            "generationConfig": {
                "temperature": 0.1,
                "topP": 0.9,
                "maxOutputTokens": 2048
            }
        }

        headers = {
            "Content-Type": "application/json"
        }

        url_with_key = f"{CONFIG['gemini_url']}?key={api_key}"
        response = requests.post(url_with_key, json=payload, headers=headers, timeout=30)
        response.raise_for_status()

        response_data = response.json()
        if "candidates" in response_data and response_data["candidates"]:
            return response_data["candidates"][0]["content"]["parts"][0]["text"]
        else:
            return "No response generated"

    except requests.exceptions.RequestException as e:
        return f"Error connecting to Gemini: {e}"
    except Exception as e:
        return f"Error generating response: {e}"

def answer_question(question: str) -> Dict[str, Any]:
    """Answer a question using RAG with enhanced Jinja2 template"""
    search_results = search_documents(question)
    
    if not search_results:
        return {
            "question": question,
            "answer": "No relevant information found in the policies.",
            "sources": []
        }
    
    # Prepare template data
    template_data = {
        "question": question,
        "sources": search_results[:3]  # Top 3 most relevant
    }
    
    try:
        template = jinja_env.get_template('insurance_query.j2')
        prompt = template.render(**template_data)
    except Exception as e:
        raise Exception(f"Template error: {e}. Make sure templates/insurance_query.j2 exists and is valid.")
    
    answer = call_gemini(prompt)
    
    return {
        "question": question,
        "answer": answer,
        "sources": [result["source"] for result in search_results[:3]],
        "relevant_sections": search_results[:3]
    }

def get_stats() -> Dict[str, Any]:
    """Get system statistics"""
    try:
        total_docs = vectorstore._collection.count()
        
        return {
            "total_chunks": total_docs,
            "embedding_model": CONFIG["embedding_model"],
            "embedding_dimensions": CONFIG["embedding_dimensions"],
            "chunk_settings": {
                "size": CONFIG["chunk_size"],
                "overlap": CONFIG["chunk_overlap"]
            }
        }
    except Exception as e:
        return {"error": str(e)}
