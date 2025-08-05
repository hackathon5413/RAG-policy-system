import os
import tempfile
import asyncio
import hashlib
import json
from pathlib import Path
from typing import List, Dict, Any
import aiofiles
import httpx
import logging
import concurrent.futures

from .vector_store import text_splitter, init_vectorstore
from .rag_core import classify_section, call_gemini
from .cache import question_cache
from config import config
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from jinja2 import Environment, FileSystemLoader

logger = logging.getLogger(__name__)
jinja_env = Environment(loader=FileSystemLoader('prompts'))

URL_CACHE_FILE = "./data/url_cache.json"
os.makedirs(os.path.dirname(URL_CACHE_FILE), exist_ok=True)

def load_url_cache() -> Dict[str, bool]:
    try:
        with open(URL_CACHE_FILE, 'r') as f:
            return json.load(f)
    except Exception:
        return {}

def save_url_cache(cache: Dict[str, bool]):
    with open(URL_CACHE_FILE, 'w') as f:
        json.dump(cache, f)

def get_file_type(url: str) -> str:
    url_lower = url.lower()
    if url_lower.endswith('.pdf') or 'pdf' in url_lower:
        return 'pdf'
    elif url_lower.endswith('.docx') or 'docx' in url_lower:
        return 'docx'
    elif url_lower.endswith('.doc') or 'doc' in url_lower:
        return 'doc'
    else:
        return 'pdf'

def get_url_hash(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()

async def download_document_from_url(url: str, timeout: int = 60) -> tuple[str, str]:
    try:
        file_type = get_file_type(url)
        
        async with httpx.AsyncClient(timeout=timeout) as client:
            logger.info(f"Downloading {file_type.upper()} from: {url}...")
            
            response = await client.get(url)
            response.raise_for_status()
            
            content_type = response.headers.get('content-type', '')
            if file_type == 'pdf' and 'pdf' not in content_type.lower() and not url.lower().endswith('.pdf'):
                logger.warning(f"Content type may not be PDF: {content_type}")
            elif file_type in ['docx', 'doc'] and 'document' not in content_type.lower() and not any(ext in url.lower() for ext in ['.docx', '.doc']):
                logger.warning(f"Content type may not be DOCX: {content_type}")
            
            temp_dir = tempfile.mkdtemp()
            if file_type == 'pdf':
                temp_file_path = os.path.join(temp_dir, "downloaded_document.pdf")
            else:
                temp_file_path = os.path.join(temp_dir, "downloaded_document.docx")
            
            async with aiofiles.open(temp_file_path, 'wb') as f:
                await f.write(response.content)
            
            file_size = len(response.content)
            logger.info(f"Downloaded {file_type.upper()}: {file_size} bytes to {temp_file_path}")
            
            return temp_file_path, file_type
            
    except httpx.TimeoutException:
        raise Exception(f"Timeout downloading document from URL: {url}")
    except httpx.HTTPStatusError as e:
        raise Exception(f"HTTP error downloading document: {e.response.status_code}")
    except Exception as e:
        raise Exception(f"Error downloading document: {str(e)}")

async def process_local_document(file_path: str, file_type: str, url_hash: str) -> Dict[str, Any]:
    try:
        loop = asyncio.get_event_loop()
        
        def sync_process_document():
            if file_type == 'pdf':
                loader = PyPDFLoader(file_path)
            else:
                loader = Docx2txtLoader(file_path)
            
            documents = loader.load()
            
            filename = Path(file_path).stem
            processed_docs = []
            
            for i, doc in enumerate(documents):
                content = doc.page_content
                if len(content) < 100:
                    continue
                
                doc.page_content = content
                doc.metadata.update({
                    "filename": filename,
                    "page": i + 1 if file_type == 'pdf' else 1,
                    "section_type": classify_section(content),
                    "url_hash": url_hash,
                    "file_type": file_type
                })
                processed_docs.append(doc)
            
            chunks = text_splitter.split_documents(processed_docs)
            
            for i, chunk in enumerate(chunks):
                chunk.metadata["chunk_id"] = f"{url_hash}_chunk_{i}"
                chunk.metadata["url_hash"] = url_hash
                chunk.metadata["file_type"] = file_type
            
            return {
                "filename": filename,
                "chunks": chunks,
                "pages_processed": len(processed_docs)
            }
        
        result = await loop.run_in_executor(None, sync_process_document)
        chunks = result["chunks"]
        logger.info(f"Adding {len(chunks)} chunks from {file_type.upper()} to vector store")
        
        vectorstore = init_vectorstore()
        vectorstore.add_documents(chunks)
        logger.info(f"Successfully added all {len(chunks)} chunks using parallel batching")
        
        cache = load_url_cache()
        cache[url_hash] = True
        save_url_cache(cache)
        
        return {
            "success": True,
            "filename": result["filename"],
            "chunks_created": len(chunks),
            "pages_processed": result["pages_processed"],
            "file_type": file_type
        }
    
    except Exception as e:
        logger.error(f"Error processing {file_type.upper()}: {e}")
        return {"success": False, "error": str(e)}

async def process_document_from_url(url: str) -> Dict[str, Any]:
    url_hash = get_url_hash(url)
    cache = load_url_cache()
    
    if url_hash in cache:
        logger.info(f"URL {url[:50]}... already processed, skipping")
        return {
            "success": True,
            "source_url": url,
            "chunks_created": 0,
            "pages_processed": 0,
            "cached": True
        }
    
    temp_file_path = None
    try:
        temp_file_path, file_type = await download_document_from_url(url)
        result = await process_local_document(temp_file_path, file_type, url_hash)
        
        if result["success"]:
            return {
                "success": True,
                "source_url": url,
                "chunks_created": result["chunks_created"],
                "pages_processed": result["pages_processed"],
                "cached": False
            }
        else:
            return result
        
    except Exception as e:
        logger.error(f"Error processing document from URL: {e}")
        return {
            "success": False,
            "error": str(e),
            "source_url": url
        }
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                temp_dir = os.path.dirname(temp_file_path)
                if os.path.exists(temp_dir) and not os.listdir(temp_dir):
                    os.rmdir(temp_dir)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp file: {e}")

async def answer_single_question(question: str) -> str:
    try:
        cached_answer = question_cache.get(question)
        if cached_answer:
            return cached_answer
            
        vectorstore = init_vectorstore()
        search_results = vectorstore.similarity_search_with_score(question, k=config.top_k)
        
        if not search_results:
            return "No relevant information found in the document."
        
        formatted_results = [{
            "content": doc.page_content,
            "metadata": doc.metadata,
            "source": f"{doc.metadata.get('filename', 'Unknown')} (Page {doc.metadata.get('page', 'N/A')})"
        } for doc, score in search_results]
        
        template = jinja_env.get_template('insurance_query.j2')
        prompt = template.render(question=question, sources=formatted_results)
        
        answer = call_gemini(prompt)
        
        if not answer or answer.strip() == "":
            return "Error: Received empty response from AI model"
        
        question_cache.set(question, answer)
        return answer
        
    except Exception as e:
        logger.error(f"Error answering question '{question}': {e}")
        return f"Error processing question: {str(e)}"

async def answer_questions(questions: List[str]) -> List[str]:
    if len(questions) == 1:
        return [await answer_single_question(questions[0])]
    
    def sync_answer_question(question: str) -> str:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(answer_single_question(question))
        finally:
            loop.close()
    
    max_workers = min(len(questions), 43)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(sync_answer_question, q) for q in questions]
        return [future.result() for future in futures]

async def process_document_and_answer(document_url: str, questions: List[str]) -> Dict[str, Any]:
    try:
        processing_result = await process_document_from_url(document_url)
        
        if not processing_result["success"]:
            return {
                "success": False,
                "error": processing_result["error"],
                "answers": [f"Error processing document: {processing_result['error']}" for _ in questions]
            }
        
        answers = await answer_questions(questions)
        
        try:
            from .vector_store import get_embeddings
            embeddings_instance = get_embeddings()
            if hasattr(embeddings_instance, 'save_cache'):
                embeddings_instance.save_cache()
        except Exception as e:
            logger.warning(f"Could not save embedding cache: {e}")
        
        return {
            "success": True,
            "answers": answers,
            "document_info": processing_result
        }
        
    except Exception as e:
        logger.error(f"Error in complete workflow: {e}")
        return {
            "success": False,
            "error": str(e),
            "answers": [f"Error: {str(e)}" for _ in questions]
        }
