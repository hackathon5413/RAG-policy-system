import os
import tempfile
import asyncio
import hashlib
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import aiofiles
import httpx
import logging

from .vector_store import text_splitter, init_vectorstore
from .rag_core import classify_section, clean_text, call_gemini
from config import CONFIG  
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
    except:
        return {}

def save_url_cache(cache: Dict[str, bool]):
    with open(URL_CACHE_FILE, 'w') as f:
        json.dump(cache, f)

def get_file_type(url: str) -> str:
    """Determine file type from URL"""
    url_lower = url.lower()
    if url_lower.endswith('.pdf') or 'pdf' in url_lower:
        return 'pdf'
    elif url_lower.endswith('.docx') or 'docx' in url_lower:
        return 'docx'
    elif url_lower.endswith('.doc') or 'doc' in url_lower:
        return 'doc'  # Treat as docx
    else:
        # Default to PDF if unclear
        return 'pdf'

def get_url_hash(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()

async def download_document_from_url(url: str, timeout: int = 60) -> tuple[str, str]:
    """Download document from blob URL and return local temporary file path and file type"""
    try:
        file_type = get_file_type(url)
        
        async with httpx.AsyncClient(timeout=timeout) as client:
            logger.info(f"Downloading {file_type.upper()} from: {url[:80]}...")
            
            response = await client.get(url)
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get('content-type', '')
            if file_type == 'pdf' and 'pdf' not in content_type.lower() and not url.lower().endswith('.pdf'):
                logger.warning(f"Content type may not be PDF: {content_type}")
            elif file_type in ['docx', 'doc'] and 'document' not in content_type.lower() and not any(ext in url.lower() for ext in ['.docx', '.doc']):
                logger.warning(f"Content type may not be DOCX: {content_type}")
            
            # Create temporary file with appropriate extension
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
    """Process local document (PDF or DOCX) and return structured data"""
    try:
        loop = asyncio.get_event_loop()
        
        def sync_process_document():
            # Choose appropriate loader based on file type
            if file_type == 'pdf':
                loader = PyPDFLoader(file_path)
            else:  # docx or doc
                loader = Docx2txtLoader(file_path)
            
            documents = loader.load()
            
            filename = Path(file_path).stem
            processed_docs = []
            
            for i, doc in enumerate(documents):
                cleaned_content = clean_text(doc.page_content)
                if len(cleaned_content) < 100:
                    continue
                
                doc.page_content = cleaned_content
                doc.metadata.update({
                    "filename": filename,
                    "page": i + 1 if file_type == 'pdf' else 1,  # DOCX is usually single "page"
                    "section_type": classify_section(cleaned_content),
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

async def enhanced_search_for_question(question: str) -> List[Tuple[Any, float]]:
    try:
        vectorstore = init_vectorstore()
        search_results = vectorstore.similarity_search_with_score(question, k=CONFIG["top_k"])
        all_results = list(search_results)
        
        key_terms = []
        q_lower = question.lower()
        
        if "grace period" in q_lower:
            key_terms = ["grace period", "premium payment"]
        elif "pre-existing" in q_lower:
            key_terms = ["pre-existing", "PED"]
        elif "maternity" in q_lower:
            key_terms = ["maternity", "pregnancy"]
        elif "cataract" in q_lower:
            key_terms = ["cataract"]
        elif "claim discount" in q_lower or "NCD" in question:
            key_terms = ["claim discount", "NCD"]
        elif "preventive" in q_lower:
            key_terms = ["preventive"]
        elif "hospital" in q_lower:
            key_terms = ["hospital definition"]
        elif "AYUSH" in question:
            key_terms = ["AYUSH"]
        elif "room rent" in q_lower or "ICU" in question:
            key_terms = ["room rent"]
        
        for term in key_terms[:1]:
            term_results = vectorstore.similarity_search_with_score(term, k=3)
            all_results.extend(term_results)
        
        seen_content = set()
        unique_results = []
        for doc, score in all_results:
            content_hash = hash(doc.page_content[:100])
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_results.append((doc, score))
        
        unique_results.sort(key=lambda x: x[1])
        return unique_results[:6]
        
    except Exception as e:
        logger.error(f"Enhanced search error: {e}")
        vectorstore = init_vectorstore()
        return vectorstore.similarity_search_with_score(question, k=CONFIG["top_k"])

async def answer_single_question(question: str) -> str:
    try:
        search_results = await enhanced_search_for_question(question)
        
        if not search_results:
            return "No relevant information found in the policy document."
        
        formatted_results = []
        for doc, score in search_results:
            formatted_results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "similarity": 1 - score,
                "source": f"{doc.metadata.get('filename', 'Unknown')} (Page {doc.metadata.get('page', 'N/A')})"
            })
        
        template_data = {
            "question": question,
            "sources": formatted_results[:4]
        }
        
        template = jinja_env.get_template('insurance_query.j2')
        prompt = template.render(**template_data)
        
        answer = call_gemini(prompt)
        return answer
        
    except Exception as e:
        logger.error(f"Error answering question '{question}': {e}")
        return f"Error processing question: {str(e)}"

async def answer_questions(questions: List[str]) -> List[str]:
    if len(questions) == 1:
        answer = await answer_single_question(questions[0])
        return [answer]
    
    import concurrent.futures
    
    def sync_answer_question(question: str) -> str:
        import asyncio
        loop = None
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(answer_single_question(question))
        finally:
            if loop:
                loop.close()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(questions), 25)) as executor:
        futures = [executor.submit(sync_answer_question, question) for question in questions]
        results = [future.result() for future in futures]
    
    return results

async def process_document_and_answer(document_url: str, questions: List[str]) -> Dict[str, Any]:
    try:
        logger.info(f"Starting document processing for {len(questions)} questions")
        processing_result = await process_document_from_url(document_url)
        
        if not processing_result["success"]:
            return {
                "success": False,
                "error": processing_result["error"],
                "answers": [f"Error processing document: {processing_result['error']}" for _ in questions]
            }
        
        if processing_result.get("cached", False):
            logger.info("Document was cached, proceeding directly to questions")
        else:
            logger.info(f"Document processed successfully: {processing_result['chunks_created']} chunks")
        
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

# Helper function for cleanup
def cleanup_temp_files():
    """Clean up any remaining temporary files"""
    try:
        temp_dir = tempfile.gettempdir()
        for item in os.listdir(temp_dir):
            if item.startswith('tmp') and ('policy' in item or 'document' in item):
                full_path = os.path.join(temp_dir, item)
                if os.path.isfile(full_path):
                    os.remove(full_path)
    except Exception as e:
        logger.warning(f"Cleanup warning: {e}")
