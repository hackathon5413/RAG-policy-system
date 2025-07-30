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

from .vector_store import text_splitter, init_openai_vectorstore, get_openai_embeddings
from .rag_core import classify_section, call_gemini
from .cache import question_cache
from config import CONFIG
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from jinja2 import Environment, FileSystemLoader

logger = logging.getLogger(__name__)
jinja_env = Environment(loader=FileSystemLoader('prompts'))

def load_common_words():
    with open('./config/common_words.json', 'r') as f:
        data = json.load(f)
        return set(data['common_words'])

common_words = load_common_words()

URL_CACHE_FILE_V2 = "./data/url_cache_v2.json"
os.makedirs(os.path.dirname(URL_CACHE_FILE_V2), exist_ok=True)

def load_url_cache_v2() -> Dict[str, bool]:
    try:
        with open(URL_CACHE_FILE_V2, 'r') as f:
            return json.load(f)
    except Exception:
        return {}

def save_url_cache_v2(cache: Dict[str, bool]):
    with open(URL_CACHE_FILE_V2, 'w') as f:
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

async def download_document_from_url_v2(url: str, timeout: int = 60) -> tuple[str, str]:
    try:
        file_type = get_file_type(url)
        
        async with httpx.AsyncClient(timeout=timeout) as client:
            logger.info(f"[V2] Downloading {file_type.upper()} from: {url[:80]}...")
            
            response = await client.get(url)
            response.raise_for_status()
            
            temp_dir = tempfile.mkdtemp()
            if file_type == 'pdf':
                temp_file_path = os.path.join(temp_dir, "downloaded_document.pdf")
            else:
                temp_file_path = os.path.join(temp_dir, "downloaded_document.docx")
            
            async with aiofiles.open(temp_file_path, 'wb') as f:
                await f.write(response.content)
            
            file_size = len(response.content)
            logger.info(f"[V2] Downloaded {file_type.upper()}: {file_size} bytes")
            
            return temp_file_path, file_type
            
    except Exception as e:
        raise Exception(f"Error downloading document: {str(e)}")

async def process_local_document_v2(file_path: str, file_type: str, url_hash: str) -> Dict[str, Any]:
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
                    "file_type": file_type,
                    "embedding_provider": "openai"
                })
                processed_docs.append(doc)
            
            chunks = text_splitter.split_documents(processed_docs)
            
            for i, chunk in enumerate(chunks):
                chunk.metadata["chunk_id"] = f"{url_hash}_chunk_{i}_v2"
                chunk.metadata["url_hash"] = url_hash
                chunk.metadata["file_type"] = file_type
                chunk.metadata["embedding_provider"] = "openai"
            
            return {
                "filename": filename,
                "chunks": chunks,
                "pages_processed": len(processed_docs)
            }
        
        result = await loop.run_in_executor(None, sync_process_document)
        chunks = result["chunks"]
        logger.info(f"[V2] Adding {len(chunks)} chunks to OpenAI vector store")
        
        vectorstore = init_openai_vectorstore()
        vectorstore.add_documents(chunks)
        logger.info(f"[V2] Successfully added {len(chunks)} chunks using OpenAI embeddings")
        
        cache = load_url_cache_v2()
        cache[url_hash] = True
        save_url_cache_v2(cache)
        
        return {
            "success": True,
            "filename": result["filename"],
            "chunks_created": len(chunks),
            "pages_processed": result["pages_processed"],
            "file_type": file_type,
            "embedding_provider": "openai"
        }
    
    except Exception as e:
        logger.error(f"[V2] Error processing {file_type.upper()}: {e}")
        return {"success": False, "error": str(e)}

async def process_document_from_url_v2(url: str) -> Dict[str, Any]:
    url_hash = get_url_hash(url)
    cache = load_url_cache_v2()
    
    if url_hash in cache:
        logger.info(f"[V2] URL already processed, skipping")
        return {
            "success": True,
            "source_url": url,
            "chunks_created": 0,
            "pages_processed": 0,
            "cached": True,
            "embedding_provider": "openai"
        }
    
    temp_file_path = None
    try:
        temp_file_path, file_type = await download_document_from_url_v2(url)
        result = await process_local_document_v2(temp_file_path, file_type, url_hash)
        
        if result["success"]:
            return {
                "success": True,
                "source_url": url,
                "chunks_created": result["chunks_created"],
                "pages_processed": result["pages_processed"],
                "cached": False,
                "embedding_provider": "openai"
            }
        else:
            return result
        
    except Exception as e:
        logger.error(f"[V2] Error processing document from URL: {e}")
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

async def enhanced_search_for_question_v2(question: str) -> List[Tuple[Any, float]]:
    try:
        vectorstore = init_openai_vectorstore()
        
        search_results = vectorstore.similarity_search_with_score(question, k=CONFIG["top_k"])
        all_results = list(search_results)
        
        import re
        question_words = re.findall(r'\b[A-Za-z]{3,}\b', question.lower())
        
        important_words = []
        for word in question_words:
            if word not in common_words and len(word) > 3:
                important_words.append(word)
        
        for term in important_words[:2]:
            term_results = vectorstore.similarity_search_with_score(term, k=2)
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
        logger.error(f"[V2] Enhanced search error: {e}")
        vectorstore = init_openai_vectorstore()
        return vectorstore.similarity_search_with_score(question, k=CONFIG["top_k"])

async def answer_single_question_v2(question: str) -> str:
    try:
        cached_answer = question_cache.get(f"v2_{question}")
        if cached_answer:
            logger.info(f"[V2] Cache hit for question")
            return cached_answer
            
        logger.info(f"[V2] Starting OpenAI search for question")
        search_results = await enhanced_search_for_question_v2(question)
        
        if not search_results:
            logger.warning(f"[V2] No search results found")
            return "No relevant information found in the policy document."
        
        logger.info(f"[V2] Found {len(search_results)} search results")
        
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
        
        logger.info(f"[V2] Sending prompt to Gemini")
        answer = call_gemini(prompt)
        
        if not answer or answer.strip() == "":
            logger.error(f"[V2] Empty answer detected")
            return "Error: Received empty response from AI model"
        
        question_cache.set(f"v2_{question}", answer)
        return answer
        
    except Exception as e:
        logger.error(f"[V2] Error answering question: {e}")
        return f"Error processing question: {str(e)}"

async def answer_questions_v2(questions: List[str]) -> List[str]:
    if len(questions) == 1:
        answer = await answer_single_question_v2(questions[0])
        return [answer]
    
    import concurrent.futures
    import time
    
    def sync_answer_question(question) -> str:
        import asyncio
        loop = None
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(answer_single_question_v2(question))
        finally:
            if loop:
                loop.close()
    
    max_workers = min(len(questions), 10)
    logger.info(f"[V2] Processing {len(questions)} questions with {max_workers} workers")
    
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(sync_answer_question, question) for question in questions]
        results = [future.result() for future in futures]
    
    total_time = time.time() - start_time
    logger.info(f"[V2] Completed {len(questions)} questions in {total_time:.2f}s")
    
    return results

async def process_document_and_answer_v2(document_url: str, questions: List[str]) -> Dict[str, Any]:
    try:
        logger.info(f"[V2] Starting OpenAI document processing for {len(questions)} questions")
        processing_result = await process_document_from_url_v2(document_url)
        
        if not processing_result["success"]:
            return {
                "success": False,
                "error": processing_result["error"],
                "answers": [f"Error processing document: {processing_result['error']}" for _ in questions]
            }
        
        if processing_result.get("cached", False):
            logger.info("[V2] Document was cached")
        else:
            logger.info(f"[V2] Document processed: {processing_result['chunks_created']} chunks")
        
        answers = await answer_questions_v2(questions)
        
        try:
            embeddings_instance = get_openai_embeddings()
            if hasattr(embeddings_instance, 'save_cache'):
                embeddings_instance.save_cache()
        except Exception as e:
            logger.warning(f"Could not save OpenAI embedding cache: {e}")
        
        return {
            "success": True,
            "answers": answers,
            "document_info": processing_result,
            "embedding_provider": "openai"
        }
        
    except Exception as e:
        logger.error(f"[V2] Error in complete workflow: {e}")
        return {
            "success": False,
            "error": str(e),
            "answers": [f"Error: {str(e)}" for _ in questions]
        }
