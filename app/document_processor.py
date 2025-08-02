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
from .rag_core import classify_section, call_gemini
from .cache import question_cache
from config import CONFIG  
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from jinja2 import Environment, FileSystemLoader
import json

logger = logging.getLogger(__name__)
jinja_env = Environment(loader=FileSystemLoader('prompts'))

def load_common_words():
    with open('./config/common_words.json', 'r') as f:
        data = json.load(f)
        return set(data['common_words'])

common_words = load_common_words()

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
            if file_type == 'pdf':
                loader = PyPDFLoader(file_path)
            else:  # docx or doc
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

async def enhanced_search_for_question(question: str) -> List[Tuple[Any, float]]:
    try:
        vectorstore = init_vectorstore()
        
        search_results = vectorstore.similarity_search_with_score(question, k=CONFIG["top_k"])
        all_results = list(search_results)
        
        import re
        question_words = re.findall(r'\b[A-Za-z]{3,}\b', question.lower())
        important_words = [word for word in question_words if word not in common_words and len(word) > 3]
        
        # Remove overly specific policy name filtering
        if len(important_words) >= 2:
            # Try compound searches without policy-specific terms
            filtered_words = [w for w in important_words if w not in ['national', 'parivar', 'mediclaim', 'plus']]
            if len(filtered_words) >= 2:
                compound_terms = [" ".join(filtered_words[i:i+2]) for i in range(len(filtered_words)-1)]
                for compound in compound_terms[:3]:
                    compound_results = vectorstore.similarity_search_with_score(compound, k=3)
                    all_results.extend(compound_results)
        
        # Individual term search
        for term in important_words[:4]:
            term_results = vectorstore.similarity_search_with_score(term, k=2)
            all_results.extend(term_results)
        
        # Numerical search if question contains numbers
        numbers = re.findall(r'\d+(?:\.\d+)?', question)
        for num in numbers[:2]:
            num_results = vectorstore.similarity_search_with_score(num, k=2)
            all_results.extend(num_results)
        
        seen_content = set()
        unique_results = []
        for doc, score in all_results:
            content_hash = hash(doc.page_content[:150])
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_results.append((doc, score))
        
        unique_results.sort(key=lambda x: x[1])
        return unique_results[:8]
        
    except Exception as e:
        logger.error(f"Enhanced search error: {e}")
        # Fallback to basic search
        vectorstore = init_vectorstore()
        return vectorstore.similarity_search_with_score(question, k=CONFIG["top_k"])

async def answer_single_question(question: str) -> str:
    try:
        cached_answer = question_cache.get(question)
        if cached_answer:
            logger.info(f"💾 Cache hit for question: {question[:50]}...")
            return cached_answer
            
        logger.info(f"🔍 Starting search for question: {question[:50]}...")
        search_results = await enhanced_search_for_question(question)
        
        if not search_results:
            logger.warning(f"⚠️ No search results found for: {question[:50]}...")
            return "No relevant information found in the policy document."
        
        logger.info(f"📊 Found {len(search_results)} search results")
        
        # Log the retrieved chunks (truncated)
        logger.info("🔍 Retrieved chunks:")
        for i, (doc, score) in enumerate(search_results[:5]): 
            chunk_preview = doc.page_content.replace('\n', ' ')
            logger.info(f"  Chunk {i+1} (score: {score:.3f}): {chunk_preview}")
        
        formatted_results = []
        for doc, score in search_results:
            formatted_results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "similarity": 1 - score,
                "source": f"{doc.metadata.get('filename', 'Unknown')} (Page {doc.metadata.get('page', 'N/A')})"
            })
        
        from .answer_processor import resolve_conflicts, enhance_answer_completeness, extract_and_validate_numbers
        
        resolved_sources = resolve_conflicts(formatted_results, question)
        
     
        logger.info(f"🎯 Final {len(resolved_sources[:6])} sources sent to LLM:")
        for i, source in enumerate(resolved_sources[:6]):
            source_preview = source['content'].replace('\n', ' ')
            logger.info(f"  Source {i+1}: {source_preview}")
        
        template_data = {
            "question": question,
            "sources": resolved_sources[:8]
        }
        
        logger.info(f"🎨 Rendering template with {len(template_data['sources'])} sources")
        template = jinja_env.get_template('insurance_query.j2')
        prompt = template.render(**template_data)
        
        logger.info(f"📤 Sending prompt to Gemini (length: {len(prompt)} chars)")
        answer = call_gemini(prompt)
        
        if answer and not answer.startswith('Error'):
            full_context = ' '.join([s['content'] for s in resolved_sources[:3]])
            answer = extract_and_validate_numbers(full_context, answer)
            answer = enhance_answer_completeness(question, answer, resolved_sources[:3])
        
        logger.info(f"📥 Received answer (length: {len(answer)} chars): '{answer[:100]}{'...' if len(answer) > 100 else ''}'")
        
        if not answer or answer.strip() == "":
            logger.error(f"❌ EMPTY ANSWER DETECTED for question: {question}")
            return "Error: Received empty response from AI model"
        
        question_cache.set(question, answer)
        return answer
        
    except Exception as e:
        logger.error(f"Error answering question '{question}': {e}")
        return f"Error processing question: {str(e)}"

async def answer_questions(questions: List[str]) -> List[str]:
    if len(questions) == 1:
        answer = await answer_single_question(questions[0])
        return [answer]
    
    import concurrent.futures
    import time
    
    def sync_answer_question(question_data) -> str:
        question = question_data
        import asyncio
        loop = None
        try:
                
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(answer_single_question(question))
        finally:
            if loop:
                loop.close()
    
    max_workers = min(len(questions), 43)  
    logger.info(f"🚀 Processing {len(questions)} questions with {max_workers} parallel workers")
    
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Add index to each question for staggering
        question_data = [question for question in questions]
        futures = [executor.submit(sync_answer_question, data) for data in question_data]
        results = [future.result() for future in futures]
    
    total_time = time.time() - start_time
    logger.info(f"✅ Completed {len(questions)} questions in {total_time:.2f}s (avg: {total_time/len(questions):.2f}s per question)")
    
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

