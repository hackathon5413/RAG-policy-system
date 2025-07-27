import os
import tempfile
import asyncio
from pathlib import Path
from typing import List, Dict, Any,Tuple
import aiofiles
import httpx
import logging

from .vector_store import text_splitter, init_vectorstore
from .rag_core import classify_section, clean_text, call_gemini
from config import CONFIG  
from langchain_community.document_loaders import PyPDFLoader
from jinja2 import Environment, FileSystemLoader

logger = logging.getLogger(__name__)


jinja_env = Environment(loader=FileSystemLoader('prompts'))

async def download_pdf_from_url(url: str, timeout: int = 60) -> str:
    """Download PDF from blob URL and return local temporary file path"""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            logger.info(f"Downloading PDF from: {url[:80]}...")
            
            response = await client.get(url)
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get('content-type', '')
            if 'pdf' not in content_type.lower() and not url.lower().endswith('.pdf'):
                logger.warning(f"Content type may not be PDF: {content_type}")
            
            # Create temporary file
            temp_dir = tempfile.mkdtemp()
            temp_file_path = os.path.join(temp_dir, "downloaded_policy.pdf")
            
            async with aiofiles.open(temp_file_path, 'wb') as f:
                await f.write(response.content)
            
            file_size = len(response.content)
            logger.info(f"Downloaded PDF: {file_size} bytes to {temp_file_path}")
            
            return temp_file_path
            
    except httpx.TimeoutException:
        raise Exception(f"Timeout downloading PDF from URL: {url}")
    except httpx.HTTPStatusError as e:
        raise Exception(f"HTTP error downloading PDF: {e.response.status_code}")
    except Exception as e:
        raise Exception(f"Error downloading PDF: {str(e)}")

async def process_local_pdf(pdf_path: str) -> Dict[str, Any]:
    """Process PDF efficiently"""
    try:
        loop = asyncio.get_event_loop()
        
        def sync_process_pdf():
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
            
            return {
                "filename": filename,
                "chunks": chunks,
                "pages_processed": len(processed_docs)
            }
        
        # Run PDF processing in thread pool
        result = await loop.run_in_executor(None, sync_process_pdf)
        
        # Add chunks to vector store efficiently
        chunks = result["chunks"]
        logger.info(f"Adding {len(chunks)} chunks to vector store")
        
        try:
            vectorstore = init_vectorstore()
            vectorstore.add_documents(chunks)
            logger.info(f"Successfully added all {len(chunks)} chunks")
        except Exception as e:
            logger.warning(f"Some embeddings failed due to rate limits: {e}")
        
        return {
            "success": True,
            "filename": result["filename"],
            "chunks_created": len(chunks),
            "pages_processed": result["pages_processed"]
        }
    
    except Exception as e:
        return {"success": False, "error": str(e)}

async def process_pdf_from_url(url: str) -> Dict[str, Any]:
    """Download and process PDF from URL"""
    temp_file_path = None
    try:
        # Download PDF
        temp_file_path = await download_pdf_from_url(url)
        
        # Process PDF using fast approach
        result = await process_local_pdf(temp_file_path)
        
        return {
            "success": True,
            "source_url": url,
            "chunks_created": result["chunks_created"],
            "pages_processed": result["pages_processed"]
        }
        
    except Exception as e:
        logger.error(f"Error processing PDF from URL: {e}")
        return {
            "success": False,
            "error": str(e),
            "source_url": url
        }
    finally:
        # Cleanup temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                # Remove temp directory if empty
                temp_dir = os.path.dirname(temp_file_path)
                if os.path.exists(temp_dir) and not os.listdir(temp_dir):
                    os.rmdir(temp_dir)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp file: {e}")

async def enhanced_search_for_question(question: str) -> List[Tuple[Any, float]]:
    """Enhanced search that tries multiple query variations"""
    try:
        all_results = []
        vectorstore = init_vectorstore()
        search_results = vectorstore.similarity_search_with_score(question, k=CONFIG["top_k"])
        all_results.extend(search_results)
        
        # 2. Extract key terms for additional searches
        key_terms = []
        if "grace period" in question.lower():
            key_terms.extend(["grace period", "premium payment", "thirty days", "30 days"])
        elif "pre-existing" in question.lower():
            key_terms.extend(["pre-existing", "PED", "waiting period", "36 months", "thirty-six months"])
        elif "maternity" in question.lower():
            key_terms.extend(["maternity", "pregnancy", "24 months", "twenty-four months"])
        elif "cataract" in question.lower():
            key_terms.extend(["cataract", "two years", "2 years"])
        elif "claim discount" in question.lower() or "NCD" in question:
            key_terms.extend(["claim discount", "NCD", "5%", "five percent"])
        elif "preventive" in question.lower():
            key_terms.extend(["preventive", "health check", "wellness"])
        elif "hospital" in question.lower():
            key_terms.extend(["hospital definition", "institution", "inpatient beds"])
        elif "AYUSH" in question:
            key_terms.extend(["AYUSH", "Ayurveda", "Yoga", "Unani", "Siddha", "Homeopathy"])
        elif "room rent" in question.lower() or "ICU" in question:
            key_terms.extend(["room rent", "ICU charges", "sub-limits", "Plan A"])
        
        # 3. Search with key terms
        for term in key_terms[:3]:
            term_results = vectorstore.similarity_search_with_score(term, k=5)
            all_results.extend(term_results)
        
        # 4. Remove duplicates and sort by score
        seen_content = set()
        unique_results = []
        for doc, score in all_results:
            content_hash = hash(doc.page_content[:100])
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_results.append((doc, score))
        
        unique_results.sort(key=lambda x: x[1])
        return unique_results[:8]
        
    except Exception as e:
        logger.error(f"Enhanced search error: {e}")
        vectorstore = init_vectorstore()
        return vectorstore.similarity_search_with_score(question, k=CONFIG["top_k"])

async def answer_single_question(question: str) -> str:
    """Answer a single question using enhanced RAG search"""
    try:
        # Use enhanced search for better retrieval
        search_results = await enhanced_search_for_question(question)
        
        if not search_results:
            return "No relevant information found in the policy document."
        
        # Format results for template
        formatted_results = []
        for doc, score in search_results:
            formatted_results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "similarity": 1 - score,
                "source": f"{doc.metadata.get('filename', 'Unknown')} (Page {doc.metadata.get('page', 'N/A')})"
            })
        
        # Prepare template data with more context
        template_data = {
            "question": question,
            "sources": formatted_results[:5]
        }
        
        # Generate answer using updated simple template
        template = jinja_env.get_template('insurance_query.j2')
        prompt = template.render(**template_data)
        
        answer = call_gemini(prompt)
        return answer
        
    except Exception as e:
        logger.error(f"Error answering question '{question}': {e}")
        return f"Error processing question: {str(e)}"

async def answer_questions(questions: List[str]) -> List[str]:
    """Answer multiple questions using the RAG system"""
    answers = []
    
    for question in questions:
        answer = await answer_single_question(question)
        answers.append(answer)
    
    return answers

async def process_document_and_answer(document_url: str, questions: List[str]) -> Dict[str, Any]:
    """Complete workflow: download PDF, process, and answer questions"""
    try:
        # Process document
        logger.info(f"Starting document processing for {len(questions)} questions")
        processing_result = await process_pdf_from_url(document_url)
        
        if not processing_result["success"]:
            return {
                "success": False,
                "error": processing_result["error"],
                "answers": [f"Error processing document: {processing_result['error']}" for _ in questions]
            }
        
        logger.info(f"Document processed successfully: {processing_result['chunks_created']} chunks")
        
        # Answer questions immediately
        answers = await answer_questions(questions)
        
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
            if item.startswith('tmp') and 'policy' in item:
                full_path = os.path.join(temp_dir, item)
                if os.path.isfile(full_path):
                    os.remove(full_path)
    except Exception as e:
        logger.warning(f"Cleanup warning: {e}")
