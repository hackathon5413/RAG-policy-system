import os
import tempfile
import asyncio
import hashlib
import json
import zipfile
import shutil
from pathlib import Path
from typing import List, Dict, Any, Tuple
import aiofiles
import httpx
import logging
import concurrent.futures
from rank_bm25 import BM25Okapi
import re

from .vector_store import text_splitter, init_vectorstore
from .rag_core import classify_section, call_gemini
from .cache import question_cache
from config import config
from langchain_community.document_loaders import (
    PyPDFLoader, 
    Docx2txtLoader,
    UnstructuredExcelLoader,
    UnstructuredPowerPointLoader,
    UnstructuredImageLoader
)
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

def get_file_type_from_url(url: str) -> str:
    """Get file type from URL - first attempt"""
    url_lower = url.lower()
    if '.pdf' in url_lower:
        return 'pdf'
    elif '.docx' in url_lower:
        return 'docx' 
    elif '.doc' in url_lower:
        return 'doc'
    elif '.xlsx' in url_lower:
        return 'xlsx'
    elif '.xls' in url_lower:
        return 'xls'
    elif '.pptx' in url_lower:
        return 'pptx'
    elif '.ppt' in url_lower:
        return 'ppt'
    elif '.png' in url_lower:
        return 'png'
    elif '.jpg' in url_lower or '.jpeg' in url_lower:
        return 'jpeg'
    elif '.zip' in url_lower:
        return 'zip'
    return 'unknown'

def get_file_type_from_content_type(content_type: str) -> str:
    """Get file type from HTTP content-type header"""
    content_type = content_type.lower()
    if 'pdf' in content_type:
        return 'pdf'
    elif 'wordprocessingml.document' in content_type:
        return 'docx'
    elif 'spreadsheetml.sheet' in content_type:
        return 'xlsx'
    elif 'presentationml.presentation' in content_type:
        return 'pptx'
    elif 'image/png' in content_type:
        return 'png'
    elif 'image/jpeg' in content_type:
        return 'jpeg'
    elif 'application/zip' in content_type:
        return 'zip'
    return 'unknown'

def get_file_type_from_signature(file_path: str) -> str:
    """Get file type from file signature (magic numbers)"""
    try:
        with open(file_path, 'rb') as f:
            header = f.read(16)
            
        if header.startswith(b'%PDF'):
            return 'pdf'
        elif header.startswith(b'PK\x03\x04'):
            # ZIP-based formats (Office docs, ZIP files)
            try:
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    filenames = zip_ref.namelist()
                    if any('word/' in f for f in filenames):
                        return 'docx'
                    elif any('xl/' in f for f in filenames):
                        return 'xlsx'
                    elif any('ppt/' in f for f in filenames):
                        return 'pptx'
                    else:
                        return 'zip'
            except Exception:
                return 'zip'
        elif header.startswith(b'\x89PNG\r\n\x1a\n'):
            return 'png'
        elif header.startswith(b'\xff\xd8\xff'):
            return 'jpeg'
            
    except Exception as e:
        logger.warning(f"Could not read file signature: {e}")
    
    return 'unknown'

def determine_file_type(url: str, content_type: str, file_path: str) -> str:
    """Determine file type using multiple methods"""
    # Try URL first
    file_type = get_file_type_from_url(url)
    if file_type != 'unknown':
        return file_type
    
    # Try content-type header
    file_type = get_file_type_from_content_type(content_type)
    if file_type != 'unknown':
        return file_type
    
    # Try file signature as last resort
    file_type = get_file_type_from_signature(file_path)
    if file_type != 'unknown':
        return file_type
    
    # Default to PDF if all else fails
    logger.warning(f"Could not determine file type for {url}, defaulting to PDF")
    return 'pdf'

def get_url_hash(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()

def process_zip_file(zip_path: str, url_hash: str, depth: int = 0, total_extracted_size: int = 0, max_extracted_size: int = 10 * 1024 * 1024) -> Dict[str, Any]:
    """Process ZIP file with strict ZIP bomb protection"""
    if depth > 3: 
        logger.error(f"ZIP nesting depth ({depth}) exceeds safe limit (3) - ZIP bomb detected")
        raise Exception(f"ZIP bomb detected: nesting depth {depth} exceeds safe limit of 3")
    
    if total_extracted_size > max_extracted_size:
        logger.error(f"Total extracted size ({total_extracted_size:,} bytes) exceeds safe limit ({max_extracted_size:,} bytes) - ZIP bomb detected")
        raise Exception(f"ZIP bomb detected: extracted size {total_extracted_size:,} exceeds limit of {max_extracted_size:,} bytes")
    
    try:
        extract_dir = tempfile.mkdtemp()
        processed_docs = []
        total_files = 0
        supported_files = 0
        
        logger.info(f"Processing ZIP at depth {depth}: {os.path.basename(zip_path)}")
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            logger.info(f"ZIP contains {len(file_list)} items: {file_list}")
            
            # Show more details about each item
            for item in file_list:
                info = zip_ref.getinfo(item)
                logger.info(f"  {item}: {info.file_size} bytes, {'folder' if item.endswith('/') else 'file'}")
            
            zip_ref.extractall(extract_dir)
            
        # Process each extracted file
        for root, dirs, files in os.walk(extract_dir):
            for file in files:
                file_path = os.path.join(root, file)
                file_lower = file.lower()
                total_files += 1
                
                logger.info(f"Processing file from ZIP: {file} (size: {os.path.getsize(file_path)} bytes)")
                
                try:
                    documents = []  # Initialize documents list
                    
                    # Only process supported file types
                    if file_lower.endswith(('.pdf', '.docx', '.doc', '.xlsx', '.xls', '.pptx', '.ppt', '.txt')):
                        supported_files += 1
                        logger.info(f"Processing supported file: {file}")
                        
                        # Determine file type and load
                        if file_lower.endswith('.pdf'):
                            loader = PyPDFLoader(file_path)
                            documents = loader.load()
                        elif file_lower.endswith(('.docx', '.doc')):
                            loader = Docx2txtLoader(file_path)
                            documents = loader.load()
                        elif file_lower.endswith(('.xlsx', '.xls')):
                            loader = UnstructuredExcelLoader(file_path)
                            documents = loader.load()
                        elif file_lower.endswith(('.pptx', '.ppt')):
                            loader = UnstructuredPowerPointLoader(file_path)
                            documents = loader.load()
                        elif file_lower.endswith('.txt'):
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                content = f.read()
                            # Create document object manually for txt files
                            from langchain.schema import Document
                            doc = Document(
                                page_content=content,
                                metadata={"source": file_path, "filename": file}
                            )
                            documents = [doc]
                        else:
                            continue
                        
                        # Process documents from this file
                        for i, doc in enumerate(documents):
                            content_length = len(doc.page_content.strip())
                            logger.info(f"Document {i+1} from {file}: {content_length} characters")
                            
                            if content_length > 50:
                                doc.metadata.update({
                                    "filename": file,
                                    "source_zip": os.path.basename(zip_path),
                                    "page": i + 1,
                                    "section_type": classify_section(doc.page_content),
                                    "url_hash": url_hash,
                                    "file_type": f"zip_content_depth_{depth}"
                                })
                                processed_docs.append(doc)
                            else:
                                logger.warning(f"Skipping short content from {file}: {content_length} chars")
                    
                    elif file_lower.endswith('.zip'):
                        # Check file size before processing nested ZIP
                        nested_size = os.path.getsize(file_path)
                        if total_extracted_size + nested_size > max_extracted_size:
                            logger.warning(f"Skipping nested ZIP {file} - would exceed size limit")
                            continue
                            
                        # Recursive ZIP processing with size tracking
                        logger.info(f"Found nested ZIP file: {file}, processing recursively at depth {depth + 1}")
                        try:
                            nested_result = process_zip_file(
                                file_path, 
                                url_hash, 
                                depth + 1, 
                                total_extracted_size + nested_size,
                                max_extracted_size
                            )
                            # Check if ZIP bomb was detected in nested processing
                            if nested_result["files_in_zip"] == 0 and nested_result["supported_files"] == 0 and not nested_result["chunks"]:
                                logger.error(f"ZIP bomb detected in nested ZIP {file} - stopping all processing")
                                raise Exception(f"ZIP bomb detected - stopping processing for safety")
                            
                            if nested_result["chunks"]:
                                processed_docs.extend(nested_result["chunks"])
                                supported_files += nested_result["supported_files"]
                                logger.info(f"Extracted {len(nested_result['chunks'])} documents from nested ZIP: {file}")
                            else:
                                logger.warning(f"No content found in nested ZIP: {file}")
                        except Exception as e:
                            if "ZIP bomb" in str(e) or "exceeds safe limit" in str(e) or "stopping processing" in str(e):
                                logger.error("ZIP bomb protection triggered - aborting entire ZIP processing")
                                raise e
                            logger.warning(f"Failed to process nested ZIP {file}: {e}")
                                
                    else:
                        logger.info(f"Skipping unsupported file type: {file}")
                                
                except Exception as e:
                    # Let ZIP bomb exceptions propagate up immediately
                    if "ZIP bomb" in str(e) or "exceeds safe limit" in str(e):
                        raise e
                    logger.warning(f"Could not process {file} from ZIP: {e}")
                    continue
        
        # Clean up extracted files
        shutil.rmtree(extract_dir, ignore_errors=True)
        
        logger.info(f"ZIP processing summary: {total_files} total files, {supported_files} supported, {len(processed_docs)} documents extracted")
        
        if not processed_docs:
            # More informative error message
            if total_files == 0:
                raise Exception("ZIP file is empty")
            elif supported_files == 0:
                raise Exception(f"ZIP contains {total_files} files but none are supported formats (PDF, DOCX, Excel, PowerPoint, TXT)")
            else:
                raise Exception(f"ZIP contains {supported_files} supported files but all content was too short or empty")
            
        return {
            "filename": os.path.basename(zip_path),
            "chunks": processed_docs,  # Will be chunked later
            "pages_processed": len(processed_docs),
            "files_in_zip": total_files,
            "supported_files": supported_files
        }
        
    except Exception as e:
        logger.error(f"Error processing ZIP file: {e}")
        raise

async def download_document_from_url(url: str, timeout: int = 60) -> Tuple[str, str]:
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            logger.info(f"Downloading from: {url}...")
            
            response = await client.get(url)
            response.raise_for_status()
            
            content_type = response.headers.get('content-type', '')
            
            # Create temp file first
            temp_dir = tempfile.mkdtemp()
            temp_file_path = os.path.join(temp_dir, "downloaded_document")
            
            async with aiofiles.open(temp_file_path, 'wb') as f:
                await f.write(response.content)
            
            file_type = determine_file_type(url, content_type, temp_file_path)
        
            extension_map = {
                'pdf': '.pdf',
                'docx': '.docx', 'doc': '.doc',
                'xlsx': '.xlsx', 'xls': '.xls',
                'pptx': '.pptx', 'ppt': '.ppt',
                'png': '.png', 'jpeg': '.jpg',
                'zip': '.zip'
            }
            
            final_extension = extension_map.get(file_type, '.pdf')
            final_file_path = temp_file_path + final_extension
            os.rename(temp_file_path, final_file_path)
            
            file_size = len(response.content)
            logger.info(f"Downloaded {file_type.upper()}: {file_size} bytes to {final_file_path}")
            
            return final_file_path, file_type
            
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
            # Choose appropriate loader based on file type
            loader = None
            documents = []
            
            if file_type == 'pdf':
                loader = PyPDFLoader(file_path)
                documents = loader.load()
            elif file_type in ['docx', 'doc']:
                loader = Docx2txtLoader(file_path)
                documents = loader.load()
            elif file_type in ['xlsx', 'xls']:
                loader = UnstructuredExcelLoader(file_path)
                documents = loader.load()
            elif file_type in ['pptx', 'ppt']:
                loader = UnstructuredPowerPointLoader(file_path)
                documents = loader.load()
            elif file_type in ['png', 'jpeg', 'jpg']:
                loader = UnstructuredImageLoader(file_path)
                documents = loader.load()
            elif file_type == 'zip':
                zip_result = process_zip_file(file_path, url_hash)

                processed_docs = zip_result["chunks"]
                chunks = text_splitter.split_documents(processed_docs)
                
                for i, chunk in enumerate(chunks):
                    chunk.metadata["chunk_id"] = f"{url_hash}_chunk_{i}"
                    chunk.metadata["url_hash"] = url_hash
                    chunk.metadata["file_type"] = file_type
                
                return {
                    "filename": zip_result["filename"],
                    "chunks": chunks,
                    "pages_processed": zip_result["pages_processed"]
                }
            else:
                raise Exception(f"Unsupported file type: {file_type}")
            
            filename = Path(file_path).stem
            processed_docs = []
            
            for i, doc in enumerate(documents):
                content = doc.page_content
                if len(content.strip()) < 50:
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
            
            if not processed_docs:
                raise Exception(f"No readable content found in {file_type} file")
            
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
        
        # Reset hybrid retriever to include new documents
        reset_hybrid_retriever()
        
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
            
        # Use hybrid search instead of just vector search
        hybrid_retriever = get_hybrid_retriever()
        search_results = hybrid_retriever.hybrid_search(question, k=config.top_k)
        
        logger.info(f"Sending {len(search_results)} chunks to LLM for question: {question[:50]}...")
        logger.info("=== CHUNKS SENT TO LLM ===")
        for i, (doc, score) in enumerate(search_results, 1):
            logger.info(f"CHUNK {i} (Score: {score:.3f}):")
            logger.info(f"Source: {doc.metadata.get('filename', 'Unknown')} - Page {doc.metadata.get('page', 'N/A')}")
            logger.info(f"Content: {doc.page_content}")
            logger.info(f"Metadata: {doc.metadata}")
            logger.info("-" * 100)
        logger.info("=== END CHUNKS ===")
        
        if not search_results:
            return "No relevant information found in the document."
        
        # Format results for the prompt
        formatted_results = []
        for doc, score in search_results:
            formatted_result = {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": score,
                "source": f"{doc.metadata.get('filename', 'Unknown')} (Page {doc.metadata.get('page', 'N/A')})"
            }
            formatted_results.append(formatted_result)
        
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

class HybridRetriever:
    """Combines vector similarity search with BM25 keyword search for better retrieval"""
    
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.bm25 = None
        self.documents = None
        self._initialize_bm25()
    
    def _initialize_bm25(self):
        """Initialize BM25 with the current document corpus"""
        try:
            # Get all documents from vectorstore
            all_docs = self.vectorstore.get()
            if all_docs and 'documents' in all_docs:
                self.documents = all_docs['documents']
                # Tokenize documents for BM25
                tokenized_docs = [self._preprocess_text(doc).split() for doc in self.documents]
                self.bm25 = BM25Okapi(tokenized_docs)
                logger.info(f"Initialized BM25 with {len(self.documents)} documents")
            else:
                logger.warning("No documents found in vectorstore for BM25 initialization")
        except Exception as e:
            logger.warning(f"Could not initialize BM25: {e}")
            self.bm25 = None
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for better BM25 matching"""
        # Convert to lowercase and remove extra whitespace
        text = re.sub(r'\s+', ' ', text.lower().strip())
        # Remove special characters but keep alphanumeric and spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        return text
    
    def _deduplicate_results(self, vector_results: List, bm25_results: List) -> List:
        """Remove duplicate documents from combined results"""
        seen_content = set()
        unique_results = []
        
        # Process vector results first (they have scores)
        for doc, score in vector_results:
            content_hash = hash(doc.page_content[:100])  # Use first 100 chars as identifier
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_results.append((doc, score, 'vector'))
        
        # Process BM25 results
        for doc, score in bm25_results:
            content_hash = hash(doc.page_content[:100])
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_results.append((doc, score, 'bm25'))
        
        return unique_results
    
    def _rerank_results(self, combined_results: List, query: str) -> List:
        """Rerank combined results using a simple scoring strategy"""
        reranked = []
        
        for doc, score, source_type in combined_results:
            # Normalize scores and combine
            if source_type == 'vector':
                # Vector similarity scores are typically 0-1, lower is better
                normalized_score = 1 - score if score <= 1 else 1 / (1 + score)
            else:
                # BM25 scores are typically positive, higher is better
                normalized_score = min(score / 15.0, 1.0)  # Normalize to 0-1 range
            
            # Give slight preference to vector results for semantic matching
            if source_type == 'vector':
                final_score = normalized_score * 1.4
            else:
                final_score = normalized_score
            
            reranked.append((doc, final_score))
        
        # Sort by final score (higher is better)
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked
    
    def hybrid_search(self, query: str, k: int = 10) -> List:
        """Perform hybrid search combining vector similarity and BM25"""
        try:
            # Vector search
            vector_results = self.vectorstore.similarity_search_with_score(query, k=k)
            
            # BM25 search
            bm25_results = []
            if self.bm25 and self.documents:
                query_tokens = self._preprocess_text(query).split()
                bm25_scores = self.bm25.get_scores(query_tokens)
                
                # Get top BM25 results
                top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:k]
                
                for idx in top_indices:
                    if bm25_scores[idx] > 0:  # Only include results with positive scores
                        # Create document object for BM25 result
                        from langchain.schema import Document
                        doc = Document(
                            page_content=self.documents[idx],
                            metadata={"bm25_score": bm25_scores[idx], "doc_index": idx}
                        )
                        bm25_results.append((doc, bm25_scores[idx]))
            
            # Combine and deduplicate
            combined_results = self._deduplicate_results(vector_results, bm25_results)
            
            # Rerank results
            final_results = self._rerank_results(combined_results, query)
            
            logger.info(f"Hybrid search: {len(vector_results)} vector + {len(bm25_results)} BM25 = {len(final_results)} unique results")
            
            return final_results[:k]
            
        except Exception as e:
            logger.warning(f"Hybrid search failed, falling back to vector search: {e}")
            # Fallback to vector search only
            return self.vectorstore.similarity_search_with_score(query, k=k)

# Global hybrid retriever instance
_hybrid_retriever = None

def get_hybrid_retriever():
    """Get or create the global hybrid retriever instance"""
    global _hybrid_retriever
    if _hybrid_retriever is None:
        vectorstore = init_vectorstore()
        _hybrid_retriever = HybridRetriever(vectorstore)
    return _hybrid_retriever

def reset_hybrid_retriever():
    """Reset the hybrid retriever (call when new documents are added)"""
    global _hybrid_retriever
    _hybrid_retriever = None
