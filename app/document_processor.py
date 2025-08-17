import asyncio
import hashlib
import json
import logging
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any, cast

import aiofiles
import httpx
from jinja2 import Environment, FileSystemLoader
from langchain.schema import Document
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    UnstructuredExcelLoader,
    UnstructuredImageLoader,
    UnstructuredPowerPointLoader,
)

from config import config

from .cache import question_cache
from .rag_core import call_gemini, classify_section
from .vector_store import init_vectorstore, text_splitter

logger = logging.getLogger(__name__)
jinja_env = Environment(loader=FileSystemLoader("prompts"))

URL_CACHE_FILE = "./data/url_cache.json"
os.makedirs(os.path.dirname(URL_CACHE_FILE), exist_ok=True)


def load_url_cache() -> dict[str, bool]:
    try:
        with open(URL_CACHE_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def save_url_cache(cache: dict[str, bool]):
    with open(URL_CACHE_FILE, "w") as f:
        json.dump(cache, f)


def get_file_type_from_url(url: str) -> str:
    """Get file type from URL - first attempt"""
    url_lower = url.lower()
    if ".pdf" in url_lower:
        return "pdf"
    elif ".docx" in url_lower:
        return "docx"
    elif ".doc" in url_lower:
        return "doc"
    elif ".xlsx" in url_lower:
        return "xlsx"
    elif ".xls" in url_lower:
        return "xls"
    elif ".pptx" in url_lower:
        return "pptx"
    elif ".ppt" in url_lower:
        return "ppt"
    elif ".png" in url_lower:
        return "png"
    elif ".jpg" in url_lower or ".jpeg" in url_lower:
        return "jpeg"
    elif ".zip" in url_lower:
        return "zip"
    return "unknown"


def get_file_type_from_content_type(content_type: str) -> str:
    """Get file type from HTTP content-type header"""
    content_type = content_type.lower()
    if "pdf" in content_type:
        return "pdf"
    elif "wordprocessingml.document" in content_type:
        return "docx"
    elif "spreadsheetml.sheet" in content_type:
        return "xlsx"
    elif "presentationml.presentation" in content_type:
        return "pptx"
    elif "image/png" in content_type:
        return "png"
    elif "image/jpeg" in content_type:
        return "jpeg"
    elif "application/zip" in content_type:
        return "zip"
    return "unknown"


def get_file_type_from_signature(file_path: str) -> str:
    """Get file type from file signature (magic numbers)"""
    try:
        with open(file_path, "rb") as f:
            header = f.read(16)

        if header.startswith(b"%PDF"):
            return "pdf"
        elif header.startswith(b"PK\x03\x04"):
            # ZIP-based formats (Office docs, ZIP files)
            try:
                with zipfile.ZipFile(file_path, "r") as zip_ref:
                    filenames = zip_ref.namelist()
                    if any("word/" in f for f in filenames):
                        return "docx"
                    elif any("xl/" in f for f in filenames):
                        return "xlsx"
                    elif any("ppt/" in f for f in filenames):
                        return "pptx"
                    else:
                        return "zip"
            except Exception:
                return "zip"
        elif header.startswith(b"\x89PNG\r\n\x1a\n"):
            return "png"
        elif header.startswith(b"\xff\xd8\xff"):
            return "jpeg"

    except Exception as e:
        logger.warning(f"Could not read file signature: {e}")

    return "unknown"


def determine_file_type(url: str, content_type: str, file_path: str) -> str:
    """Determine file type using multiple methods"""
    # Try URL first
    file_type = get_file_type_from_url(url)
    if file_type != "unknown":
        return file_type

    # Try content-type header
    file_type = get_file_type_from_content_type(content_type)
    if file_type != "unknown":
        return file_type

    # Try file signature as last resort
    file_type = get_file_type_from_signature(file_path)
    if file_type != "unknown":
        return file_type

    # Default to PDF if all else fails
    logger.warning(f"Could not determine file type for {url}, defaulting to PDF")
    return "pdf"


def get_url_hash(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()


def process_zip_file(
    zip_path: str,
    url_hash: str,
    depth: int = 0,
    total_extracted_size: int = 0,
    max_extracted_size: int = 10 * 1024 * 1024,
) -> dict[str, Any]:
    """Process ZIP file with strict ZIP bomb protection"""
    if depth > 3:
        logger.error(
            f"ZIP nesting depth ({depth}) exceeds safe limit (3) - ZIP bomb detected"
        )
        raise Exception(
            f"ZIP bomb detected: nesting depth {depth} exceeds safe limit of 3"
        )

    if total_extracted_size > max_extracted_size:
        logger.error(
            f"Total extracted size ({total_extracted_size:,} bytes) exceeds safe limit ({max_extracted_size:,} bytes) - ZIP bomb detected"
        )
        raise Exception(
            f"ZIP bomb detected: extracted size {total_extracted_size:,} exceeds limit of {max_extracted_size:,} bytes"
        )

    try:
        extract_dir = tempfile.mkdtemp()
        processed_docs = []
        total_files = 0
        supported_files = 0

        logger.info(f"Processing ZIP at depth {depth}: {os.path.basename(zip_path)}")

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            file_list = zip_ref.namelist()
            logger.info(f"ZIP contains {len(file_list)} items: {file_list}")

            # Show more details about each item
            for item in file_list:
                info = zip_ref.getinfo(item)
                logger.info(
                    f"  {item}: {info.file_size} bytes, {'folder' if item.endswith('/') else 'file'}"
                )

            zip_ref.extractall(extract_dir)

        # Process each extracted file
        for root, _dirs, files in os.walk(extract_dir):
            for file in files:
                file_path = os.path.join(root, file)
                file_lower = file.lower()
                total_files += 1

                logger.info(
                    f"Processing file from ZIP: {file} (size: {os.path.getsize(file_path)} bytes)"
                )

                try:
                    documents = []  # Initialize documents list

                    # Only process supported file types
                    if file_lower.endswith(
                        (
                            ".pdf",
                            ".docx",
                            ".doc",
                            ".xlsx",
                            ".xls",
                            ".pptx",
                            ".ppt",
                            ".txt",
                        )
                    ):
                        supported_files += 1
                        logger.info(f"Processing supported file: {file}")

                        # Determine file type and load
                        if file_lower.endswith(".pdf"):
                            loader = PyPDFLoader(file_path)
                            documents = loader.load()
                        elif file_lower.endswith((".docx", ".doc")):
                            loader = Docx2txtLoader(file_path)
                            documents = loader.load()
                        elif file_lower.endswith((".xlsx", ".xls")):
                            loader = UnstructuredExcelLoader(file_path)
                            documents = loader.load()
                        elif file_lower.endswith((".pptx", ".ppt")):
                            loader = UnstructuredPowerPointLoader(file_path)
                            documents = loader.load()
                        elif file_lower.endswith(".txt"):
                            with open(
                                file_path, encoding="utf-8", errors="ignore"
                            ) as f:
                                content = f.read()
                            # Create document object manually for txt files
                            from langchain.schema import Document

                            doc = Document(
                                page_content=content,
                                metadata={"source": file_path, "filename": file},
                            )
                            documents = [doc]
                        else:
                            continue

                        # Process documents from this file
                        for i, doc in enumerate(documents):
                            content_length = len(doc.page_content.strip())
                            logger.info(
                                f"Document {i + 1} from {file}: {content_length} characters"
                            )

                            if content_length > 50:
                                doc.metadata.update(
                                    {
                                        "filename": file,
                                        "source_zip": os.path.basename(zip_path),
                                        "page": i + 1,
                                        "section_type": classify_section(
                                            doc.page_content
                                        ),
                                        "url_hash": url_hash,
                                        "file_type": f"zip_content_depth_{depth}",
                                    }
                                )
                                processed_docs.append(doc)
                            else:
                                logger.warning(
                                    f"Skipping short content from {file}: {content_length} chars"
                                )

                    elif file_lower.endswith(".zip"):
                        # Check file size before processing nested ZIP
                        nested_size = os.path.getsize(file_path)
                        if total_extracted_size + nested_size > max_extracted_size:
                            logger.warning(
                                f"Skipping nested ZIP {file} - would exceed size limit"
                            )
                            continue

                        # Recursive ZIP processing with size tracking
                        logger.info(
                            f"Found nested ZIP file: {file}, processing recursively at depth {depth + 1}"
                        )
                        try:
                            nested_result = process_zip_file(
                                file_path,
                                url_hash,
                                depth + 1,
                                total_extracted_size + nested_size,
                                max_extracted_size,
                            )
                            # Check if ZIP bomb was detected in nested processing
                            if (
                                nested_result["files_in_zip"] == 0
                                and nested_result["supported_files"] == 0
                                and not nested_result["chunks"]
                            ):
                                logger.error(
                                    f"ZIP bomb detected in nested ZIP {file} - stopping all processing"
                                )
                                raise Exception(
                                    "ZIP bomb detected - stopping processing for safety"
                                )

                            if nested_result["chunks"]:
                                processed_docs.extend(nested_result["chunks"])
                                supported_files += nested_result["supported_files"]
                                logger.info(
                                    f"Extracted {len(nested_result['chunks'])} documents from nested ZIP: {file}"
                                )
                            else:
                                logger.warning(
                                    f"No content found in nested ZIP: {file}"
                                )
                        except Exception as e:
                            if (
                                "ZIP bomb" in str(e)
                                or "exceeds safe limit" in str(e)
                                or "stopping processing" in str(e)
                            ):
                                logger.error(
                                    "ZIP bomb protection triggered - aborting entire ZIP processing"
                                )
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

        logger.info(
            f"ZIP processing summary: {total_files} total files, {supported_files} supported, {len(processed_docs)} documents extracted"
        )

        if not processed_docs:
            # More informative error message
            if total_files == 0:
                raise Exception("ZIP file is empty")
            elif supported_files == 0:
                raise Exception(
                    f"ZIP contains {total_files} files but none are supported formats (PDF, DOCX, Excel, PowerPoint, TXT)"
                )
            else:
                raise Exception(
                    f"ZIP contains {supported_files} supported files but all content was too short or empty"
                )

        return {
            "filename": os.path.basename(zip_path),
            "chunks": processed_docs,  # Will be chunked later
            "pages_processed": len(processed_docs),
            "files_in_zip": total_files,
            "supported_files": supported_files,
        }

    except Exception as e:
        logger.error(f"Error processing ZIP file: {e}")
        raise


async def download_document_from_url(url: str, timeout: int = 60) -> tuple[str, str]:
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            logger.info(f"Downloading from: {url}...")

            response = await client.get(url)
            response.raise_for_status()

            content_type = response.headers.get("content-type", "")

            # Create temp file first
            temp_dir = tempfile.mkdtemp()
            temp_file_path = os.path.join(temp_dir, "downloaded_document")

            async with aiofiles.open(temp_file_path, "wb") as f:
                await f.write(response.content)

            file_type = determine_file_type(url, content_type, temp_file_path)

            extension_map = {
                "pdf": ".pdf",
                "docx": ".docx",
                "doc": ".doc",
                "xlsx": ".xlsx",
                "xls": ".xls",
                "pptx": ".pptx",
                "ppt": ".ppt",
                "png": ".png",
                "jpeg": ".jpg",
                "zip": ".zip",
            }

            final_extension = extension_map.get(file_type, ".pdf")
            final_file_path = temp_file_path + final_extension
            os.rename(temp_file_path, final_file_path)

            file_size = len(response.content)
            logger.info(
                f"Downloaded {file_type.upper()}: {file_size} bytes to {final_file_path}"
            )

            return final_file_path, file_type

    except httpx.TimeoutException:
        raise Exception(f"Timeout downloading document from URL: {url}")
    except httpx.HTTPStatusError as e:
        raise Exception(f"HTTP error downloading document: {e.response.status_code}")
    except Exception as e:
        raise Exception(f"Error downloading document: {e!s}")


async def process_local_document(
    file_path: str, file_type: str, url_hash: str
) -> dict[str, Any]:
    try:
        loop = asyncio.get_event_loop()

        def sync_process_document():
            # Choose appropriate loader based on file type
            loader = None
            documents = []

            if file_type == "pdf":
                loader = PyPDFLoader(file_path)
                documents = loader.load()
            elif file_type in ["docx", "doc"]:
                loader = Docx2txtLoader(file_path)
                documents = loader.load()
            elif file_type in ["xlsx", "xls"]:
                loader = UnstructuredExcelLoader(file_path)
                documents = loader.load()
            elif file_type in ["pptx", "ppt"]:
                loader = UnstructuredPowerPointLoader(file_path)
                documents = loader.load()
            elif file_type in ["png", "jpeg", "jpg"]:
                loader = UnstructuredImageLoader(file_path)
                documents = loader.load()
            elif file_type == "zip":
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
                    "pages_processed": zip_result["pages_processed"],
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
                doc.metadata.update(
                    {
                        "filename": filename,
                        "page": i + 1 if file_type == "pdf" else 1,
                        "section_type": classify_section(content),
                        "url_hash": url_hash,
                        "file_type": file_type,
                    }
                )
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
                "pages_processed": len(processed_docs),
            }

        result = await loop.run_in_executor(None, sync_process_document)
        chunks = cast(list[Document], result["chunks"])
        logger.info(
            f"Adding {len(chunks)} chunks from {file_type.upper()} to vector store"
        )

        vectorstore = init_vectorstore()
        vectorstore.add_documents(chunks)

        logger.info(
            f"Successfully added all {len(chunks)} chunks using parallel batching"
        )

        cache = load_url_cache()
        cache[url_hash] = True
        save_url_cache(cache)

        return {
            "success": True,
            "filename": result["filename"],
            "chunks_created": len(chunks),
            "pages_processed": result["pages_processed"],
            "file_type": file_type,
        }

    except Exception as e:
        logger.error(f"Error processing {file_type.upper()}: {e}")
        return {"success": False, "error": str(e)}


async def process_document_from_url(url: str) -> dict[str, Any]:
    url_hash = get_url_hash(url)
    cache = load_url_cache()

    if url_hash in cache:
        logger.info(f"URL {url[:50]}... already processed, skipping")
        return {
            "success": True,
            "source_url": url,
            "chunks_created": 0,
            "pages_processed": 0,
            "cached": True,
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
                "cached": False,
            }
        else:
            return result

    except Exception as e:
        logger.error(f"Error processing document from URL: {e}")
        return {"success": False, "error": str(e), "source_url": url}
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                temp_dir = os.path.dirname(temp_file_path)
                if os.path.exists(temp_dir) and not os.listdir(temp_dir):
                    os.rmdir(temp_dir)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp file: {e}")


async def process_question_batch(questions: list[str]) -> list[str]:
    try:
        # Step 1: Check cache first to avoid unnecessary processing
        cached_answers = {}
        uncached_questions = []
        question_order = {}  # Track original order

        for i, question in enumerate(questions):
            question_order[question] = i
            cached_answer = question_cache.get(question)
            if cached_answer:
                cached_answers[question] = cached_answer
                logger.info(f"📦 Using cached answer for: {question[:50]}...")
            else:
                uncached_questions.append(question)
                logger.info(f"🔄 Will process: {question[:50]}...")

        # If all questions are cached, return cached answers in order
        if not uncached_questions:
            logger.info(f"📦 All {len(questions)} questions found in cache!")
            return [cached_answers[question] for question in questions]

        # Step 2: Process only uncached questions
        logger.info(
            f"🔄 Processing {len(uncached_questions)} uncached questions, {len(cached_answers)} cached"
        )

        from .task_classifier import get_batch_task_classifications

        classifications = get_batch_task_classifications(uncached_questions)
        logger.info("Classifications obtained for uncached questions", classifications)

        # Step 3: Enhanced vector search with query transformations
        from .vector_store import semantic_similarity_search

        question_chunk_map = []

        for i, (question, classification) in enumerate(
            zip(uncached_questions, classifications, strict=False), 1
        ):
            task_type = classification.get("task_type", "RETRIEVAL_QUERY")
            transformed_queries = classification.get("transformed_queries", [question])

            # Use all transformed queries for enhanced retrieval
            all_chunks = []
            seen_chunk_ids = set()

            for query in transformed_queries:
                search_results = semantic_similarity_search(
                    query, k=config.top_k, task_type=task_type
                )

                if search_results:
                    for doc, score in search_results:
                        chunk_id = doc.metadata.get("chunk_id", id(doc))
                        if chunk_id not in seen_chunk_ids:
                            all_chunks.append((doc, score))
                            seen_chunk_ids.add(chunk_id)

            # Sort by relevance score and take top results
            all_chunks.sort(key=lambda x: x[1], reverse=True)
            top_chunks = all_chunks[: config.top_k]

            question_chunks = []
            if top_chunks:
                question_chunks = [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "source": f"{doc.metadata.get('filename', 'Unknown')} (Page {doc.metadata.get('page', 'N/A')})",
                    }
                    for doc, _ in top_chunks
                ]

            # Store question with its specific chunks
            task_type = classification.get("task_type", "QUESTION_ANSWERING")
            question_chunk_map.append(
                {
                    "question_num": i,
                    "question": question,
                    "task_type": task_type,
                    "chunks": question_chunks,
                }
            )

        # Step 4: Generate answers for uncached questions only
        new_answers = []
        if question_chunk_map:  # Only call LLM if there are uncached questions
            prompt = create_structured_prompt_with_mapping(question_chunk_map)

            # Call LLM and handle response
            response = call_gemini(prompt)

            if not response or response.strip() == "":
                new_answers = [
                    "Error: Received empty response from AI model"
                    for _ in uncached_questions
                ]
            else:
                new_answers = parse_multi_question_response(
                    response, uncached_questions
                )

            # Cache the new answers
            for question, answer in zip(uncached_questions, new_answers, strict=False):
                if answer and not answer.startswith("Error:"):
                    question_cache.set(question, answer)
                    logger.info(f"💾 Cached answer for: {question[:50]}...")

        # Step 5: Combine cached and new answers in original order
        final_answers = []
        new_answer_index = 0

        for question in questions:
            if question in cached_answers:
                final_answers.append(cached_answers[question])
            else:
                final_answers.append(new_answers[new_answer_index])
                new_answer_index += 1

        logger.info(
            f"✅ Returned {len(final_answers)} answers ({len(cached_answers)} cached, {len(new_answers)} new)"
        )
        return final_answers

    except Exception as e:
        logger.error(f"Error processing question batch: {e}")
        return [f"Error processing questions: {e!s}" for _ in questions]


def create_structured_prompt_with_mapping(question_chunk_map: list) -> str:
    """Create a structured prompt that maps each question to its relevant chunks"""
    template = jinja_env.get_template("insurance_query.j2")

    return template.render(question_chunk_map=question_chunk_map)


def parse_multi_question_response(response: str, questions: list[str]) -> list[str]:
    """Parse LLM response expecting strict JSON with an 'answers' list."""
    try:
        data = json.loads(response.strip())
        answers = data.get("answers", [])
        # Pad missing answers
        if len(answers) < len(questions):
            answers += ["Information not available"] * (len(questions) - len(answers))
        return [str(ans) for ans in answers[: len(questions)]]
    except Exception:
        # On any parse error, return 'Information not available' for each question
        return ["Information not available"] * len(questions)


async def answer_questions(questions: list[str]) -> list[str]:
    total_questions = len(questions)
    # Use configurable batch size for questions
    max_batch_size = config.question_batch_size

    if total_questions <= max_batch_size:
        return await process_question_batch(questions)

    num_batches = (total_questions + max_batch_size - 1) // max_batch_size
    base_batch_size = total_questions // num_batches
    remainder = total_questions % num_batches

    batches = []
    start_idx = 0

    for i in range(num_batches):
        batch_size = base_batch_size + (1 if i < remainder else 0)
        batch = questions[start_idx : start_idx + batch_size]
        batches.append(batch)
        start_idx += batch_size

    logger.info(
        f"Processing {total_questions} questions in {len(batches)} batches: {[len(b) for b in batches]}"
    )

    # Process all batches in parallel
    batch_tasks = [process_question_batch(batch) for batch in batches]
    batch_results = await asyncio.gather(*batch_tasks)

    # Flatten results back into single list
    all_answers = []
    for batch_answers in batch_results:
        all_answers.extend(batch_answers)

    return all_answers


async def process_document_and_answer(
    document_url: str, questions: list[str]
) -> dict[str, Any]:
    if config.is_agentic_url(document_url):
        from .hackrx_agentic import process_hackrx_agentic

        return await process_hackrx_agentic(document_url, questions)

    file_type = get_file_type_from_url(document_url)

    if file_type == "unknown":
        from .hackrx_agentic import process_api_url

        return await process_api_url(document_url, questions)

    try:
        processing_result = await process_document_from_url(document_url)

        if not processing_result["success"]:
            return {
                "success": False,
                "error": processing_result["error"],
                "answers": [
                    f"Error processing document: {processing_result['error']}"
                    for _ in questions
                ],
            }

        answers = await answer_questions(questions)

        try:
            from .vector_store import get_embeddings

            embeddings_instance = get_embeddings()
            if hasattr(embeddings_instance, "save_cache"):
                embeddings_instance.save_cache()
        except Exception as e:
            logger.warning(f"Could not save embedding cache: {e}")

        return {"success": True, "answers": answers, "document_info": processing_result}

    except Exception as e:
        logger.error(f"Error in complete workflow: {e}")
        return {
            "success": False,
            "error": str(e),
            "answers": [f"Error: {e!s}" for _ in questions],
        }
