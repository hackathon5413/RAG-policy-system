import asyncio
import logging
import os
from pathlib import Path
from typing import Any, cast

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

from .rag_core import call_gemini, classify_section
from .utils import (
    create_structured_prompt_with_mapping,
    download_document_from_url,
    get_file_type_from_url,
    get_url_hash,
    load_url_cache,
    parse_multi_question_response,
    process_zip_file,
    question_cache,
    save_url_cache,
)
from .vector_store import init_vectorstore, text_splitter

logger = logging.getLogger(__name__)
jinja_env = Environment(loader=FileSystemLoader("prompts"))


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

        # Step 2.5: Get classifications (conditionally)
        if config.task_classification_enabled:
            from .task_classifier import get_batch_task_classifications

            classifications = get_batch_task_classifications(uncached_questions)
            logger.info(f"🎯 Classifications obtained: {classifications}")
        else:
            # Create default classifications when task classification is disabled
            logger.info(
                "⚡ Task classification disabled - using default classifications"
            )
            classifications = []
            for question in uncached_questions:
                classifications.append(
                    {
                        "question": question,
                        "task_type": "QUESTION_ANSWERING",  # Default task type
                        "transformed_queries": [question],  # Only original question
                    }
                )
            logger.info(f"⚡ Created {len(classifications)} default classifications")

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

            # Re-rank using cross-encoder for better relevance (if enabled)
            if config.reranking_enabled:
                from .vector_store import rerank_chunks_async

                # Take 2x more candidates for re-ranking, then select top_k
                initial_candidates = min(len(all_chunks), config.top_k * 2)
                all_chunks.sort(key=lambda x: x[1], reverse=True)
                top_candidates = all_chunks[:initial_candidates]

                top_chunks = await rerank_chunks_async(
                    question, top_candidates, config.top_k
                )
                logger.info(
                    f"🔄 Reranking enabled: Using {len(top_chunks)} reranked chunks (async)"
                )
            else:
                # Use simple sorting by similarity score when reranking is disabled
                all_chunks.sort(key=lambda x: x[1], reverse=True)
                top_chunks = all_chunks[: config.top_k]
                logger.info(
                    f"⚡ Reranking disabled: Using top {len(top_chunks)} chunks by similarity score"
                )

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

            # Log final chunks going to LLM
            logger.info(
                f"📋 Question {i}: '{question}...' -> {len(question_chunks)} final chunks"
            )
            for idx, chunk in enumerate(question_chunks):  # Show all chunks
                logger.info(
                    f"  Chunk {idx + 1}: {chunk['source']} - {chunk['content']}"
                )

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
