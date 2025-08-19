"""File type detection and hash utilities."""

import hashlib
import logging
import os
import shutil
import tempfile
import zipfile
from typing import Any

import aiofiles
import httpx
from langchain.schema import Document
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    UnstructuredExcelLoader,
    UnstructuredPowerPointLoader,
)

logger = logging.getLogger(__name__)


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
    """Generate MD5 hash of URL for caching purposes"""
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
                                # Import classify_section when needed to avoid circular imports
                                from ..rag_core import classify_section

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
    """Download document from URL and return file path and type"""
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
