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


class FileDownloadError(Exception):
    """Raised when file download fails."""

    pass


logger = logging.getLogger(__name__)


def determine_file_type_from_content(file_path: str) -> str:
    """
    Determine file type by examining file content/magic numbers.

    Args:
        file_path: Path to the file to analyze

    Returns:
        str: The detected file type or 'unknown' if type cannot be determined
    """
    with open(file_path, "rb") as f:
        # Read first few bytes for magic number detection
        header = f.read(8)

        # PDF: %PDF
        if header.startswith(b"%PDF"):
            return "pdf"

        # ZIP: PK
        if header.startswith(b"PK\x03\x04"):
            return "zip"

        # Office files (docx, xlsx, pptx) are zip files
        if header.startswith(b"PK"):
            # Need to check internal structure
            return "zip"  # For now return zip

        # JPEG: FF D8
        if header.startswith(b"\xff\xd8"):
            return "jpeg"

        # PNG: 89 50 4E 47
        if header.startswith(b"\x89PNG"):
            return "png"

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


def determine_file_type(
    url: str | None = None,
    content_type: str | None = None,
    file_path: str | None = None,
) -> str:
    """
    Determine file type from URL, content-type header, and/or file content.
    At least one of url, content_type, or file_path must be provided.

    Args:
        url: Optional URL or filename to analyze
        content_type: Optional MIME type from Content-Type header
        file_path: Optional path to the downloaded file for content analysis

    Returns:
        str: The detected file type ('pdf', 'docx', etc.) or 'unknown' if type cannot be determined
    """
    detected_type = None

    # Try each method in order until we find a match
    if url:
        url_lower = url.lower()
        if ".pdf" in url_lower:
            detected_type = "pdf"
        elif ".docx" in url_lower:
            detected_type = "docx"
        elif ".doc" in url_lower:
            detected_type = "doc"
        elif ".xlsx" in url_lower:
            detected_type = "xlsx"
        elif ".xls" in url_lower:
            detected_type = "xls"
        elif ".pptx" in url_lower:
            detected_type = "pptx"
        elif ".ppt" in url_lower:
            detected_type = "ppt"
        elif ".png" in url_lower:
            detected_type = "png"
        elif ".jpg" in url_lower or ".jpeg" in url_lower:
            detected_type = "jpeg"
        elif ".zip" in url_lower:
            detected_type = "zip"

    # If URL didn't help and we have content type, try that
    if not detected_type and content_type:
        content_type_lower = content_type.lower()
        if "pdf" in content_type_lower:
            detected_type = "pdf"
        elif "docx" in content_type_lower or "msword" in content_type_lower:
            detected_type = "docx"
        elif "xlsx" in content_type_lower or "spreadsheet" in content_type_lower:
            detected_type = "xlsx"
        elif "pptx" in content_type_lower or "presentation" in content_type_lower:
            detected_type = "pptx"

    # If still no match and we have a file, try content analysis
    if not detected_type and file_path:
        detected_type = determine_file_type_from_content(file_path)

    if not detected_type:
        detected_type = "unknown"

    return detected_type
    # Try using content-type if provided
    if content_type:
        content_type_lower = content_type.lower()
        if "pdf" in content_type_lower:
            return "pdf"
        elif "docx" in content_type_lower or "msword" in content_type_lower:
            return "docx"
        elif "xlsx" in content_type_lower or "spreadsheet" in content_type_lower:
            return "xlsx"
        elif "pptx" in content_type_lower or "presentation" in content_type_lower:
            return "pptx"


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


async def download_from_url(url: str, target_path: str) -> None:
    """Download a file from URL to a target path."""
    logger.info(f"Downloading from: {url}...")

    from .s3_utils import download_from_s3, is_s3_url

    if is_s3_url(url):
        # Download using S3 client
        if not download_from_s3(url, target_path):
            raise FileDownloadError("Failed to download from S3")
        logger.info(f"S3 download completed: {target_path}")
    else:
        # Regular HTTP download
        async with httpx.AsyncClient(follow_redirects=True) as client:
            try:
                # Set a longer timeout for large files
                response = await client.get(url, timeout=30.0)
                response.raise_for_status()

                async with aiofiles.open(target_path, "wb") as f:
                    await f.write(response.content)
                logger.info(f"HTTP download completed: {target_path}")
            except httpx.HTTPStatusError as e:
                raise FileDownloadError(
                    f"HTTP error downloading document: {e.response.status_code}"
                )
            except Exception as e:
                raise FileDownloadError(f"Error downloading document: {e!s}")


async def download_document_from_url(url: str) -> tuple[str, str]:
    """Download a document from URL and return its path and type."""
    # Create a temporary file
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, "downloaded_document")

    try:
        # Download the file
        await download_from_url(url, temp_path)

        # Determine file type
        file_type = determine_file_type(url=url, file_path=temp_path)
        if file_type == "unknown":
            raise FileDownloadError("Could not determine file type")

        # Rename file with proper extension
        final_path = f"{temp_path}.{file_type}"
        os.rename(temp_path, final_path)

        return final_path, file_type

    except Exception as e:
        # Clean up on error
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise e
