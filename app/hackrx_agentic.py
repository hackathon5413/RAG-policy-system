import json
import asyncio
import logging
from typing import List, Dict, Any
from langchain_community.document_loaders import PyPDFLoader
from .document_processor import download_document_from_url
from .api_tools import call_any_url
from .rag_core import call_gemini
from jinja2 import Environment, FileSystemLoader

logger = logging.getLogger(__name__)
jinja_env = Environment(loader=FileSystemLoader('prompts'))

async def process_hackrx_agentic(document_url: str, questions: List[str]) -> Dict[str, Any]:
    temp_file_path = None
    try:
        temp_file_path, _ = await download_document_from_url(document_url)
        
        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()
        pdf_content = "\n".join([doc.page_content for doc in documents])
        
        apis_needed = await classify_apis_needed(pdf_content, questions)
        
        api_results = {}
        if apis_needed:
            tasks = [call_any_url(api_url) for api_url in apis_needed]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for api_url, result in zip(apis_needed, results):
                api_results[api_url] = result if not isinstance(result, Exception) else {"error": str(result)}
        
        answers = await generate_final_answers(pdf_content, api_results, questions)
        
        return {"success": True, "answers": answers}
        
    except Exception as e:
        logger.error(f"Error in hackrx agentic processing: {e}")
        return {
            "success": False,
            "error": str(e),
            "answers": [f"Error: {str(e)}" for _ in questions]
        }
    finally:
        if temp_file_path:
            import os
            try:
                os.remove(temp_file_path)
            except Exception:
                pass

async def classify_apis_needed(pdf_content: str, questions: List[str]) -> List[str]:
    try:
        template = jinja_env.get_template('agentic_classifier.j2')
        prompt = template.render(
            pdf_content=pdf_content,
            questions=questions
        )
        
        response = call_gemini(prompt)
        
        if response.startswith("```json"):
            response = response[7:-3].strip()
        
        result = json.loads(response)
        return result.get("api_urls", [])
        
    except Exception as e:
        logger.warning(f"API classification failed: {e}")
        return []

async def generate_final_answers(pdf_content: str, api_results: Dict[str, Any], questions: List[str]) -> List[str]:
    try:
        template = jinja_env.get_template('agentic_answerer.j2')
        prompt = template.render(
            pdf_content=pdf_content,
            api_results=api_results,
            questions=questions
        )
        
        response = call_gemini(prompt)
        
        if response.startswith("```json"):
            response = response[7:-3].strip()
        
        result = json.loads(response)
        return result.get("answers", ["Error parsing response" for _ in questions])
        
    except Exception as e:
        logger.error(f"Final answer generation failed: {e}")
        return [f"Error: {str(e)}" for _ in questions]

async def process_api_url(document_url: str, questions: List[str]) -> Dict[str, Any]:
    try:
        api_result = await call_any_url(document_url)
        
        answers = await generate_final_answers(
            pdf_content="",
            api_results={"api_response": api_result},
            questions=questions
        )
        
        return {"success": True, "answers": answers}
        
    except Exception as e:
        logger.error(f"Error processing API URL: {e}")
        return {
            "success": False,
            "error": str(e),
            "answers": [f"Error: {str(e)}" for _ in questions]
        }
