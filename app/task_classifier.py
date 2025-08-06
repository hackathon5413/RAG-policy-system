
import logging
import json
from .rag_core import call_gemini
from config import config

logger = logging.getLogger(__name__)



def _fallback_classify(question: str) -> str:

    q = question.lower()
    
    # Verification patterns
    verification_words = ["can i", "am i eligible", "customer for", "been covered", 
                         "years", "is covered", "covered?", "does cover", 
                         "allowed?", "eligible?", "remaining", "balance"]
    
    if any(word in q for word in verification_words):
        logger.info("🔄 Fallback: FACT_VERIFICATION")
        return "FACT_VERIFICATION"
    
    # Question patterns  
    elif any(q.startswith(word) for word in ["what", "when", "how", "why", "where", "which"]):
        logger.info("🔄 Fallback: QUESTION_ANSWERING")
        return "QUESTION_ANSWERING"
    
    # Default
    else:
        logger.info("🔄 Fallback: RETRIEVAL_QUERY")
        return "RETRIEVAL_QUERY"

def get_task_and_queries(question: str) -> dict:
    try:
        expansion_count = getattr(config, 'query_expansion_count', 3)
        
        prompt = f"""You must respond with ONLY valid JSON. No explanation, no additional text.

Analyze this question and classify the task type, then generate {expansion_count} total questions (including the original).

QUESTION: "{question}"

Task types:
- QUESTION_ANSWERING: What/When/How questions, procedures, document lists
- FACT_VERIFICATION: Can I/Am I/Is covered questions, eligibility checks
- RETRIEVAL_QUERY: General search, unclear questions

Generate {expansion_count-1} alternative questions that ask the same thing using:
- Different terminology
- Different phrasing
- Domain-specific terms
- Different question structures

Respond with this exact JSON format:
{{
  "task_type": "QUESTION_ANSWERING",
  "expanded_questions": [
    "{question}",
    "variant question 1",
    "variant question 2"
  ]
}}"""
        
        response = call_gemini(prompt).strip()
        
        # Remove markdown code blocks if present
        if response.startswith('```json'):
            response = response[7:]  # Remove '```json'
        if response.startswith('```'):
            response = response[3:]   # Remove '```'
        if response.endswith('```'):
            response = response[:-3]  # Remove trailing '```'
        response = response.strip()
        
        try:
            result = json.loads(response)
            task_type = result.get('task_type', '').upper()
            expanded_questions = result.get('expanded_questions', [question])
            
            valid_types = ["QUESTION_ANSWERING", "FACT_VERIFICATION", "RETRIEVAL_QUERY"]
            if task_type not in valid_types:
                task_type = _fallback_classify(question)
            
            if not expanded_questions or len(expanded_questions) == 0:
                expanded_questions = [question]
                
            logger.info(f"🎯 Combined: {task_type}, {len(expanded_questions)} queries")
            
            return {
                "task_type": task_type,
                "expanded_questions": expanded_questions
            }
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON: {e}. Response was: {response[:200]}...")
            return {
                "task_type": _fallback_classify(question),
                "expanded_questions": [question]
            }
            
    except Exception as e:
        logger.error(f"Combined classification failed: {e}")
        return {
            "task_type": _fallback_classify(question),
            "expanded_questions": [question]
        }
