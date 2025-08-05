"""
Task Type Classifier

Improved classification system for determining optimal task types based on user questions.
Uses a combination of LLM classification with robust fallback mechanisms and caching.
"""

import logging
import re
from enum import Enum
from typing import Optional, Dict, List, Tuple
from functools import lru_cache
from dataclasses import dataclass

from .rag_core import call_gemini

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Enumeration of supported task types"""
    QUESTION_ANSWERING = "QUESTION_ANSWERING"
    FACT_VERIFICATION = "FACT_VERIFICATION" 
    RETRIEVAL_QUERY = "RETRIEVAL_QUERY"


@dataclass
class ClassificationPattern:
    """Pattern definition for fallback classification"""
    keywords: List[str]
    prefixes: List[str]
    patterns: List[str]
    weight: float = 1.0


class TaskTypeClassifier:
    """Enhanced task type classifier with improved accuracy and caching"""
    
    def __init__(self):
        self._classification_patterns = self._initialize_patterns()
        self._gemini_prompt = self._build_classification_prompt()
    
    def _initialize_patterns(self) -> Dict[TaskType, ClassificationPattern]:
        """Initialize classification patterns for fallback mechanism"""
        return {
            TaskType.FACT_VERIFICATION: ClassificationPattern(
                keywords=[
                    "can i", "am i eligible", "am i allowed", "do i qualify",
                    "is covered", "does cover", "covered by", "eligible for",
                    "allowed to", "permitted to", "authorized", "entitled",
                    "remaining", "balance", "limit", "maximum", "minimum",
                    "years", "months", "days", "period", "duration",
                    "customer for", "been covered", "coverage includes",
                    "policy covers", "plan includes", "benefits include"
                ],
                prefixes=[
                    "can i", "am i", "do i", "will i", "should i",
                    "is it", "does it", "would it"
                ],
                patterns=[
                    r"\b(can|am|do|will|should)\s+i\b",
                    r"\bis\s+(covered|eligible|allowed|included)\b",
                    r"\bdoes\s+(cover|include|allow|permit)\b",
                    r"\b(remaining|balance|left|available)\b",
                    r"\b\d+\s+(years?|months?|days?)\b"
                ],
                weight=2.0
            ),
            
            TaskType.QUESTION_ANSWERING: ClassificationPattern(
                keywords=[
                    "what is", "what are", "what does", "what would",
                    "how to", "how do", "how does", "how can", "how much",
                    "when do", "when does", "when can", "when should",
                    "where do", "where can", "where is",
                    "why does", "why would", "which one", "which are",
                    "explain", "describe", "tell me", "show me",
                    "procedure", "process", "steps", "instructions",
                    "definition", "meaning", "difference", "comparison"
                ],
                prefixes=[
                    "what", "how", "when", "where", "why", "which",
                    "explain", "describe", "tell", "show", "list"
                ],
                patterns=[
                    r"\b(what|how|when|where|why|which)\b",
                    r"\b(explain|describe|tell|show|list)\b",
                    r"\b(procedure|process|steps|instructions)\b",
                    r"\b(definition|meaning|difference)\b"
                ],
                weight=1.5
            ),
            
            TaskType.RETRIEVAL_QUERY: ClassificationPattern(
                keywords=[
                    "find", "search", "look for", "get me", "show all",
                    "list all", "give me", "provide", "retrieve",
                    "information about", "details about", "data on"
                ],
                prefixes=[
                    "find", "search", "look", "get", "show", "list",
                    "give", "provide", "retrieve"
                ],
                patterns=[
                    r"\b(find|search|look\s+for|get\s+me)\b",
                    r"\b(show|list|give|provide)\s+(all|me)\b",
                    r"\b(information|details|data)\s+(about|on)\b"
                ],
                weight=1.0
            )
        }
    
    def _build_classification_prompt(self) -> str:
        """Build optimized prompt for LLM classification"""
        return """Classify this question into exactly ONE category:

QUESTION_ANSWERING: Questions seeking explanations, procedures, or factual information
- What/How/When/Where/Why questions
- Requests for explanations, definitions, or procedures
- Examples: "What is X?", "How do I do X?", "When does X happen?"

FACT_VERIFICATION: Questions about eligibility, coverage, or specific conditions
- Can/Am I/Is questions about eligibility or coverage
- Verification of specific facts or conditions
- Examples: "Can I do X?", "Am I eligible for X?", "Is X covered?"

RETRIEVAL_QUERY: General search requests or unclear questions
- Broad search requests
- Ambiguous or incomplete questions
- Examples: "Find information about X", "Tell me about X"

Question: "{question}"

Classification (respond with only the category name):"""
    
    @lru_cache(maxsize=1000)
    def classify(self, question: str) -> TaskType:
        """
        Classify question into optimal task type with caching
        
        Args:
            question: User question to classify
            
        Returns:
            TaskType enum value
        """
        if not question or not question.strip():
            logger.warning("Empty question provided, defaulting to RETRIEVAL_QUERY")
            return TaskType.RETRIEVAL_QUERY
        
        question = question.strip()
        
        try:
            # Try LLM classification first
            llm_result = self._classify_with_llm(question)
            if llm_result:
                logger.info(f"🎯 LLM classified as {llm_result.value}")
                return llm_result
            
        except Exception as e:
            logger.warning(f"⚠️ LLM classification failed: {e}")
        
        # Fall back to pattern-based classification
        fallback_result = self._classify_with_patterns(question)
        logger.info(f"🔄 Pattern classified as {fallback_result.value}")
        return fallback_result
    
    def _classify_with_llm(self, question: str) -> Optional[TaskType]:
        """Classify using LLM with improved prompt"""
        try:
            prompt = self._gemini_prompt.format(question=question)
            response = call_gemini(prompt).strip().upper()
            
            # More flexible matching
            for task_type in TaskType:
                if (task_type.value in response or 
                    response in task_type.value or
                    task_type.value.replace('_', ' ') in response):
                    return task_type
            
            logger.warning(f"⚠️ Invalid LLM response '{response}'")
            return None
            
        except Exception as e:
            logger.error(f"❌ LLM classification error: {e}")
            return None
    
    def _classify_with_patterns(self, question: str) -> TaskType:
        """Enhanced pattern-based classification with scoring"""
        question_lower = question.lower().strip()
        scores = {task_type: 0.0 for task_type in TaskType}
        
        for task_type, pattern in self._classification_patterns.items():
            score = self._calculate_pattern_score(question_lower, pattern)
            scores[task_type] = score
        
        # Get the highest scoring task type
        best_match = max(scores.items(), key=lambda x: x[1])
        
        # If no clear winner, use additional heuristics
        if best_match[1] == 0:
            return self._apply_fallback_heuristics(question_lower)
        
        return best_match[0]
    
    def _calculate_pattern_score(self, question: str, pattern: ClassificationPattern) -> float:
        """Calculate confidence score for a pattern match"""
        score = 0.0
        
        # Keyword matching
        keyword_matches = sum(1 for keyword in pattern.keywords if keyword in question)
        score += keyword_matches * pattern.weight
        
        # Prefix matching (higher weight for question starters)
        for prefix in pattern.prefixes:
            if question.startswith(prefix):
                score += 2.0 * pattern.weight
                break
        
        # Regex pattern matching
        for regex_pattern in pattern.patterns:
            if re.search(regex_pattern, question, re.IGNORECASE):
                score += 1.5 * pattern.weight
        
        return score
    
    def _apply_fallback_heuristics(self, question: str) -> TaskType:
        """Apply heuristic rules when pattern matching fails"""
        # Question words at the start typically indicate QUESTION_ANSWERING
        question_starters = ["what", "how", "when", "where", "why", "which", "who"]
        if any(question.startswith(starter) for starter in question_starters):
            return TaskType.QUESTION_ANSWERING
        
        # Modal verbs often indicate FACT_VERIFICATION
        modal_patterns = ["can ", "could ", "should ", "would ", "may ", "might "]
        if any(pattern in question for pattern in modal_patterns):
            return TaskType.FACT_VERIFICATION
        
        # Questions ending with ? are likely QUESTION_ANSWERING
        if question.endswith("?") and len(question.split()) > 2:
            return TaskType.QUESTION_ANSWERING
        
        # Default to RETRIEVAL_QUERY for unclear cases
        return TaskType.RETRIEVAL_QUERY
    
    def get_classification_confidence(self, question: str) -> Dict[str, float]:
        """
        Get confidence scores for all task types
        
        Args:
            question: Question to analyze
            
        Returns:
            Dictionary with task types and their confidence scores
        """
        question_lower = question.lower().strip()
        scores = {}
        
        for task_type, pattern in self._classification_patterns.items():
            score = self._calculate_pattern_score(question_lower, pattern)
            scores[task_type.value] = score
        
        # Normalize scores to percentages
        total_score = sum(scores.values())
        if total_score > 0:
            scores = {k: (v / total_score) * 100 for k, v in scores.items()}
        
        return scores
    
    def clear_cache(self) -> None:
        """Clear the classification cache"""
        self.classify.cache_clear()
        logger.info("🧹 Classification cache cleared")


# Global classifier instance
_classifier = TaskTypeClassifier()


def get_optimal_task_type(question: str) -> str:
    """
    Main entry point for task type classification
    
    Args:
        question: User question to classify
        
    Returns:
        Task type as string (for backward compatibility)
    """
    try:
        task_type = _classifier.classify(question)
        return task_type.value
    except Exception as e:
        logger.error(f"❌ Classification failed with unexpected error: {e}")
        return TaskType.RETRIEVAL_QUERY.value


def get_classification_confidence(question: str) -> Dict[str, float]:
    """
    Get confidence scores for all task types
    
    Args:
        question: Question to analyze
        
    Returns:
        Dictionary with task types and their confidence scores
    """
    return _classifier.get_classification_confidence(question)


def clear_classification_cache() -> None:
    """Clear the classification cache"""
    _classifier.clear_cache()


# Backward compatibility
def _fallback_classify(question: str) -> str:
    """Legacy fallback function for backward compatibility"""
    logger.warning("Using deprecated _fallback_classify function")
    return _classifier._classify_with_patterns(question).value