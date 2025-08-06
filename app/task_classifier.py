import logging
import re
from enum import Enum
from typing import Optional, Dict, List
from dataclasses import dataclass
from jinja2 import Template

from .rag_core import call_gemini

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Enumeration of supported task types."""
    QUESTION_ANSWERING = "QUESTION_ANSWERING"
    FACT_VERIFICATION = "FACT_VERIFICATION"
    RETRIEVAL_QUERY = "RETRIEVAL_QUERY"


@dataclass
class ClassificationPattern:
    """Pattern definition for fallback classification."""
    keywords: List[str]
    prefixes: List[str]
    patterns: List[str]
    weight: float = 1.0


class TaskTypeClassifier:
    """Enhanced task type classifier using LLM and pattern-based fallback."""

    def __init__(self):
        self._classification_patterns = self._init_patterns()
        self._gemini_template = self._build_llm_template()

    def _init_patterns(self) -> Dict[TaskType, ClassificationPattern]:
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
                weight=3.0
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
                weight=2.0
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

    def _build_llm_template(self) -> Template:
        return Template(
            """
<task>
Classify the following question into exactly ONE category based on the user's intent.
</task>

<categories>
<category name="QUESTION_ANSWERING">
<description>Questions seeking explanations, procedures, or factual information</description>
<examples>
- What is term life insurance?
- How do I file a claim?
- When does my policy expire?
</examples>
</category>

<category name="FACT_VERIFICATION">
<description>Questions about eligibility, coverage, or specific conditions</description>
<examples>
- Can I add my spouse to this policy?
- Am I eligible for this coverage?
- Do I have remaining benefits?
</examples>
</category>

<category name="RETRIEVAL_QUERY">
<description>General search requests or unclear questions</description>
<examples>
- Find information about health insurance
- Tell me about available policies
</examples>
</category>
</categories>

<question>{{ question }}</question>

<instructions>
Respond with ONLY the category name.
</instructions>
            """.strip()
        )

    def classify(self, question: str) -> TaskType:
        if not question or not question.strip():
            logger.warning("Empty question provided, defaulting to RETRIEVAL_QUERY")
            return TaskType.RETRIEVAL_QUERY

        question = question.strip()

        llm_result = self._classify_with_llm(question)
        if llm_result:
            logger.info(f"🎯 LLM classified as {llm_result.value}")
            return llm_result

        pattern_result = self._classify_with_patterns(question)
        logger.info(f"🔄 Pattern classified as {pattern_result.value}")
        return pattern_result

    def _classify_with_llm(self, question: str) -> Optional[TaskType]:
        try:
            prompt = self._gemini_template.render(question=question)
            response = call_gemini(prompt).strip().upper()

            for task_type in TaskType:
                if response == task_type.value:
                    return task_type

            logger.warning(f"⚠️ Unexpected LLM response: {response}")
        except Exception as e:
            logger.error(f"❌ LLM classification error: {e}")
        return None

    def _classify_with_patterns(self, question: str) -> TaskType:
        scores = {
            task_type: self._score_pattern(question.lower(), pattern)
            for task_type, pattern in self._classification_patterns.items()
        }

        best_type, best_score = max(scores.items(), key=lambda item: item[1])
        if best_score == 0:
            return self._fallback_heuristic(question.lower())
        return best_type

    def _score_pattern(self, question: str, pattern: ClassificationPattern) -> float:
        score = sum(1 for kw in pattern.keywords if kw in question) * pattern.weight

        if any(question.startswith(pre) for pre in pattern.prefixes):
            score += 2.0 * pattern.weight

        for regex in pattern.patterns:
            if re.search(regex, question, re.IGNORECASE):
                score += 1.5 * pattern.weight

        return score

    def _fallback_heuristic(self, question: str) -> TaskType:
        starters = ["what", "how", "when", "where", "why", "which", "who"]
        if any(question.startswith(w) for w in starters):
            return TaskType.QUESTION_ANSWERING

        if any(modal in question for modal in ["can ", "could ", "should ", "would ", "may ", "might"]):
            return TaskType.FACT_VERIFICATION

        if question.endswith("?") and len(question.split()) > 2:
            return TaskType.QUESTION_ANSWERING

        return TaskType.RETRIEVAL_QUERY


_classifier = TaskTypeClassifier()

def get_optimal_task_type(question: str) -> str:
    try:
        return _classifier.classify(question).value
    except Exception as e:
        logger.error(f"❌ Classification failed: {e}")
        return TaskType.RETRIEVAL_QUERY.value
