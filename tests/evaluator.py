"""
Answer Evaluator Module
Handles LLM-based evaluation of model answers against expected answers
"""

import json
import logging
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

logger = logging.getLogger(__name__)


class AnswerEvaluator:
    """Evaluates model answers against expected answers using LLM-based scoring"""

    def __init__(self):
        # Setup Jinja environment for prompts
        templates_path = Path(__file__).parent / "prompts"
        self.jinja_env = Environment(loader=FileSystemLoader(str(templates_path)))

    def _create_batch_evaluation_prompt(self, questions_and_answers: list[dict[str, str]], expected_answers: list[str]) -> str:
        """Create a batch evaluation prompt using Jinja template"""
        template = self.jinja_env.get_template('answer_evaluator.j2')
        return template.render(
            batch_mode=True,
            questions_and_answers=questions_and_answers,
            expected_answers=expected_answers
        )

    async def evaluate_multiple_answers(self, questions: list[str], expected_answers: list[str], actual_answers: list[str]) -> dict[str, Any]:
        """Evaluate multiple answers using single batch LLM call"""
        if len(questions) != len(expected_answers) or len(questions) != len(actual_answers):
            raise ValueError("Questions, expected answers, and actual answers must have the same length")

        # Filter out questions without expected answers for batch evaluation
        questions_and_answers = []
        valid_indices = []

        for i, (question, expected, actual) in enumerate(zip(questions, expected_answers, actual_answers, strict=True)):
            if expected.strip():
                questions_and_answers.append({
                    "question": question,
                    "actual_answer": actual,
                    "index": i + 1
                })
                valid_indices.append(i)

        evaluations = []

        if questions_and_answers:
            try:
                # Import here to avoid circular imports
                from app.rag_core import call_gemini

                # Get expected answers for valid questions only
                valid_expected_answers = [expected_answers[i] for i in valid_indices]

                # Create batch evaluation prompt
                prompt = self._create_batch_evaluation_prompt(questions_and_answers, valid_expected_answers)

                # Single LLM call for all questions
                response = call_gemini(prompt)

                # Clean up the response if it has markdown formatting
                if response.startswith("```json"):
                    response = response[7:-3].strip()
                elif response.startswith("```"):
                    response = response[3:-3].strip()

                batch_result = json.loads(response)

                # Process batch results
                if "evaluations" in batch_result:
                    batch_evaluations = batch_result["evaluations"]
                else:
                    batch_evaluations = batch_result if isinstance(batch_result, list) else [batch_result]

                # Create evaluations list with all questions
                for i, (question, expected, actual) in enumerate(zip(questions, expected_answers, actual_answers, strict=True)):
                    if expected.strip() and i in valid_indices:
                        # Find corresponding evaluation from batch result
                        valid_index = valid_indices.index(i)
                        if valid_index < len(batch_evaluations):
                            evaluation = batch_evaluations[valid_index]
                            evaluation["question_index"] = i + 1
                            evaluation["question"] = question
                            evaluation["actual_answer"] = actual
                            evaluation["expected_answer"] = expected
                        else:
                            evaluation = {
                                "accuracy_score": 0,
                                "intent_fulfilled": False,
                                "explanation": "Evaluation failed in batch processing",
                                "key_points_covered": 0,
                                "areas_for_improvement": "Batch evaluation error",
                                "question_index": i + 1,
                                "question": question,
                                "actual_answer": actual,
                                "expected_answer": expected
                            }
                    else:
                        evaluation = {
                            "accuracy_score": -1,
                            "intent_fulfilled": None,
                            "explanation": "No expected answer provided for evaluation",
                            "key_points_covered": -1,
                            "areas_for_improvement": "Add expected answer to enable evaluation",
                            "question_index": i + 1,
                            "question": question,
                            "actual_answer": actual,
                            "expected_answer": expected
                        }
                    evaluations.append(evaluation)

            except Exception as e:
                logger.error(f"Batch evaluation failed: {e}")
                # Create default evaluations for all questions since batch failed
                for i, (question, expected, actual) in enumerate(zip(questions, expected_answers, actual_answers, strict=True)):
                    evaluation = {
                        "accuracy_score": 0 if expected.strip() else -1,
                        "intent_fulfilled": False if expected.strip() else None,
                        "explanation": f"Batch evaluation failed: {e}" if expected.strip() else "No expected answer provided for evaluation",
                        "key_points_covered": 0 if expected.strip() else -1,
                        "areas_for_improvement": "Retry evaluation" if expected.strip() else "Add expected answer to enable evaluation",
                        "question_index": i + 1,
                        "question": question,
                        "actual_answer": actual,
                        "expected_answer": expected
                    }
                    evaluations.append(evaluation)
        else:
            # No valid questions to evaluate
            for i, (question, expected, actual) in enumerate(zip(questions, expected_answers, actual_answers, strict=True)):
                evaluations.append({
                    "accuracy_score": -1,
                    "intent_fulfilled": None,
                    "explanation": "No expected answer provided for evaluation",
                    "key_points_covered": -1,
                    "areas_for_improvement": "Add expected answer to enable evaluation",
                    "question_index": i + 1,
                    "question": question,
                    "actual_answer": actual,
                    "expected_answer": expected
                })

        # Calculate overall statistics
        valid_evaluations = [e for e in evaluations if e["accuracy_score"] >= 0]
        total_score = sum(e["accuracy_score"] for e in valid_evaluations)
        total_intent_fulfilled = sum(1 for e in valid_evaluations if e["intent_fulfilled"])

        if valid_evaluations:
            avg_accuracy = total_score / len(valid_evaluations)
            intent_fulfillment_rate = (total_intent_fulfilled / len(valid_evaluations)) * 100
        else:
            avg_accuracy = 0
            intent_fulfillment_rate = 0

        return {
            "evaluations": evaluations,
            "summary": {
                "total_questions": len(questions),
                "evaluated_questions": len(valid_evaluations),
                "skipped_questions": len(questions) - len(valid_evaluations),
                "average_accuracy": round(avg_accuracy, 2),
                "intent_fulfillment_rate": round(intent_fulfillment_rate, 2),
                "questions_with_intent_fulfilled": total_intent_fulfilled
            }
        }
