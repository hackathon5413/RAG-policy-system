"""Response parsing and prompt creation utilities."""

import json
import logging

from jinja2 import Environment, FileSystemLoader

logger = logging.getLogger(__name__)
jinja_env = Environment(loader=FileSystemLoader("prompts"))


def create_structured_prompt_with_mapping(question_chunk_map: list) -> str:
    template = jinja_env.get_template("insurance_query.j2")
    return template.render(question_chunk_map=question_chunk_map)


def parse_multi_question_response(response: str, questions: list[str]) -> list[str]:
    """Parse LLM response expecting strict JSON with an 'answers' list."""
    try:
        data = json.loads(response.strip())
        answers = data.get("answers", [])
        logger.info(f"📊 Parsed {len(answers)} answers from LLM response")
        for i, answer in enumerate(answers[:3]):  # Log first 3 answers
            logger.info(f"   Answer {i + 1}: {answer}...")

        # Pad missing answers
        if len(answers) < len(questions):
            raise Exception(
                f"LLM returned {len(answers)} answers but expected {len(questions)} questions"
            )
        return [str(ans) for ans in answers[: len(questions)]]
    except Exception as e:
        logger.error(f"❌ Failed to parse LLM response as JSON: {e}")
        logger.error(f"❌ Raw response: {response}")

        # Check if response looks like a valid answer but not in JSON format
        response_text = response.strip()
        if (
            response_text
            and len(response_text) > 10
            and not response_text.startswith("{")
        ):
            logger.warning(
                "📝 LLM returned valid text but not JSON format. Wrapping in JSON structure."
            )
            # For single question, wrap the response
            if len(questions) == 1:
                return [response_text]
            else:
                raise Exception("Failed to parse multi-question response as JSON")

        # If it's truly malformed, re-raise the exception
        raise e
