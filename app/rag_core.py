import json
import logging
import os
import threading

import requests
from dotenv import load_dotenv
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_delay,
    wait_exponential_jitter,
)
from ollama import Client


load_dotenv()


class RateLimitError(Exception):
    pass


class ServerErrorError(Exception):
    pass


class GeminiAPIRotator:
    def __init__(self):
        self.api_keys = [os.getenv(f"GEMINI_API_KEY_{i}") for i in range(1, 44)]
        self.api_keys = [
            key for key in self.api_keys if key and not key.startswith("YOUR_API_KEY")
        ]

        if not self.api_keys:
            fallback_key = os.getenv("GEMINI_API_KEY")
            if fallback_key:
                self.api_keys = [fallback_key]
        self.current_index = 0
        self.lock = threading.Lock()

    def get_next_key(self):
        with self.lock:
            key = self.api_keys[self.current_index]
            key_num = self.current_index + 1
            self.current_index = (self.current_index + 1) % len(self.api_keys)
            return key, key_num


api_rotator = GeminiAPIRotator()


def load_keywords():
    with open("./config/section_keywords.json") as f:
        section_keywords = json.load(f)
    return section_keywords


section_keywords = load_keywords()


def classify_section(text: str) -> str:
    text_lower = text.lower()

    for section_type, keywords in section_keywords.items():
        if any(keyword in text_lower for keyword in keywords):
            return section_type

    return "general"


@retry(
    stop=stop_after_delay(420),  # 7 minutes maximum
    wait=wait_exponential_jitter(
        initial=1, max=30, jitter=2
    ),  # 1s, 2s, 4s, 8s... up to 30s with jitter
    retry=retry_if_exception_type(
        (RateLimitError, ServerErrorError, requests.exceptions.RequestException)
    ),
    reraise=True,
)
def call_gemini(prompt: str, turbo: bool = True) -> str:
    """
    Generate a response using the local open-source GPT model via Ollama.
    Turbo mode is enabled by default for faster responses.
    """
    try:
        model_name = "gpt-oss:20b" if turbo else "gpt-oss:20b"
        logging.info(f"🦾 [LLM] Using model: {model_name} (Turbo={turbo})")

        client = Client(
            host="https://ollama.com",
            headers={'Authorization': '96a54c567d8c4a90a60497bd2c4e87e6.QdoVFFd_LlRGUR_S-iqR_a47'}
        )
        response = client.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
        )

        # Ollama's response structure: {'message': {'role': 'assistant', 'content': ...}}
        raw_response = ""
        if isinstance(response, dict) and "message" in response:
            raw_response = response["message"].get("content", "")
        elif hasattr(response, "message") and hasattr(response.message, "content"):
            raw_response = response.message.content

        if not raw_response or raw_response.strip() == "":
            logging.error(f"❌ LLM returned empty text! Full response: {response}")
            return "Error: AI model returned empty response"
        
        return raw_response.strip()

    except Exception as e:
        logging.error(f"❌ Unexpected error: {e}")
        return f"Error generating response: {e}"

    return "Error: Unexpected response handling"
