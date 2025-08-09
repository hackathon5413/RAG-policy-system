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

from config import config

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
def call_gemini(prompt: str) -> str:
    try:
        api_key, key_num = api_rotator.get_next_key()
        logging.info(f"🔑 [LLM] Using API key #{key_num}")

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.1,
                "topP": 0.9,
                "maxOutputTokens": 10000,
            },
        }

        headers = {"Content-Type": "application/json"}

        url_with_key = f"{config.gemini_url}?key={api_key}"
        response = requests.post(
            url_with_key, json=payload, headers=headers, timeout=30
        )

        if response.status_code == 200:
            response_data = response.json()
            if response_data.get("candidates"):
                raw_response = response_data["candidates"][0]["content"]["parts"][0][
                    "text"
                ]

                if not raw_response or raw_response.strip() == "":
                    logging.error(
                        f"❌ GEMINI RETURNED EMPTY TEXT! Full response: {response_data}"
                    )
                    return "Error: AI model returned empty response"

                return raw_response.strip()
            else:
                logging.error(f"❌ No candidates in Gemini response: {response_data}")
                return "No response generated"

        elif response.status_code in [429, 503]:
            logging.warning(
                f"⚠️ Rate limited (HTTP {response.status_code}), tenacity will retry with different key"
            )
            raise RateLimitError(f"Rate limited: HTTP {response.status_code}")

        elif response.status_code in [500, 502, 504]:
            logging.warning(
                f"⚠️ Server error (HTTP {response.status_code}), tenacity will retry"
            )
            raise ServerErrorError(f"Server error: HTTP {response.status_code}")

        else:
            response.raise_for_status()

    except (RateLimitError, ServerErrorError):
        raise
    except requests.exceptions.Timeout:
        logging.warning("⏰ Request timeout, tenacity will retry")
        raise requests.exceptions.Timeout("Request timeout")
    except requests.exceptions.RequestException as e:
        logging.warning(f"🌐 Connection error, tenacity will retry: {e}")
        raise e
    except Exception as e:
        logging.error(f"❌ Unexpected error: {e}")
        return f"Error generating response: {e}"

    return "Error: Unexpected response handling"
