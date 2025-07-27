

import json
import requests
import os
from typing import Dict, Any
from config import config

def load_keywords():
    """Load section and cleanup keywords"""
    with open('./config/section_keywords.json', 'r') as f:
        section_keywords = json.load(f)
    with open('./config/cleanup_keywords.json', 'r') as f:
        cleanup_keywords = json.load(f)
    return section_keywords, cleanup_keywords

# Load keywords once at import time
section_keywords, cleanup_keywords = load_keywords()

def classify_section(text: str) -> str:
    """Classify text into section type"""
    text_lower = text.lower()
    
    for section_type, keywords in section_keywords.items():
        if any(keyword in text_lower for keyword in keywords):
            return section_type
    
    return "general"

def clean_text(text: str) -> str:
    """Clean text by removing headers, footers, irrelevant content"""
    lines = text.split('\n')
    cleaned_lines = []
    
    # Get all skip patterns
    all_skip_patterns = []
    for category in cleanup_keywords.values():
        all_skip_patterns.extend(category)
    
    for line in lines:
        line = line.strip()
        if not line or len(line) < 10:
            continue
        if any(skip in line.lower() for skip in all_skip_patterns):
            continue
        cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)

def call_gemini(prompt: str) -> str:
    """Call Gemini API with environment variable for API key"""
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return "Error: GEMINI_API_KEY environment variable not set"
        
        payload = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }],
            "generationConfig": {
                "temperature": 0.1,
                "topP": 0.9,
                "maxOutputTokens": 2048
            }
        }

        headers = {
            "Content-Type": "application/json"
        }

        url_with_key = f"{config.gemini_url}?key={api_key}"
        response = requests.post(url_with_key, json=payload, headers=headers, timeout=30)
        response.raise_for_status()

        response_data = response.json()
        if "candidates" in response_data and response_data["candidates"]:
            return response_data["candidates"][0]["content"]["parts"][0]["text"]
        else:
            return "No response generated"

    except requests.exceptions.RequestException as e:
        return f"Error connecting to Gemini: {e}"
    except Exception as e:
        return f"Error generating response: {e}"
