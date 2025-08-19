"""Agentic RAG core system with LLM-driven tool selection"""

import json
import os
import re
from dataclasses import dataclass
from typing import Any

import requests
from google import genai
from jinja2 import Environment, FileSystemLoader

API_KEY = os.getenv("GEMINI_API_KEY")


@dataclass
class ToolResult:
    tool_name: str
    success: bool
    data: Any
    error: str | None = None


@dataclass
class SearchChunk:
    content: str
    url: str
    metadata: dict[str, Any]


TOOL_REGISTRY = {
    "api_call": {
        "description": "Make HTTP requests to APIs found in document content",
        "when_to_use": "When content contains API endpoints, URLs, or references to external services",
        "function": "make_api_call",
    },
    "calculation": {
        "description": "Perform mathematical calculations and formula evaluation",
        "when_to_use": "When content contains numbers, formulas, or calculation requirements",
        "function": "perform_calculation",
    },
    "web_scrape": {
        "description": "Extract data from web pages referenced in content",
        "when_to_use": "When content references external websites or web resources",
        "function": "scrape_web_data",
    },
}


def extract_urls(text: str) -> list[str]:
    """Extract URLs from text content"""
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    return re.findall(url_pattern, text)


def make_api_call(
    url: str, method: str = "GET", params: dict | None = None
) -> ToolResult:
    """Make HTTP API call"""
    try:
        response = requests.request(method, url, params=params, timeout=10)
        response.raise_for_status()
        return ToolResult("api_call", True, response.json())
    except Exception as e:
        return ToolResult("api_call", False, None, str(e))


def perform_calculation(expression: str) -> ToolResult:
    """Safely evaluate mathematical expressions"""
    try:
        allowed_chars = set("0123456789+-*/.() ")
        if not all(c in allowed_chars for c in expression):
            raise ValueError("Invalid characters in expression")
        result = eval(expression, {"__builtins__": {}})
        return ToolResult("calculation", True, result)
    except Exception as e:
        return ToolResult("calculation", False, None, str(e))


def scrape_web_data(url: str) -> ToolResult:
    """Extract data from web page"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return ToolResult("web_scrape", True, response.text[:1000])
    except Exception as e:
        return ToolResult("web_scrape", False, None, str(e))


def plan_tools(content: str, question: str) -> list[str]:
    """Use LLM to determine which tools to use"""
    env = Environment(loader=FileSystemLoader("prompts"))
    template = env.get_template("tool_planner.j2")
    prompt = template.render(tools=TOOL_REGISTRY, content=content, question=question)

    try:
        client = genai.Client(api_key=API_KEY)
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp", contents=prompt
        )
        if response and response.text:
            tools = json.loads(response.text.strip())
            return [tool for tool in tools if tool in TOOL_REGISTRY]
        return []
    except Exception:
        return []


def execute_tools(tools: list[str], content: str) -> dict[str, ToolResult]:
    """Execute selected tools on content"""
    results = {}

    for tool_name in tools:
        if tool_name == "api_call":
            urls = extract_urls(content)
            if urls:
                results[tool_name] = make_api_call(urls[0])

        elif tool_name == "calculation":
            numbers = re.findall(r"\d+\.?\d*", content)
            if len(numbers) >= 2:
                expr = f"{numbers[0]} + {numbers[1]}"
                results[tool_name] = perform_calculation(expr)

        elif tool_name == "web_scrape":
            urls = extract_urls(content)
            if urls:
                results[tool_name] = scrape_web_data(urls[0])

    return results


def enhance_chunks_with_tools(
    chunks: list[SearchChunk], question: str
) -> list[SearchChunk]:
    """Enhance search chunks with tool results"""
    enhanced_chunks = []

    for chunk in chunks:
        tools_needed = plan_tools(chunk.content, question)

        if tools_needed:
            tool_results = execute_tools(tools_needed, chunk.content)

            enhanced_content = chunk.content
            for tool_name, result in tool_results.items():
                if result.success:
                    enhanced_content += f"\n[{tool_name.upper()}_RESULT]: {result.data}"

            enhanced_chunk = SearchChunk(
                content=enhanced_content,
                url=chunk.url,
                metadata={**chunk.metadata, "tools_used": list(tool_results.keys())},
            )
            enhanced_chunks.append(enhanced_chunk)
        else:
            enhanced_chunks.append(chunk)

    return enhanced_chunks
