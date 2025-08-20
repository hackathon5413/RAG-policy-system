import json
import os
import re
from typing import Any

import requests
from google import genai
from jinja2 import Environment, FileSystemLoader
from langchain.tools import BaseTool
from pydantic import BaseModel

from .utils.models import (
    AgentConfig,
    APICallInput,
    CalculationInput,
    SearchChunk,
    ToolResult,
    WebScrapeInput,
)

API_KEY = os.getenv("GEMINI_API_KEY")


class APICallTool(BaseTool):
    """Langchain tool for making API calls"""

    name: str = "api_call"
    description: str = "Make HTTP requests to APIs found in document content"
    args_schema: type[BaseModel] | dict[str, Any] | None = APICallInput

    def _run(
        self, url: str, method: str = "GET", params: dict | None = None
    ) -> ToolResult:
        """Execute the API call"""
        try:
            response = requests.request(method, url, params=params, timeout=10)
            response.raise_for_status()
            return ToolResult(tool_name="api_call", success=True, data=response.json())
        except Exception as e:
            return ToolResult(
                tool_name="api_call", success=False, data=None, error=str(e)
            )


class CalculationTool(BaseTool):
    """Langchain tool for performing calculations"""

    name: str = "calculation"
    description: str = "Perform mathematical calculations and formula evaluation"
    args_schema: type[BaseModel] | dict[str, Any] | None = CalculationInput

    def _run(self, expression: str) -> ToolResult:
        """Execute the calculation"""
        try:
            # Safely evaluate mathematical expressions
            allowed_chars = set("0123456789+-*/.() ")
            if not all(c in allowed_chars for c in expression):
                raise ValueError("Invalid characters in expression")
            result = eval(expression, {"__builtins__": {}})
            return ToolResult(tool_name="calculation", success=True, data=result)
        except Exception as e:
            return ToolResult(
                tool_name="calculation", success=False, data=None, error=str(e)
            )


class WebScrapeTool(BaseTool):
    """Langchain tool for web scraping"""

    name: str = "web_scrape"
    description: str = "Extract data from web pages referenced in content"
    args_schema: type[BaseModel] | dict[str, Any] | None = WebScrapeInput

    def _run(self, url: str) -> ToolResult:
        """Execute the web scraping"""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return ToolResult(
                tool_name="web_scrape", success=True, data=response.text[:1000]
            )
        except Exception as e:
            return ToolResult(
                tool_name="web_scrape", success=False, data=None, error=str(e)
            )


# Initialize tools
api_call_tool = APICallTool()
calculation_tool = CalculationTool()
web_scrape_tool = WebScrapeTool()

LANGCHAIN_TOOLS = [api_call_tool, calculation_tool, web_scrape_tool]

TOOL_REGISTRY = {
    "api_call": {
        "description": "Make HTTP requests to APIs found in document content",
        "when_to_use": "When content contains API endpoints, URLs, or references to external services",
        "tool": api_call_tool,
    },
    "calculation": {
        "description": "Perform mathematical calculations and formula evaluation",
        "when_to_use": "When content contains numbers, formulas, or calculation requirements",
        "tool": calculation_tool,
    },
    "web_scrape": {
        "description": "Extract data from web pages referenced in content",
        "when_to_use": "When content references external websites or web resources",
        "tool": web_scrape_tool,
    },
}


def extract_urls(text: str) -> list[str]:
    """Extract URLs from text content"""
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    return re.findall(url_pattern, text)


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


def execute_tools_with_langchain(
    tools: list[str], content: str
) -> dict[str, ToolResult]:
    """Execute selected Langchain tools on content"""
    results = {}

    for tool_name in tools:
        if tool_name == "api_call":
            urls = extract_urls(content)
            if urls:
                results[tool_name] = api_call_tool.run({"url": urls[0]})

        elif tool_name == "calculation":
            numbers = re.findall(r"\d+\.?\d*", content)
            if len(numbers) >= 2:
                expr = f"{numbers[0]} + {numbers[1]}"
                results[tool_name] = calculation_tool.run({"expression": expr})

        elif tool_name == "web_scrape":
            urls = extract_urls(content)
            if urls:
                results[tool_name] = web_scrape_tool.run({"url": urls[0]})

    return results


def enhance_chunks_with_tools(
    chunks: list[SearchChunk], question: str
) -> list[SearchChunk]:
    """Enhance search chunks with tool results using Langchain tools"""
    enhanced_chunks = []

    for chunk in chunks:
        tools_needed = plan_tools(chunk.content, question)

        if tools_needed:
            tool_results = execute_tools_with_langchain(tools_needed, chunk.content)

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


class AgenticRAGProcessor:
    """Main class for agentic RAG processing with Langchain tools"""

    def __init__(self, config: AgentConfig = AgentConfig()):
        self.config = config
        self.tools = LANGCHAIN_TOOLS
        self.tool_registry = TOOL_REGISTRY

    def process_chunks(
        self, chunks: list[SearchChunk], question: str
    ) -> list[SearchChunk]:
        """Process chunks with agentic capabilities"""
        return enhance_chunks_with_tools(chunks, question)

    def get_available_tools(self) -> list[str]:
        """Get list of available tool names"""
        return list(self.tool_registry.keys())

    def execute_single_tool(self, tool_name: str, **kwargs) -> ToolResult:
        """Execute a single tool with given parameters"""
        if tool_name not in self.tool_registry:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                data=None,
                error=f"Tool '{tool_name}' not found",
            )

        tool_instance = self.tool_registry[tool_name]["tool"]
        if not isinstance(tool_instance, BaseTool):
            return ToolResult(
                tool_name=tool_name,
                success=False,
                data=None,
                error=f"Invalid tool type for '{tool_name}'",
            )

        try:
            return tool_instance.run(kwargs)
        except Exception as e:
            return ToolResult(
                tool_name=tool_name, success=False, data=None, error=str(e)
            )
