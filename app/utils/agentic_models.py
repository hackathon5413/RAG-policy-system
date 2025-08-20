from typing import Any

from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    """Pydantic model for tool execution results"""

    tool_name: str = Field(description="Name of the executed tool")
    success: bool = Field(description="Whether the tool execution was successful")
    data: Any = Field(description="Data returned by the tool")
    error: str | None = Field(
        default=None, description="Error message if execution failed"
    )


class SearchChunk(BaseModel):
    """Pydantic model for search chunk data"""

    content: str = Field(description="The content of the search chunk")
    url: str = Field(description="URL source of the content")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class APICallInput(BaseModel):
    """Input schema for API call tool"""

    url: str = Field(description="The URL to make the API call to")
    method: str = Field(default="GET", description="HTTP method to use")
    params: dict | None = Field(
        default=None, description="Query parameters for the request"
    )


class CalculationInput(BaseModel):
    """Input schema for calculation tool"""

    expression: str = Field(description="Mathematical expression to evaluate")


class WebScrapeInput(BaseModel):
    """Input schema for web scraping tool"""

    url: str = Field(description="The URL to scrape data from")


class AgentConfig(BaseModel):
    """Configuration for the agentic RAG system"""

    model_name: str = Field(
        default="gemini-2.0-flash-exp", description="LLM model to use"
    )
    max_tools_per_chunk: int = Field(
        default=3, description="Maximum tools to execute per chunk"
    )
    tool_timeout: int = Field(
        default=10, description="Timeout for tool execution in seconds"
    )
    enable_parallel_execution: bool = Field(
        default=False, description="Enable parallel tool execution"
    )
