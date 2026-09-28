from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    reported: bool = False


class ToolRequest(BaseModel):
    id: str
    name: str
    arguments: Dict[str, Any]
    provider_metadata: Dict[str, Any] = Field(default_factory=dict)


class NormalizedResponse(BaseModel):
    text: Optional[str] = None
    tool_requests: List[ToolRequest] = Field(default_factory=list)
    stop_reason: Optional[str] = None
    usage: TokenUsage


class ModelClient(ABC):
    @abstractmethod
    def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> NormalizedResponse:
        """Generates a normalized response from the LLM provider."""
        pass
