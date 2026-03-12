from __future__ import annotations

from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field

from ai_agent.generator.schema import ToolSelection, CandidateDoc

class ToolRunLog(BaseModel):
    tool: str
    inputs: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    timestamp: Optional[str] = None

class UsageStats(BaseModel):
    """Token usage statistics from the agent."""
    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

class AgentToolSelection(ToolSelection):
    tool_calls: List[ToolRunLog] = Field(default_factory=list)
    usage: Optional[UsageStats] = None

    def to_legacy_dict(self) -> Dict[str, Any]:
        # Map to legacy pipeline result shape expected by UI (subset)
        return {
            "conversation": self.conversation.model_dump(mode="python"),
            "choices": [c.model_dump(mode="python") for c in self.choices],
            "reason": self.reason,
            "explanation": self.explanation,
            "tool_calls": [c.model_dump(mode="python") for c in self.tool_calls],
            "usage": self.usage.model_dump(mode="python") if self.usage else None,
        }

__all__ = [
    "AgentToolSelection",
    "ToolRunLog",
    "UsageStats",
    "CandidateDoc",
]
