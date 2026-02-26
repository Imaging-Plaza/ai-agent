from __future__ import annotations

import json
import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class ChatState:
    """Encapsulates all conversation state for the agent."""
    conversation_history: List[str] = field(default_factory=list)
    banlist: set = field(default_factory=set)
    last_choices: Dict[str, Any] = field(default_factory=dict)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    pending_demo_tool: Optional[str] = None
    pending_demo_url: Optional[str] = None
    last_preview_path: Optional[str] = None
    last_files: List[str] = field(default_factory=list)
    last_image_meta: Optional[str] = None
    
    # Tool approval system
    pending_tool_approval: Optional[str] = None  # Tool name waiting for approval
    pending_tool_params: Dict[str, Any] = field(default_factory=dict)  # Tool parameters
    agent_result: Optional[Dict[str, Any]] = None  # Cached agent result before tool execution
    
    def to_dict(self) -> dict:
        """Serialize state for Gradio State component."""
        return {
            "conversation_history": self.conversation_history,
            "banlist": list(self.banlist),
            "last_choices": self.last_choices,
            "tool_calls": self.tool_calls,
            "pending_demo_tool": self.pending_demo_tool,
            "pending_demo_url": self.pending_demo_url,
            "last_preview_path": self.last_preview_path,
            "last_files": self.last_files,
            "last_image_meta": self.last_image_meta,
            "pending_tool_approval": self.pending_tool_approval,
            "pending_tool_params": self.pending_tool_params,
            "agent_result": self.agent_result,
        }
    
    @staticmethod
    def from_dict(d: dict) -> 'ChatState':
        """Deserialize state from Gradio State component."""
        if not d:
            return ChatState()
        return ChatState(
            conversation_history=d.get("conversation_history", []),
            banlist=set(d.get("banlist", [])),
            last_choices=d.get("last_choices", {}),
            tool_calls=d.get("tool_calls", []),
            pending_demo_tool=d.get("pending_demo_tool"),
            pending_demo_url=d.get("pending_demo_url"),
            last_preview_path=d.get("last_preview_path"),
            last_files=d.get("last_files", []),
            last_image_meta=d.get("last_image_meta"),
            pending_tool_approval=d.get("pending_tool_approval"),
            pending_tool_params=d.get("pending_tool_params", {}),
            agent_result=d.get("agent_result"),
        )


@dataclass
class ChatMessage:
    """Represents a rich message in the chat."""
    text: str = ""
    images: List[str] = field(default_factory=list)  # file paths
    files: List[Tuple[str, str]] = field(default_factory=list)  # (path, label)
    json_data: Optional[Dict[str, Any]] = None
    code_blocks: List[Tuple[str, str]] = field(default_factory=list)  # (lang, code)
    tool_traces: List[Dict[str, Any]] = field(default_factory=list)
    stats: Optional[Dict[str, Any]] = None  # Performance stats (time, tokens, etc.)
    
    def to_markdown(self) -> str:
        """Convert message to markdown with media."""
        parts = []
        
        if self.text:
            parts.append(self.text)
        
        # Render stats if available
        if self.stats:
            parts.append("\n---\n**📊 Performance Stats:**\n")
            if "compute_time" in self.stats:
                parts.append(f"⏱️ Compute time: {self.stats['compute_time']:.2f}s\n")
            if "total_time" in self.stats:
                parts.append(f"⏱️ Total time: {self.stats['total_time']:.2f}s\n")
            if "tokens" in self.stats:
                tok = self.stats["tokens"]
                parts.append(f"🎫 Tokens: {tok.get('total', 0)} (in: {tok.get('input', 0)}, out: {tok.get('output', 0)})\n")
        
        # Render file links
        for file_path, label in self.files:
            if os.path.exists(file_path):
                parts.append(f"\n📎 [{label}]({file_path})")
        
        # Render JSON in code block
        if self.json_data:
            json_str = json.dumps(self.json_data, indent=2)
            parts.append(f"\n```json\n{json_str}\n```")
        
        # Render code blocks
        for lang, code in self.code_blocks:
            parts.append(f"\n```{lang}\n{code}\n```")
        
        return "\n".join(parts)
