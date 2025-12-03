from __future__ import annotations

import json
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
    
    def to_markdown(self) -> str:
        """Convert message to markdown with media."""
        parts = []
        
        if self.text:
            parts.append(self.text)
        
        # Render file links
        for file_path, label in self.files:
            import os
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
