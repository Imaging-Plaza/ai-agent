# generator/generator.py
from __future__ import annotations
import json, os, logging
from typing import List, Optional, Dict, Any
from openai import OpenAI
from datetime import datetime
from pathlib import Path
import base64

from generator.schema import CandidateDoc, ToolSelection
from generator.prompts import SELECTOR_SYSTEM
from utils.image_analyzer import _to_supported_png_dataurl as to_data_url


# -------- VLM selector --------
class VLMToolSelector:
    """
    Single-call VLM selector:
      (text + image + candidates + original-image metadata) -> {choice, alternates, why}
    Writes a readable prompt + the exact PNG sent to the model when LOG_PROMPTS=1.
    """
    def __init__(self, model: Optional[str] = None):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = (
            model
            or os.getenv("OPENAI_MODEL")
            or "gpt-4o-mini"
        )
        self.log = logging.getLogger("selector.vlm")
        self.last_usage: Optional[dict] = None
        self.last_logfile: Optional[str] = None


    @staticmethod
    def _save_prompt(kind: str, model: str, system: str, user_text: str,
                    data_url: Optional[str], response_text: Optional[str] = None,
                    usage: Optional[dict] = None) -> Optional[str]:
        """Save a .txt with system+user (+ image note) and, if present, the response and usage."""
        if str(os.getenv("LOG_PROMPTS", "")).lower() not in ("1", "true", "yes", "on"):
            return None

        Path("logs").mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        txt_path = Path("logs") / f"{kind}_{ts}.txt"
        png_path = None

        if data_url and data_url.startswith("data:image/png;base64,"):
            try:
                img_b64 = data_url.split(",", 1)[1]
                img_bytes = base64.b64decode(img_b64)
                png_path = Path("logs") / f"{kind}_{ts}.png"
                png_path.write_bytes(img_bytes)
            except Exception:
                png_path = None

        with txt_path.open("w", encoding="utf-8") as f:
            f.write(f"MODEL: {model}\n\n")
            f.write("--- SYSTEM ---\n" + system.strip() + "\n\n")
            f.write("--- USER (text) ---\n" + user_text.strip() + "\n\n")
            if png_path:
                f.write(f"--- IMAGE ---\nSaved PNG: {png_path}\n\n")
            elif data_url:
                f.write(f"--- IMAGE ---\n(data-url length: {len(data_url)})\n\n")
            if response_text is not None:
                f.write("--- RESPONSE ---\n" + response_text.strip() + "\n\n")
            if usage:
                f.write("--- USAGE ---\n" + str(usage) + "\n")
        return str(txt_path)

    def select(
        self,
        user_task: str,
        candidates: List["CandidateDoc"],
        image_path: Optional[str],
        image_meta: Optional[str] = None,
    ) -> "ToolSelection":
        """
        Build one chat.completions request with:
        - system: selection rules
        - user: text block (task + candidates + optional metadata) + optional image (PNG data URL)
        Returns ToolSelection(choice, alternates, why).
        Robust to empty/non-JSON responses and always logs prompt (+ response if available).
        """
        # ---- Build compact candidate list
        def _csv(seq) -> str:
            return ",".join(str(x) for x in (seq or []) if x not in (None, ""))

        def fmt(c: CandidateDoc) -> str:
            dims_str = ",".join(f"{d}D" for d in (c.dims or []))
            return (
                f"- {c.name or ''} | "
                f"tasks={_csv(c.tasks)} | "
                f"modality={_csv(c.modality)} | "
                f"dims={dims_str} | "
                f"lang={c.programming_language or ''} | "
                f"gpu={'' if c.gpu_required is None else c.gpu_required} | "
                f"url={c.url or ''}"
            )

        cand_block = "\n".join(fmt(c) for c in candidates)
        meta_note = (
            "NOTE: Prefer candidates matching the original file extension indicated in 'Image metadata'.\n"
            if image_meta else ""
        )
        meta_block = f"\nImage metadata: {image_meta}" if image_meta else ""
        user_text = (
            f"User task: {user_task}\n"
            f"{meta_note}"
            f"Candidates:\n{cand_block}\n"
            f"Return STRICT JSON with:\n"
            f"1. 'conversation.status': Must be 'needs_clarification' if task is unclear\n"
            f"2. 'choices': Up to {os.getenv('NUM_CHOICES', 3)} ranked tools if task is clear\n"
            f"Each choice MUST include:\n"
            f"- name: tool name\n"
            f"- rank: position (1=best)\n"
            f"- accuracy: score (0-100) calculated from task/compatibility/features\n"
            f"- why: explanation with score breakdown (40/40 task + 30/30 compat + 30/30 features)\n"
            f"{meta_block}"
        )

        # ---- Build message parts (include image if available)
        data_url: Optional[str] = None
        parts: List[Dict[str, Any]] = [{"type": "text", "text": user_text}]
        if image_path:
            url = to_data_url(image_path)
            if url:
                data_url = url
                parts.append({"type": "image_url", "image_url": {"url": url}})
            else:
                self.log.warning("VLM selector: could not convert image; proceeding text-only.")

        # ---- Call model safely
        self.log.info("VLM selector model=%s", self.model)

        text: Optional[str] = None
        usage: Optional[dict] = None
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SELECTOR_SYSTEM},
                    {"role": "user", "content": parts},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            )

            # Extract response content and usage
            text = (resp.choices[0].message.content or "")
            try:
                usage = resp.usage.model_dump()
            except Exception:
                usage = getattr(resp, "usage", None)
            self.last_usage = usage if isinstance(usage, dict) else None

            # Guard against empty/whitespace responses
            if not text.strip():
                raise RuntimeError("Model returned empty content for JSON response.")

            # Parse JSON and validate required keys
            try:
                data = json.loads(text)
            except json.JSONDecodeError as e:
                preview = text[:200].replace("\n", " ")
                raise RuntimeError(f"Model returned non-JSON content: {preview!r}") from e

            # Handle empty or missing choices
            if "choices" not in data:
                data["choices"] = []
            if not data["choices"] and "reason" not in data:
                data["reason"] = "no_suitable_tool"

            # Add default values for missing required fields
            if data["choices"]:
                for i, choice in enumerate(data["choices"], 1):
                    # Required fields must be present
                    if "name" not in choice:
                        raise RuntimeError("Choice missing required field: name")
                    if "rank" not in choice:
                        choice["rank"] = i
                    if "accuracy" not in choice:
                        raise RuntimeError("Choice missing required field: accuracy")
                    if "why" not in choice:
                        raise RuntimeError("Choice missing required field: why")

                    # Validate values
                    if not choice["name"]:
                        raise RuntimeError("Choice name cannot be empty")
                    if not isinstance(choice["rank"], (int, float)):
                        raise RuntimeError(f"Invalid rank value: {choice['rank']}")
                    if not isinstance(choice["accuracy"], (int, float)):
                        raise RuntimeError(f"Invalid accuracy value: {choice['accuracy']}")
                    
                    # Ensure accuracy is in valid range
                    choice["accuracy"] = max(0.0, min(100.0, float(choice["accuracy"])))

            # Drop empty strings
            if "reason" in data and not str(data["reason"]).strip():
                data.pop("reason", None)
            if "explanation" in data and not str(data["explanation"]).strip():
                data.pop("explanation", None)

            # Ensure clean shape
            conv = data.get("conversation") or {}
            if str(conv.get("status", "complete")).lower() == "needs_clarification":
                data["choices"] = []
                data.pop("reason", None)
                data.pop("explanation", None)

            # If choices exist, do not carry a global reason
            if data.get("choices"):
                data.pop("reason", None)

            if not data.get("conversation"):
                data["conversation"] = {"status": "complete"}

            return ToolSelection(**data)

        finally:
            # Always attempt to log prompt (+ response/usage if present); never let logging crash the flow
            try:
                self.last_logfile = self._save_prompt(
                    kind="vlm_selector",
                    model=self.model,
                    system=SELECTOR_SYSTEM,
                    user_text=user_text,
                    data_url=data_url,
                    response_text=text,  # may be None/empty on failure
                    usage=self.last_usage if hasattr(self, "last_usage") else usage,
                )
                if self.last_logfile:
                    self.log.info("Saved VLM selector prompt → %s", self.last_logfile)
            except Exception:
                self.log.exception("Failed to save VLM selector prompt log")

