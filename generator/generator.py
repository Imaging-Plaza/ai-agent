# generator/generator.py
from __future__ import annotations

import json
import os
import logging
from pathlib import Path
from datetime import datetime
import re
from typing import List, Optional, Dict, Any

from pydantic import ValidationError
from generator.schema import PerceptionCues, CandidateDoc, PlanAndCode
from generator.prompts import PLANNER_SYSTEM, build_user_prompt


# ---------- Provider interface ----------

class LLMProvider:
    def generate_json(self, system: str, user: str, response_schema_name: str = "json") -> str:
        """Return a JSON string (model should be steered to respond with JSON)."""
        raise NotImplementedError


# ---------- OpenAI provider (default) ----------

class OpenAIProvider(LLMProvider):
    def __init__(self, model: Optional[str] = None, timeout: float = 60.0):
        from openai import OpenAI  # lazy import
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.timeout = timeout
        self.log = logging.getLogger("generator.openai")

        # stash last call for debugging
        self.last_request: Optional[Dict[str, Any]] = None
        self.last_response: Optional[str] = None
        self.last_usage: Optional[Dict[str, Any]] = None
        self.last_logfile: Optional[str] = None

    def _maybe_log_to_file(self, system: str, user: str, response_text: str, usage: Dict[str, Any] | None):
        """
        Write full prompts to logs/ if LOG_PROMPTS=1/true.
        Returns the log file path or None.
        """
        if str(os.getenv("LOG_PROMPTS", "")).lower() not in ("1", "true", "yes", "on"):
            return None
        Path("logs").mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = Path("logs") / f"openai_{ts}.txt"
        with path.open("w", encoding="utf-8") as f:
            f.write(f"MODEL: {self.model}\n\n")
            f.write("--- SYSTEM ---\n")
            f.write(system.strip() + "\n\n")
            f.write("--- USER ---\n")
            f.write(user.strip() + "\n\n")
            f.write("--- RESPONSE ---\n")
            f.write((response_text or "").strip() + "\n\n")
            f.write("--- USAGE ---\n")
            f.write(str(usage or {}))
        return str(path)

    def generate_json(self, system: str, user: str, response_schema_name: str = "json") -> str:
        self.last_request = {"model": self.model, "system": system, "user": user}

        # small console preview (truncate to keep logs tidy)
        self.log.info("OpenAI chat.completions model=%s", self.model)
        self.log.debug("SYSTEM: %s", (system[:300] + "…") if len(system) > 300 else system)
        self.log.debug("USER: %s", (user[:300] + "…") if len(user) > 300 else user)

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
            timeout=self.timeout,
        )

        text = resp.choices[0].message.content

        # usage dict (SDK differences)
        try:
            usage = resp.usage.model_dump()
        except Exception:
            usage = getattr(resp, "usage", None)

        self.last_response = text
        self.last_usage = usage if isinstance(usage, dict) else None

        # write full prompts if enabled
        self.last_logfile = self._maybe_log_to_file(system, user, text, self.last_usage)
        if self.last_logfile:
            self.log.info("Saved prompt log → %s", self.last_logfile)

        return text


# ---------- Fallback generic HTTP provider (for local LLM servers) ----------

class GenericHTTPProvider(LLMProvider):
    """
    Very simple JSON POST to a local server.
    Configure with:
      RAG_GEN_ENDPOINT (e.g., http://localhost:8000/generate)
    Expected server contract:
      POST body: {"system":..., "user":...}
      Response: {"text": "...JSON..."}
    """
    def __init__(self, endpoint: Optional[str] = None):
        import requests  # pip install requests
        self.requests = requests
        self.endpoint = endpoint or os.getenv("RAG_GEN_ENDPOINT", "http://localhost:8000/generate")

    def generate_json(self, system: str, user: str, response_schema_name: str = "json") -> str:
        r = self.requests.post(self.endpoint, json={"system": system, "user": user}, timeout=120)
        r.raise_for_status()
        data = r.json()
        return data.get("text", "")


# ---------- Utilities ----------

def extract_first_json(text: str) -> str:
    """
    Robustly pull first JSON object from a string, in case the model adds wrapper text.
    """
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model output.")
    return match.group(0)


# ---------- Public API ----------

class PlanAndCodeGenerator:
    def __init__(self, provider: Optional[LLMProvider] = None):
        # Default to OpenAI provider; fall back to generic HTTP if key isn't set.
        if provider is not None:
            self.provider = provider
        elif os.getenv("OPENAI_API_KEY"):
            print("Using OpenAIProvider with OPENAI_API_KEY.")
            self.provider = OpenAIProvider()
        else:
            self.provider = GenericHTTPProvider()

        self.last_provider: Optional[str] = None
        self.last_request: Optional[Dict[str, Any]] = None
        self.last_raw: Optional[str] = None
        self.last_usage: Optional[Dict[str, Any]] = None
        self.last_logfile: Optional[str] = None  # <-- needed for logging
        self.log = logging.getLogger("generator")

    def generate(
        self,
        user_task: str,
        candidates: List[CandidateDoc],
        image_path: str,
        out_mask_path: str = "mask.nii.gz",
        overlay_png_path: str = "overlay.png",
        cues: Optional[PerceptionCues] = None,
    ) -> PlanAndCode:
        # Build the user prompt
        prompt = build_user_prompt(
            user_task=user_task,
            cues=cues,
            candidates=candidates,
            image_path=image_path,
            out_mask_path=out_mask_path,
            overlay_png_path=overlay_png_path,
        )

        # Call provider
        self.last_provider = self.provider.__class__.__name__
        raw = self.provider.generate_json(PLANNER_SYSTEM, prompt)
        self.last_raw = raw

        # capture request/usage/log file from provider
        self.last_request = getattr(self.provider, "last_request", None)
        self.last_usage = getattr(self.provider, "last_usage", None)
        self.last_logfile = getattr(self.provider, "last_logfile", None)

        # quick console summary
        tokens = (self.last_usage or {}).get("total_tokens") if isinstance(self.last_usage, dict) else self.last_usage
        self.log.info("Provider=%s tokens=%s", self.last_provider, tokens)
        if self.last_logfile:
            self.log.info("Full prompt saved at %s", self.last_logfile)

        # Parse JSON (with fallback if model wrapped it)
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = json.loads(extract_first_json(raw))

        # Validate against schema; do a one-shot repair if needed
        try:
            return PlanAndCode(**payload)
        except ValidationError:
            try:
                schema_json = PlanAndCode.schema_json(indent=2)  # pydantic v1
            except Exception:
                import json as _json
                schema_json = _json.dumps(PlanAndCode.model_json_schema(), indent=2)  # pydantic v2

            repair_instructions = (
                "The prior JSON did not match this schema:\n"
                f"{schema_json}\nReturn corrected JSON only."
            )

            repaired = self.provider.generate_json(
                PLANNER_SYSTEM,
                prompt + "\n\n" + repair_instructions
            )
            repaired_payload = json.loads(extract_first_json(repaired))
            return PlanAndCode(**repaired_payload)
