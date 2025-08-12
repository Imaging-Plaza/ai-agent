# generator/generator.py
from __future__ import annotations
import json
import os
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

# generator/generator.py
class OpenAIProvider(LLMProvider):
    def __init__(self, model: Optional[str] = None, timeout: float = 60.0):
        from openai import OpenAI
        import os
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.timeout = timeout
        self.last_request = None
        self.last_response = None
        self.last_usage = None

    def generate_json(self, system: str, user: str, response_schema_name: str = "json") -> str:
        self.last_request = {"model": self.model, "system": system, "user": user}
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            response_format={"type": "json_object"},
            temperature=0.2,
            timeout=self.timeout,
        )
        text = resp.choices[0].message.content
        self.last_response = text
        # New SDK returns usage; keep it if present
        try:
            self.last_usage = resp.usage.model_dump()
        except Exception:
            self.last_usage = getattr(resp, "usage", None)
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
        self.last_provider = None
        self.last_request = None
        self.last_raw = None
        self.last_usage = None

    def generate(
        self,
        user_task: str,
        candidates: List[CandidateDoc],
        image_path: str,
        out_mask_path: str = "mask.nii.gz",
        overlay_png_path: str = "overlay.png",
        cues: Optional[PerceptionCues] = None,
    ) -> PlanAndCode:
        prompt = build_user_prompt(
            user_task=user_task,
            cues=cues,
            candidates=candidates,
            image_path=image_path,
            out_mask_path=out_mask_path,
            overlay_png_path=overlay_png_path,
        )
        self.last_provider = self.provider.__class__.__name__
        raw = self.provider.generate_json(PLANNER_SYSTEM, prompt)
        self.last_raw = raw
        # capture request/usage if available
        self.last_request = getattr(self.provider, "last_request", None)
        self.last_usage = getattr(self.provider, "last_usage", None)

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = json.loads(extract_first_json(raw))

        try:
            return PlanAndCode(**payload)
        except ValidationError as e:
            # Optional repair round: ask the model to fix to schema (one shot)
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
                PLANNER_SYSTEM, prompt + "\n\n" + repair_instructions
            )
            payload = json.loads(extract_first_json(repaired))
            return PlanAndCode(**payload)
