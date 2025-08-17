# generator/generator.py
from __future__ import annotations
import json, os, logging, re
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

from generator.schema import CandidateDoc, ToolSelection
from generator.prompts import SELECTOR_SYSTEM

class LLMProvider:
    def generate_json(self, system: str, user: str, response_schema_name: str = "json") -> str:
        raise NotImplementedError

class OpenAIProvider(LLMProvider):
    def __init__(self, model: Optional[str] = None, timeout: float = 60.0):
        from openai import OpenAI
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = (
            model
            or os.getenv("OPENAI_SELECTOR_MODEL")
            or os.getenv("OPENAI_MODEL")
            or "gpt-4o-mini"
        )
        self.timeout = timeout
        self.log = logging.getLogger("generator.openai")
        self.last_request = None
        self.last_response = None
        self.last_usage = None
        self.last_logfile = None

    def _maybe_log_to_file(self, system: str, user: str, response_text: str, usage: dict | None):
        if str(os.getenv("LOG_PROMPTS", "")).lower() not in ("1", "true", "yes", "on"):
            return None
        Path("logs").mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = Path("logs") / f"openai_{ts}.txt"
        with path.open("w", encoding="utf-8") as f:
            f.write(f"MODEL: {self.model}\n\n--- SYSTEM ---\n{system.strip()}\n\n--- USER ---\n{user.strip()}\n\n--- RESPONSE ---\n{(response_text or '').strip()}\n\n--- USAGE ---\n{str(usage or {})}")
        return str(path)

    def generate_json(self, system: str, user: str, response_schema_name: str = "json") -> str:
        self.last_request = {"model": self.model, "system": system, "user": user}
        self.log.info("OpenAI chat.completions model=%s", self.model)
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role":"system","content":system},{"role":"user","content":user}],
            response_format={"type":"json_object"},
            temperature=0.2,
            timeout=self.timeout,
        )
        text = resp.choices[0].message.content
        try:
            usage = resp.usage.model_dump()
        except Exception:
            usage = getattr(resp, "usage", None)
        self.last_response = text
        self.last_usage = usage
        self.last_logfile = self._maybe_log_to_file(system, user, text, usage)
        if self.last_logfile:
            self.log.info("Saved prompt log → %s", self.last_logfile)
        return text

# won't be useful for now, but kept for reference
class GenericHTTPProvider(LLMProvider):
    def __init__(self, endpoint: Optional[str] = None):
        import requests
        self.requests = requests
        self.endpoint = endpoint or os.getenv("RAG_GEN_ENDPOINT", "http://localhost:8000/generate")
    def generate_json(self, system: str, user: str, response_schema_name: str = "json") -> str:
        r = self.requests.post(self.endpoint, json={"system": system, "user": user}, timeout=120)
        r.raise_for_status()
        return r.json().get("text", "")

def extract_first_json(text: str) -> str:
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError("No JSON object found in model output.")
    return m.group(0)

# -------- Single-call VLM selector --------
class VLMToolSelector:
    """
    Single-call VLM selector:
      (text + image + candidates + original-image metadata) -> {choice, alternates, why}
    Writes a readable prompt + the exact PNG sent to the model when LOG_PROMPTS=1.
    """
    def __init__(self, model: Optional[str] = None):
        from openai import OpenAI
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = (
            model
            or os.getenv("OPENAI_SELECTOR_MODEL")
            or os.getenv("OPENAI_VLM_MODEL")
            or os.getenv("OPENAI_MODEL")
            or "gpt-4o-mini"
        )
        self.log = logging.getLogger("selector.vlm")
        self.last_usage: Optional[dict] = None
        self.last_logfile: Optional[str] = None

    @staticmethod
    def _save_prompt(kind: str, model: str, system: str, user_text: str,
                     data_url: Optional[str]) -> Optional[str]:
        """Save a small .txt with system+user and, if present, a .png of the exact image."""
        if str(os.getenv("LOG_PROMPTS", "")).lower() not in ("1", "true", "yes", "on"):
            return None
        from datetime import datetime
        from pathlib import Path
        import base64
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
            f.write("--- SYSTEM ---\n")
            f.write(system.strip() + "\n\n")
            f.write("--- USER (text) ---\n")
            f.write(user_text.strip() + "\n\n")
            if png_path:
                f.write(f"--- IMAGE ---\nSaved PNG: {png_path}\n")
            elif data_url:
                f.write(f"--- IMAGE ---\n(data-url length: {len(data_url)})\n")
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
          - user content: text block (task + candidates + metadata) + image (PNG data URL)
        Returns ToolSelection(choice, alternates, why).
        """
        from generator.prompts import SELECTOR_SYSTEM
        from generator.schema import ToolSelection, CandidateDoc
        from utils.image_analyzer import _to_supported_png_dataurl as to_data_url

        # compact candidate lines
        def fmt(c: CandidateDoc) -> str:
            return (
                f"- {c.name} | tasks={','.join(c.tasks)} | modality={','.join(c.modality)} | "
                f"dims={','.join(c.dims)} | inputs={','.join(c.input_formats)} | "
                f"outputs={','.join(c.output_types)}"
            )
        cand_block = "\n".join(fmt(c) for c in candidates)

        meta_note = "NOTE: Prefer candidates matching the original file extension indicated in 'Image metadata'.\n" if image_meta else ""
        meta_block = f"\nImage metadata: {image_meta}" if image_meta else ""

        user_text = (
            f"User task: {user_task}\n"
            f"{meta_note}"
            f"Candidates (choose exactly one by name):\n{cand_block}\n"
            f"Return STRICT JSON with keys: choice, alternates, why.{meta_block}"
        )

        # add image preview (converted to PNG data URL)
        data_url = None
        parts: List[Dict[str, Any]] = [{"type": "text", "text": user_text}]
        if image_path:
            url = to_data_url(image_path)
            if url:
                data_url = url
                parts.append({"type": "image_url", "image_url": {"url": url}})
            else:
                self.log.warning("VLM selector: could not convert image; proceeding text-only.")

        # log prompt files if enabled
        self.last_logfile = self._save_prompt(
            kind="vlm_selector",
            model=self.model,
            system=SELECTOR_SYSTEM,
            user_text=user_text,
            data_url=data_url,
        )
        if self.last_logfile:
            self.log.info("Saved VLM selector prompt → %s", self.last_logfile)

        # call VLM
        self.log.info("VLM selector model=%s", self.model)
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SELECTOR_SYSTEM},
                {"role": "user", "content": parts},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        text = resp.choices[0].message.content
        try:
            usage = resp.usage.model_dump()
        except Exception:
            usage = getattr(resp, "usage", None)
        self.last_usage = usage if isinstance(usage, dict) else None

        data = json.loads(text)
        return ToolSelection(**data)
