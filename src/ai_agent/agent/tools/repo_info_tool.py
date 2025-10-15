from __future__ import annotations

import requests
from pydantic import BaseModel

class RepoInfoInput(BaseModel):
    url: str
    max_chars: int = 4000

class RepoInfoOutput(BaseModel):
    url: str
    content: str
    truncated: bool = False

def tool_repo_info(inp: RepoInfoInput) -> RepoInfoOutput:
    """Very lightweight fallback to fetch README or index page. For GitHub repo URLs,
    attempt raw README retrieval. This can be replaced later with repo-to-text embedding service.
    """
    url = inp.url
    text = ""
    try:
        if "github.com" in url and not url.endswith(".md"):
            parts = url.rstrip("/").split("/")
            if len(parts) >= 2:
                # derive user/repo
                user = parts[-2]
                repo = parts[-1]
                candidate = f"https://raw.githubusercontent.com/{user}/{repo}/main/README.md"
                r = requests.get(candidate, timeout=10)
                if r.status_code == 200 and len(r.text) > 200:
                    text = r.text
        if not text:
            r = requests.get(url, timeout=10)
            text = r.text
    except Exception as e:
        return RepoInfoOutput(url=url, content=f"ERROR: {e}")
    trunc = False
    if len(text) > inp.max_chars:
        text = text[: inp.max_chars]
        trunc = True
    return RepoInfoOutput(url=url, content=text, truncated=trunc)