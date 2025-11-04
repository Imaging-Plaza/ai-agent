from retriever.software_doc import SoftwareDoc
from typing import Optional


def _best_runnable_link(doc: SoftwareDoc) -> Optional[str]:
    """Return the most user-friendly runnable link.

    Preference order:
      1. Hugging Face Space (hf.space or huggingface.co/spaces)
      2. Other interactive demo hosts (gradio.live, replicate.run, etc.)
      3. Executable notebook links (.ipynb, colab)
      4. Fallback to first runnable example / notebook URL (GitHub last)
    Explicit `priority` values in catalog still respected (lower is better), but
    host preference can override large default values.
    """
    def base_priority(item) -> float:
        if isinstance(item, dict) and "priority" in item:
            try:
                return float(item["priority"])
            except Exception:
                pass
        return 100.0  # neutral base

    def extract_url(item) -> Optional[str]:
        if isinstance(item, str):
            u = item.strip()
            return u or None
        if isinstance(item, dict):
            for k in ("url", "href", "link", "contentUrl"):
                u = item.get(k)
                if isinstance(u, str) and u.strip():
                    return u.strip()
        return None

    def host_bonus(u: str) -> float:
        lu = u.lower()
        if "huggingface.co/spaces" in lu or lu.startswith("https://hf.space"):
            return -60.0
        if "gradio.live" in lu:
            return -40.0
        if "replicate.run" in lu or "replicate.com" in lu:
            return -30.0
        if lu.endswith(".ipynb") or "colab.research.google.com" in lu:
            return -10.0
        if "github.com" in lu:
            return +10.0  # de-prioritize plain GitHub vs real demos
        return 0.0

    collected = []
    for items in (getattr(doc, "runnable_example", None) or [], getattr(doc, "has_executable_notebook", None) or []):
        for it in items:
            url = extract_url(it)
            if not url:
                continue
            pr = base_priority(it) + host_bonus(url)
            collected.append((pr, url))

    if not collected:
        return None
    collected.sort(key=lambda x: x[0])
    return collected[0][1]