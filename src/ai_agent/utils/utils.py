from retriever.embedders import SoftwareDoc
from typing import Optional


def _best_runnable_link(doc: SoftwareDoc) -> Optional[str]:
        def priority(item) -> float:
            if isinstance(item, dict) and "priority" in item:
                try:
                    return float(item["priority"])
                except Exception:
                    pass
            return 1e9

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

        for items in (getattr(doc, "runnable_example", None) or [], getattr(doc, "has_executable_notebook", None) or []):
            try:
                items_sorted = sorted(items, key=priority)
            except Exception:
                items_sorted = items
            for it in items_sorted:
                url = extract_url(it)
                if url:
                    return url

        return None