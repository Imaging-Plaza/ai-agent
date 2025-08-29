"""
software_jsonld_to_jsonl.py

Produce one JSONL row per software entity (schema.org/SoftwareSourceCode) from a JSON-LD file.

Pipeline:
  1) Load JSON-LD and build an index by @id (deep-merge duplicates).
  2) Pick only roots whose @type includes SoftwareSourceCode.
  3) Recursively dereference @id references (incl. blank nodes), avoiding cycles.
  4) Unwrap JSON-LD value objects {"@value": ... , "@type": ...} to scalars (cast xsd types).
  5) Strip JSON-LD control keys (@context/@language...) and rename @id->id, @type->type.
  6) Strip known vocab prefixes from KEYS at any depth (schema.org / imaging-plaza / w3id OKN / biomedit SPHN).
  7) Optionally drop keys in EXCLUDE_KEYS.
  8) Write one cleaned record per software root as JSONL.

Set INPUT_FILE and OUTPUT_FILE, then run.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Set
import json

# ---- configure here ----
INPUT_FILE = "dataset/full_graph_new.jsonld"
OUTPUT_FILE = "dataset/filtered_dataset.jsonl"
# Optionally drop certain properties anywhere (AFTER prefix stripping). Example:
EXCLUDE_KEYS: Set[str] = set([
    # "bodySite",
])
# ------------------------

# Prefixes to strip from KEYS (order matters: more specific first)
PREFIXES: Tuple[str, ...] = (
    # schema.org
    "http://schema.org/",
    "https://schema.org/",
    # imaging-plaza
    "https://imaging-plaza.epfl.ch/ontology#",
    "http://imaging-plaza.epfl.ch/ontology#",
    # OKN
    "https://w3id.org/okn/o/sd#",
    "http://w3id.org/okn/o/sd#",
    # SPHN / biomedit
    "https://biomedit.ch/rdf/sphn-schema/sphn#",
    "http://biomedit.ch/rdf/sphn-schema/sphn#",
)

SOFTWARE_TYPES: Set[str] = {
    "http://schema.org/SoftwareSourceCode",
    "https://schema.org/SoftwareSourceCode",
    "schema:SoftwareSourceCode",
    "SoftwareSourceCode",
}

# Known XML Schema datatypes for safe casting
XSD_BOOLEAN = {"http://www.w3.org/2001/XMLSchema#boolean", "xsd:boolean"}
XSD_INTEGERS = {
    "http://www.w3.org/2001/XMLSchema#integer",
    "http://www.w3.org/2001/XMLSchema#long",
    "http://www.w3.org/2001/XMLSchema#int",
    "xsd:integer",
    "xsd:int",
    "xsd:long",
}
XSD_FLOATS = {
    "http://www.w3.org/2001/XMLSchema#float",
    "http://www.w3.org/2001/XMLSchema#double",
    "http://www.w3.org/2001/XMLSchema#decimal",
    "xsd:float",
    "xsd:double",
    "xsd:decimal",
}


# ---------------- utilities ----------------

def deep_merge(a: Any, b: Any) -> Any:
    """Deeply merge two JSON values (dict/list/scalars)."""
    if a is b or a == b:
        return a
    if isinstance(a, dict) and isinstance(b, dict):
        out = dict(a)
        for k, v in b.items():
            if k in out:
                out[k] = deep_merge(out[k], v)
            else:
                out[k] = v
        return out
    if isinstance(a, list) and isinstance(b, list):
        out = list(a)
        for x in b:
            if x not in out:
                out.append(x)
        return out
    if isinstance(a, list):
        return deep_merge(a, [b])
    if isinstance(b, list):
        return deep_merge([a], b)
    # scalar vs dict -> list; scalar vs scalar -> 2-item list
    return [a, b] if a != b else a


def normalize_types(t: Any) -> List[str]:
    """Return a list of type strings."""
    if t is None:
        return []
    if isinstance(t, list):
        return [str(x) for x in t]
    return [str(t)]


def is_software(node: Dict[str, Any]) -> bool:
    """True if node's @type includes SoftwareSourceCode (accepting http/https/compact)."""
    types = set(normalize_types(node.get("@type")))
    if types & SOFTWARE_TYPES:
        return True
    for t in types:
        if t.endswith("SoftwareSourceCode"):
            return True
    return False


def strip_key_prefix(key: Any) -> Any:
    """Strip known prefixes from string keys."""
    if not isinstance(key, str):
        return key
    for p in PREFIXES:
        if key.startswith(p):
            return key[len(p):]
    return key


def cast_typed_value(value: Any, vtype: str) -> Any:
    """Cast a JSON-LD typed literal to a Python scalar when safe."""
    if not isinstance(value, str):
        # value might already be numeric/bool
        return value
    low = value.strip().lower()
    if vtype in XSD_BOOLEAN:
        if low in ("true", "1"):
            return True
        if low in ("false", "0"):
            return False
        return value
    if vtype in XSD_INTEGERS:
        try:
            return int(value)
        except Exception:
            return value
    if vtype in XSD_FLOATS:
        try:
            return float(value)
        except Exception:
            return value
    # For dates and unknown types, leave as string
    return value


def unwrap_value_object(obj: Dict[str, Any]) -> Any:
    """
    Unwrap JSON-LD value objects like:
      {"@value": "10", "@type": "xsd:integer"} -> 10
      {"@value": "true", "@type": "xsd:boolean"} -> True
      {"@value": "2023-01-01"} -> "2023-01-01"
    """
    val = obj.get("@value")
    vtype = obj.get("@type")
    if vtype:
        return cast_typed_value(val, vtype)
    return val


def strip_jsonld_control(obj: Any) -> Any:
    """
    Remove JSON-LD control keys and rename @id/@type at any depth,
    **but first unwrap value objects** so we don't lose @value.

    - Value objects: {"@value":..., "@type":...} -> scalar (cast)
    - @id -> id
    - @type -> type (list or string; localize IRIs to tail segment)
    - other "@..." keys are dropped
    """
    if isinstance(obj, dict):
        # 1) Value object handling: must come first
        if "@value" in obj:
            return strip_jsonld_control(unwrap_value_object(obj))

        out: Dict[str, Any] = {}
        for k, v in obj.items():
            if k == "@id":
                out["id"] = strip_jsonld_control(v)
            elif k == "@type":
                types = normalize_types(v)
                out["type"] = [localize_iri(x) for x in types] if len(types) > 1 else localize_iri(types[0]) if types else types
            elif isinstance(k, str) and k.startswith("@"):
                # drop @context, @language, etc.
                continue
            else:
                out[k] = strip_jsonld_control(v)
        return out
    if isinstance(obj, list):
        return [strip_jsonld_control(x) for x in obj]
    return obj


def localize_iri(s: Any) -> Any:
    """Return last token after '#' or '/', otherwise the string itself."""
    if not isinstance(s, str):
        return s
    if "#" in s:
        return s.rsplit("#", 1)[-1]
    if "/" in s:
        return s.rstrip("/").rsplit("/", 1)[-1]
    return s


def strip_prefixes_and_merge(obj: Any) -> Any:
    """
    Recursively strip vocab prefixes from DICT KEYS and deep-merge collisions.
    (Run this AFTER strip_jsonld_control so we don't touch '@...' keys.)
    """
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            nk = strip_key_prefix(k)
            nv = strip_prefixes_and_merge(v)
            if nk in EXCLUDE_KEYS:
                continue
            if nk in out:
                out[nk] = deep_merge(out[nk], nv)
            else:
                out[nk] = nv
        return out
    if isinstance(obj, list):
        return [strip_prefixes_and_merge(x) for x in obj]
    return obj


# ------------- core pipeline -------------

def build_index(graph_nodes: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Build and deep-merge an index of nodes by @id.
    If multiple nodes share the same @id, their properties are merged.
    Nodes without @id are ignored in the index (they'll be captured by deref from parents).
    """
    idx: Dict[str, Dict[str, Any]] = {}
    for n in graph_nodes:
        if not isinstance(n, dict):
            continue
        nid = n.get("@id")
        if isinstance(nid, str):
            if nid in idx:
                idx[nid] = deep_merge(idx[nid], n)
            else:
                idx[nid] = dict(n)
    return idx


def deref(node: Any, idx: Dict[str, Dict[str, Any]], seen: Set[str] | None = None) -> Any:
    """
    Recursively dereference objects with '@id' by replacing them with their full node,
    merged with any inline properties. Avoid infinite loops with `seen`.
    """
    if isinstance(node, dict):
        node_id = node.get("@id")
        base = dict(node)
        if isinstance(node_id, str) and node_id in idx:
            if seen is None:
                seen = set()
            if node_id in seen:
                return {"@id": node_id}
            seen.add(node_id)
            merged = deep_merge(idx[node_id], base)
            out: Dict[str, Any] = {}
            for k, v in merged.items():
                out[k] = deref(v, idx, seen=set(seen))
            return out
        else:
            out: Dict[str, Any] = {}
            for k, v in node.items():
                out[k] = deref(v, idx, seen=set(seen) if seen is not None else None)
            return out

    if isinstance(node, list):
        return [deref(x, idx, seen=set(seen) if seen is not None else None) for x in node]

    return node


def extract_graph(doc: Any) -> List[Dict[str, Any]]:
    """Return the list of nodes from a JSON-LD document regardless of shape."""
    if isinstance(doc, dict) and "@graph" in doc:
        g = doc["@graph"]
        return [x for x in g if isinstance(x, dict)]
    if isinstance(doc, list):
        return [x for x in doc if isinstance(x, dict)]
    if isinstance(doc, dict):
        return [doc]
    return []


def drop_empties(obj: Any) -> Any:
    """Remove dict keys with None/''/[]/{} and empty list items recursively (keeps 0/False)."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            vv = drop_empties(v)
            if _is_empty(vv):
                continue
            out[k] = vv
        return out
    if isinstance(obj, list):
        new = [drop_empties(x) for x in obj]
        return [x for x in new if not _is_empty(x)]
    return obj


def _is_empty(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, str) and v == "":
        return True
    if isinstance(v, dict) and len(v) == 0:
        return True
    if isinstance(v, list) and len(v) == 0:
        return True
    return False


def main() -> None:
    # 1) Load
    data = json.loads(Path(INPUT_FILE).read_text(encoding="utf-8"))

    # 2) Index by @id (deep-merge duplicates)
    nodes = extract_graph(data)
    index = build_index(nodes)

    # 3) Root selection: only SoftwareSourceCode
    software_ids: List[str] = [nid for nid, node in index.items() if is_software(node)]

    # 4) For each software root: deref -> unwrap values -> strip controls -> strip prefixes -> drop empties
    out_path = Path(OUTPUT_FILE)
    count = 0
    with out_path.open("w", encoding="utf-8") as fw:
        for sid in software_ids:
            resolved = deref(index[sid], index)
            # Unwrap value objects FIRST, then drop @-keys / rename, then strip prefixes & merge
            cleaned = strip_jsonld_control(resolved)
            cleaned = strip_prefixes_and_merge(cleaned)
            cleaned = drop_empties(cleaned)
            fw.write(json.dumps(cleaned, ensure_ascii=False) + "\n")
            count += 1

    print(f"Wrote {count} software records to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
