from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict

import rdflib
from SPARQLWrapper import SPARQLWrapper, JSONLD, TURTLE
from urllib.error import HTTPError
import logging

from ai_agent.utils.full_processing import full_processing

import hashlib
from ai_agent.retriever.software_doc import SoftwareDoc
from ai_agent.retriever.text_embedder import LocalBGEEmbedder
from ai_agent.retriever.vector_index import IndexItem, VectorIndex

log = logging.getLogger("ai_agent.catalog.sync")


def _index_artifacts_present(index_dir: Path) -> bool:
    """Return True when minimal FAISS artifacts exist."""
    return (index_dir / "index.faiss").exists() and (index_dir / "meta.json").exists()


def _count_jsonl_rows(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    except Exception:
        return -1


def _maybe_skip_sync_if_fresh(out_jsonl: Path, out_jsonld: Path, index_dir: Path) -> Dict[str, Any] | None:
    """Skip remote sync when local catalog/index are fresh enough for startup."""
    freshness_s = int(os.getenv("SYNC_SKIP_IF_FRESH_SECONDS", "0") or 0)
    force_sync = str(os.getenv("SYNC_FORCE", "0")).lower() in (
        "1",
        "true",
        "yes",
        "on",
    )

    if force_sync or freshness_s <= 0:
        return None

    if not out_jsonl.exists() or not _index_artifacts_present(index_dir):
        return None

    age_s = time.time() - out_jsonl.stat().st_mtime
    if age_s > freshness_s:
        return None

    digest_path = out_jsonl.with_suffix(out_jsonl.suffix + ".sha1")
    digest = ""
    if digest_path.exists():
        try:
            digest = digest_path.read_text(encoding="utf-8").strip()
        except Exception:
            digest = ""

    count = _count_jsonl_rows(out_jsonl)
    log.info(
        "Skipping sync (fresh local catalog age=%ss <= %ss; force with SYNC_FORCE=1)",
        int(age_s),
        freshness_s,
    )
    return {
        "jsonld_path": str(out_jsonld),
        "jsonl_path": str(out_jsonl),
        "count": count,
        "changed": False,
        "digest": digest,
        "index_dir": str(index_dir),
        "skipped": True,
        "skip_reason": "fresh_local_catalog",
    }


# --------------------------- helpers ---------------------------
def _load_query() -> str:
    """
    Load a SPARQL query from query file.
    """
    fname = os.getenv("GRAPHDB_QUERY_FILE", "get_relevant_software.rq").strip()
    p = Path(fname)

    if p.exists():
        template = p.read_text(encoding="utf-8")
        log.debug("Loaded SPARQL query from %s", p)
    else:
        log.error(
            "Unable to load query. Set GRAPHDB_QUERY_FILE to a valid path or place '%s' under ai_agent/queries/",
            fname,
        )
        raise RuntimeError(
            f"Unable to load query. Set GRAPHDB_QUERY_FILE to a valid path or place '{fname}' under ai_agent/queries/."
        )

    graph = (os.getenv("GRAPHDB_GRAPH") or "").strip()
    if not graph or "://" not in graph:
        log.error("GRAPHDB_GRAPH must be an absolute IRI; got: %r", graph)
        raise RuntimeError(
            "Set GRAPHDB_GRAPH (or GRAPH_NAME/GRAPH_URI) to an absolute IRI for {graph}."
        )

    q = template.format_map({"graph": graph})
    log.debug("Final SPARQL query prepared (length=%d chars)", len(q))
    return q


def _norm_doc_for_diff(d: SoftwareDoc) -> Dict[str, Any]:
    def _sorted_unique(xs):
        return sorted({str(x).strip() for x in (xs or []) if str(x).strip()})

    return {
        "name": (d.name or "").strip(),
        "url": (d.url or "").strip(),
        "repo_url": (d.repo_url or "").strip(),
        "documentation": (getattr(d, "documentation", None) or "").strip(),
        "description": (d.description or "").strip(),
        "category": _sorted_unique(d.category),
        "tasks": _sorted_unique(d.tasks),
        "modality": _sorted_unique(d.modality),
        "keywords": _sorted_unique(d.keywords),
        "dims": sorted(set(d.dims or [])),
        "anatomy": _sorted_unique(d.anatomy),
        "programming_language": (d.programming_language or "").strip(),
        "software_requirements": _sorted_unique(d.software_requirements),
        "gpu_required": bool(d.gpu_required) if d.gpu_required is not None else None,
        "is_free": bool(d.is_free) if d.is_free is not None else None,
        "is_based_on": _sorted_unique(d.is_based_on),
        "plugin_of": _sorted_unique(d.plugin_of),
        "related_organizations": _sorted_unique(d.related_organizations),
        "license": (d.license or "").strip(),
        "os": _sorted_unique(d.os),
    }


def _sha1_docs(docs: list[SoftwareDoc]) -> str:
    norm = [_norm_doc_for_diff(x) for x in docs]
    s = json.dumps(
        sorted(norm, key=lambda x: x["name"].lower()),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _diff_norm_docs(
    old_norm: list[Dict[str, Any]], new_norm: list[Dict[str, Any]]
) -> Dict[str, Any]:
    old_map = {(x.get("name") or "").strip(): x for x in old_norm}
    new_map = {(x.get("name") or "").strip(): x for x in new_norm}

    added = sorted(set(new_map) - set(old_map))
    removed = sorted(set(old_map) - set(new_map))

    changed = []
    details: Dict[str, list[str]] = {}
    shared = sorted(set(old_map) & set(new_map))
    for name in shared:
        a, b = old_map[name], new_map[name]
        changed_fields = []
        for k in sorted(set(a.keys()) | set(b.keys())):
            av, bv = a.get(k), b.get(k)
            if isinstance(av, list) or isinstance(bv, list):
                avs = set(av or [])
                bvs = set(bv or [])
                if avs != bvs:
                    changed_fields.append(k)
            else:
                if av != bv:
                    changed_fields.append(k)
        if changed_fields:
            changed.append(name)
            details[name] = changed_fields

    return {"added": added, "removed": removed, "changed": changed, "details": details}


def _read_docs(jsonl_path: Path) -> list[SoftwareDoc]:
    def _first_str(v):
        if v is None:
            return ""
        if isinstance(v, list):
            for x in v:
                s = str(x).strip()
                if s:
                    return s
            return ""
        return str(v).strip()

    docs: list[SoftwareDoc] = []
    total = 0
    made = 0
    invalid = 0

    try:
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                total += 1
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except Exception:
                    invalid += 1
                    continue

                payload = {
                    "name": _first_str(data.get("name")),
                    "url": data.get("url"),
                    "repo_url": data.get("repo_url") or data.get("codeRepository"),
                    "description": data.get("description"),
                    "applicationCategory": data.get("applicationCategory"),
                    "featureList": data.get("featureList"),
                    "imagingModality": data.get("imagingModality")
                    or data.get("modality"),
                    "keywords": data.get("keywords"),
                    "programmingLanguage": data.get("programmingLanguage"),
                    "softwareRequirements": data.get("softwareRequirements"),
                    "requiresGPU": data.get("requiresGPU"),
                    "isAccessibleForFree": data.get("isAccessibleForFree"),
                    "isBasedOn": data.get("isBasedOn"),
                    "isPluginModuleOf": data.get("isPluginModuleOf"),
                    "relatedToOrganization": data.get("relatedToOrganization"),
                    "license": data.get("license"),
                    "supportingData": data.get("supportingData"),
                    "os": data.get("operatingSystem") or data.get("os"),
                    "runnableExample": data.get("runnableExample"),
                    "hasExecutableNotebook": data.get("hasExecutableNotebook"),
                    "documentation": data.get("hasDocumentation"),
                }

                try:
                    d = SoftwareDoc(**payload)
                    docs.append(d)
                    made += 1
                except Exception:
                    invalid += 1
                    continue
    except FileNotFoundError:
        log.warning("JSONL not found at %s", jsonl_path)
    except Exception:
        log.exception("Error reading JSONL: %s", jsonl_path)

    log.info(
        "[jsonl->docs] %s: total=%d, docs=%d, invalid=%d",
        jsonl_path,
        total,
        made,
        invalid,
    )
    return docs


# --------------------------- fetch + convert ---------------------------
def fetch_jsonld(endpoint: str, query: str) -> Any:
    """Use SPARQLWrapper. Prefer JSON-LD, fall back to TURTLE → rdflib → JSON-LD."""
    sparql = SPARQLWrapper(endpoint)
    sparql.setQuery(query)

    # Auth
    user = os.getenv("GRAPHDB_USER")
    pwd = os.getenv("GRAPHDB_PASSWORD")
    if user and pwd:
        sparql.setCredentials(user=user, passwd=pwd)

    try:
        sparql.setReturnFormat(JSONLD)  # type: ignore[arg-type]
        log.info("Executing SPARQL (JSON-LD)…")
        res = sparql.query().convert()

        if isinstance(res, rdflib.graph.Graph):
            as_jsonld = res.serialize(format="json-ld", indent=None)
            log.debug("JSON-LD graph returned; serialized via rdflib")
            return json.loads(as_jsonld)

        if isinstance(res, (str, bytes)):
            return json.loads(res.decode("utf-8") if isinstance(res, bytes) else res)
        return res
    except Exception:
        log.info("JSON-LD not available; falling back to TURTLE → rdflib…")
        sparql.setReturnFormat(TURTLE)  # type: ignore[arg-type]
        try:
            ttl = sparql.query().convert()
            if isinstance(ttl, bytes):
                ttl = ttl.decode("utf-8")
            g = rdflib.Graph()
            g.parse(data=ttl, format="turtle")
            as_jsonld = g.serialize(format="json-ld", indent=None)
            log.debug("Converted TURTLE → JSON-LD via rdflib")
            return json.loads(as_jsonld)
        except HTTPError as e:
            log.error("HTTP error: %s", e)
            raise


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def convert_jsonld_to_jsonl(in_jsonld: Path, out_jsonl: Path) -> int:
    tmp_out = out_jsonl.with_suffix(out_jsonl.suffix + ".tmp")
    full_processing(str(in_jsonld), str(tmp_out))
    tmp_out.replace(out_jsonl)
    try:
        with out_jsonl.open("r", encoding="utf-8") as f:
            return sum(1 for _ in f)
    except Exception:
        return -1


# --------------------------- public API ---------------------------
def sync_once(
    *, out_jsonld: Path | None = None, out_jsonl: Path | None = None
) -> Dict[str, Any]:
    out_jsonld = Path(
        out_jsonld or os.getenv("OUTPUT_JSONLD", "dataset/catalog.jsonld")
    )
    out_jsonl = Path(out_jsonl or os.getenv("OUTPUT_JSONL", "dataset/catalog.jsonl"))

    index_dir = Path(os.getenv("RAG_INDEX_DIR", "artifacts/rag_index"))
    index_dir.mkdir(parents=True, exist_ok=True)

    skipped = _maybe_skip_sync_if_fresh(out_jsonl, out_jsonld, index_dir)
    if skipped is not None:
        return skipped

    endpoint = os.getenv("GRAPHDB_URL")
    query = _load_query()

    log.info("Fetching JSON-LD from %s", endpoint)
    data = fetch_jsonld(endpoint, query)

    log.info("Saving snapshot: %s", out_jsonld)
    write_json(data, out_jsonld)

    log.info("Running full_processing → %s", out_jsonl)
    count = convert_jsonld_to_jsonl(out_jsonld, out_jsonl)

    log.info("Sync complete: %s records → %s", count if count >= 0 else "?", out_jsonl)

    docs = _read_docs(out_jsonl)

    norm_docs = [_norm_doc_for_diff(d) for d in docs]
    norm_docs_sorted = sorted(norm_docs, key=lambda x: x["name"].lower())
    norm_path = out_jsonl.with_suffix(".semantic.json")
    prev_norm_path = out_jsonl.with_suffix(".semantic.prev.json")
    diff_path = out_jsonl.with_suffix(".diff.json")

    prev_norm: list[Dict[str, Any]] = []
    if norm_path.exists():
        try:
            prev_norm = json.loads(norm_path.read_text(encoding="utf-8"))
        except Exception:
            prev_norm = []

    digest_path = out_jsonl.with_suffix(out_jsonl.suffix + ".sha1")
    prev_digest = ""
    if digest_path.exists():
        try:
            prev_digest = digest_path.read_text(encoding="utf-8").strip()
        except Exception:
            prev_digest = ""

    digest = _sha1_docs(docs)
    changed = digest != prev_digest

    try:
        if norm_path.exists():
            norm_path.replace(prev_norm_path)
        norm_path.write_text(
            json.dumps(
                norm_docs_sorted,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        digest_path.write_text(digest, encoding="utf-8")
    except Exception:
        log.debug("Could not write semantic snapshot or digest")

    if not changed:
        # Important: catalog may be unchanged while embedding model/dim changed.
        # In that case, loading old FAISS artifacts fails and we must rebuild.
        faiss_rebuilt = False
        faiss_delta: Dict[str, int] = {"added": 0, "updated": 0, "removed": 0}
        try:
            embedder = LocalBGEEmbedder()
            VectorIndex.load(index_dir, embedder)
            log.info(
                "Catalog unchanged (semantic sha1=%s); keeping FAISS index", digest[:12]
            )
        except Exception as e:
            log.warning(
                "Catalog unchanged but FAISS index is missing/incompatible; rebuilding index (%s)",
                e,
            )
            embedder = LocalBGEEmbedder()
            idx = VectorIndex(embedder)
            items = [
                IndexItem(id=d.name, doc=d) for d in docs if getattr(d, "name", None)
            ]
            faiss_delta = idx.sync_with_catalog(items)
            idx.save(index_dir)
            faiss_rebuilt = True
            log.info(
                "FAISS index rebuilt in %s: added=%d, updated=%d, removed=%d",
                index_dir,
                faiss_delta["added"],
                faiss_delta["updated"],
                faiss_delta["removed"],
            )

        return {
            "jsonld_path": str(out_jsonld),
            "jsonl_path": str(out_jsonl),
            "count": count,
            "changed": False,
            "digest": digest,
            "index_dir": str(index_dir),
            "faiss_rebuilt": faiss_rebuilt,
            "faiss_delta": faiss_delta,
        }

    diff = _diff_norm_docs(prev_norm, norm_docs_sorted)
    try:
        diff_path.write_text(
            json.dumps(diff, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception:
        log.debug("Could not write diff to %s", diff_path)

    add_n = len(diff["added"])
    rem_n = len(diff["removed"])
    chg_n = len(diff["changed"])
    if add_n or rem_n or chg_n:
        add_s = ", ".join(diff["added"][:5]) + (" ..." if add_n > 5 else "")
        rem_s = ", ".join(diff["removed"][:5]) + (" ..." if rem_n > 5 else "")
        chg_s = ", ".join(diff["changed"][:5]) + (" ..." if chg_n > 5 else "")
        log.info(
            "Catalog changes: added=%d, removed=%d, changed=%d", add_n, rem_n, chg_n
        )
        if add_s:
            log.info("  added (sample): %s", add_s)
        if rem_s:
            log.info("  removed (sample): %s", rem_s)
        if chg_s:
            log.info("  changed (sample): %s", chg_s)
        log.info("Full diff written to %s", diff_path)

    embedder = LocalBGEEmbedder()
    try:
        idx = VectorIndex.load(index_dir, embedder)
    except Exception as e:
        log.warning(
            "Could not load existing FAISS index; rebuilding from catalog (%s)", e
        )
        idx = VectorIndex(embedder)

    items = [IndexItem(id=d.name, doc=d) for d in docs if getattr(d, "name", None)]
    delta = idx.sync_with_catalog(items)

    if any(delta.values()):
        idx.save(index_dir)
        log.info(
            "FAISS index updated in %s: added=%d, updated=%d, removed=%d",
            index_dir,
            delta["added"],
            delta["updated"],
            delta["removed"],
        )
    else:
        log.info("FAISS index unchanged after diff")

    return {
        "jsonld_path": str(out_jsonld),
        "jsonl_path": str(out_jsonl),
        "count": count,
        "changed": True,
        "digest": digest,
        "index_dir": str(index_dir),
        "faiss_delta": delta,
        "faiss_docs": len(items),
        "catalog_diff": {
            "added": add_n,
            "removed": rem_n,
            "changed": chg_n,
            "diff_path": str(diff_path),
        },
    }
