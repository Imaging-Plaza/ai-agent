"""SPARQL query tool — lets the agent ask the catalog's GraphDB anything the
RAG can't answer cleanly (counts, distincts, structural filters, etc.).

Safety:
    · read-only by syntax check — UPDATE / INSERT / DELETE / DROP / CLEAR /
      CREATE / LOAD / COPY / MOVE / ADD are rejected before the query goes
      anywhere near the endpoint
    · result row count is capped (we inject / override LIMIT)
    · network and query time are bounded (SPARQLWrapper timeout)
    · auth comes from the same GRAPHDB_USER / GRAPHDB_PASSWORD env the
      catalog sync uses
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from SPARQLWrapper import JSON, SPARQLWrapper

log = logging.getLogger("agent.tools.sparql")

# Default ceiling for any SELECT — overrideable via input.
DEFAULT_LIMIT = 50
HARD_MAX_LIMIT = 200

# Update keywords that must NOT appear at the top level of the query. We
# match on word boundaries (case-insensitive) outside of `<...>` IRIs and
# quoted strings — a real parser would be safer but for an agent-authored
# query this catches the common cases.
_UPDATE_KEYWORDS = re.compile(
    r"\b("
    r"INSERT|DELETE|UPDATE|DROP|CLEAR|CREATE|LOAD|COPY|MOVE|ADD|MODIFY"
    r")\b",
    re.IGNORECASE,
)

_LIMIT_RE = re.compile(r"\blimit\s+(\d+)\b", re.IGNORECASE)


class SparqlQueryInput(BaseModel):
    query: str = Field(..., description="A SPARQL SELECT / ASK query.")
    limit: int = Field(
        default=DEFAULT_LIMIT,
        ge=1,
        le=HARD_MAX_LIMIT,
        description="Maximum row count (we'll add / cap LIMIT in the query).",
    )


class SparqlQueryOutput(BaseModel):
    columns: List[str] = Field(default_factory=list)
    rows: List[Dict[str, Any]] = Field(default_factory=list)
    row_count: int = 0
    truncated: bool = False
    boolean: Optional[bool] = None  # only for ASK queries
    query_executed: str = ""
    error: Optional[str] = None


def _strip_strings_and_iris(q: str) -> str:
    """Remove `"..."`, `'...'`, and `<...>` so keyword scanning isn't fooled
    by literal text inside the query."""
    q = re.sub(r'"(?:[^"\\]|\\.)*"', '""', q)
    q = re.sub(r"'(?:[^'\\]|\\.)*'", "''", q)
    q = re.sub(r"<[^>]*>", "<>", q)
    return q


def _looks_readonly(query: str) -> bool:
    scrubbed = _strip_strings_and_iris(query)
    return _UPDATE_KEYWORDS.search(scrubbed) is None


def _enforce_limit(query: str, limit: int) -> str:
    """Inject LIMIT, or cap an existing one to `limit`. SELECT only — ASK,
    CONSTRUCT, DESCRIBE pass through unchanged (LIMIT is moot for ASK; the
    other two are blocked upstream)."""
    if not re.search(r"\bselect\b", query, re.IGNORECASE):
        return query
    m = _LIMIT_RE.search(query)
    if m:
        requested = int(m.group(1))
        if requested <= limit:
            return query
        return _LIMIT_RE.sub(f"LIMIT {limit}", query)
    trimmed = query.rstrip(" \t\n;")
    return trimmed + "\nLIMIT " + str(limit)


def tool_sparql_query(inp: SparqlQueryInput) -> SparqlQueryOutput:
    """Execute a SPARQL SELECT or ASK against GRAPHDB_URL and return a flat
    table of bindings."""
    endpoint = (os.getenv("GRAPHDB_URL") or "").strip()
    if not endpoint:
        return SparqlQueryOutput(
            error="GRAPHDB_URL is not configured on the server.",
            query_executed=inp.query,
        )

    if not _looks_readonly(inp.query):
        return SparqlQueryOutput(
            error="rejected_update_query — only read-only SPARQL is allowed",
            query_executed=inp.query,
        )

    # We currently only normalise SELECT/ASK. CONSTRUCT/DESCRIBE return RDF
    # graphs which would need a different output schema — bail with a clear
    # message rather than silently misformatting.
    qkind = re.search(
        r"\b(select|ask|construct|describe)\b", inp.query, re.IGNORECASE
    )
    if not qkind:
        return SparqlQueryOutput(
            error="missing_query_form — start the query with SELECT or ASK",
            query_executed=inp.query,
        )
    form = qkind.group(1).lower()
    if form in ("construct", "describe"):
        return SparqlQueryOutput(
            error=f"unsupported_form — {form.upper()} not supported by this tool yet; rephrase as SELECT",
            query_executed=inp.query,
        )

    final_query = _enforce_limit(inp.query, inp.limit) if form == "select" else inp.query

    sparql = SPARQLWrapper(endpoint)
    sparql.setQuery(final_query)
    sparql.setReturnFormat(JSON)
    user = os.getenv("GRAPHDB_USER")
    pwd = os.getenv("GRAPHDB_PASSWORD")
    if user and pwd:
        sparql.setCredentials(user=user, passwd=pwd)
    sparql.setTimeout(20)

    try:
        result = sparql.query().convert()
    except Exception as exc:
        log.warning("sparql_query failed: %s", exc)
        return SparqlQueryOutput(
            error=f"sparql_error: {exc}", query_executed=final_query
        )

    # ASK → boolean
    if "boolean" in result:
        return SparqlQueryOutput(
            boolean=bool(result["boolean"]),
            row_count=0,
            query_executed=final_query,
        )

    cols: List[str] = list(result.get("head", {}).get("vars", []))
    bindings = result.get("results", {}).get("bindings", []) or []
    rows: List[Dict[str, Any]] = []
    for b in bindings:
        row: Dict[str, Any] = {}
        for c in cols:
            cell = b.get(c)
            if cell is None:
                row[c] = None
            else:
                row[c] = cell.get("value")
        rows.append(row)

    truncated = len(rows) >= inp.limit
    return SparqlQueryOutput(
        columns=cols,
        rows=rows,
        row_count=len(rows),
        truncated=truncated,
        query_executed=final_query,
    )


__all__ = ["SparqlQueryInput", "SparqlQueryOutput", "tool_sparql_query"]
