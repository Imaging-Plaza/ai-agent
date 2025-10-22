from __future__ import annotations

import base64
import fnmatch
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

import requests
from pydantic import BaseModel

# ----------------------------- Public I/O models ------------------------------

class RepoSummaryInput(BaseModel):
    url: str


class RepoSummaryOutput(BaseModel):
    truncated: bool
    ref: Optional[str] = None
    summary: str

# ------------------------------ GitHub fetcher --------------------------------

GITHUB_API = "https://api.github.com"

# Keep the fetch set tight and human-written; avoid binaries & CI/docker noise.
INCLUDE_GLOBS = [
    "README.*",
    "docs/**/*.md", "doc/**/*.md", "documentation/**/*.md",
    "examples/**/*.md", "examples/**/*.py",
    "demo/**/*.md", "demo/**/*.py",
    "pyproject.toml", "setup.cfg", "setup.py",
    "requirements*.txt", "environment*.yml", "Pipfile", "Pipfile.lock",
    "**/cli*.py", "**/main.py", "scripts/**",
]
EXCLUDE_GLOBS = [
    ".git/**",".github/**","**/.venv/**","venv/**","env/**","data/**","datasets/**","docs/_build/**",
    "Dockerfile","docker/**","**/Dockerfile",
    "CHANGELOG*","changelog*","**/CHANGELOG*","**/changelog*",
    "tests/**","test/**","**/*.ipynb_checkpoints/**",
    # binaries / large
    "**/*.png","**/*.jpg","**/*.jpeg","**/*.gif","**/*.tif","**/*.tiff",
    "**/*.pdf","**/*.zip","**/*.tar*","**/*.7z","**/*.rar",
    "**/*.onnx","**/*.pt","**/*.pth","**/*.h5","**/*.ckpt",
]
TEXTY_EXTS = {".md",".rst",".txt",".toml",".cfg",".ini",".py",".yml",".yaml",".json",".sh",".bat",".ps1"}


@dataclass
class FetchedFile:
    path: str
    content: str  # decoded text


@dataclass
class RepoSnapshot:
    owner: str
    name: str
    ref: str
    description: Optional[str]
    license_spdx: Optional[str]
    topics: List[str]
    files: List[FetchedFile]
    truncated: bool


def _auth_headers(token: Optional[str]) -> Dict[str, str]:
    h = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def _parse_repo_url(url: str) -> Tuple[str, str, Optional[str]]:
    """
    Supports https://github.com/owner/repo[.git][#/tree/<ref>|@<ref>]
    Returns (owner, repo, ref or None)
    """
    u = urlparse(url)
    if u.netloc not in {"github.com", "www.github.com"}:
        raise ValueError("Only GitHub URLs are supported.")
    parts = [p for p in u.path.strip("/").split("/") if p]
    if len(parts) < 2:
        raise ValueError("Invalid GitHub repo URL.")
    owner, repo = parts[0], parts[1].removesuffix(".git")
    ref = None
    if len(parts) >= 4 and parts[2] == "tree":
        ref = "/".join(parts[3:])
    if u.fragment:
        ref = u.fragment
    return owner, repo, ref


def _matches_any(path: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatch(path, pat) for pat in patterns)


def _looks_texty(path: str) -> bool:
    _, dot, ext = path.rpartition(".")
    return ("." + ext.lower()) in TEXTY_EXTS


def _fetch_repo_meta(session: requests.Session, owner: str, name: str) -> Tuple[str, Optional[str], Optional[str], List[str]]:
    """Return (default_branch, description, license_spdx, topics)."""
    r = session.get(f"{GITHUB_API}/repos/{owner}/{name}")
    r.raise_for_status()
    repo = r.json()
    default_branch = repo.get("default_branch", "main")
    description = repo.get("description")
    license_spdx = (repo.get("license") or {}).get("spdx_id") or None

    # Topics (best-effort)
    topics: List[str] = []
    rt = session.get(f"{GITHUB_API}/repos/{owner}/{name}/topics")
    if rt.ok:
        topics = (rt.json().get("names") or [])[:10]
    return default_branch, description, license_spdx, topics


def fetch_repo_snapshot_via_api(
    repo_url: str,
    token: Optional[str] = None,
    max_files: int = 100,
    max_bytes_per_file: int = 300_000,
) -> RepoSnapshot:
    """
    Fetch a curated subset of files via GitHub REST API (no git clone).
    """
    owner, name, ref = _parse_repo_url(repo_url)
    s = requests.Session()
    s.headers.update(_auth_headers(token))

    # Repo meta & default branch
    default_branch, desc, spdx, topics = _fetch_repo_meta(s, owner, name)
    if not ref:
        ref = default_branch

    truncated = False

    # List tree; fall back to resolving branch -> sha if needed
    r = s.get(f"{GITHUB_API}/repos/{owner}/{name}/git/trees/{ref}", params={"recursive": "1"})
    if r.status_code == 422:
        rb = s.get(f"{GITHUB_API}/repos/{owner}/{name}/branches/{ref}")
        rb.raise_for_status()
        sha = rb.json()["commit"]["sha"]
        r = s.get(f"{GITHUB_API}/repos/{owner}/{name}/git/trees/{sha}", params={"recursive": "1"})
    r.raise_for_status()
    tree = r.json().get("tree", [])

    candidate_paths: List[str] = []
    for node in tree:
        if node.get("type") != "blob":
            continue
        path = node["path"]
        if _matches_any(path, EXCLUDE_GLOBS):
            continue
        if _matches_any(path, INCLUDE_GLOBS) or _looks_texty(path):
            candidate_paths.append(path)

    if len(candidate_paths) > max_files:
        candidate_paths = candidate_paths[:max_files]
        truncated = True

    files: List[FetchedFile] = []
    for path in candidate_paths:
        rr = s.get(f"{GITHUB_API}/repos/{owner}/{name}/contents/{path}", params={"ref": ref})
        if rr.status_code == 404:
            continue
        rr.raise_for_status()
        meta = rr.json()
        if isinstance(meta, list):
            continue
        size = int(meta.get("size") or 0)
        if size > max_bytes_per_file:
            truncated = True
            continue
        enc = meta.get("encoding")
        content_b64 = meta.get("content") or ""
        text = ""
        if enc == "base64":
            try:
                text = base64.b64decode(content_b64).decode("utf-8", errors="replace")
            except Exception:
                continue
        else:
            text = content_b64
        files.append(FetchedFile(path=path, content=text))

    return RepoSnapshot(
        owner=owner,
        name=name,
        ref=ref,
        description=desc,
        license_spdx=spdx,
        topics=topics,
        files=files,
        truncated=truncated,
    )

def _dedupe_str(seq: Iterable[str]) -> List[str]:
    """Dedupe strings, trimming and preserving order."""
    seen, out = set(), []
    for x in seq:
        s = x.strip()
        if not s:
            continue
        if s not in seen:
            out.append(s)
            seen.add(s)
    return out

def _dedupe_pairs(seq: Iterable[tuple[str, str]]) -> List[tuple[str, str]]:
    """Dedupe (text, source) pairs; trims text, preserves source and order."""
    seen, out = set(), []
    for text, src in seq:
        t = text.strip()
        if not t:
            continue
        key = (t, src)
        if key not in seen:
            out.append((t, src))
            seen.add(key)
    return out

# -------------------------- Extractors (verbatim-only) ------------------------

# Headings we consider for extracting usage/API text
INSTALL_PAT = re.compile(r"(?i)\b(pip|pipx|conda)\s+install\b[^\n]*")
CLI_LINE_RE = re.compile(
    r"(?m)^\s*(?:\$|>)?\s*(?:python(?:\s+-m)?\s+\S+|pip\s+install\s+\S+|pipx\s+run\s+\S+|conda\s+install\s+\S+|poetry\s+run\s+\S+|uvicorn\s+\S+|streamlit\s+run\s+\S+|gradio\s+\S+|pytest\s+\S+)\b[^\n]*"
)
FENCED_PY_RE = re.compile(r"```python(.*?)```", re.S | re.I)
HEAD_INPUTS = re.compile(r"(?im)^#{1,6}\s*inputs?\b")
HEAD_OUTPUTS = re.compile(r"(?im)^#{1,6}\s*outputs?\b")
URL_RE = re.compile(r"https?://[^\s\]\)>\}\"\']+")

# (Removed duplicate _dedupe function; use _dedupe_str instead)

def _first_readme_para(files: List[FetchedFile]) -> Optional[str]:
    for f in files:
        if re.search(r"(^|/|\\)readme(\.|$)", f.path, re.I):
            blob = f.content.strip()
            parts = re.split(r"\n\s*\n", blob, maxsplit=1)
            para = parts[0].strip()
            para = re.sub(r"^#+\s*", "", para)
            return para[:500] if para else None
    return None

def _is_readme_docs_examples(path: str) -> bool:
    return (re.search(r"readme", path, re.I) is not None) or path.startswith(("docs/","doc/","documentation/","examples/","demo/"))

def _extract_install(files: List[FetchedFile]) -> List[tuple[str, str]]:
    out: List[tuple[str,str]] = []
    for f in files:
        if not _is_readme_docs_examples(f.path):
            continue
        for m in INSTALL_PAT.finditer(f.content):
            out.append((m.group(0).strip(), f.path))
    return _dedupe_pairs(out)[:3]

def _extract_cli(files: List[FetchedFile]) -> List[tuple[str, str]]:
    out: List[tuple[str,str]] = []
    for f in files:
        if not _is_readme_docs_examples(f.path):
            continue
        for m in CLI_LINE_RE.finditer(f.content):
            cmd = re.sub(r"^\s*(\$|>)\s*", "", m.group(0)).strip()
            out.append((cmd, f.path))
    return _dedupe_pairs(out)[:5]   # Limit to 5 CLI command examples for brevity and relevance

def _extract_api(files: List[FetchedFile]) -> List[tuple[str, str]]:
    out: List[tuple[str,str]] = []
    for f in files:
        if not _is_readme_docs_examples(f.path) and not f.path.endswith(".py"):
            continue
        for m in FENCED_PY_RE.finditer(f.content):
            block = m.group(1).strip()
            if block:
                # keep short
                lines = block.splitlines()
                if len(lines) > 40:
                    block = "\n".join(lines[:40]) + "\n# …"
                out.append((block, f.path))
                if len(out) >= 2:
                    return out
    return out

def _extract_section_text(files: List[FetchedFile], head_re: re.Pattern, max_lines: int = 10) -> List[tuple[str, str]]:
    """Grab paragraphs immediately under 'Inputs'/'Outputs' headings in README/docs."""
    out: List[tuple[str,str]] = []
    for f in files:
        if not _is_readme_docs_examples(f.path):
            continue
        txt = f.content
        for m in head_re.finditer(txt):
            start = m.end()
            # capture until next heading or EOF
            rest = txt[start:]
            stop = re.search(r"(?m)^\s*#{1,6}\s+\S", rest)
            chunk = rest[: stop.start() if stop else len(rest)]
            # take first N lines
            lines = [ln.rstrip() for ln in chunk.strip().splitlines() if ln.strip()]
            if lines:
                out.append(("\n".join(lines[:max_lines]).strip(), f.path))
    # dedupe identical blocks
    seen, ded = set(), []
    for block, src in out:
        key = (block, src)
        if key not in seen:
            ded.append((block, src)); seen.add(key)
    return ded[:2]

def _extract_requirements(files: List[FetchedFile]) -> List[tuple[str, str]]:
    out: List[tuple[str,str]] = []
    for f in files:
        if not re.search(r"(requirements|pyproject|setup\.cfg|setup\.py|environment.*\.yml|Pipfile)", f.path, re.I):
            continue
        lines = []
        for line in f.content.splitlines():
            s = line.strip()
            if not s or len(s) > 120:
                continue
            # only keep plausible requirement lines
            if re.match(r"^[A-Za-z0-9_.\-]+(\[.*\])?([=><!~].*)?$", s):
                lines.append(s)
        if lines:
            # keep first few lines to avoid huge dumps
            out.append(("\n".join(lines[:20]), f.path))
    return out[:3]

def _extract_urls(files: List[FetchedFile]) -> List[tuple[str, str]]:
    out: List[tuple[str,str]] = []
    for f in files:
        if not _is_readme_docs_examples(f.path):
            continue
        for m in URL_RE.finditer(f.content):
            u = m.group(0).rstrip(").,;:'\"*")
            out.append((u, f.path))
    # dedupe by URL
    seen, ded = set(), []
    for u, src in out:
        if u not in seen:
            ded.append((u, src)); seen.add(u)
    return ded[:10]

def _notable_paths(files: List[FetchedFile]) -> List[str]:
    candidates = [f.path for f in files if f.path.count("/") <= 2]
    keep = []
    for p in candidates:
        if p.startswith(("examples/","scripts/","docs/","demo/","README")):
            keep.append(p)
    return _dedupe_str(keep)[:10]   # <-- ensure string dedupe

# ---------------------------- Normalizer for GitHub links ----------------------------

_OWNER_REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")

def _coerce_owner_repo_ref(input_str: str) -> Tuple[str, str, Optional[str]]:
    """
    Accepts:
      - https://github.com/owner/repo[.git][/tree/<ref>|#<ref>...]
      - http(s)://www.github.com/owner/repo...
      - github.com/owner/repo...
      - owner/repo
    Returns (owner, repo, ref|None) or raises ValueError with a helpful message.
    """
    s = (input_str or "").strip()

    # Short form: owner/repo
    if _OWNER_REPO_RE.match(s):
        owner, repo = s.split("/", 1)
        repo = repo.removesuffix(".git")
        return owner, repo, None

    # Missing scheme but github domain
    if s.startswith("github.com/") or s.startswith("www.github.com/"):
        s = "https://" + s.lstrip("www.")

    # Full URL?
    try:
        u = urlparse(s)
    except Exception:
        u = None

    if u and u.netloc.lower() in {"github.com", "www.github.com"}:
        parts = [p for p in u.path.strip("/").split("/") if p]
        if len(parts) >= 2:
            owner, repo = parts[0], parts[1].removesuffix(".git")
            ref = None
            # /tree/<ref> pattern
            if len(parts) >= 4 and parts[2] == "tree":
                ref = "/".join(parts[3:])
            # fragment as ref (e.g., #main)
            if u.fragment:
                ref = u.fragment
            return owner, repo, ref

    # As a last chance, try to extract owner/repo from a github-looking string
    m = re.search(r"github\.com[:/]+([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)", s, re.I)
    if m:
        owner, repo = m.group(1), m.group(2).removesuffix(".git")
        return owner, repo, None

    raise ValueError(
        "[BAD_REPO_URL] Provide a GitHub repo as 'owner/repo' or a GitHub URL, "
        f"got: {input_str!r}"
    )

def coerce_github_url_or_none(s: str) -> str | None:
    """
    Returns a canonical GitHub URL ('https://github.com/owner/repo' or with #ref)
    if input is a valid GitHub repo URL or 'owner/repo'. Otherwise returns None.
    """
    try:
        owner, repo, ref = _coerce_owner_repo_ref(s)
        base = f"https://github.com/{owner}/{repo}"
        return f"{base}#{ref}" if ref else base
    except Exception:
        return None

# ---------------------------- Summary construction ----------------------------

def build_markdown_summary(repo: RepoSnapshot, repo_url: str) -> str:
    lines: List[str] = []
    header = f"## {repo.owner}/{repo.name}"
    meta_bits = [f"[Repo]({repo_url})", f"Ref: `{repo.ref}`"]
    if repo.license_spdx and repo.license_spdx != "NOASSERTION":
        meta_bits.append(f"License: {repo.license_spdx}")
    if repo.topics:
        meta_bits.append("Topics: " + ", ".join(repo.topics[:8]))
    lines += [header, " • ".join(meta_bits), ""]

    if repo.description:
        lines += [repo.description.strip(), ""]

    overview = _first_readme_para(repo.files)
    if overview:
        lines += ["### Overview (README)", overview, ""]

    installs = _extract_install(repo.files)
    if installs:
        lines += ["### Installation"]
        for cmd, src in installs:
            lines.append(f"- `{cmd}`  — _source: {src}_")
        lines.append("")

    cli = _extract_cli(repo.files)
    if cli:
        lines += ["### Quickstart — CLI"]
        for cmd, src in cli:
            lines.append(f"- `{cmd}`  — _source: {src}_")
        lines.append("")

    api = _extract_api(repo.files)
    if api:
        lines += ["### Quickstart — Python"]
        for code, src in api:
            lines += ["```python", code, "```", f"_— source: {src}_", ""]
        # remove trailing blank line to keep tidy
        if lines and lines[-1] == "":
            lines.pop()

    inputs = _extract_section_text(repo.files, HEAD_INPUTS)
    if inputs:
        lines += ["", "### Inputs"]
        for txt, src in inputs:
            lines += [txt, f"_— source: {src}_", ""]

    outputs = _extract_section_text(repo.files, HEAD_OUTPUTS)
    if outputs:
        lines += ["", "### Outputs"]
        for txt, src in outputs:
            lines += [txt, f"_— source: {src}_", ""]

    reqs = _extract_requirements(repo.files)
    if reqs:
        lines += ["", "### Key dependencies"]
        for block, src in reqs:
            lines += ["```text", block, "```", f"_— source: {src}_"]

    links = _extract_urls(repo.files)
    if links:
        lines += ["", "### Helpful Links"]
        for u, src in links:
            lines.append(f"- {u}  — _source: {src}_")

    notable = _notable_paths(repo.files)
    if notable:
        lines += ["", "### Notable files/dirs"]
        for p in notable:
            lines.append(f"- `{p}`")

    if repo.truncated:
        lines += ["", "_Note: summary may be incomplete due to fetch limits._"]

    # Final tidy: ensure no duplicate blank lines at end
    out = "\n".join(lines).rstrip() + "\n"
    return out

# ----------------------------- Tool entry point -------------------------------

def tool_repo_summary(input: RepoSummaryInput) -> RepoSummaryOutput:
    """
    Summarize a GitHub repo conservatively (verbatim + provenance):
      - Fetches README/docs/examples/requirements/pyproject via GitHub REST API.
      - Builds a single Markdown string with quoted commands/snippets & sources.
      - No inferred tasks/modalities/formats → avoids misleading the agent.
    """
    token = os.getenv("GITHUB_TOKEN")
    snap = fetch_repo_snapshot_via_api(input.url, token=token)
    summary = build_markdown_summary(snap, input.url)
    return RepoSummaryOutput(truncated=snap.truncated, ref=snap.ref, summary=summary)