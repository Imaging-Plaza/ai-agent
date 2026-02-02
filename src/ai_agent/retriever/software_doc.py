from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class SoftwareDoc(BaseModel):
    """
    Minimal software doc used for retrieval and ranking.
    Tolerant to catalog variation (extra='ignore') and normalizes common types.
    Also derives dims/anatomy/modality from supportingData if missing.
    """

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    # Identity
    name: str
    url: Optional[str] = None
    repo_url: Optional[str] = None
    description: Optional[str] = None
    documentation: Optional[str] = None
    
    @field_validator("name", mode="before")
    @classmethod
    def _coerce_name_from_list(cls, v):
        """Handle name field that might be a list (common in catalog)."""
        if isinstance(v, list):
            return v[0] if v else "unknown"
        return v

    # Semantics
    category: List[str] = Field(default_factory=list, alias="applicationCategory")
    tasks: List[str] = Field(default_factory=list, alias="featureList")
    modality: List[str] = Field(default_factory=list, alias="imagingModality")
    keywords: List[str] = Field(default_factory=list)

    # Anatomy / dims
    dims: List[int] = Field(default_factory=list)  # derived from supportingData[*].hasDimensionality when absent
    anatomy: List[str] = Field(default_factory=list)  # derived from supportingData[*].bodySite when absent

    # Tech details
    programming_language: Optional[str] = Field(default=None, alias="programmingLanguage")
    software_requirements: List[str] = Field(default_factory=list, alias="softwareRequirements")
    gpu_required: Optional[bool] = Field(default=None, alias="requiresGPU")
    is_free: Optional[bool] = Field(default=None, alias="isAccessibleForFree")
    is_based_on: List[str] = Field(default_factory=list, alias="isBasedOn")
    plugin_of: List[str] = Field(default_factory=list, alias="isPluginModuleOf")
    related_organizations: List[str] = Field(default_factory=list, alias="relatedToOrganization")
    license: Optional[str] = None

    # Misc
    os: List[str] = Field(default_factory=list)

    # Demo / Spaces info
    runnable_example: List[Union[str, Dict[str, Any]]] = Field(
        default_factory=list, alias="runnableExample"
    )
    has_executable_notebook: List[Union[str, Dict[str, Any]]] = Field(
        default_factory=list, alias="hasExecutableNotebook"
    )

    @field_validator("runnable_example", "has_executable_notebook", mode="before")
    @classmethod
    def _coerce_list_any(cls, v):
        if v is None:
            return []
        return v if isinstance(v, list) else [v]

    # --------- Derive dims/anatomy/modality from supportingData BEFORE field validators ---------
    @model_validator(mode="before")
    @classmethod
    def _derive_from_supporting_data(cls, data: Any):
        if not isinstance(data, dict):
            return data

        sd = data.get("supportingData")
        if not sd:
            return data

        # normalize to list of dicts
        items = sd if isinstance(sd, list) else [sd]

        # collect from nested records
        dims_collected: List[int] = []
        anatomy_collected: List[str] = []
        mod_extra: List[str] = []
        fmt_tokens: List[str] = [] 

        def push_dim(x):
            try:
                xi = int(x)
                if xi not in dims_collected:
                    dims_collected.append(xi)
            except Exception:
                # tolerate "3D"/"2-D"/"volumetric" etc.
                if isinstance(x, str):
                    s = x.strip().lower().replace(" ", "")
                    if s in {"2", "2d", "2-d"}:
                        if 2 not in dims_collected:
                            dims_collected.append(2)
                            return
                    if s in {"3", "3d", "3-d", "volume", "volumetric", "stack"}:
                        if 3 not in dims_collected:
                            dims_collected.append(3)
                            return
                    if s in {"4", "4d", "4-d", "timeseries", "time-series", "temporal"}:
                        if 4 not in dims_collected:
                            dims_collected.append(4)
                            return
                    digits = "".join(ch for ch in s if ch.isdigit())
                    if digits:
                        try:
                            xi = int(digits)
                            if xi not in dims_collected:
                                dims_collected.append(xi)
                        except Exception:
                            pass

        for it in items:
            if not isinstance(it, dict):
                continue

            # hasDimensionality
            hd = it.get("hasDimensionality")
            if hd is not None:
                if isinstance(hd, list):
                    for v in hd:
                        push_dim(v)
                else:
                    push_dim(hd)

            # bodySite
            bs = it.get("bodySite")
            if bs is not None:
                vals = bs if isinstance(bs, list) else [bs]
                for v in vals:
                    s = str(v).strip()
                    if s and s not in anatomy_collected:
                        anatomy_collected.append(s)

            # imagingModality (nested)
            im = it.get("imagingModality")
            if im is not None:
                vals = im if isinstance(im, list) else [im]
                for v in vals:
                    s = str(v).strip()
                    if s and s not in mod_extra:
                        mod_extra.append(s)

            # datasetFormat
            fm = it.get("datasetFormat")
            if fm is not None:
                vals = fm if isinstance(fm, list) else [fm]
                for v in vals:
                    s = str(v or "").strip().lower()
                    if not s:
                        continue
                    # accept mime types, dotted or bare extensions
                    s = s.split("/")[-1]
                    if s.startswith("."):
                        s = s[1:]
                    if s:
                        tok = f"format:{s}"
                        if tok not in fmt_tokens:
                            fmt_tokens.append(tok)

        # populate only if missing/empty at top-level
        if not data.get("dims") and dims_collected:
            data["dims"] = dims_collected
        if not data.get("anatomy") and anatomy_collected:
            data["anatomy"] = anatomy_collected
        # merge nested imagingModality into top-level if present
        if mod_extra:
            top = data.get("imagingModality") or data.get("modality") or []
            top_list = top if isinstance(top, list) else [top]
            merged = []
            for v in top_list + mod_extra:
                s = str(v).strip()
                if s and s not in merged:
                    merged.append(s)
            data["imagingModality"] = merged

        if fmt_tokens:
            kws = data.get("keywords") or []
            kws = kws if isinstance(kws, list) else [kws]
            for t in fmt_tokens:
                if t not in kws:
                    kws.append(t)
            data["keywords"] = kws

        return data

    # ---------- validators / coercers ----------
    @staticmethod
    def _canon_lang(s: str) -> str:
        t = (s or "").strip()
        low = t.lower().replace(" ", "").replace("-", "")
        mapping = {
            "cpp": "C++",
            "cplusplus": "C++",
            "c++": "C++",
            "csharp": "C#",
            "c#": "C#",
            "py": "Python",
            "python": "Python",
            "matlab": "MATLAB",
            "javascript": "JavaScript",
            "js": "JavaScript",
            "julia": "Julia",
            "r": "R",
            "java": "Java",
            "cuda": "CUDA",
            "go": "Go",
            "rust": "Rust",
        }
        return mapping.get(low, t)

    @staticmethod
    def _norm_url_one(v: Any) -> Optional[str]:
        if not v:
            return None
        s = str(v).strip()
        if not s:
            return None
        try:
            u = urlsplit(s)
            scheme = (u.scheme or "").lower()
            netloc = (u.netloc or "").lower()
            if netloc.startswith("www."):
                netloc = netloc[4:]
            path = (u.path or "").rstrip("/")
            if path.endswith(".git"):
                path = path[:-4]
            return urlunsplit((scheme, netloc, path, "", ""))
        except Exception:
            return s

    @staticmethod
    def _as_list_of_str(v) -> List[str]:
        if v is None:
            return []
        if isinstance(v, list):
            out = []
            for x in v:
                if x is None:
                    continue
                s = str(x).strip()
                if s and s not in out:
                    out.append(s)
            return out
        s = str(v).strip()
        return [s] if s else []

    @field_validator(
        "category",
        "tasks",
        "modality",
        "keywords",
        "software_requirements",
        "is_based_on",
        "related_organizations",
        "os",
        "plugin_of",
        mode="before",
    )
    @classmethod
    def _coerce_list_strs(cls, v):
        return cls._as_list_of_str(v)

    @field_validator("programming_language", "license", mode="before")
    @classmethod
    def _coerce_scalar_from_list(cls, v):
        if isinstance(v, list):
            vals = sorted({str(x).strip() for x in v if isinstance(x, str) and x.strip()})
            if not vals:
                return None
            pick = vals[0]
            return cls._canon_lang(pick) if "programming_language" in cls.__fields__ else pick
        if isinstance(v, str):
            return cls._canon_lang(v) if "programming_language" in cls.__fields__ else v
        return v

    @field_validator("url", "repo_url", "documentation", mode="before")
    @classmethod
    def _coerce_and_normalize_url(cls, v):
        if isinstance(v, list):
            cands = [cls._norm_url_one(x) for x in v]
            cands = sorted({x for x in cands if x})
            return cands[0] if cands else None
        return cls._norm_url_one(v)

    @field_validator("description", mode="before")
    @classmethod
    def _coerce_description(cls, v):
        if v is None:
            return None
        if isinstance(v, list):
            for x in v:
                s = str(x).strip()
                if s:
                    return s
            return None
        return str(v).strip()

    @field_validator("gpu_required", "is_free", mode="before")
    @classmethod
    def _coerce_bool(cls, v):
        if isinstance(v, list):
            b = cls._coerce_bool(v[0])
            if b is not None:
                return b
            return None
        if isinstance(v, bool) or v is None:
            return v
        if isinstance(v, (int, float)):
            return bool(v)
        if isinstance(v, str):
            s = v.strip().lower()
            if s in {"true", "1", "yes", "y", "on"}:
                return True
            if s in {"false", "0", "no", "n", "off"}:
                return False
        return None

    @field_validator("dims", mode="before")
    @classmethod
    def _coerce_dims(cls, v):
        if v is None:
            return []
        items = v if isinstance(v, list) else [v]
        out: List[int] = []

        def push(x):
            try:
                xi = int(x)
                if xi not in out:
                    out.append(xi)
            except Exception:
                pass

        for it in items:
            if isinstance(it, (int, float)):
                push(it)
                continue
            if not isinstance(it, str):
                continue
            s = it.strip().lower().replace(" ", "")
            if s in {"2", "2d", "2-d"}:
                push(2)
                continue
            if s in {"3", "3d", "3-d", "volume", "volumetric", "stack"}:
                push(3)
                continue
            if s in {"4", "4d", "4-d", "timeseries", "time-series", "temporal"}:
                push(4)
                continue
            digits = "".join(ch for ch in s if ch.isdigit())
            if digits:
                push(digits)
        return out

    def to_retrieval_text(self) -> str:
        """
        Generate text representation for retrieval.
        
        Strategy:
        1. Include all semantic fields without expansion (expansion happens at query-time)
        2. Repeat critical fields (tasks, modality, anatomy) for better matching
        3. Keep less critical metadata at the end for context
        """
        parts = []
        
        # Name (high importance)
        if self.name:
            parts.append(self.name)
        
        # Tasks (repeated 3x) - HIGHEST PRIORITY
        if self.tasks:
            tasks_str = " ".join(self.tasks)
            parts.extend([tasks_str, tasks_str, tasks_str])
        
        # Anatomy (repeated 2x)
        if self.anatomy:
            anatomy_str = " ".join(self.anatomy)
            parts.extend([anatomy_str, anatomy_str])
        
        # Modality (repeated 2x)
        if self.modality:
            modality_str = " ".join(self.modality)
            parts.extend([modality_str, modality_str])
        
        # Dimensions (as-is from catalog)
        if self.dims:
            dim_terms = [f"{d}D" for d in self.dims]
            parts.append(" ".join(dim_terms))
        
        # Category and keywords
        if self.category:
            parts.append(" ".join(self.category))
        if self.keywords:
            parts.append(" ".join(self.keywords))
        
        # Description (provides context)
        if self.description:
            parts.append(self.description)
        
        # Secondary metadata
        if self.programming_language:
            parts.append(f"language:{self.programming_language}")
        if self.plugin_of:
            parts.append(f"plugin:{' '.join(self.plugin_of)}")
        if self.is_based_on:
            parts.append(f"based_on:{' '.join(self.is_based_on)}")
        
        return " ".join(p for p in parts if p)
