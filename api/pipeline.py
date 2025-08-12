# api/pipeline.py
from __future__ import annotations
import shutil, os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from retriever.embedders import LocalBGEEmbedder, VectorIndex, IndexItem, CrossEncoderReranker, SoftwareDoc
from generator.generator import PlanAndCodeGenerator
from generator.schema import CandidateDoc, PerceptionCues
from executor.space_runner import call_space_flow, call_space_with_file, SpaceRunError


class RAGImagingPipeline:
    def __init__(self, docs: list[SoftwareDoc], workdir: str = "runs", hf_token: Optional[str] = None):
        self.workdir = Path(workdir); self.workdir.mkdir(parents=True, exist_ok=True)
        self.embedder = LocalBGEEmbedder()
        self.index = VectorIndex(self.embedder)
        self.index.upsert([IndexItem(id=d.name, doc=d) for d in docs])
        self.reranker = CrossEncoderReranker()
        self.gen = PlanAndCodeGenerator()
        self.hf_token = hf_token

    def recommend(self, query: str, top_k: int = 5) -> list[dict]:
        hits = self.index.search(query, k=20, reranker=self.reranker, rerank_top_k=top_k)
        return hits

    def select_with_generator(self, query: str, hits: list[dict], cues: Optional[PerceptionCues], image_path: str) -> Tuple[dict, dict]:
        # Prepare candidates for the generator
        candidates = [CandidateDoc(**h["doc"].model_dump()) for h in hits[:5]]
        result = self.gen.generate(
            user_task=query,
            candidates=candidates,
            image_path=image_path,
            out_mask_path=str(self.workdir / "mask.nii.gz"),
            overlay_png_path=str(self.workdir / "overlay.png"),
            cues=cues,
        )
        # Find the chosen doc
        chosen = next((h for h in hits if h["doc"].name == result.choice), hits[0])
        return chosen, result.model_dump()

    def run_space_for_choice(self, chosen, image_path):
        doc = chosen["doc"]
        timeout = getattr(doc, "space_timeout", None) or 1200
        if getattr(doc, "hf_calls", None):
            return call_space_flow(
                hf_space=doc.hf_space,
                image_path=image_path,
                calls=doc.hf_calls,
                timeout=timeout,
                hf_token=self.hf_token,
            )
        # fallback single-call (only for Spaces that really have one endpoint)
        return call_space_with_file(
            hf_space=doc.hf_space,
            image_path=image_path,
            api_name=getattr(doc, "hf_api_name", None),
            timeout=timeout,
            hf_token=self.hf_token,
        )


    def recommend_and_run(
        self,
        image_path: str,
        user_task: str,
        cues: Optional[PerceptionCues] = None,
    ) -> Dict[str, Any]:
        hits = self.recommend(user_task, top_k=5)
        if not hits:
            return {"error": "No candidates found."}

        chosen, gen_json = self.select_with_generator(user_task, hits, cues, image_path)

        # Try primary; if it fails, try alternates with hf_space
        errors = []
        alt_hits = [chosen] + [h for h in hits if h["doc"].name != chosen["doc"].name]
        for cand in alt_hits:
            try:
                image_out, raw_outputs = self.run_space_for_choice(cand, image_path)
                return {
                    "choice": cand["doc"].name,
                    "alternates": [h["doc"].name for h in hits if h["doc"].name != cand["doc"].name][:2],
                    "why": gen_json.get("why", ""),
                    "result_image": image_out,
                    "raw_outputs": raw_outputs,
                    "generator": gen_json,
                }
            except Exception as e:
                errors.append(str(e))
                continue

        return {"error": "All Spaces failed.", "errors": errors, "generator": gen_json}
