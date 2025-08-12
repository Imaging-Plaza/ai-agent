"""
examples/wireup.py

End-to-end demo:
1) build a small software index (FAISS over BGE-M3 embeddings),
2) run a query + cross-encoder reranking,
3) pass top candidates to the generator (LLM) to pick a tool and emit a tiny demo script.

Notes:
- The generator defaults to an OpenAI backend if OPENAI_API_KEY is set.
- Your sample SoftwareDoc supports TIFF/TIF inputs, while your perception cue prefers NIfTI.
  This mismatch is intentional here; it helps you observe how the generator justifies choices.
"""

import logging
from retriever.embedders import (
    LocalBGEEmbedder, VectorIndex, IndexItem, CrossEncoderReranker, SoftwareDoc
)
from generator.generator import PlanAndCodeGenerator
from generator.schema import CandidateDoc, PerceptionCues

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=False) 

# --- logging setup (feel free to remove if you don't want console logs) ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("wireup")

# 1) Build the index ----------------------------------------------------------
# Create a local embedding model (BAAI/bge-m3) and a cosine-similarity FAISS index
log.info("Initializing embedder and vector index...")
embedder = LocalBGEEmbedder()
index = VectorIndex(embedder)

# Define a tiny catalog of software entries (normally you'd load many from JSONL)
# This single entry claims: CT, 3D, segmentation, Python, CPU OK, TIFF/TIF inputs.
docs = [
    SoftwareDoc(
        name="lungs-segmentation",
        tasks=["segmentation"],
        modality=["CT"],
        dims=["3D"],
        input_formats=["TIFF", "TIF"],
        output_types=["mask"],
        language="Python",
        weights_available=True,
        gpu_required=False,
        os=["Linux", "Windows"],
        license="MIT",
        description=(
            "A Python package for lungs segmentation in CT scan images using deep learning."
        ),
    ),
    # ... add more SoftwareDoc() entries as you expand the catalog
]

# Embed the documents and add them to the FAISS index
log.info("Upserting %d document(s) into the index...", len(docs))
index.upsert([IndexItem(id=d.name, doc=d) for d in docs])

# 2) Query + rerank -----------------------------------------------------------
# Craft a user query; the reranker (bge-reranker-v2-m3) sharpens the top-K results.
query = "CT lungs segmentation with output mask. Python preferred."
log.info("Searching for: %s", query)
reranker = CrossEncoderReranker()

# Search top-20 via FAISS, then rerank to top-5 via cross-encoder
hits = index.search(query, k=20, reranker=reranker, rerank_top_k=5)
log.info("Retrieved %d hit(s) after reranking.", len(hits))
for i, h in enumerate(hits, 1):
    log.info(
        "Hit %d | id=%s | sim=%.3f | rerank=%.3f",
        i,
        h["id"],
        h.get("score", float("nan")),
        h.get("rerank_score", float("nan")),
    )

# 3) Prepare candidates for the generator ------------------------------------
# Convert the top hits into lightweight CandidateDoc objects
top_k = min(5, len(hits))
candidates = [CandidateDoc(**h["doc"].model_dump()) for h in hits[:top_k]]
log.info("Prepared %d candidate(s) for the generator.", len(candidates))

# 4) (Optional) Perception cues from your VLM front-end ----------------------
# These are structured hints extracted from the query image (or user input).
# Here we hardcode them for the example.
cues = PerceptionCues(
    modality="CT",
    dims="3D",
    anatomy="lung",
    task="segmentation",
    io_hint="NIfTI",  # note: your only doc currently lists TIFF/TIF inputs
)
log.info("Perception cues: %s", cues.model_dump())

# 5) Generate plan + code -----------------------------------------------------
# Instantiate the plan/code generator. By default it uses OpenAI if OPENAI_API_KEY is set.
gen = PlanAndCodeGenerator()
log.info(f"Calling the generator using provider {gen.provider.__class__.__name__} to select the best tool and emit code...")

# Ask the generator to produce {choice, alternates, why, steps, code}
result = gen.generate(
    user_task=query,
    candidates=candidates,
    image_path="input_image.nii.gz",  # input path the snippet should read
    out_mask_path="mask.nii.gz",      # where the snippet should write the mask
    overlay_png_path="overlay.png",   # where the snippet should save a visual overlay
    cues=cues,
)

print("Provider:", gen.last_provider)
print("\n--- SYSTEM PROMPT ---\n", gen.last_request["system"])
print("\n--- USER PROMPT ---\n", gen.last_request["user"])
print("\n--- USAGE ---\n", gen.last_usage)

# 6) Display results ----------------------------------------------------------
# Print the structured answer for quick inspection.
print("\n=== GENERATOR RESULT ===")
print("Choice:", result.choice)
print("Alternates:", result.alternates)
print("Why:", result.why)
print("Steps:")
for s in result.steps:
    print(" -", s)
print("\nCode:\n", result.code)
print("========================\n")

# Optional tip: If you wire an executor, you can now run `result.code` in your Docker sandbox
# to produce `mask.nii.gz` and `overlay.png`, and capture logs/errors for the UI.