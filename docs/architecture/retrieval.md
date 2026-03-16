# Retrieval Pipeline

The retrieval stage is the first phase of the AI Imaging Agent's two-stage pipeline. It performs fast text-based search to find candidate tools from the software catalog.

## Overview

**Goal**: Quickly narrow down the software catalog to most relevant candidates

**Characteristics**:

- ⚡ Fast (~100-300ms)
- 🔢 Deterministic and reproducible
- 🚫 No LLM calls
- 💰 Low cost (no API fees)

## Pipeline Stages

```mermaid
graph LR
    A[User Query] --> B[Query Enhancement]
    B --> C[Embedding]
    C --> D[FAISS Search]
    D --> E[CrossEncoder Rerank]
    E --> F[Top-K Candidates]
```

## Step 1: Query Enhancement

### Format Token Injection

When users upload files, format tokens are added to the query:

```python
# User uploads: scan.dcm
# User query: "segment lungs"

# Enhanced query:
"segment lungs format:DICOM format:CT format:3D"
```

**Format tokens added**:

- File extension (`format:DICOM`, `format:NIfTI`)
- Image modality from metadata (`format:CT`, `format:MRI`)
- Dimensionality (`format:2D`, `format:3D`, `format:4D`)

**Why this helps**:

- Matches tools that support specific formats
- Boosts DICOM-compatible tools for DICOM input
- Ensures dimension compatibility (3D tools for volumes)

### Control Tag Processing

Special tags are extracted and processed:

```python
query = "segment lungs [EXCLUDE:tool1|tool2]"

# Extracted:
clean_query = "segment lungs"
excluded_tools = ["tool1", "tool2"]
```

**Supported tags**:
- `[EXCLUDE:tool1|tool2]`: Filter tools from results

## Step 2: Metadata-Aware Querying

The pipeline does not perform semantic vocabulary expansion. Instead, retrieval combines:

- cleaned task text
- format tokens inferred from uploaded files (for example `format:DICOM`)
- compact image metadata hints (modality/anatomy/dimensionality when available)

This keeps retrieval deterministic and closely tied to the user's data.

### Alternative Query Generation

On retry (when initial results < 5 tools):

```python
# Initial query
query1 = "segment rare pulmonary structure"
results = 2 tools  # Too few!

# Retry 1: Broader formulation (keep first 2-3 words)
query2 = "segment rare pulmonary"
results = 7 tools  # Better!

# Retry 2: If still insufficient, repeat with same broadening strategy
```

**Max retries**: 2

## Step 3: Embedding

### BGE-M3 Model

**Model**: `BAAI/bge-m3`

**Characteristics**:

- Multilingual (but used for English)
- 1024-dimensional embeddings
- Trained for retrieval tasks
- Fast inference (~10ms per query)

**Embedding process**:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-m3")
query_vector = model.encode(
    query,
    normalize_embeddings=True  # L2 normalization for cosine similarity
)
# Returns: np.array of shape (1024,)
```

### Catalog Embedding

Software tools are pre-embedded during indexing:

```python
# For each tool in catalog:
tool_text = f"{tool.name} {tool.description} {' '.join(tool.keywords)}"
tool_vector = model.encode(tool_text, normalize_embeddings=True)

# Store in FAISS index
faiss_index.add(tool_vector)
```

**Index structure**:

- FAISS IndexFlatIP (inner product = cosine similarity for normalized vectors)
- ~150 tools in current catalog
- Index size: ~600KB

## Step 4: FAISS Search

### Vector Search

FAISS performs fast similarity search:

```python
import faiss

# Search for top 20 most similar tools
scores, indices = faiss_index.search(
    query_vector.reshape(1, -1),
    k=20
)

# Returns:
# scores: [0.85, 0.82, 0.79, ...]  # Cosine similarities
# indices: [42, 17, 89, ...]        # Tool IDs in catalog
```

**Search algorithm**:

- IndexFlatIP: Exact search (brute force)
- Fast for catalog size (~150 tools)
- Could use IVF for larger catalogs (>10k tools)

**Why top-20**:

- More candidates than needed (default final: 8)
- Provides options for reranking
- Balances recall vs. later stage cost

### Candidate Retrieval

```python
candidates = [catalog[idx] for idx in indices[:20]]
candidate_scores = scores[:20].tolist()

# Example candidates:
[
    {
        "name": "TotalSegmentator",
        "score": 0.85,
        "description": "Automated multi-organ segmentation...",
        ...
    },
    ...
]
```

## Step 5: CrossEncoder Reranking

### Why Rerank?

**BiEncoder (BGE-M3)** limitations:

- Encodes query and documents independently
- No query-document interaction
- Misses subtle relevance signals

**CrossEncoder** benefits:

- Jointly encodes query + document
- Cross-attention between query and doc
- More accurate relevance scoring
- Slower (not suitable for entire catalog)

### Reranking Model

**Model**: `BAAI/bge-reranker-v2-m3`

**Characteristics**:

- Trained on MS-MARCO passage ranking
- 6 layers, fast inference (~50ms per pair)
- Direct relevance score (no embedding)

**Reranking process**:

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")

# Score each (query, candidate) pair
pairs = [(query, candidate.description) for candidate in candidates]
rerank_scores = reranker.predict(pairs)

# Re-sort by rerank scores
sorted_indices = np.argsort(rerank_scores)[::-1]
reranked_candidates = [candidates[i] for i in sorted_indices][:8]
```

**Output**: Top-8 candidates with refined ranking

## Output Format

### Candidate Schema

Each candidate passed to Stage 2:

```python
{
    "name": "TotalSegmentator",
    "description": "Automated multi-organ segmentation for CT and MRI",
    "url": "https://github.com/wasserth/TotalSegmentator",
    "keywords": ["segmentation", "medical-imaging", "CT", "MRI"],
    "license": "Apache-2.0",
    "supporting_data": {
        "modalities": ["CT", "MRI"],
        "dimensions": ["3D"],
        "formats": ["DICOM", "NIfTI"],
        "demo_url": "https://huggingface.co/spaces/..."
    },
    "retrieval_score": 0.85,  # FAISS or rerank score
}
```

**Fields used by VLM**:

- Essential for understanding tool capability
- Formatted as table in VLM prompt
- Enables comparative reasoning

## Index Management

### Building the Index

Done during catalog sync:

```bash
ai_agent sync
```

**Process**:

1. Load catalog JSONL
2. Embed each tool description
3. Build FAISS index
4. Save to disk: `artifacts/rag_index/`

**Files**:
```
artifacts/rag_index/
├── index.faiss          # FAISS binary index
└── meta.json            # Metadata (tool IDs, config)
```

### Loading the Index

At startup:

```python
from retriever.vector_index import VectorIndex

index = VectorIndex()
index.load("artifacts/rag_index")

# Ready for queries
results = index.search(query, k=20)
```

### Updating the Index

When catalog changes:
1. Sync detects new/modified tools
2. Re-embed entire catalog (fast, ~2 seconds)
3. Rebuild FAISS index
4. Reload in pipeline (no restart needed)

## Performance Optimization

### Caching

**Model loading**:

- BGE-M3 and CrossEncoder loaded once at startup
- Kept in memory for entire session

**Index loading**:

- FAISS index loaded once
- Small enough to fit in memory (~MB)

### Batch Processing

For multiple queries (testing, batch mode):

```python
# Batch embed multiple queries
query_vectors = model.encode(queries, batch_size=32)

# Batch FAISS search
scores, indices = index.search(query_vectors, k=20)
```

### GPU Acceleration

Models can use GPU if available:

```python
model = SentenceTransformer("BAAI/bge-m3", device="cuda")
reranker = CrossEncoder("...", device="cuda")
```

## Retrieval Metrics

### Monitored Metrics

During retrieval:

- **Number of candidates found**: Should be ≥8 for good coverage
- **Average similarity score**: Higher = better match
- **Reranking impact**: Score change after reranking
- **Retry usage**: Whether broadening retry was triggered

### Logging

Retrieval events logged:

```
INFO retriever.vector_index: FAISS search: query="segment lungs", results=20, top_score=0.85
INFO retriever.reranker: Reranking 20 candidates, top_score_change=+0.12
INFO retriever.pipeline: Final candidates: 8, avg_score=0.78
```

## Limitations

### Current Limitations

1. **English only**: No multilingual support (though model is capable)
2. **Small catalog**: ~150 tools (FAISS overkill, but scales)
3. **No filtering**: Can't filter by license, modality in retrieval (done in Stage 2)
4. **Heuristic retries**: Broadening strategy is simple prefix-based shortening

### Future Enhancements

- **Hybrid search**: Combine semantic + keyword (BM25)
- **Metadata filters**: Pre-filter by modality, license, format
- **Personalization**: User history, preferences
- **Adaptive retries**: Learn better broadening formulations from query logs

## Next Steps

- Learn about [Agent & VLM Selection](agent.md)
- Explore [Software Catalog](catalog.md)
- Return to [Architecture Overview](overview.md)
