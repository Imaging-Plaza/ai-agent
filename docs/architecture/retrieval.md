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

### Embedder

The embedder is configured in `config.yaml` under `retrieval.embedder`. Two backends are supported:

| Backend | Description | Default |
|---------|-------------|---------|
| `remote` | OpenAI-compatible HTTP embeddings endpoint | ✅ Yes |
| `local` | Local `sentence-transformers` model | No |

**Default configuration (remote)**:

```yaml
retrieval:
  embedder:
    backend: "remote"
    model_name: "Qwen/Qwen3-Embedding-8B"
    base_url: "https://inference-rcp.epfl.ch/v1"
    api_key_env: "EPFL_API_KEY_EMBEDDER"
    timeout_s: 20
```

**Local backend example**:

```yaml
retrieval:
  embedder:
    backend: "local"
    model_name: "BAAI/bge-m3"
```

The local backend uses `sentence-transformers` directly on the host machine. The remote backend sends requests to any OpenAI-compatible embeddings endpoint.

**Query and corpus prefixes** (applied automatically):

- Query: `"Represent the query for retrieving relevant software: <text>"`
- Corpus: `"Represent the software for retrieval: <text>"`

### Catalog Embedding

Software tools are pre-embedded at startup (unless `EMBED_CATALOG_ON_START=0`):

1. The pipeline reads `SOFTWARE_CATALOG` (default: `dataset/catalog.jsonl`)
2. Each tool is converted to an `IndexItem` and passed to `VectorIndex.sync_with_catalog()`
3. The index is saved to `artifacts/rag_index/`

**Index structure**:

- FAISS IndexFlatIP (inner product, works with normalized vectors)
- Contains all tools from the catalog
- Saved as `index.faiss` + `meta.json`

## Step 4: FAISS Search

### Vector Search

FAISS performs fast similarity search using the IndexFlatIP algorithm:

- **IndexFlatIP**: Exact (brute force) inner product search — suitable for catalog sizes up to ~10k tools
- The top-N candidates (default: 12 per tool call) are retrieved by cosine similarity

### Candidate Retrieval

FAISS returns candidate indices which are resolved to `SoftwareDoc` objects. These are filtered to remove any tools in the excluded list, then passed to the reranker.

## Step 5: CrossEncoder Reranking

### Why Rerank?

**Bi-encoder** limitations:

- Encodes query and documents independently
- No query-document interaction
- Misses subtle relevance signals

**CrossEncoder** benefits:

- Jointly encodes query + document
- Cross-attention between query and doc
- More accurate relevance scoring
- Slower (suitable only for a small candidate set)

### Reranking Model

The reranker is configured in `config.yaml` under `retrieval.reranker`. Two backends are supported:

| Backend | Description | Default |
|---------|-------------|---------|
| `remote` | OpenAI-compatible HTTP reranking endpoint | ✅ Yes |
| `local` | Local `sentence-transformers` CrossEncoder | No |

**Default configuration (remote)**:

```yaml
retrieval:
  reranker:
    backend: "remote"
    model_name: "BAAI/bge-reranker-v2-m3"
    base_url: "https://inference-rcp.epfl.ch/v1"
    api_key_env: "EPFL_API_KEY_EMBEDDER"
    timeout_s: 20
```

**Local backend example**:

```yaml
retrieval:
  reranker:
    backend: "local"
    model_name: "BAAI/bge-reranker-v2-m3"
```

!!! note
    If the reranker API key (`EPFL_API_KEY_EMBEDDER`) is not set, reranking is **disabled** and original FAISS scores are used instead.

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

The FAISS index is built automatically at startup by `RAGImagingPipeline` (see `EMBED_CATALOG_ON_START`). You can also force a rebuild via:

```bash
ai_agent sync
```

The `sync` command queries a GraphDB SPARQL endpoint (see [Catalog Sync](catalog.md)), converts the results to JSONL, and rebuilds the FAISS index.

**Stored artifacts**:

```
artifacts/rag_index/
├── index.faiss          # FAISS binary index
└── meta.json            # Metadata (tool IDs, embedding config)
```

### Hot-Reload

When catalog contents change (detected via SHA-1 hash), the pipeline reloads the index without restarting:

```python
ok = pipeline.reload_index()   # returns True on success
```

The auto-refresh background thread calls this automatically when `SYNC_EVERY_HOURS > 0`.
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
