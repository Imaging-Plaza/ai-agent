# Advanced Features (not tested for now..)

The AI Imaging Agent includes several advanced features for power users and specialized use cases.

## Control Tags

Control tags modify agent behavior using special syntax in your queries.

### Exclude Tools

Filter out specific tools from results:

```
Find lung segmentation tools [EXCLUDE:totalsegmentator|medicalsam]
```

**Syntax**: `[EXCLUDE:tool1|tool2|tool3]`

**Use cases**:

- You've already tried certain tools
- Exclude tools you don't have access to
- Filter by licensing (exclude proprietary tools)
- Remove tools with specific limitations

**Example**:
```
You: Segment kidneys [EXCLUDE:totalsegmentator]
Agent: [Returns kidney segmentation tools except TotalSegmentator]

You: Find open-source options [EXCLUDE:proprietarytool1|proprietarytool2]
Agent: [Returns only open-source tools]
```

### Disable Reranking

Skip the CrossEncoder reranking step for faster results:

```
Find segmentation tools [NO_RERANK]
```

**Benefits**:

- ✅ Faster retrieval (~2x speedup)
- ✅ Lower computational cost
- ✅ Good for broad exploratory queries

**Trade-offs**:

- ❌ Potentially less accurate ranking
- ❌ May include less relevant tools
- ❌ Semantic matching only (no cross-attention)

**When to use**:

- Quick exploration
- Very specific queries (already well-targeted)
- When speed matters more than precision

### Force Refinement

Request clarification even with good results:

```
Segment this image [REFINE]
```

**Results in**:

- Agent asks clarifying questions
- More focused recommendations
- Opportunity to specify requirements

**Example**:
```
You: Analyze this CT scan [REFINE]
Agent: I can see this is a CT scan. What specific analysis are you interested in?
      - Organ segmentation
      - Tumor detection
      - Bone analysis
      - Vascular analysis

You: Tumor detection
Agent: [Provides tumor detection tools specifically]
```

## Alternative Searches

Request the agent to search with different strategies.

### Requesting Alternatives

Use natural language:

```
Can you search for alternatives?
Show me other options
Find different tools
What else is available?
```

**What happens**:

- Agent formulates alternative query
- Uses semantic neighbors for expansion
- Searches with different emphasis
- Returns new set of recommendations

**Limit**: Up to 3 alternative searches per conversation

### When to Use

- Initial results don't quite match
- Want to see different approaches
- Exploring the catalog
- Looking for specialized tools

**Example conversation**:
```
You: Segment lungs from this CT
Agent: [Provides general lung segmentation tools]

You: Can you search for alternatives?
Agent: [Searches with emphasis on "airway segmentation", "pulmonary analysis"]

You: Show me other options
Agent: [Searches with emphasis on "CT thorax processing", "respiratory imaging"]
```

## Multi-Model Support

### Selecting Different Models

The UI provides a model selector dropdown:

Available models (configurable in `config.yaml`):

- **gpt-4o-mini**: Faster, lower cost
- **gpt-4o**: Higher accuracy, multimodal
- **gpt-5.1**: Latest capabilities (if available)
- **Custom endpoints**: EPFL, local servers, etc.

### Model Trade-offs

| Model | Speed | Cost | Accuracy | Vision |
|-------|-------|------|----------|--------|
| gpt-4o-mini | ⚡⚡⚡ | 💰 | ⭐⭐⭐ | ✅ |
| gpt-4o | ⚡⚡ | 💰💰 | ⭐⭐⭐⭐ | ✅✅ |
| gpt-5.1 | ⚡ | 💰💰💰 | ⭐⭐⭐⭐⭐ | ✅✅✅ |

### When to Switch Models

**Use gpt-4o-mini when**:

- Doing quick explorations
- Cost is a concern
- Tasks are straightforward
- Query is well-specified

**Use gpt-4o when**:

- Complex visual analysis needed
- Accuracy is critical
- Ambiguous queries
- Multi-step reasoning required

**Use gpt-5.1 when**:

- Maximum accuracy needed
- Complex multi-modal tasks
- Research/publication work

## Repository Info Tool

### What It Does

The agent can fetch detailed information about GitHub repositories:

```
You: Tell me about TotalSegmentator
Agent: [Fetches repo info from GitHub via DeepWiki or repocards]

Repository: wasserth/TotalSegmentator
Description: Automated multi-organ segmentation in CT and MR images
Stars: 1.2k
Language: Python
Topics: segmentation, medical-imaging, deep-learning
Last Updated: 2024-03-15
License: Apache-2.0
```

### Data Sources

1. **DeepWiki MCP** (primary): Fast, pre-indexed repository documentation
2. **Repocards** (fallback): Direct library-based fetch

### Usage

Ask about tools naturally:

```
What is [tool name]?
Tell me more about [repository]
Show me details for [tool]
```

## Conversation State Management

### State Tracking

The agent maintains state across conversation:

- **Uploaded files**: All files in session
- **Preview images**: Converted images for VLM
- **Excluded tools**: Tools filtered via `[EXCLUDE:]`
- **Conversation history**: Previous messages and context
- **Turn counter**: Current conversation turn

### Viewing State

In the sidebar (debug mode):

```json
{
  "conversation_turn": 3,
  "uploaded_files": ["scan.dcm", "brain.nii"],
  "excluded_tools": ["tool1", "tool2"],
  "preview_images": ["/tmp/scan_preview.png"]
}
```

### Resetting State

To start fresh:
- Refresh the page
- Clear uploaded files
- Start new conversation

## Query Expansion

### How It Works

Your query is automatically expanded with semantic neighbors:

```
Original: "segment brain"
Expanded: "segment brain segmentation parcellation extraction
           anatomy neuroimaging cranial"
```

**Based on**:

- BGE-M3 embeddings
- Catalog vocabulary
- Cosine similarity >0.75
- Top 10 neighbors

### Benefits

- ✅ Finds tools using different terminology
- ✅ Broader coverage of catalog
- ✅ Handles synonyms automatically
- ✅ No manual synonym dictionaries

### Customization

Expansion is automatic but considers:
- Your exact query terms (boosted weight)
- Semantically similar terms
- Format tokens from uploaded files

## Format-Aware Matching

### Input Format Tokens

File uploads add format tokens to queries:

```
Uploaded: scan.dcm (DICOM)
Query enhancement: "segment lungs format:DICOM format:CT format:3D"
```

### How It Helps

- **Narrows results**: Shows compatible tools first
- **Boosts relevance**: DICOM tools rank higher for DICOM
- **Compatibility check**: Agent verifies format support

### Supported Formats

Tokens added for:
- File extension (`.dcm`, `.nii`, `.png`)
- Detected format (DICOM, NIfTI, TIFF)
- Modality for medical images (CT, MRI, XR)
- Dimensions (2D, 3D, 4D)

## Iterative Retrieval

### Auto-Retry on Low Results

If initial search returns <5 candidates:

1. **Retry #1**: Alternative query with semantic expansion
2. **Retry #2**: Further expansion with broader terms
3. **Max 2 retries**: Then return best available

### Why It Matters

- Handles rare/specialized queries
- Finds tools even with limited matches
- Automatic - no user action needed

### Example

```
Query: "segment rare anatomical structure"
Initial: 2 candidates found
Retry 1: Expanded to "segment anatomy structure region organ"
Result: 7 candidates found ✓
```

## Debug Features

### Prompt Logging

Enable in `.env`:

```dotenv
LOG_PROMPTS=1
```

**Saves**:

- VLM prompts sent to API
- Images included in prompts
- Response JSON
- Timestamp and metadata

**Location**: `logs/prompts/YYYYMMDD_HHMMSS/`

**Contents**:
```
logs/prompts/20240315_143022/
├── prompt.txt          # Text prompt
├── image_0.png         # Uploaded image
├── response.json       # API response
└── metadata.json       # Request metadata
```

### Execution Traces

Always shown in chat (expandable):

```html
<details>
<summary>🔧 Execution Trace</summary>
...detailed logs...
</details>
```

Shows:

- Tool calls made
- Parameters used
- API responses
- Timing information

## Catalog Synchronization

### Auto-Refresh

Configured via `.env`:

```dotenv
SYNC_EVERY_HOURS=24
```

**Behavior**:

- Background thread checks for catalog updates
- Reloads FAISS index if changed
- No UI interruption
- Logs refresh activity

### Manual Sync

Force synchronization:

```bash
ai_agent sync
```

Updates:

- Software catalog
- Embeddings
- FAISS index
- Vocabulary for expansion

## Advanced Configuration

### Custom Catalog

Use your own tool catalog:

```dotenv
SOFTWARE_CATALOG=/path/to/custom_catalog.jsonl
```

**Format**: JSONL with schema.org SoftwareSourceCode

### API Endpoints

Configure custom OpenAI-compatible endpoints in `config.yaml`:

```yaml
available_models:
  - display_name: "Local LLM"
    name: "llama-3.1"
    base_url: "http://localhost:8000/v1"
    api_key_env: "LOCAL_API_KEY"
```

### Pipeline Parameters

Fine-tune retrieval:

```dotenv
TOP_K=8              # Candidates to retrieve
NUM_CHOICES=3        # Final recommendations
RERANK_TOP_N=20      # Candidates before reranking
```

## Next Steps

- Dive into [Architecture Overview](../architecture/overview.md)
- Learn about [Development and Contributing](../development/contributing.md)
- Check [Environment Variables Reference](../reference/environment.md)
