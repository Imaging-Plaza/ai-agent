# Software Catalog

The software catalog is the foundation of the AI Imaging Agent, containing curated information about imaging analysis tools.

## Overview

**Format**: JSON Lines (JSONL)
**Location**: `dataset/catalog.jsonl`
**Schema**: Based on schema.org SoftwareSourceCode
**Size**: ~150 tools currently

## Catalog Schema

### Core Fields

Based on [schema.org/SoftwareSourceCode](https://schema.org/SoftwareSourceCode):

```json
{
  "@type": "SoftwareSourceCode",
  "name": "TotalSegmentator",
  "description": "Tool for automated segmentation of 104 anatomical structures",
  "url": "https://github.com/wasserth/TotalSegmentator",
  "codeRepository": "https://github.com/wasserth/TotalSegmentator",
  "programmingLanguage": "Python",
  "runtimePlatform": "PyTorch",
  "license": "Apache-2.0",
  "keywords": ["segmentation", "CT", "MRI", "medical-imaging"],
  "applicationCategory": "Medical Imaging",
  "operatingSystem": ["Linux", "Windows", "macOS"],
  "softwareVersion": "2.0.0",
  "datePublished": "2022-09-01",
  "dateModified": "2024-01-15",
  "author": {
    "@type": "Person",
    "name": "Jakob Wasserthal"
  }
}
```

### Extended Fields

Custom fields in `supportingData`:

```json
{
  "supportingData": {
    "modalities": ["CT", "MRI"],
    "dimensions": ["3D"],
    "formats": ["DICOM", "NIfTI", "PNG"],
    "tasks": ["segmentation", "organ-segmentation"],
    "demo_url": "https://huggingface.co/spaces/username/totalsegmentator",
    "paper_url": "https://doi.org/10.1000/example",
    "citations": 150,
    "github_stars": 1200
  }
}
```

### Field Descriptions

#### name
Canonical tool name (matches repository or published name)

**Example**: `"TotalSegmentator"`, `"nnU-Net"`, `"MedSAM"`

#### description
Brief description of tool's purpose and capabilities

**Guidelines**:
- 1-2 sentences
- Mention key features
- Include domain/modality if specific

#### url
Primary landing page (usually GitHub repo)

#### codeRepository  
Source code repository URL (GitHub, GitLab, etc.)

#### programmingLanguage
Primary language(s)

**Common values**: `"Python"`, `"C++"`, `"JavaScript"`, `"Jupyter Notebook"`

#### license
Software license identifier (SPDX format)

**Common values**:
- `"Apache-2.0"`: Permissive, commercial OK
- `"MIT"`: Very permissive
- `"GPL-3.0"`: Copyleft
- `"BSD-3-Clause"`: Permissive
- `"Proprietary"`: Restricted

#### keywords
Array of relevant tags/keywords

**Categories**:
- **Tasks**: segmentation, classification, registration, detection
- **Modalities**: CT, MRI, X-ray, ultrasound, microscopy
- **Techniques**: deep-learning, traditional-cv, machine-learning
- **Domains**: medical-imaging, scientific-imaging, neuroscience

#### supportingData.modalities
Medical imaging modalities supported

**Standard values**:
- `"CT"`: Computed Tomography
- `"MRI"`: Magnetic Resonance Imaging
- `"XR"`: X-ray radiography
- `"US"`: Ultrasound
- `"PET"`: Positron Emission Tomography
- `"SPECT"`: Single-Photon Emission CT
- `"OCT"`: Optical Coherence Tomography
- `"Microscopy"`: Various microscopy types

#### supportingData.dimensions
Spatial dimensions supported

**Values**: `["2D"]`, `["3D"]`, `["2D", "3D"]`, `["4D"]`

- **2D**: Single slice images
- **3D**: Volumetric data
- **4D**: Time-series volumes (3D + time)

#### supportingData.formats
File formats supported for input/output

**Common values**:
- Medical: `"DICOM"`, `"NIfTI"`, `"NRRD"`, `"Analyze"`
- Standard: `"PNG"`, `"JPEG"`, `"TIFF"`, `"BMP"`
- Scientific: `"HDF5"`, `"Zarr"`, `"OME-TIFF"`
- Other: `"NumPy"`, `"MAT"`

#### supportingData.tasks
Analysis tasks the tool performs

**Common values**:
- `"segmentation"`: Image segmentation
- `"classification"`: Image classification
- `"detection"`: Object detection
- `"registration"`: Image registration/alignment
- `"reconstruction"`: 3D reconstruction
- `"enhancement"`: Image enhancement
- `"analysis"`: General analysis

#### supportingData.demo_url
Link to runnable demo (HuggingFace Space, Colab, web app)

**Preferred**: HuggingFace Gradio Spaces (best integration)

**Example**: `"https://huggingface.co/spaces/username/toolname"`

## Catalog Structure

### File Format

JSON Lines (JSONL): Each line is a complete JSON object

```jsonl
{"@type": "SoftwareSourceCode", "name": "Tool1", ...}
{"@type": "SoftwareSourceCode", "name": "Tool2", ...}
{"@type": "SoftwareSourceCode", "name": "Tool3", ...}
```

**Benefits**:
- Easy to append new tools
- Stream processing for large catalogs
- Each line independently parseable
- Git-friendly (line-based diffs)

### Catalog Loading

```python
import json

def load_catalog(path: str) -> list[dict]:
    tools = []
    with open(path) as f:
        for line in f:
            if line.strip():
                tools.append(json.loads(line))
    return tools
```

### Validation

Tools are validated on load:

```python
from pydantic import BaseModel, HttpUrl

class SoftwareSourceCode(BaseModel):
    name: str
    description: str
    url: HttpUrl
    license: str
    keywords: list[str]
    supportingData: dict
    
    class Config:
        extra = "allow"  # Allow additional schema.org fields
```

## Catalog Management

### Adding New Tools

1. **Create entry** following schema:

```json
{
  "@type": "SoftwareSourceCode",
  "name": "NewTool",
  "description": "Brief description of the tool",
  "url": "https://github.com/user/newtool",
  "codeRepository": "https://github.com/user/newtool",
  "programmingLanguage": "Python",
  "license": "MIT",
  "keywords": ["segmentation", "CT"],
  "supportingData": {
    "modalities": ["CT"],
    "dimensions": ["3D"],
    "formats": ["DICOM", "NIfTI"],
    "tasks": ["segmentation"],
    "demo_url": "https://huggingface.co/spaces/user/newtool"
  }
}
```

2. **Append to catalog.jsonl** (as single line, no pretty printing)

3. **Update checksum**:

```bash
shasum dataset/catalog.jsonl > dataset/catalog.jsonl.sha1
```

4. **Sync catalog**:

```bash
ai_agent sync
```

This rebuilds the embeddings and FAISS index.

### Updating Existing Tools

1. **Find tool** in `catalog.jsonl`
2. **Edit JSON** (update fields)
3. **Validate JSON** syntax
4. **Update checksum** and **sync**

### Removing Tools

1. **Delete line** from `catalog.jsonl`
2. **Update checksum** and **sync**

## Catalog Sources

### Current Catalog

Built from:
- **Medical Imaging Tools**: TotalSegmentator, nnU-Net, MedSAM, etc.
- **Computer Vision Libraries**: OpenCV, scikit-image
- **Deep Learning Frameworks**: PyTorch, TensorFlow tools
- **Specialized Tools**: ITK, SimpleITK, 3D Slicer modules
- **HuggingFace Spaces**: Gradio apps for imaging

### Curation Process

Tools are included based on:
1. **Relevance**: Imaging analysis tasks
2. **Quality**: Actively maintained, documented
3. **Accessibility**: Open-source or free demos
4. **Runnable**: Has demo or clear usage examples

### Catalog Growth

**Current**: ~150 tools
**Target**: 500+ tools covering:
- Medical imaging (CT, MRI, X-ray, ultrasound, pathology)
- Scientific imaging (microscopy, astronomy, remote sensing)
- Computer vision (general object detection, segmentation, etc.)

## Synchronization

### Auto-Sync

Configured via `.env`:

```dotenv
SYNC_EVERY_HOURS=24
```

**Process**:
1. Background thread checks catalog every 24h
2. Compares SHA1 checksum
3. If changed:
    - Reload catalog
    - Re-embed all tools
    - Rebuild FAISS index
    - Update vocabulary for query expansion

### Manual Sync

```bash
ai_agent sync
```

**Output**:
```
[sync] 150 → dataset/catalog.jsonl
[sync] Rebuilding embeddings...
[sync] Embedding 150 tools... (5.2s)
[sync] Building FAISS index...
[sync] Saved to artifacts/rag_index/
[sync] Updating vocabulary...
[sync] Sync complete.
```

## Embeddings and Index

### Embedding Process

For each tool, create text representation:

```python
tool_text = f"{tool['name']} {tool['description']} {' '.join(tool['keywords'])}"

# Optional: Include supportingData
if 'supportingData' in tool:
    sd = tool['supportingData']
    tool_text += f" {' '.join(sd.get('modalities', []))}"
    tool_text += f" {' '.join(sd.get('tasks', []))}"

# Embed
embedding = embedder.encode(tool_text, normalize_embeddings=True)
```

### Index Storage

```
artifacts/rag_index/
├── index.faiss          # FAISS IndexFlatIP
└── meta.json            # Tool IDs, config, timestamps
```

**meta.json** structure:

```json
{
  "tool_ids": ["tool1", "tool2", ...],
  "version": "1.0",
  "embedding_model": "BAAI/bge-m3",
  "embedding_dim": 1024,
  "num_tools": 150,
  "created_at": "2024-03-01T12:00:00Z",
  "catalog_sha1": "abc123..."
}
```

## Vocabulary Extraction

### Purpose

Extract terms for query expansion:

```python
vocabulary = set()

for tool in catalog:
    vocabulary.add(tool['name'].lower())
    vocabulary.update(tool['description'].lower().split())
    vocabulary.update(tool.get('keywords', []))
    
    if 'supportingData' in tool:
        sd = tool['supportingData']
        vocabulary.update(sd.get('modalities', []))
        vocabulary.update(sd.get('tasks', []))

# Result: ~5000 unique terms
```

### Vocabulary Embeddings

Pre-embed vocabulary for fast query expansion:

```python
vocab_list = list(vocabulary)
vocab_embeddings = embedder.encode(vocab_list, normalize_embeddings=True)

# Save for query expansion
np.save("artifacts/vocab_embeddings.npy", vocab_embeddings)
```

At query time, find nearest neighbors efficiently.

## Quality Assurance

### Validation Rules

1. **Required fields**: name, description, url, license
2. **Valid URLs**: Well-formed HTTP/HTTPS URLs
3. **Standard licenses**: SPDX identifiers preferred
4. **Consistent keywords**: Use standard terminology
5. **Demo URLs**: Verify demos are live and accessible

### Automated Checks

```python
def validate_catalog(catalog_path):
    errors = []
    
    with open(catalog_path) as f:
        for i, line in enumerate(f, 1):
            try:
                tool = json.loads(line)
                
                # Required fields
                for field in ['name', 'description', 'url']:
                    if field not in tool:
                        errors.append(f"Line {i}: Missing {field}")
                
                # URL validation
                if not tool['url'].startswith('http'):
                    errors.append(f"Line {i}: Invalid URL")
                
                # supportingData structure
                if 'supportingData' in tool:
                    sd = tool['supportingData']
                    if 'demo_url' in sd and sd['demo_url']:
                        if not sd['demo_url'].startswith('http'):
                            errors.append(f"Line {i}: Invalid demo_url")
                            
            except json.JSONDecodeError as e:
                errors.append(f"Line {i}: JSON syntax error - {e}")
    
    return errors
```

## Best Practices

### Tool Descriptions

✅ **Good**:
```
"Automated multi-organ segmentation for CT and MRI supporting 104 anatomical structures"
```

❌ **Bad**:
```
"A tool"  # Too vague
"The best segmentation tool ever created with amazing accuracy..."  # Too marketing-y
```

### Keywords

✅ **Good**:
```
["segmentation", "CT", "MRI", "medical-imaging", "deep-learning", "organ-segmentation"]
```

❌ **Bad**:
```
["cool", "awesome", "the best"]  # Not searchable terms
```

### Demo URLs

✅ **Preferred**:
- HuggingFace Gradio Spaces
- Google Colab notebooks
- Live web demos

❌ **Avoid**:
- Dead links
- Paywalled demos
- Demos requiring registration

## Next Steps

- Return to [Architecture Overview](overview.md)
- Learn about [Retrieval Pipeline](retrieval.md)
- Explore [Agent & VLM Selection](agent.md)
