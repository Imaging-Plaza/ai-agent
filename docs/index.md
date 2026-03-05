# AI Imaging Agent

**An intelligent RAG + AI agent system that helps users discover the right imaging software for their images and tasks.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/imaging-plaza/ai-agent/blob/main/LICENSE)

---

## What is AI Imaging Agent?

AI Imaging Agent (also known as **Imaging Plaza**) is a conversational AI assistant that helps researchers and practitioners find the right imaging analysis tools for their specific needs. Simply upload an image, describe what you want to do, and get ranked software recommendations with links to runnable demos.

## ✨ Key Features

- **🤖 Conversational AI Agent**: Natural language interaction with multi-turn context
- **🔍 Smart Retrieval**: BGE-M3 embeddings + FAISS + CrossEncoder reranking
- **👁️ Vision-Aware Selection**: VLM-based tool selection considering both image content and metadata
- **🏥 Medical Imaging Focus**: Specialized support for CT, MRI, DICOM, NIfTI, and other medical formats
- **🎯 Format-Aware Matching**: IO compatibility scoring based on file formats and dimensions
- **🚀 Demo Integration**: Direct execution of Gradio Space demos on your images
- **📊 Rich UI**: Chat interface with image previews, file management, and execution traces

## Quick Example

```bash
# Install and run
pip install -e .
ai_agent chat
```

Then in the web interface:

1. Upload an image (e.g., a CT scan, or a PNG)
2. Type your request (e.g., _"I want to segment the lungs from this image"_ or _"I want to deblur this image"_)
3. Get ranked tool recommendations with accuracy scores
4. Click "Run demo" to execute tools directly

## Use Cases

### Medical Imaging
- Segment organs from CT/MRI scans
- Register brain images
- Detect tumors and anomalies
- Analyze DICOM files

### Scientific Imaging
- Process microscopy images
- Analyze multidimensional TIFF stacks
- Extract features from scientific images

### General Computer Vision
- Object detection and segmentation
- Image classification
- OCR and text extraction
- Image enhancement

## How It Works

The system uses a **two-stage pipeline**:

1. **Retrieval Stage**: Fast text search using BGE-M3 embeddings and FAISS to find candidate tools from a curated catalog
2. **Agent Selection**: Vision-language model (GPT-4o) analyzes your image and task to rank the best tools with explanations

Learn more in the [Architecture Overview](architecture/overview.md).

## Getting Started

Ready to try it out? Head over to the [Installation Guide](getting-started/installation.md) to get started!

## Project Status

This project is actively developed and maintained by the Imaging Plaza team. Check the [Changelog](reference/changelog.md) for recent updates.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](https://github.com/imaging-plaza/ai-agent/blob/main/LICENSE) file for details.
