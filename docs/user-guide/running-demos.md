# Running Demos (This area is still under construction..)

The AI Imaging Agent can execute tool demos directly on your uploaded images. This guide explains how demo execution works and how to use it effectively.

## What Are Demos?

Demos are **runnable examples** of imaging tools, typically hosted as:

- **HuggingFace Spaces**: Interactive Gradio or Streamlit applications
- **Jupyter Notebooks**: Google Colab or similar notebook environments
- **Web Applications**: Hosted web interfaces
- **GitHub Examples**: Code repositories with example scripts

## Demo Execution Flow

### 1. Agent Offers to Run Demo

After providing recommendations, the agent may offer:

```
Agent: Would you like me to run the demo with your image?
```

This appears when:

- Tool has a compatible Gradio Space demo
- Your image format is compatible
- Demo's API is accessible

### 2. You Confirm

Respond with affirmative language:

- "yes"
- "sure"
- "ok"
- "please"
- "go ahead"
- "run it"

The agent detects these patterns and proceeds.

### 3. Execution Happens

The agent:

1. Uploads your image to the demo space
2. Configures any required parameters
3. Triggers execution
4. Monitors progress
5. Retrieves results

### 4. Results Display

You receive:

- **Success message** with output
- **Result images** or files
- **Execution trace** showing what happened

## Demo Types

### Gradio Space Demos

**Best supported** - Direct API integration:

```
🚀 Demo: https://huggingface.co/spaces/username/toolname
```

**Features**:

- ✅ Automatic execution
- ✅ Progress monitoring
- ✅ Result retrieval
- ✅ Error handling

**Example**:
```
Running TotalSegmentator on your CT scan...
✓ Image uploaded
✓ Processing started
✓ Segmentation complete
✓ Results downloaded
```

### Notebook Demos

**Partially supported** - Links provided for manual execution:

```
📓 Notebook: https://colab.research.google.com/...
```

**Process**:

1. Click the notebook link
2. Open in Google Colab
3. Upload your image to the notebook
4. Run cells sequentially
5. Download results

### Web Application Demos

**Manual execution** - Opens in browser:

```
🌐 Web Demo: https://example.com/tool
```

**Process**:

1. Click the demo link
2. Web app opens in new tab
3. Upload your image via the web UI
4. Configure settings
5. Run and download results

### GitHub Repository Examples

**Code-based** - Requires local setup:

```
💻 Repository: https://github.com/user/repo
```

**Process**:

1. Clone the repository
2. Install dependencies
3. Run example scripts
4. Adapt for your data

## Execution Traces

When demos run, you see detailed traces:

```html
<details>
<summary>🔧 Tool Execution Trace</summary>

Step 1: Uploading image to Gradio Space
  ✓ Connected to space: username/toolname
  ✓ Image uploaded: 2.3 MB

Step 2: Configuring parameters
  ✓ Set task: lung-segmentation
  ✓ Set format: DICOM

Step 3: Running inference
  ⏳ Processing... (estimated 30s)
  ✓ Completed in 28s

Step 4: Retrieving results
  ✓ Downloaded segmentation mask: 1.1 MB
  ✓ Downloaded visualization: 0.8 MB

Status: ✅ Success
</details>
```

Click to expand and see full details.

## Supported Gradio Spaces

### Auto-Detected Parameters

The agent automatically configures:

#### Image Input
- Detects image input component(s)
- Uploads your file
- Converts format if needed

#### Task Selection
Common task parameters:

- **Task dropdown**: Matches your query to task option
- **Model selection**: Chooses appropriate model
- **Mode**: Inference, predict, analyze, etc.

#### Format Options
- **Input format**: DICOM, NIfTI, PNG, etc.
- **Output format**: Segmentation mask, visualization, etc.
- **Data type**: 2D, 3D, specific modality

### Manual Parameters

Some demos require manual interaction:

```
Agent: This demo has additional parameters. Please visit the link to configure:
- Segmentation threshold: 0.5
- Post-processing: enabled
```

## Demo Execution Best Practices

!!! tip "Check Compatibility First"
    Verify the tool supports your file format in the recommendation metadata.

!!! tip "Use Standard Formats"
    Demos work best with standard formats (PNG, JPEG for general; DICOM, NIfTI for medical).

!!! tip "Be Patient"
    Some demos take time, especially for:
    - Large images or volumes
    - Deep learning models
    - 3D processing
    
    Typical times: 10 seconds to 2 minutes.

!!! tip "Save Results Immediately"
    Download result files promptly - they may not persist after closing the browser.

!!! warning "Rate Limits"
    Public Gradio Spaces may have rate limits or queue systems during high usage.

## Troubleshooting Demo Execution

### Demo Fails to Run

**Error**: Connection timeout or failed upload

**Solutions**:

- Check internet connection
- Try again (server may be busy)
- Visit demo link manually
- Try alternative recommendation

### Wrong Results

**Error**: Output doesn't match expectations

**Solutions**:

- Check if correct parameters were used
- Verify image uploaded correctly
- Try adjusting task settings manually
- Compare with demo's example images

### Incompatible Format

**Error**: "Format not supported"

**Solutions**:

- Convert image to supported format
- Use tool that accepts your format
- Try alternative recommendation

### Demo Link Broken

**Error**: 404 or space not found

**Solutions**:

- Space may be temporarily down
- Check GitHub repo for alternative demo
- Try different tool recommendation
- Report broken link

## Manual Demo Execution

If automatic execution isn't available:

### For Gradio Spaces

1. Click the demo link
2. The space opens in your browser
3. Upload your image via the UI
4. Select appropriate options
5. Click "Submit" or "Run"
6. Download results

### For Colab Notebooks

1. Click the notebook link
2. Open in Google Colab
3. Run setup cells (install dependencies)
4. Upload your image when prompted:
   ```python
   from google.colab import files
   uploaded = files.upload()
   ```
5. Run processing cells
6. Download results:
   ```python
   files.download('result.png')
   ```

### For Local Execution

1. Clone the repository:
   ```bash
   git clone https://github.com/user/repo
   cd repo
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run example script:
   ```bash
   python run_demo.py --input your_image.png --output result.png
   ```

4. Check output directory for results

## Understanding Results

### Segmentation Results

Typically includes:

- **Segmentation mask**: Binary or multi-class mask
- **Overlay visualization**: Mask overlaid on original image
- **Statistics**: Volume, area, counts

### Detection Results

Usually provides:

- **Bounding boxes**: Coordinates of detected objects
- **Annotated image**: Visual with boxes/labels
- **Confidence scores**: Detection confidence

### Registration Results

Common outputs:

- **Transformed image**: Registered/aligned image
- **Transformation matrix**: Spatial transform parameters
- **Quality metrics**: Similarity scores

### Classification Results

Typical outputs:

- **Class labels**: Predicted categories
- **Probabilities**: Confidence per class
- **Visualization**: Class activation maps

## Demo Feedback

Help improve the agent by reporting:

### Successful Demos
When demos work well, this validates:

- Tool compatibility
- Parameter auto-configuration
- Format handling

### Issues
Report when:

- Demo fails unexpectedly
- Results are incorrect
- Parameters were misconfigured
- Format conversion was wrong

Feedback helps refine the agent's demo execution capabilities.

## Next Steps

- Explore [Advanced Features](advanced-features.md)
- Learn about the [Architecture](../architecture/overview.md)
- Check [CLI Commands](../reference/cli.md)
