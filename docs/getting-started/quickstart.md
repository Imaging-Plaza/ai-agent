# Quick Start

This guide will walk you through your first interaction with the AI Imaging Agent.

## Starting the Application

Once you've [installed](installation.md) and [configured](configuration.md) the agent, start the chat interface:

```bash
ai_agent chat
```

You should see output like:

```
[startup-sync] 150 → dataset/catalog.jsonl
[startup-refresh] catalog unchanged; keeping existing FAISS index
Running on local URL:  http://127.0.0.1:7860
```

Open your web browser and navigate to **http://127.0.0.1:7860**

## Your First Query

### Example 1: Object Segmentation

Let's try a simple segmentation task:

1. **Upload an Image**: Click the upload area or drag and drop an image (e.g., a photo of a cat)

2. **Type Your Request**: In the chat input, type:
   ```
   I want to segment the cat from this image
   ```

3. **Review Recommendations**: The agent will return ranked tool recommendations with:
    - Tool names and descriptions
    - Accuracy scores
    - Explanations for why each tool matches your task
    - Links to runnable demos

4. **Run a Demo** (optional): Click "Would you like me to run the demo?" and respond with "yes" to execute the tool directly on your image

### Example 2: Medical Image Analysis

For medical imaging tasks:

1. **Upload a Medical Image**: Upload a DICOM file, NIfTI volume, or medical image

2. **Describe Your Task**:
   ```
   Segment the lungs from this CT scan
   ```

3. **Get Format-Aware Results**: The agent considers:
    - Your image format (DICOM, NIfTI, etc.)
    - Image dimensions (2D, 3D, 4D)
    - Medical imaging modality (CT, MRI, etc.)

### Example 3: General Computer Vision

For general tasks:

```
Detect all objects in this image
```

```
Extract text from this document image
```

```
Classify what type of animal is in this picture
```

## Understanding the Interface

### Chat Panel

- **Message History**: Scroll to see previous interactions
- **Rich Media**: Images, files, and tool cards are rendered inline
- **Code Blocks**: Formatted code and JSON responses

### Sidebar

- **Uploaded Files**: View all files you've uploaded in the session
- **Preview Images**: See converted image previews
- **Debug Info**: View conversation state and excluded tools (if in debug mode)

### Tool Recommendation Cards

Each recommended tool shows:

- **Rank**: Priority order (1 = best match)
- **Name**: Tool/software name
- **Accuracy Score**: Confidence level (0-100%)
- **Description**: What the tool does
- **Explanation**: Why it matches your request
- **Metadata**:
    - Supported modalities (CT, MRI, etc.)
    - Dimensions (2D, 3D, etc.)
    - File formats (DICOM, NIfTI, PNG, etc.)
    - License information
    - Tags and categories
- **Demo Link**: Direct link to runnable example

## Advanced Usage

### Multi-Turn Conversations

The agent maintains conversation context:

```
You: I have a lung CT scan
Agent: [Provides general information about lung CT analysis tools]

You: I want to segment the airways
Agent: [Provides specific airway segmentation tools]

You: Show me alternatives
Agent: [Provides different tool options]
```

### Excluding Tools

Exclude specific tools from results:

```
Find lung segmentation tools [EXCLUDE:totalsegmentator|medicalsam]
```

### Requesting Alternatives

If initial results don't match your needs:

```
Show me alternative tools

Can you search for other options?

What else is available?
```

## CLI Commands

The agent provides two main commands:

### Launch Chat Interface

```bash
ai_agent chat
```

Starts the Gradio web interface with automatic catalog synchronization.

### Sync Catalog

```bash
ai_agent sync
```

Manually synchronize the software catalog without launching the UI.

## Tips for Best Results

!!! tip "Be Specific"
    The more specific your request, the better the recommendations:
    
    - ❌ "Process this image"
    - ✅ "Segment the liver from this abdominal CT scan"

!!! tip "Upload First"
    Upload your image before describing the task. The agent can see image content and metadata.

!!! tip "Mention Formats"
    If you need specific format support, mention it:
    
    "I need a tool that works with DICOM files"

!!! tip "Use Natural Language"
    No need to use technical jargon - conversational language works fine:
    
    "Help me find tumors in this MRI" works just as well as "Tumor detection in MRI volumes"

## Next Steps

Now that you've run your first queries:

- Learn more about [Using the Chat Interface](../user-guide/chat-interface.md)
- Explore [Supported File Formats](../user-guide/file-formats.md)
- Understand [How Recommendations Work](../user-guide/recommendations.md)
- Dive into the [Architecture Overview](../architecture/overview.md)
