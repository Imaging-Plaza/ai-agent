# Using the Chat Interface

The AI Imaging Agent provides a conversational interface for discovering and using imaging software. This guide explains how to interact with the chat interface effectively.

## Interface Layout

The interface consists of three main areas:

### Left Panel: Chat Conversation
- **Message History**: Your conversation with the agent
- **Rich Media Rendering**: Images, tool cards, and files are displayed inline
- **Input Box**: Type your messages at the bottom
- **File Upload**: Attach files via the paperclip icon or drag-and-drop

### Right Panel: Sidebar
- **Files Tab**: View uploaded files with format information
- **State Tab**: Debug information showing conversation state

### Header
- **Model Selector**: Choose which AI model to use
- **Settings**: Access configuration options

## Basic Workflow

### 1. Upload Files

Upload images or other files in several ways:

- **Drag and Drop**: Drag files directly onto the upload area
- **Click to Browse**: Click the upload area to select files
- **Attach to Message**: Use the paperclip icon in the input box

Files are automatically processed and metadata is extracted.

### 2. Describe Your Task

Use natural language to describe what you want to do:

!!! example "Good Task Descriptions"
    - "I want to segment the lungs from this CT scan"
    - "Help me detect tumors in this MRI"
    - "I need to register these two brain images"
    - "Extract text from this medical report"
    - "Classify the organ shown in this ultrasound"

### 3. Review Recommendations

The agent returns ranked tool recommendations with:

- **Tool Cards**: Each tool is presented in a card format
- **Accuracy Scores**: Confidence levels for each recommendation
- **Explanations**: Why each tool matches your request
- **Metadata**: Technical details about compatibility

### 4. Run Demos (Optional)

The agent may offer to run demos:

```
Agent: Would you like me to run the demo with your image?
```

Respond with affirmative language:
- "yes"
- "sure"
- "ok"
- "please"
- "go ahead"

The agent will execute the tool and show results.

## Multi-Turn Conversations

The agent maintains context across multiple messages:

!!! example "Multi-Turn Example"
    ```
    You: I have a lung CT scan [uploads file]
    
    Agent: I can see you have a DICOM CT image. What would you like to do with it?
    
    You: Segment the airways
    
    Agent: [Provides airway segmentation tool recommendations]
    
    You: What about segmenting the whole lung?
    
    Agent: [Provides lung segmentation tools, remembering you're working with CT]
    
    You: Show me alternatives
    
    Agent: [Provides additional options]
    ```

## Advanced Features

### Excluding Tools

Exclude specific tools using the `[EXCLUDE:...]` tag:

```
Find segmentation tools [EXCLUDE:totalsegmentator|medicalsam]
```

You can exclude multiple tools separated by `|`.

### Disabling Reranking

For faster (but potentially less accurate) results:

```
Find lung segmentation tools [NO_RERANK]
```

### Force Refinement

Request clarification even when results are available:

```
Segment this image [REFINE]
```

### Requesting Alternatives

Ask the agent to search with different strategies:

```
Can you search for alternatives?

Show me other options

Find different tools for this task
```

The agent can perform up to 3 alternative searches per conversation.

## Understanding Agent Responses

### Recommendation Cards

Each recommendation includes:

#### Header
- **Rank Number**: 1, 2, 3 (1 = best match)
- **Tool Name**: Software/tool identifier
- **Accuracy Score**: 0-100% confidence

#### Body
- **Description**: What the tool does
- **Explanation**: Why it matches your task
- **Demo Link**: Click to visit runnable example

#### Footer Metadata
- **Modalities**: CT, MRI, X-ray, etc.
- **Dimensions**: 2D, 3D, 4D
- **Formats**: Supported file formats (DICOM, NIfTI, etc.)
- **License**: Software license information
- **Tags**: Categorization and keywords

### Execution Traces

When demos run, you'll see execution details:

```
<details>
<summary>Tool Execution Trace</summary>

Image uploaded to Gradio Space
Processing started...
Result: Success
Output saved to: result.png
</details>
```

Click to expand and see full execution logs.

### Clarification Questions

Sometimes the agent needs more information:

```
Agent: I found several segmentation tools. Which organ are you trying to segment?

You: The liver

Agent: [Provides liver-specific segmentation tools]
```

## File Management

### Uploaded Files List

The sidebar shows all uploaded files with:

- **Filename**: Original file name
- **Format**: File type/extension
- **Size**: File size
- **Preview**: Thumbnail (for images)

### Image Previews

Medical images are automatically converted:

- **DICOM**: PNG previews; 3D series use orthogonal composite views (MIPs + central slices) rather than a single slice
- **NIfTI**: PNG previews built from orthogonal composite views of the volume
- **TIFF Stacks**: PNG previews built from orthogonal composite views of the stack
- **Standard 2D Images**: Resized PNG preview of the original image

Previews are used for VLM analysis while preserving original format metadata.

### Removing Files

Click the 'X' button next to a file to remove it from the current session.

## Conversation State

The debug sidebar shows:

### Current State
- **Status**: idle, processing, waiting
- **Conversation Turn**: Current turn number
- **Excluded Tools**: Tools filtered from results

### Preview Images
- Images prepared for VLM analysis
- Format conversions applied

## Tips for Effective Interaction

!!! tip "Be Specific About Requirements"
    Mention specific needs:
    
    - "I need a tool that works with NIfTI files"
    - "Must support 3D volumes"
    - "Looking for open-source options"

!!! tip "Use Conversational Language"
    Natural language works best:
    
    - ✅ "Help me find tool that segments kidneys"
    - ❌ "kidney_segmentation_tool filter:3D"

!!! tip "Iterate Based on Results"
    If initial results aren't perfect, refine:
    
    - "Can you find tools with higher accuracy?"
    - "Show me open-source alternatives"
    - "What about tools that support DICOM?"

!!! tip "Ask Follow-Up Questions"
    The agent maintains context:
    
    - "What about the second recommendation?"
    - "Can you compare these two tools?"
    - "Which one is fastest?"

## Troubleshooting

### No Recommendations

If the agent can't find suitable tools:

- Try rephrasing your query
- Be more specific about the task
- Check that your file uploaded successfully
- Ensure your task matches the catalog domain (imaging/medical)

### Wrong Recommendations

If recommendations don't match:

- Provide more context about your specific needs
- Mention required file format support
- Specify modality or domain
- Use the exclude feature to filter out irrelevant tools

### Demo Execution Fails

If a demo doesn't run:

- Check your internet connection
- Verify the demo link is still active
- Try a different recommended tool
- Check file format compatibility

## Next Steps

- Learn about [Supported File Formats](file-formats.md)
- Understand [How Recommendations Work](recommendations.md)
- Explore [Running Demos](running-demos.md)
- Check out [Advanced Features](advanced-features.md)
