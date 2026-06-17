# Quick Start

This guide walks through your first local session with the React frontend and FastAPI backend.

## Start The Application

After [installation](installation.md) and [configuration](configuration.md), start the backend from the repository root:

```bash
ai_agent serve
```

By default this starts FastAPI on `http://localhost:8000`.

In a second terminal, start the Vite frontend:

```bash
cd src/frontend
npm run dev
```

Open `http://localhost:5173`. Vite proxies `/api/*` requests to the backend.

!!! note "Production mode"
    If `src/frontend/dist` exists, `ai_agent serve` serves the built React app directly. In Docker, the frontend is built into the image and exposed on port `7860`.

## Sign In

If `APP_PASSWORD` is set in `.env`, enter that passphrase on the login page. If `APP_PASSWORD` is unset, the frontend skips enforced auth for local development.

## Your First Query

### Example 1: Medical Volume

1. Click the attach control or drag a file into the composer.
2. Upload a DICOM file, NIfTI volume, TIFF stack, or standard image.
3. Ask:

```text
Segment the lungs from this CT scan
```

The app will upload the asset, extract metadata, generate previews, and stream status updates while the agent searches the catalog.

### Example 2: Standard Image

Upload a PNG or JPG and ask:

```text
I want to segment the foreground object from this image
```

### Example 3: General Discovery

You can also ask without files:

```text
What tools can register two brain MRI images?
```

## Review Results

Recommendation cards include:

- Rank and accuracy score
- Tool name and short explanation
- Format, modality, dimension, license, and category metadata when available
- Demo or repository links
- Optional pending actions, such as approving a demo run

## Interface Basics

- **Sidebar**: start a new chat, reopen stored local conversations, and open the asset gallery.
- **Header**: choose the model, adjust top-k retrieval and number of recommendations, switch theme, or sign out.
- **Composer**: attach files, select session assets, use example prompts, or submit follow-ups.
- **Queue banner**: messages sent while the agent is busy are queued and can be canceled.
- **Minimap**: jump through longer conversations.
- **Asset views**: preview uploaded files, inspect metadata, view slices/MIPs, and render supported volumes in 3D.

## Useful Commands

```bash
/help
/img <asset-id | name | url>
/audio <url>
/video <url>
/youtube <id | url>
/embed <url>
```

Slash commands add inline media/embed turns to the conversation without invoking the recommendation agent.

## Advanced Usage

### Multi-Turn Conversations

The agent keeps conversational context:

```text
You: I have a lung CT scan.
Agent: What would you like to do with it?

You: Segment the airways.
Agent: [Provides airway segmentation tools]

You: Show me alternatives.
Agent: [Searches again with a different strategy]
```

### Excluding Tools

Use control tags to remove specific tools from retrieval:

```text
Find lung segmentation tools [EXCLUDE:totalsegmentator|medicalsam]
```

### Running The Legacy UI

The older Gradio interface remains available:

```bash
ai_agent chat
```

Use the React frontend for new development unless you are specifically testing legacy Gradio behavior.

## Tips

!!! tip "Upload before asking"
    File metadata and previews improve format-aware recommendations.

!!! tip "Be specific"
    "Segment the liver from an abdominal CT volume" gives the agent more signal than "process this image".

!!! tip "Mention constraints"
    Include requirements such as DICOM support, 3D volume support, open-source license, or GPU availability.

## Next Steps

- Learn more about [Using the Chat Interface](../user-guide/chat-interface.md)
- Explore [Supported File Formats](../user-guide/file-formats.md)
- Understand [How Recommendations Work](../user-guide/recommendations.md)
- Dive into the [Architecture Overview](../architecture/overview.md)
