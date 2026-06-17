# Using The Chat Interface

The React chat interface is the primary way to use AI Imaging Agent. It combines file upload, conversation history, recommendation cards, media previews, and controlled demo actions in one workspace.

## Layout

### Sidebar

- Start a new conversation.
- Reopen conversations stored in browser local storage.
- Open the session asset gallery.
- Keep prior turns available when you resume a conversation.

### Header

- View the current conversation title.
- Select the agent model from `config.yaml`.
- Adjust retrieval `top_k` and the number of recommendations.
- Switch light/dark theme.
- Sign out when password auth is enabled.

### Message Area

- Shows user turns, agent replies, recommendation cards, tool traces, generated media, and clarification prompts.
- Streams status updates from the backend while the synchronous agent run is in progress.
- Includes a minimap for navigating longer conversations.

### Composer

- Type natural-language requests.
- Attach new files from disk.
- Reattach previous session assets from the gallery.
- Prefill from example prompts.
- Use slash commands for inline media embeds.

If you send another message while the agent is busy, it is added to a queue. Queued messages appear in a banner and can be canceled before they run.

## Basic Workflow

1. **Upload files** by dragging them into the composer or using the attach control.
2. **Describe the task**, for example:

   ```text
   Segment the lungs from this CT scan
   ```

3. **Watch status updates** as the backend validates files, searches the catalog, ranks candidates, and prepares the response.
4. **Review recommendations** with rank, accuracy, explanation, metadata, and demo links.
5. **Answer clarification prompts** if the agent needs more context.
6. **Approve or decline pending actions** such as demo execution.

## File And Asset Handling

Uploaded files are stored in the current server-side session. The frontend receives asset IDs and preview URLs, then can request additional views from the backend.

Supported views include:

- Cached PNG preview for VLM and UI display.
- Raw original file serving for supported media/PDF formats.
- Slice views for volume assets.
- Maximum intensity projections for volume assets.
- Downsampled Float32 volume bytes for browser-side Three.js rendering.

The asset gallery lets you reuse already-uploaded files without selecting them from disk again.

## Supported Interactions

### Natural-Language Requests

Good requests include both the imaging task and constraints:

```text
Find open-source tools that segment kidneys in 3D NIfTI MRI data.
```

```text
Register these two brain MRI images and prefer tools with runnable demos.
```

### Alternatives

Ask for another search strategy:

```text
Show me alternatives.
```

The agent can perform limited alternative searches in a conversation.

### Excluding Tools

Use the `[EXCLUDE:...]` control tag:

```text
Find lung segmentation tools [EXCLUDE:totalsegmentator|medicalsam]
```

### Slash Embeds

Slash commands add media/embed turns without running the recommendation agent:

```text
/help
/img <asset-id | name | url>
/audio <url>
/video <url>
/youtube <id | url>
/embed <url>
```

`/img` can resolve an uploaded session asset by exact ID, ID prefix, or filename substring.

## Recommendation Cards

Each card can include:

- Rank and tool name.
- Accuracy score.
- Explanation for the match.
- Catalog metadata such as modality, anatomy, dimension, format, license, and categories.
- Demo URL or repository link.

The agent may also emit tool traces. These show which tools ran and what intermediate actions were taken.

## Demo Actions

Some recommendations include runnable examples. When the agent can run or prepare a demo action, the UI shows a pending action panel. You can approve, decline, or confirm the demo flow.

!!! warning
    Running external demos can send your uploaded data to third-party services such as public Hugging Face Spaces. Review the destination before approving.

## Conversation Persistence

The frontend stores conversation transcripts in browser local storage and restores them into server-side sessions when you continue a chat. Uploaded files themselves live in the server process temporary upload area, so restarting the backend can invalidate old asset IDs.

## Tips

!!! tip "Upload first"
    The agent gets better results when it can inspect image previews and original metadata.

!!! tip "Use constraints"
    Mention DICOM/NIfTI/TIFF, 2D/3D/4D, modality, anatomy, license, or GPU constraints when they matter.

!!! tip "Use the model controls"
    Switch to a faster model for exploration and a stronger model for more ambiguous visual reasoning.

!!! tip "Queue follow-ups"
    You can type follow-up messages while the agent is still working; they run in order after the current response finishes.

## Troubleshooting

### Login Does Not Appear

`APP_PASSWORD` is probably unset, so auth is disabled for local development.

### Frontend Cannot Reach Backend

- Confirm `ai_agent serve` is running.
- In dev, confirm Vite is running on `http://localhost:5173`.
- Check `DEV_CORS_ORIGINS` and `VITE_API_TARGET` if you changed ports.

### Old Asset Previews Fail

The backend may have restarted and cleared the in-memory session store. Reupload the files.

### No Recommendations

- Rephrase the task with more domain detail.
- Upload the relevant file before asking.
- Mention format, modality, or anatomy explicitly.
- Make sure the local catalog path is correct.

## Next Steps

- Learn about [Supported File Formats](file-formats.md)
- Understand [How Recommendations Work](recommendations.md)
- Explore [Running Demos](running-demos.md)
- Check [Advanced Features](advanced-features.md)
