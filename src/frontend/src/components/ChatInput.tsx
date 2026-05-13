import { useEffect, useRef, useState } from "react";
import { useTranscription } from "../hooks/useTranscription";
import { api, type Asset } from "../lib/api";
import { isSlashCommand, parseSlashCommand, type ParsedCommand } from "../lib/slashCommands";
import AssetModal from "./AssetModal";
import AttachModal from "./AttachModal";
import SpeechModal from "./SpeechModal";

export type StagedAsset = Asset;

type Props = {
  sessionId: string | null;
  onSessionId: (id: string) => void;
  onSend: (message: string, attachments: StagedAsset[]) => void;
  busy: boolean;
  /** Triggered when the user clicks "↗" on an expired gallery card. */
  onOpenConversation?: (conv: import("../hooks/useConversations").StoredConversation) => void;
  /** Slash command intercepted in the composer — runs locally, never hits the agent. */
  onSlashCommand?: (command: ParsedCommand, raw: string) => void;
};

type ExternalRefs = {
  openGallery: () => void;
  /** Drop text + attachments into the composer without sending. */
  prefill: (text: string, attachments: StagedAsset[]) => void;
};

export default function ChatInput({
  sessionId,
  onSessionId,
  onSend,
  busy,
  onOpenConversation,
  onSlashCommand,
  externalRefs,
}: Props & { externalRefs?: React.MutableRefObject<ExternalRefs | null> }) {
  const [text, setText] = useState("");
  const [staged, setStaged] = useState<StagedAsset[]>([]);
  const [uploading, setUploading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [showSpeech, setShowSpeech] = useState(false);
  const [showAttach, setShowAttach] = useState<null | "upload" | "gallery">(
    null
  );

  // Expose imperative actions to the page (sidebar opens the gallery,
  // example cards prefill the composer).
  useEffect(() => {
    if (!externalRefs) return;
    externalRefs.current = {
      openGallery: () => setShowAttach("gallery"),
      prefill: (incomingText, attachments) => {
        setText((cur) =>
          cur && incomingText
            ? cur.trimEnd() + " " + incomingText
            : incomingText
        );
        setStaged((prev) => {
          const ids = new Set(prev.map((p) => p.asset_id));
          return [...prev, ...attachments.filter((a) => !ids.has(a.asset_id))];
        });
        // Defer focus to the next frame so the textarea has resized first.
        requestAnimationFrame(() => taRef.current?.focus());
      },
    };
    return () => {
      if (externalRefs) externalRefs.current = null;
    };
  }, [externalRefs]);
  const [inspectAsset, setInspectAsset] = useState<{
    id: string;
    name?: string;
    previewUrl?: string | null;
  } | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const taRef = useRef<HTMLTextAreaElement>(null);
  const speech = useTranscription();

  useEffect(() => {
    const el = taRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 200) + "px";
  }, [text]);

  useEffect(() => {
    let dragDepth = 0;
    function onEnter(e: DragEvent) {
      if (!e.dataTransfer?.types.includes("Files")) return;
      dragDepth++;
      setDragging(true);
      e.preventDefault();
    }
    function onLeave() {
      dragDepth = Math.max(0, dragDepth - 1);
      if (dragDepth === 0) setDragging(false);
    }
    function onOver(e: DragEvent) {
      if (e.dataTransfer?.types.includes("Files")) e.preventDefault();
    }
    async function onDrop(e: DragEvent) {
      dragDepth = 0;
      setDragging(false);
      e.preventDefault();
      const files = e.dataTransfer?.files;
      if (files && files.length > 0) await upload(Array.from(files));
    }
    window.addEventListener("dragenter", onEnter);
    window.addEventListener("dragleave", onLeave);
    window.addEventListener("dragover", onOver);
    window.addEventListener("drop", onDrop);
    return () => {
      window.removeEventListener("dragenter", onEnter);
      window.removeEventListener("dragleave", onLeave);
      window.removeEventListener("dragover", onOver);
      window.removeEventListener("drop", onDrop);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId]);

  async function upload(files: File[]) {
    if (!files.length) return;
    setUploading(true);
    try {
      const r = await api.uploadFiles(files, sessionId ?? undefined);
      if (!sessionId) onSessionId(r.session_id);
      setStaged((prev) => [...prev, ...r.assets]);
    } catch (e) {
      console.error("upload failed", e);
      alert("upload_failed — check file type and try again");
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  }

  function submit() {
    const t = text.trim();
    if (!t && staged.length === 0) return;
    if (t && isSlashCommand(t) && onSlashCommand) {
      const parsed = parseSlashCommand(t);
      if (parsed) {
        onSlashCommand(parsed, t);
        setText("");
        // Keep staged files — the user may want to send them in a follow-up.
        return;
      }
    }
    onSend(t, staged);
    setText("");
    setStaged([]);
  }

  function onKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  }

  const [micToast, setMicToast] = useState<string | null>(null);

  // Surface short transient mic errors as a small banner above the composer
  // (alert() is gross and getUserMedia errors should NOT block the user).
  function flashToast(msg: string, ms = 4500) {
    setMicToast(msg);
    window.setTimeout(() => setMicToast((t) => (t === msg ? null : t)), ms);
  }

  async function onMicClick() {
    // 1. If we know the browser can't record, jump straight to the modal so
    //    the user can read the explanation.
    if (
      speech.micState === "unsupported" ||
      speech.micState === "no_device" ||
      speech.micState === "permission_denied"
    ) {
      setShowSpeech(true);
      return;
    }

    // 2. If the model isn't ready, open the modal so the user can download or
    //    watch progress — never start recording before the worker can decode.
    if (!speech.hasModel || speech.phase.kind === "downloading") {
      setShowSpeech(true);
      return;
    }
    if (speech.phase.kind !== "ready" && speech.phase.kind !== "recording") {
      setShowSpeech(true);
      return;
    }

    // 3. Toggle recording.
    if (speech.phase.kind === "recording") {
      try {
        const transcribed = await speech.stopAndTranscribe();
        if (transcribed.trim()) {
          setText((cur) =>
            cur ? cur.trimEnd() + " " + transcribed.trim() : transcribed.trim()
          );
          taRef.current?.focus();
        } else {
          flashToast("no speech detected — try again");
        }
      } catch (e: any) {
        console.error("transcription failed", e);
        flashToast(`transcription failed · ${e?.message || e}`);
      }
    } else {
      try {
        await speech.startRecording();
      } catch (e: any) {
        const msg = String(e?.message || e);
        if (msg.includes("permission_denied")) {
          flashToast("microphone permission denied — opening settings");
          setShowSpeech(true);
        } else if (msg.includes("no_device")) {
          flashToast("no microphone detected");
          setShowSpeech(true);
        } else if (msg.includes("unsupported")) {
          flashToast("this browser can't capture audio");
          setShowSpeech(true);
        } else {
          flashToast(`mic failed · ${msg}`);
        }
      }
    }
  }

  const recording = speech.phase.kind === "recording";
  const transcribing = speech.phase.kind === "transcribing";

  // Compute mic button title + variant
  const micUnavailable =
    speech.micState === "unsupported" ||
    speech.micState === "no_device" ||
    speech.micState === "permission_denied";
  const micTitle = recording
    ? "stop recording"
    : micUnavailable
      ? "microphone unavailable — click for help"
      : !speech.hasModel
        ? "set up dictation"
        : "dictate";

  return (
    <>
      {dragging && <div className="dropzone-overlay">drop_to_upload</div>}

      <SpeechModal open={showSpeech} onClose={() => setShowSpeech(false)} />
      <AttachModal
        open={showAttach !== null}
        initialMode={showAttach || "upload"}
        sessionId={sessionId}
        onSessionId={onSessionId}
        onPick={(assets) => {
          setStaged((prev) => {
            const ids = new Set(prev.map((p) => p.asset_id));
            return [...prev, ...assets.filter((a) => !ids.has(a.asset_id))];
          });
        }}
        onOpenConversation={onOpenConversation}
        onClose={() => setShowAttach(null)}
      />
      <AssetModal
        assetId={inspectAsset?.id ?? null}
        fallbackPreviewUrl={inspectAsset?.previewUrl ?? null}
        fallbackName={inspectAsset?.name ?? null}
        onClose={() => setInspectAsset(null)}
      />

      <div className="composer-wrap">
        {micToast && (
          <div className="mic-toast mono" role="status">
            {micToast}
          </div>
        )}
        {staged.length > 0 && (
          <div className="staged-files">
            {staged.map((a) => (
              <div key={a.asset_id} className="staged-card">
                <button
                  type="button"
                  className="staged-thumb"
                  onClick={() =>
                    setInspectAsset({
                      id: a.asset_id,
                      name: a.display_name,
                      previewUrl: a.preview_url,
                    })
                  }
                  title={a.display_name || a.asset_id}
                  aria-label="Preview"
                >
                  {a.preview_url ? (
                    <img src={a.preview_url} alt="" />
                  ) : (
                    <span className="staged-glyph">▣</span>
                  )}
                  <span className="staged-format mono">
                    {a.original_format || "file"}
                  </span>
                </button>
                <div className="staged-meta">
                  <span className="staged-name" title={a.display_name || ""}>
                    {a.display_name}
                  </span>
                  {a.metadata_text && (
                    <span className="staged-sub mono" title={a.metadata_text}>
                      {a.metadata_text.length > 64
                        ? a.metadata_text.slice(0, 63) + "…"
                        : a.metadata_text}
                    </span>
                  )}
                </div>
                <button
                  type="button"
                  className="staged-remove"
                  onClick={() =>
                    setStaged((prev) =>
                      prev.filter((x) => x.asset_id !== a.asset_id)
                    )
                  }
                  aria-label="Remove"
                  title="Remove"
                >
                  ×
                </button>
              </div>
            ))}
          </div>
        )}

        <div className="composer">
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept="image/*,video/*,audio/*,.dcm,.nii,.nii.gz,.tiff,.tif,.png,.jpg,.jpeg,.webp,.bmp,.gif,.mp4,.mov,.webm,.mkv,.mp3,.wav,.ogg,.flac,.m4a,.pdf"
            style={{ display: "none" }}
            onChange={(e) => upload(Array.from(e.target.files || []))}
          />
          <button
            type="button"
            className="btn-attach"
            title="Attach files · upload or pick from gallery"
            disabled={uploading}
            onClick={() => setShowAttach("upload")}
          >
            +
          </button>
          <textarea
            ref={taRef}
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder={
              uploading
                ? "uploading…"
                : recording
                  ? "listening…"
                  : transcribing
                    ? "transcribing…"
                    : busy
                      ? "> type next message — it'll queue while the agent replies"
                      : "> describe a task · /help for slash commands · shift+enter newline"
            }
            rows={1}
          />
          <button
            type="button"
            className={
              "btn-mic" +
              (recording ? " is-recording" : "") +
              (micUnavailable ? " is-unavailable" : "")
            }
            onClick={() => void onMicClick()}
            disabled={busy || transcribing}
            title={micTitle}
            aria-label="Voice input"
          >
            {recording ? (
              <span className="mic-stop" aria-hidden />
            ) : (
              <MicIcon />
            )}
          </button>
          <button
            type="button"
            className={"btn-send" + (busy ? " is-queueing" : "")}
            onClick={submit}
            disabled={!text.trim() && staged.length === 0}
            title={busy ? "Queue for after this reply" : "Send"}
          >
            {busy ? "+" : "→"}
          </button>
        </div>
        <div className="composer-disclaimer mono">
          ai-generated suggestions · verify tool capabilities, licensing and
          inputs before use · not a medical device, not for clinical or
          diagnostic decisions
        </div>
      </div>
    </>
  );
}

function MicIcon() {
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
    >
      <rect x="9" y="3" width="6" height="11" rx="3" />
      <path d="M5 11a7 7 0 0 0 14 0" />
      <path d="M12 18v3" />
    </svg>
  );
}
