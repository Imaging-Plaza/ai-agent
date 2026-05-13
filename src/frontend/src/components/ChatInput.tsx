import { useEffect, useRef, useState } from "react";
import { api, type Asset } from "../lib/api";

export type StagedAsset = Asset;

type Props = {
  sessionId: string | null;
  onSessionId: (id: string) => void;
  onSend: (message: string, attachments: StagedAsset[]) => void;
  busy: boolean;
};

export default function ChatInput({ sessionId, onSessionId, onSend, busy }: Props) {
  const [text, setText] = useState("");
  const [staged, setStaged] = useState<StagedAsset[]>([]);
  const [uploading, setUploading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const taRef = useRef<HTMLTextAreaElement>(null);

  // Auto-grow textarea
  useEffect(() => {
    const el = taRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 200) + "px";
  }, [text]);

  // Drag-and-drop on the document body (so dropping anywhere over the chat
  // surface works, not just on the input).
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
      alert("Upload failed. Check the file type and try again.");
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  }

  function submit() {
    const t = text.trim();
    if (!t && staged.length === 0) return;
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

  return (
    <>
      {dragging && <div className="dropzone-overlay">Drop files to upload</div>}

      <div className="composer-wrap">
        {staged.length > 0 && (
          <div className="staged-files">
            {staged.map((a) => (
              <div key={a.asset_id} className="staged-file">
                {a.preview_url ? (
                  <img src={a.preview_url} alt="" />
                ) : (
                  <span>📎</span>
                )}
                <span>{a.display_name}</span>
                <button
                  type="button"
                  onClick={() =>
                    setStaged((prev) => prev.filter((x) => x.asset_id !== a.asset_id))
                  }
                  aria-label="Remove"
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
            accept="image/*,.dcm,.nii,.nii.gz,.tiff,.tif,.png,.jpg,.jpeg,.webp,.bmp"
            style={{ display: "none" }}
            onChange={(e) => upload(Array.from(e.target.files || []))}
          />
          <button
            type="button"
            className="btn-attach"
            title="Attach files"
            disabled={uploading || busy}
            onClick={() => fileInputRef.current?.click()}
          >
            📎
          </button>
          <textarea
            ref={taRef}
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder={
              uploading
                ? "Uploading…"
                : "Describe what you want to do (Shift+Enter for newline)…"
            }
            rows={1}
            disabled={busy}
          />
          <button
            type="button"
            className="btn-send"
            onClick={submit}
            disabled={busy || (!text.trim() && staged.length === 0)}
            title="Send"
          >
            ↑
          </button>
        </div>
      </div>
    </>
  );
}
