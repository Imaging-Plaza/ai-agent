/** Unified "Add to conversation" picker.
 *
 *  Two stacked sources:
 *    1. Drop / pick new files from disk (re-uses the standard upload flow).
 *    2. A gallery of every asset the user has already uploaded — server-side
 *       (current session) plus every asset id referenced by their stored
 *       conversations (locally cached). Expired ones are marked as such.
 *
 *  Selections accumulate; "Add N" commits them all to the composer.
 */

import { useEffect, useMemo, useRef, useState } from "react";
import {
  useConversations,
  type StoredConversation,
} from "../hooks/useConversations";
import { api, type Asset } from "../lib/api";
import { formatRelativeDate } from "../lib/dates";

type Props = {
  open: boolean;
  sessionId: string | null;
  initialMode?: "upload" | "gallery";
  onSessionId: (id: string) => void;
  onPick: (assets: Asset[]) => void;
  /** Open the original conversation an expired asset was used in. */
  onOpenConversation?: (conv: StoredConversation) => void;
  onClose: () => void;
};

type GalleryItem = Asset & {
  _origin: "server" | "local";
  _expired?: boolean;
  _fromConv?: StoredConversation;
  /** Display timestamp in ms (regardless of source). */
  _timestamp?: number;
};

export default function AttachModal({
  open,
  sessionId,
  initialMode = "upload",
  onSessionId,
  onPick,
  onOpenConversation,
  onClose,
}: Props) {
  const { conversations } = useConversations();
  const [serverAssets, setServerAssets] = useState<Asset[]>([]);
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [selected, setSelected] = useState<Map<string, Asset>>(new Map());
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  useEffect(() => {
    if (!open) {
      setSelected(new Map());
      setError(null);
      return;
    }
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
    }
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  useEffect(() => {
    if (!open || !sessionId) {
      setServerAssets([]);
      return;
    }
    setLoading(true);
    void (async () => {
      try {
        const r = await fetch(`/api/files/sessions/${sessionId}`, {
          credentials: "include",
        });
        if (!r.ok) throw new Error(`gallery ${r.status}`);
        const data: Asset[] = await r.json();
        setServerAssets(data);
      } catch (e: any) {
        setServerAssets([]);
        console.warn("gallery load failed", e);
      } finally {
        setLoading(false);
      }
    })();
  }, [open, sessionId]);

  /** Pulled from local conversation history — every asset id we've ever seen
   *  the user attach, regardless of session. Also tracks WHICH conversation
   *  each asset came from so expired cards can offer to jump back. */
  const localItems = useMemo<GalleryItem[]>(() => {
    const seen = new Map<string, StoredConversation>();
    for (const conv of conversations) {
      for (const t of conv.turns) {
        if (t.role !== "user") continue;
        for (const a of t.attachments) {
          if (!seen.has(a.asset_id)) seen.set(a.asset_id, conv);
        }
      }
    }
    const out: GalleryItem[] = [];
    for (const [assetId, conv] of seen.entries()) {
      const turn = conv.turns.find(
        (t) => t.role === "user" && t.attachments.some((a) => a.asset_id === assetId)
      );
      const attach = turn?.role === "user"
        ? turn.attachments.find((a) => a.asset_id === assetId)
        : undefined;
      out.push({
        asset_id: assetId,
        display_name: attach?.display_name,
        preview_url: attach?.preview_url ?? null,
        _origin: "local",
        _fromConv: conv,
        // No per-attachment timestamp — use the conversation's createdAt
        // as a reasonable proxy.
        _timestamp: conv.createdAt,
      });
    }
    return out;
  }, [conversations]);

  /** Merge server assets (authoritative) with local-only refs (likely
   *  expired). Server entries dedupe local ones with the same id, BUT we
   *  still attach _fromConv when we have it locally so users can jump back
   *  to the originating conversation. */
  const gallery = useMemo<GalleryItem[]>(() => {
    const fromConvById = new Map<string, StoredConversation>();
    for (const li of localItems) {
      if (li._fromConv) fromConvById.set(li.asset_id, li._fromConv);
    }
    const localTsById = new Map<string, number>();
    for (const li of localItems) {
      if (li._timestamp != null) localTsById.set(li.asset_id, li._timestamp);
    }
    const serverIds = new Set(serverAssets.map((a) => a.asset_id));
    const a: GalleryItem[] = serverAssets.map((x) => ({
      ...x,
      _origin: "server",
      _fromConv: fromConvById.get(x.asset_id),
      // Server gives seconds; convert to ms for the formatter.
      _timestamp:
        x.created_at != null
          ? x.created_at * 1000
          : localTsById.get(x.asset_id),
    }));
    const b: GalleryItem[] = localItems
      .filter((x) => !serverIds.has(x.asset_id))
      .map((x) => ({ ...x, _expired: true }));
    // Newest first across both sources.
    const all = [...a, ...b];
    all.sort((x, y) => (y._timestamp ?? 0) - (x._timestamp ?? 0));
    return all;
  }, [serverAssets, localItems]);

  function toggleSelect(item: GalleryItem) {
    if (item._expired) return;
    setSelected((prev) => {
      const next = new Map(prev);
      if (next.has(item.asset_id)) next.delete(item.asset_id);
      else next.set(item.asset_id, item);
      return next;
    });
  }

  async function uploadFiles(files: File[]) {
    if (!files.length) return;
    setUploading(true);
    setError(null);
    try {
      const r = await api.uploadFiles(files, sessionId ?? undefined);
      if (!sessionId) onSessionId(r.session_id);
      // Auto-select the freshly uploaded items so the user can keep stacking.
      setSelected((prev) => {
        const next = new Map(prev);
        for (const a of r.assets) next.set(a.asset_id, a);
        return next;
      });
      setServerAssets((prev) => {
        const ids = new Set(prev.map((x) => x.asset_id));
        return [...r.assets.filter((x) => !ids.has(x.asset_id)), ...prev];
      });
    } catch (e: any) {
      setError(String(e?.message || e));
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  }

  function onDrop(e: React.DragEvent) {
    e.preventDefault();
    setDragging(false);
    const files = e.dataTransfer?.files;
    if (files && files.length > 0) void uploadFiles(Array.from(files));
  }

  function commit() {
    if (selected.size === 0) return;
    onPick(Array.from(selected.values()));
    setSelected(new Map());
    onClose();
  }

  if (!open) return null;

  const total = gallery.length;
  const availableCount = gallery.filter((g) => !g._expired).length;

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div
        className="modal attach-modal"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
      >
        <header className="modal-head">
          <div className="modal-tag mono">add · {initialMode}</div>
          <button className="modal-x" onClick={onClose} aria-label="Close">
            ✕
          </button>
        </header>

        <h2 className="modal-title">add to conversation</h2>
        <p className="modal-sub">
          drop a new file or pick from your gallery — uploads stay until the
          session expires, references are remembered locally.
        </p>

        {/* Upload zone */}
        <div
          className={"attach-dropzone" + (dragging ? " is-active" : "")}
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
        >
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept="image/*,video/*,audio/*,.dcm,.nii,.nii.gz,.tiff,.tif,.png,.jpg,.jpeg,.webp,.bmp,.gif,.mp4,.mov,.webm,.mkv,.mp3,.wav,.ogg,.flac,.m4a,.pdf"
            style={{ display: "none" }}
            onChange={(e) =>
              uploadFiles(Array.from(e.target.files || []))
            }
          />
          <div className="attach-drop-text">
            <strong>↑ drop files here</strong>
            <span className="mono">
              {uploading ? "uploading…" : "or click to browse"}
            </span>
          </div>
          <button
            type="button"
            className="btn-primary"
            disabled={uploading}
            onClick={() => fileInputRef.current?.click()}
          >
            ↳ choose files
          </button>
        </div>

        {error && (
          <div className="error-banner" style={{ marginTop: 12 }}>
            {error}
          </div>
        )}

        {/* Gallery */}
        <div className="attach-gallery-head">
          <span className="mono attach-section-label">gallery</span>
          <span className="mono attach-gallery-count">
            {availableCount} available · {total - availableCount} expired
          </span>
        </div>

        {loading ? (
          <div className="attach-loading mono">loading…</div>
        ) : gallery.length === 0 ? (
          <div className="attach-empty mono">
            nothing uploaded yet — drop a file above
          </div>
        ) : (
          <div className="attach-grid">
            {gallery.map((g) => {
              const isSel = selected.has(g.asset_id);
              const conv = g._fromConv;
              const canJump = !!(g._expired && conv && onOpenConversation);
              return (
                <div
                  key={g.asset_id}
                  className={
                    "attach-card" +
                    (isSel ? " selected" : "") +
                    (g._expired ? " expired" : "")
                  }
                >
                  <button
                    type="button"
                    className="attach-card-main"
                    onClick={() => {
                      if (g._expired) return;
                      toggleSelect(g);
                    }}
                    disabled={g._expired}
                    title={
                      g._expired
                        ? `expired — used in "${conv?.title ?? "old conversation"}"`
                        : g.display_name || g.asset_id
                    }
                  >
                    <div className="attach-thumb">
                      {g.preview_url ? (
                        <img
                          src={g.preview_url}
                          alt=""
                          onError={(ev) => {
                            (ev.target as HTMLImageElement).style.display = "none";
                          }}
                        />
                      ) : (
                        <span className="attach-glyph">▣</span>
                      )}
                      {g._expired && (
                        <span className="attach-badge expired-badge mono">
                          expired
                        </span>
                      )}
                      {g.original_format && (
                        <span className="attach-format mono">
                          {g.original_format}
                        </span>
                      )}
                      {isSel && <span className="attach-check">✓</span>}
                    </div>
                    <div className="attach-card-bottom">
                      <div
                        className="attach-name mono"
                        title={g.display_name || ""}
                      >
                        {g.display_name || g.asset_id.slice(0, 8)}
                      </div>
                      {g._timestamp && (
                        <div
                          className="attach-date mono"
                          title={new Date(g._timestamp).toLocaleString()}
                        >
                          {formatRelativeDate(g._timestamp)}
                        </div>
                      )}
                    </div>
                  </button>
                  {conv && (
                    <button
                      type="button"
                      className={
                        "attach-conv-link mono" +
                        (canJump ? " is-jump" : "")
                      }
                      onClick={() => {
                        if (canJump) {
                          onOpenConversation!(conv);
                          onClose();
                        }
                      }}
                      disabled={!canJump}
                      title={
                        canJump
                          ? `open conversation: ${conv.title}`
                          : `from: ${conv.title}`
                      }
                    >
                      <span className="attach-conv-arrow">↗</span>
                      <span className="attach-conv-name">{conv.title}</span>
                    </button>
                  )}
                </div>
              );
            })}
          </div>
        )}

        <div className="modal-actions attach-actions">
          <span className="mono attach-summary">
            {selected.size > 0
              ? `${selected.size} selected`
              : "nothing selected"}
          </span>
          <button
            className="btn-primary"
            onClick={commit}
            disabled={selected.size === 0}
          >
            ↳ add {selected.size > 0 ? `(${selected.size})` : ""}
          </button>
          <button className="btn-ghost" onClick={onClose}>
            cancel
          </button>
        </div>
      </div>
    </div>
  );
}
