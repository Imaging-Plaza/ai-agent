/** Modal that handles BOTH the parakeet model lifecycle AND microphone
 *  permissions, so the user can fix every prerequisite from one place.
 */

import { useEffect } from "react";
import { useTranscription, type MicState } from "../hooks/useTranscription";

type Props = {
  open: boolean;
  onClose: () => void;
};

function fmtBytes(b: number): string {
  if (!Number.isFinite(b) || b <= 0) return "—";
  const u = ["B", "KB", "MB", "GB"];
  let i = 0;
  while (b >= 1024 && i < u.length - 1) {
    b /= 1024;
    i++;
  }
  return `${b.toFixed(b >= 100 ? 0 : 1)} ${u[i]}`;
}

function micLabel(s: MicState): { icon: string; label: string; tone: "ok" | "warn" | "err" } {
  switch (s) {
    case "permission_granted":
      return { icon: "●", label: "mic_ready", tone: "ok" };
    case "permission_prompt":
      return { icon: "○", label: "mic_will_prompt", tone: "warn" };
    case "permission_denied":
      return { icon: "✕", label: "mic_blocked", tone: "err" };
    case "no_device":
      return { icon: "✕", label: "no_microphone_detected", tone: "err" };
    case "unsupported":
      return { icon: "✕", label: "browser_unsupported", tone: "err" };
    case "unknown":
    default:
      return { icon: "·", label: "checking…", tone: "warn" };
  }
}

export default function SpeechModal({ open, onClose }: Props) {
  const {
    phase,
    hasModel,
    micState,
    refreshMic,
    requestMic,
    ensureLoaded,
    deleteModel,
  } = useTranscription();

  useEffect(() => {
    if (!open) return;
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
    }
    document.addEventListener("keydown", onKey);
    void refreshMic();
    return () => document.removeEventListener("keydown", onKey);
  }, [open, onClose, refreshMic]);

  if (!open) return null;

  const downloading = phase.kind === "downloading";
  const ready = phase.kind === "ready";
  const errored = phase.kind === "error";

  const pct =
    downloading && phase.total > 0
      ? Math.min(100, Math.round((phase.loaded / phase.total) * 100))
      : 0;

  const mic = micLabel(micState);
  const modelTone = ready
    ? "ok"
    : hasModel || downloading
      ? "warn"
      : "warn";

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div
        className="modal speech-modal"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
      >
        <header className="modal-head">
          <div className="modal-tag mono">model · parakeet-tdt-0.6b-v3</div>
          <button className="modal-x" onClick={onClose} aria-label="Close">
            ✕
          </button>
        </header>

        <h2 className="modal-title">on-device speech-to-text</h2>
        <p className="modal-sub">
          NVIDIA Parakeet TDT 0.6B v3 runs entirely in your browser (WebGPU →
          WASM fallback). 25 European languages, word-level timestamps, no
          audio leaves the page.
        </p>

        <div className="modal-stats mono">
          <div>
            <span>size</span> ~2.5 GB
          </div>
          <div>
            <span>cache</span> browser storage
          </div>
          <div>
            <span>backend</span> {"gpu" in navigator ? "webgpu" : "wasm"}
          </div>
        </div>

        {/* Prerequisites — two rows: model + mic */}
        <div className="prereq-grid">
          <div className={"prereq " + modelTone}>
            <span className="prereq-dot" aria-hidden />
            <div className="prereq-text">
              <div className="prereq-label mono">model</div>
              <div className="prereq-state">
                {downloading
                  ? `downloading… ${pct}%`
                  : ready
                    ? "ready"
                    : hasModel
                      ? "cached · idle"
                      : "not downloaded"}
              </div>
            </div>
          </div>
          <div className={"prereq " + mic.tone}>
            <span className="prereq-dot" aria-hidden />
            <div className="prereq-text">
              <div className="prereq-label mono">microphone</div>
              <div className="prereq-state">{mic.label}</div>
            </div>
          </div>
        </div>

        {downloading && (
          <div className="modal-progress">
            <div className="bar">
              <div className="bar-fill" style={{ width: `${pct}%` }} />
            </div>
            <div className="mono progress-meta">
              {phase.file ? <span>{phase.file}</span> : <span>downloading…</span>}
              <span>
                {fmtBytes(phase.loaded)} / {fmtBytes(phase.total)} · {pct}%
              </span>
            </div>
            <p className="modal-note">
              First-time download only. Future sessions load from cache and
              start instantly.
            </p>
          </div>
        )}

        {errored && (
          <div className="error-banner" style={{ marginTop: 12 }}>
            {phase.kind === "error" ? phase.message : ""}
          </div>
        )}

        {micState === "permission_denied" && (
          <div className="modal-note help">
            <strong>microphone blocked.</strong> click the lock icon in your
            browser's address bar, allow microphone access for this site, then
            press <em>refresh mic</em>.
          </div>
        )}
        {micState === "no_device" && (
          <div className="modal-note help">
            <strong>no microphone found.</strong> plug one in (or enable an
            internal one in your OS sound settings), then press{" "}
            <em>refresh mic</em>.
          </div>
        )}
        {micState === "unsupported" && (
          <div className="modal-note help">
            this browser doesn't expose audio capture. try a recent Chrome,
            Edge, or Safari.
          </div>
        )}

        <div className="modal-actions">
          {!ready && !downloading && (
            <button className="btn-primary" onClick={() => void ensureLoaded()}>
              {hasModel ? "↳ reload model" : "↳ download model"}
            </button>
          )}
          {(micState === "permission_prompt" || micState === "unknown") && (
            <button className="btn-primary" onClick={() => void requestMic()}>
              ↳ grant mic access
            </button>
          )}
          {(micState === "permission_denied" ||
            micState === "no_device" ||
            micState === "unsupported") && (
            <button className="btn-ghost" onClick={() => void refreshMic()}>
              refresh mic
            </button>
          )}
          {(hasModel || ready) && (
            <button
              className="btn-decline"
              onClick={async () => {
                if (
                  confirm(
                    "Delete the cached speech model? You'll have to re-download next time."
                  )
                ) {
                  await deleteModel();
                }
              }}
            >
              delete cached model
            </button>
          )}
          <button className="btn-ghost" onClick={onClose}>
            close
          </button>
        </div>
      </div>
    </div>
  );
}
