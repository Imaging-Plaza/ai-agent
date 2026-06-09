/** Browser-side speech-to-text powered by parakeet.js.
 *
 * Lifecycle states:
 *   idle        - worker not booted (lazy: first call to ensureWorker)
 *   downloading - first-time model download (~2.5 GB, cached afterwards)
 *   ready       - model loaded and warm
 *   recording   - mic open, accumulating PCM
 *   transcribing- decoded clip is being sent to the model
 *   error       - last operation failed (see message)
 *
 * The worker only mounts when the user opens the modal or clicks the mic, so
 * a visitor who never touches voice never downloads anything.
 *
 * Cache management: parakeet.js downloads through @huggingface/transformers
 * which uses the browser Cache Storage API under the "transformers-cache"
 * name. ``deleteModel`` clears that cache (best-effort across renames).
 */

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";

type Phase =
  | { kind: "idle" }
  | { kind: "downloading"; loaded: number; total: number; file: string }
  | { kind: "ready" }
  | { kind: "recording"; startedAt: number }
  | { kind: "transcribing" }
  | { kind: "error"; message: string };

/** Microphone availability + permission state, surfaced so the UI can give
 *  honest feedback before the user clicks a button that's never going to work
 *  on their machine. */
export type MicState =
  | "unknown"          // not yet probed
  | "unsupported"      // no getUserMedia / no MediaDevices in this browser
  | "no_device"        // navigator detects no audio input
  | "permission_prompt"// permission state is "prompt" (will ask on use)
  | "permission_denied"// user previously blocked the mic
  | "permission_granted"; // ready to record

type ContextValue = {
  phase: Phase;
  hasModel: boolean;
  micState: MicState;
  /** Re-probe device + permission. Cheap; safe to call from a modal. */
  refreshMic: () => Promise<void>;
  /** Boot the worker and trigger model download / warm-up. */
  ensureLoaded: () => Promise<void>;
  /** Trigger the browser's mic permission prompt explicitly (separate from
   *  starting a recording, so the user can grant access from the modal). */
  requestMic: () => Promise<MicState>;
  /** Open the mic and start accumulating audio. */
  startRecording: () => Promise<void>;
  /** Stop the mic, send the buffered audio to the model, resolve with the
   *  decoded text once it comes back. */
  stopAndTranscribe: () => Promise<string>;
  /** Wipe the cached model from the browser. */
  deleteModel: () => Promise<void>;
};

const TranscriptionContext = createContext<ContextValue | null>(null);
const CACHED_FLAG = "ai_agent.parakeet.cached.v1";

export function TranscriptionProvider({ children }: { children: ReactNode }) {
  const [phase, setPhase] = useState<Phase>({ kind: "idle" });
  const [micState, setMicState] = useState<MicState>("unknown");
  const [hasModel, setHasModel] = useState<boolean>(() => {
    try {
      return localStorage.getItem(CACHED_FLAG) === "1";
    } catch {
      return false;
    }
  });
  const workerRef = useRef<Worker | null>(null);
  const reqIdRef = useRef(0);
  const pendingRef = useRef<
    Map<number, { resolve: (s: string) => void; reject: (e: Error) => void }>
  >(new Map());

  // Audio capture
  const streamRef = useRef<MediaStream | null>(null);
  const ctxRef = useRef<AudioContext | null>(null);
  const procRef = useRef<ScriptProcessorNode | null>(null);
  const chunksRef = useRef<Float32Array[]>([]);
  const captureRateRef = useRef<number>(48000);

  // ---------- Mic probing -------------------------------------------------
  const refreshMic = useCallback(async () => {
    if (typeof navigator === "undefined" || !navigator.mediaDevices) {
      setMicState("unsupported");
      return;
    }
    if (typeof navigator.mediaDevices.getUserMedia !== "function") {
      setMicState("unsupported");
      return;
    }
    // Permissions API is best-effort (Safari historically didn't expose
    // "microphone" — we fall back to inspecting devices in that case).
    let permState: PermissionState | null = null;
    try {
      // @ts-ignore — "microphone" is a valid name on Chromium/Firefox
      const p = await navigator.permissions?.query?.({ name: "microphone" });
      permState = (p?.state as PermissionState) ?? null;
    } catch {
      permState = null;
    }
    let hasMic = true;
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      hasMic = devices.some((d) => d.kind === "audioinput");
    } catch {
      // If enumerateDevices fails before any permission, assume yes; we'll
      // discover the truth on getUserMedia.
      hasMic = true;
    }
    if (!hasMic) {
      setMicState("no_device");
      return;
    }
    if (permState === "granted") setMicState("permission_granted");
    else if (permState === "denied") setMicState("permission_denied");
    else setMicState("permission_prompt");
  }, []);

  useEffect(() => {
    void refreshMic();
  }, [refreshMic]);

  // React to permission changes (e.g., user toggling site permissions).
  useEffect(() => {
    if (!navigator.permissions?.query) return;
    let p: PermissionStatus | null = null;
    let cancelled = false;
    (async () => {
      try {
        // @ts-ignore
        p = await navigator.permissions.query({ name: "microphone" });
        if (cancelled || !p) return;
        const onChange = () => void refreshMic();
        p.addEventListener("change", onChange);
      } catch {
        /* ignore */
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [refreshMic]);

  const requestMic = useCallback(async (): Promise<MicState> => {
    if (typeof navigator === "undefined" || !navigator.mediaDevices?.getUserMedia) {
      setMicState("unsupported");
      return "unsupported";
    }
    try {
      // Open + immediately close a probe stream to trigger the permission
      // prompt without keeping the mic hot.
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      stream.getTracks().forEach((t) => t.stop());
      setMicState("permission_granted");
      return "permission_granted";
    } catch (err: any) {
      const name = err?.name || "";
      if (name === "NotAllowedError" || name === "SecurityError") {
        setMicState("permission_denied");
        return "permission_denied";
      }
      if (name === "NotFoundError" || name === "OverconstrainedError") {
        setMicState("no_device");
        return "no_device";
      }
      // Generic failure — report as denied so the modal explains.
      setMicState("permission_denied");
      return "permission_denied";
    }
  }, []);

  const ensureWorker = useCallback(() => {
    if (workerRef.current) return workerRef.current;
    const w = new Worker(
      new URL("../workers/parakeet.worker.ts", import.meta.url),
      { type: "module" }
    );
    w.onmessage = (e) => {
      const m = e.data || {};
      switch (m.type) {
        case "progress":
          setPhase({
            kind: "downloading",
            loaded: m.loaded ?? 0,
            total: m.total ?? 0,
            file: m.file || "",
          });
          break;
        case "ready":
          setPhase({ kind: "ready" });
          setHasModel(true);
          try {
            localStorage.setItem(CACHED_FLAG, "1");
          } catch {}
          break;
        case "transcription": {
          const p = pendingRef.current.get(m.requestId);
          pendingRef.current.delete(m.requestId);
          p?.resolve(m.text || "");
          setPhase({ kind: "ready" });
          break;
        }
        case "error": {
          if (m.requestId !== undefined) {
            const p = pendingRef.current.get(m.requestId);
            pendingRef.current.delete(m.requestId);
            p?.reject(new Error(m.message || "transcription failed"));
          }
          setPhase({ kind: "error", message: m.message || "unknown error" });
          break;
        }
      }
    };
    workerRef.current = w;
    return w;
  }, []);

  const ensureLoaded = useCallback(async () => {
    const w = ensureWorker();
    setPhase((p) =>
      p.kind === "ready" || p.kind === "downloading"
        ? p
        : { kind: "downloading", loaded: 0, total: 0, file: "" }
    );
    w.postMessage({
      type: "load",
      data: {
        backend: ("gpu" in navigator) ? "webgpu" : "wasm",
      },
    });
    // Resolve when phase becomes "ready" (handled by message handler).
  }, [ensureWorker]);

  /** Downsample a Float32 buffer from ``inRate`` to 16 kHz mono. */
  function downsampleTo16k(buf: Float32Array, inRate: number): Float32Array {
    if (inRate === 16000) return buf;
    const ratio = inRate / 16000;
    const outLen = Math.floor(buf.length / ratio);
    const out = new Float32Array(outLen);
    let inIdx = 0;
    let outIdx = 0;
    while (outIdx < outLen) {
      const next = Math.floor((outIdx + 1) * ratio);
      // Box-mean sampler — good enough for ASR pre-processing.
      let sum = 0;
      let count = 0;
      for (let i = inIdx; i < next && i < buf.length; i++) {
        sum += buf[i];
        count++;
      }
      out[outIdx] = count > 0 ? sum / count : 0;
      inIdx = next;
      outIdx++;
    }
    return out;
  }

  const startRecording = useCallback(async () => {
    if (phase.kind === "recording") return;
    if (phase.kind !== "ready") {
      throw new Error("model_not_ready");
    }
    if (!navigator.mediaDevices?.getUserMedia) {
      setMicState("unsupported");
      throw new Error("mic_unsupported");
    }
    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          sampleRate: 16000,
        },
      });
    } catch (err: any) {
      const name = err?.name || "";
      if (name === "NotAllowedError" || name === "SecurityError") {
        setMicState("permission_denied");
        throw new Error("mic_permission_denied");
      }
      if (name === "NotFoundError" || name === "OverconstrainedError") {
        setMicState("no_device");
        throw new Error("mic_no_device");
      }
      throw new Error(`mic_open_failed: ${err?.message || err}`);
    }
    setMicState("permission_granted");
    streamRef.current = stream;

    const ctx = new (window.AudioContext ||
      (window as any).webkitAudioContext)();
    ctxRef.current = ctx;
    captureRateRef.current = ctx.sampleRate;

    const src = ctx.createMediaStreamSource(stream);
    // ScriptProcessorNode is deprecated but universally supported. AudioWorklet
    // would be cleaner; this version keeps the wiring straightforward.
    const proc = ctx.createScriptProcessor(4096, 1, 1);
    procRef.current = proc;
    chunksRef.current = [];
    proc.onaudioprocess = (ev) => {
      const ch = ev.inputBuffer.getChannelData(0);
      // Copy out — the buffer is re-used by the audio thread.
      const copy = new Float32Array(ch.length);
      copy.set(ch);
      chunksRef.current.push(copy);
    };
    src.connect(proc);
    proc.connect(ctx.destination);

    setPhase({ kind: "recording", startedAt: Date.now() });
  }, [phase]);

  const stopAndTranscribe = useCallback(async () => {
    if (phase.kind !== "recording") {
      throw new Error("not recording");
    }
    // Tear down audio graph
    try {
      procRef.current?.disconnect();
    } catch {}
    try {
      streamRef.current?.getTracks().forEach((t) => t.stop());
    } catch {}
    try {
      await ctxRef.current?.close();
    } catch {}
    const rate = captureRateRef.current;
    const chunks = chunksRef.current;
    chunksRef.current = [];
    procRef.current = null;
    streamRef.current = null;
    ctxRef.current = null;

    const totalLen = chunks.reduce((a, c) => a + c.length, 0);
    const merged = new Float32Array(totalLen);
    let off = 0;
    for (const c of chunks) {
      merged.set(c, off);
      off += c.length;
    }
    const audio = downsampleTo16k(merged, rate);

    if (audio.length < 16000 * 0.3) {
      setPhase({ kind: "ready" });
      return "";
    }

    setPhase({ kind: "transcribing" });
    const w = ensureWorker();
    const requestId = ++reqIdRef.current;
    const text = await new Promise<string>((resolve, reject) => {
      pendingRef.current.set(requestId, { resolve, reject });
      w.postMessage({ type: "transcribe", data: { audio, requestId } });
    });
    return text;
  }, [phase, ensureWorker]);

  const deleteModel = useCallback(async () => {
    try {
      // Drop the worker so any held references release the model.
      workerRef.current?.terminate();
      workerRef.current = null;
    } catch {}
    try {
      const names = await caches.keys();
      for (const n of names) {
        if (/transformers|huggingface|parakeet|onnx/i.test(n)) {
          await caches.delete(n);
        }
      }
    } catch {}
    try {
      // Best-effort IndexedDB wipe — some HF caches use it as well.
      const dbs = await (indexedDB as any).databases?.();
      if (Array.isArray(dbs)) {
        for (const db of dbs) {
          if (
            db?.name &&
            /transformers|huggingface|parakeet|onnx/i.test(db.name)
          ) {
            indexedDB.deleteDatabase(db.name);
          }
        }
      }
    } catch {}
    try {
      localStorage.removeItem(CACHED_FLAG);
    } catch {}
    setHasModel(false);
    setPhase({ kind: "idle" });
  }, []);

  useEffect(
    () => () => {
      try {
        workerRef.current?.terminate();
      } catch {}
    },
    []
  );

  const value = useMemo<ContextValue>(
    () => ({
      phase,
      hasModel,
      micState,
      refreshMic,
      ensureLoaded,
      requestMic,
      startRecording,
      stopAndTranscribe,
      deleteModel,
    }),
    [
      phase,
      hasModel,
      micState,
      refreshMic,
      ensureLoaded,
      requestMic,
      startRecording,
      stopAndTranscribe,
      deleteModel,
    ]
  );

  return (
    <TranscriptionContext.Provider value={value}>
      {children}
    </TranscriptionContext.Provider>
  );
}

export function useTranscription(): ContextValue {
  const ctx = useContext(TranscriptionContext);
  if (!ctx) throw new Error("useTranscription must be used inside <TranscriptionProvider>");
  return ctx;
}
