/**
 * Web Worker wrapping parakeet.js so the ~2.5 GB model download + ONNX
 * inference never block the main thread.
 *
 * Protocol (postMessage → from main thread):
 *   { type: 'load', data: { backend?: 'webgpu' | 'wasm' } }
 *   { type: 'transcribe', data: { audio: Float32Array, requestId: number } }
 *
 * Replies (postMessage → to main thread):
 *   { type: 'progress', file, loaded, total }
 *   { type: 'ready' }
 *   { type: 'transcription', requestId, text, words, metrics }
 *   { type: 'error', message, requestId? }
 */

// @ts-ignore — no types shipped
import { fromHub } from "parakeet.js";

declare const self: DedicatedWorkerGlobalScope;

let model: any = null;
let loading = false;

async function load(backend: "webgpu" | "wasm" = "webgpu") {
  if (model) {
    self.postMessage({ type: "ready" });
    return;
  }
  if (loading) return;
  loading = true;
  try {
    model = await fromHub("parakeet-tdt-0.6b-v3", {
      backend,
      progress: (p: { file: string; loaded: number; total: number }) => {
        self.postMessage({
          type: "progress",
          file: p.file,
          loaded: p.loaded,
          total: p.total,
        });
      },
    });
    self.postMessage({ type: "ready" });
  } catch (err: any) {
    self.postMessage({ type: "error", message: String(err?.message ?? err) });
  } finally {
    loading = false;
  }
}

async function transcribe(audio: Float32Array, requestId: number) {
  if (!model) {
    self.postMessage({
      type: "error",
      requestId,
      message: "Model not loaded yet",
    });
    return;
  }
  try {
    const result = await model.transcribe(audio, 16000);
    self.postMessage({
      type: "transcription",
      requestId,
      text: result.utterance_text || "",
      words: result.words || [],
      metrics: result.metrics || null,
    });
  } catch (err: any) {
    self.postMessage({
      type: "error",
      requestId,
      message: String(err?.message ?? err),
    });
  }
}

self.onmessage = (e: MessageEvent) => {
  const { type, data } = e.data || {};
  if (type === "load") void load(data?.backend);
  else if (type === "transcribe") void transcribe(data.audio, data.requestId);
};

export {};
