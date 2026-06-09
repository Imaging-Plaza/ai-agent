/** Typed wrapper around the FastAPI surface.
 *
 * In dev, Vite proxies /api/* to :8000.
 * In prod, the same FastAPI serves both /api/* and the static SPA.
 */

export type ModelOption = {
  display_name: string;
  name: string;
  provider?: string | null;
};

export type Asset = {
  asset_id: string;
  display_name?: string;
  original_format?: string;
  preview_url?: string | null;
  metadata_text?: string | null;
  /** Unix epoch seconds. Server populates this on upload; older clients won't have it. */
  created_at?: number | null;
};

export type UploadResponse = {
  session_id: string;
  assets: Asset[];
};

export type Health = {
  ok: boolean;
  catalog_docs: number;
  sessions: number;
};

class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function jsonGet<T>(path: string): Promise<T> {
  const r = await fetch(path, { credentials: "include" });
  if (!r.ok) throw new ApiError(r.status, await safeText(r));
  return (await r.json()) as T;
}

async function jsonPost<T>(path: string, body: unknown): Promise<T> {
  const r = await fetch(path, {
    method: "POST",
    credentials: "include",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new ApiError(r.status, await safeText(r));
  return (await r.json()) as T;
}

async function safeText(r: Response): Promise<string> {
  try {
    return await r.text();
  } catch {
    return r.statusText;
  }
}

export const api = {
  // Auth
  authStatus: () => jsonGet<{ required: boolean }>("/api/auth/status"),
  login: (password: string) =>
    jsonPost<{ ok: boolean }>("/api/auth/login", { password }),
  logout: () => jsonPost<{ ok: boolean }>("/api/auth/logout", {}),

  // Catalog / models / health
  models: () => jsonGet<ModelOption[]>("/api/models"),
  healthz: () => jsonGet<Health>("/api/healthz"),

  // Files
  uploadFiles: async (files: File[], sessionId?: string): Promise<UploadResponse> => {
    const form = new FormData();
    for (const f of files) form.append("files", f);
    if (sessionId) form.append("session_id", sessionId);
    const r = await fetch("/api/files", {
      method: "POST",
      credentials: "include",
      body: form,
    });
    if (!r.ok) throw new ApiError(r.status, await safeText(r));
    return (await r.json()) as UploadResponse;
  },

  previewUrl: (assetId: string) => `/api/files/preview/${assetId}`,
};

export { ApiError };
