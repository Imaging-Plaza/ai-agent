/** Browser-local backup / restore.
 *
 *  Export: bundles every `localStorage` key the app uses into a single
 *  `data.json`, wraps it in a ZIP, and triggers a download.
 *
 *  Import: takes a `File` (the user picks one from disk), unzips it,
 *  parses `data.json`, writes its keys back into localStorage. The caller
 *  is expected to reload the page afterwards so React hooks re-read.
 *
 *  Everything is browser-side; nothing hits the server.
 */

import { strFromU8, strToU8, unzipSync, zipSync } from "fflate";

/** localStorage keys we care about. Any unknown key in an import is
 *  ignored to keep the surface predictable. */
const KEYS = [
  "ai_agent.conversations.v1",
  "ai_agent.parakeet.cached.v1",
  "theme",
];

type Backup = {
  app: "ai_plaza";
  version: 1;
  exported_at: string;
  data: Record<string, string>;
};

function snapshotLocal(): Backup {
  const data: Record<string, string> = {};
  for (const k of KEYS) {
    try {
      const v = localStorage.getItem(k);
      if (v != null) data[k] = v;
    } catch {
      /* skip */
    }
  }
  return {
    app: "ai_plaza",
    version: 1,
    exported_at: new Date().toISOString(),
    data,
  };
}

export function exportLocal(): void {
  const payload = snapshotLocal();
  const json = JSON.stringify(payload, null, 2);
  const bytes = zipSync(
    { "data.json": strToU8(json) },
    { level: 9 }
  );
  // `bytes` is a Uint8Array; the Blob constructor accepts it but our TS
  // lib's `BlobPart` union is narrower than the runtime spec, so cast.
  const blob = new Blob([bytes as BlobPart], { type: "application/zip" });
  const url = URL.createObjectURL(blob);
  const stamp = payload.exported_at.replace(/[:.]/g, "-").slice(0, 19);
  const a = document.createElement("a");
  a.href = url;
  a.download = `ai_plaza-backup-${stamp}.zip`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  // Free the blob URL after a tick.
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export type ImportResult = {
  restored_keys: string[];
  skipped_keys: string[];
  exported_at?: string;
};

export async function importLocal(file: File): Promise<ImportResult> {
  const buf = new Uint8Array(await file.arrayBuffer());
  let entries: Record<string, Uint8Array>;
  try {
    entries = unzipSync(buf);
  } catch (err: any) {
    throw new Error(`not_a_zip: ${err?.message || err}`);
  }
  const json = entries["data.json"];
  if (!json) throw new Error("missing_data_json");
  let parsed: Backup;
  try {
    parsed = JSON.parse(strFromU8(json));
  } catch (err: any) {
    throw new Error(`bad_json: ${err?.message || err}`);
  }
  if (parsed?.app !== "ai_plaza" || typeof parsed.data !== "object") {
    throw new Error("not_an_ai_plaza_backup");
  }
  const restored: string[] = [];
  const skipped: string[] = [];
  for (const [k, v] of Object.entries(parsed.data)) {
    if (!KEYS.includes(k)) {
      skipped.push(k);
      continue;
    }
    try {
      localStorage.setItem(k, String(v));
      restored.push(k);
    } catch {
      skipped.push(k);
    }
  }
  return {
    restored_keys: restored,
    skipped_keys: skipped,
    exported_at: parsed.exported_at,
  };
}
