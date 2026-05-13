/** A minimal SSE-over-fetch client.
 *
 * EventSource doesn't support POST bodies, so we drive our own SSE parser
 * over `fetch().body`. The backend emits named events in the standard
 * `event:` / `data:` format; this util yields `{ event, data }` per record.
 */

export type ChatEvent =
  | { event: "session"; data: { session_id: string } }
  | { event: "status"; data: { phase: string; message: string } }
  | { event: "text"; data: { content: string } }
  | {
      event: "recommendation";
      data: {
        rank: number;
        name: string;
        accuracy: number;
        why: string;
        doc: Record<string, unknown> | null;
        demo_url: string | null;
      };
    }
  | {
      event: "tool_trace";
      data: { trace: Record<string, unknown> };
    }
  | {
      event: "pending_action";
      data: {
        type: "demo_confirm" | "tool_approval";
        tool_name: string;
        display_name?: string | null;
        icon?: string | null;
        image_name?: string | null;
        demo_url?: string | null;
        prompt: string;
      };
    }
  | {
      event: "clarification";
      data: { question: string; context: string | null; options: string[] };
    }
  | { event: "images"; data: { paths: string[] } }
  | {
      event: "files";
      data: { items: { path: string; label: string }[] };
    }
  | { event: "usage"; data: { total: number; input: number; output: number } }
  | { event: "error"; data: { message: string; code: string } }
  | { event: "done"; data: { status: string } };

export type ChatStartBody = {
  session_id?: string | null;
  message: string;
  asset_ids?: string[];
  model?: string | null;
  top_k?: number | null;
  num_choices?: number | null;
  /** Used when resuming a stored conversation — seeds the server-side session
   *  with the prior transcript so the agent has context. */
  seed_history?: string[] | null;
};

export async function* streamChat(
  path: string,
  body: ChatStartBody | Record<string, unknown>,
  signal?: AbortSignal
): AsyncGenerator<ChatEvent> {
  const r = await fetch(path, {
    method: "POST",
    credentials: "include",
    headers: { "content-type": "application/json", accept: "text/event-stream" },
    body: JSON.stringify(body),
    signal,
  });
  if (!r.ok || !r.body) {
    throw new Error(`SSE start failed: ${r.status} ${r.statusText}`);
  }

  const reader = r.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buf = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });

    // SSE records are separated by a blank line ("\n\n"). Some servers send
    // "\r\n\r\n"; normalize to "\n".
    buf = buf.replace(/\r\n/g, "\n");
    let idx = buf.indexOf("\n\n");
    while (idx !== -1) {
      const record = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      const parsed = parseRecord(record);
      if (parsed) yield parsed;
      idx = buf.indexOf("\n\n");
    }
  }

  // Flush any trailing record without final newline pair.
  if (buf.trim()) {
    const parsed = parseRecord(buf);
    if (parsed) yield parsed;
  }
}

function parseRecord(record: string): ChatEvent | null {
  let event = "message";
  const dataLines: string[] = [];
  for (const line of record.split("\n")) {
    if (!line || line.startsWith(":")) continue;
    const colon = line.indexOf(":");
    const field = colon === -1 ? line : line.slice(0, colon);
    const value = colon === -1 ? "" : line.slice(colon + 1).replace(/^ /, "");
    if (field === "event") event = value;
    else if (field === "data") dataLines.push(value);
  }
  if (!dataLines.length) return null;
  const raw = dataLines.join("\n");
  let data: unknown = raw;
  try {
    data = JSON.parse(raw);
  } catch {
    // Leave as string; some events may be plain strings in the future.
  }
  return { event, data } as ChatEvent;
}
