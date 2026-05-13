/** A small custom chat hook tailored to our SSE protocol.
 *
 * Why not the Vercel AI SDK's useChat directly?
 *   Their wire format is opinionated (text-delta / tool-call / tool-result /
 *   finish). Our protocol carries first-class events for recommendations,
 *   pending actions, and clarifications that don't map cleanly onto the SDK
 *   shapes. A 100-LoC custom hook is more elegant than an adapter layer.
 */

import { useCallback, useRef, useState } from "react";
import { ChatEvent, ChatStartBody, streamChat } from "../lib/sse";

export type Recommendation = {
  rank: number;
  name: string;
  accuracy: number;
  why: string;
  doc: Record<string, any> | null;
  demo_url: string | null;
};

export type PendingAction = {
  type: "demo_confirm" | "tool_approval";
  tool_name: string;
  display_name?: string | null;
  icon?: string | null;
  image_name?: string | null;
  demo_url?: string | null;
  prompt: string;
};

export type AssistantTurn = {
  id: string;
  role: "assistant";
  text: string;
  statusMessage: string | null;
  recommendations: Recommendation[];
  pending: PendingAction | null;
  clarification: { question: string; options: string[] } | null;
  toolTraces: Record<string, any>[];
  imageUrls: string[];
  files: { path: string; label: string }[];
  usage: { total: number; input: number; output: number } | null;
  status: "streaming" | "done" | "error";
  error?: string;
};

export type UserTurn = {
  id: string;
  role: "user";
  text: string;
  attachments: { asset_id: string; display_name?: string; preview_url?: string | null }[];
};

export type EmbedTurn = {
  id: string;
  role: "embed";
  kind: "image" | "audio" | "video" | "iframe" | "youtube" | "info";
  src?: string;
  label?: string;
  command: string;
};

export type Turn = UserTurn | AssistantTurn | EmbedTurn;

type StartArgs = {
  message: string;
  attachments?: UserTurn["attachments"];
  model?: string | null;
  topK?: number | null;
  numChoices?: number | null;
  /** Only used by the first turn after resuming a stored conversation. */
  seedHistory?: string[] | null;
};

export function useChat() {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  const updateLast = useCallback(
    (mut: (t: AssistantTurn) => AssistantTurn) => {
      setTurns((prev) => {
        const next = [...prev];
        for (let i = next.length - 1; i >= 0; i--) {
          if (next[i].role === "assistant") {
            next[i] = mut(next[i] as AssistantTurn);
            break;
          }
        }
        return next;
      });
    },
    [setTurns]
  );

  const consumeStream = useCallback(
    async (gen: AsyncGenerator<ChatEvent>) => {
      try {
        for await (const ev of gen) {
          switch (ev.event) {
            case "session":
              setSessionId(ev.data.session_id);
              break;
            case "status":
              updateLast((t) => ({ ...t, statusMessage: ev.data.message }));
              break;
            case "text":
              // Real text arriving — drop the placeholder status line.
              updateLast((t) => ({
                ...t,
                text: t.text + ev.data.content,
                statusMessage: null,
              }));
              break;
            case "recommendation":
              updateLast((t) => ({
                ...t,
                recommendations: [...t.recommendations, ev.data],
              }));
              break;
            case "tool_trace":
              updateLast((t) => ({
                ...t,
                toolTraces: [...t.toolTraces, ev.data.trace],
              }));
              break;
            case "pending_action":
              updateLast((t) => ({ ...t, pending: ev.data }));
              break;
            case "clarification":
              updateLast((t) => ({
                ...t,
                clarification: {
                  question: ev.data.question,
                  options: ev.data.options,
                },
              }));
              break;
            case "images":
              updateLast((t) => ({
                ...t,
                imageUrls: [...t.imageUrls, ...ev.data.paths],
              }));
              break;
            case "files":
              updateLast((t) => ({
                ...t,
                files: [...t.files, ...ev.data.items],
              }));
              break;
            case "usage":
              updateLast((t) => ({ ...t, usage: ev.data }));
              break;
            case "error":
              updateLast((t) => ({
                ...t,
                status: "error",
                error: ev.data.message,
              }));
              break;
            case "done":
              updateLast((t) => ({ ...t, status: "done" }));
              break;
          }
        }
      } finally {
        setBusy(false);
      }
    },
    [setSessionId, updateLast]
  );

  const send = useCallback(
    async ({
      message,
      attachments = [],
      model,
      topK,
      numChoices,
      seedHistory,
    }: StartArgs) => {
      const userTurn: UserTurn = {
        id: `u-${Date.now()}`,
        role: "user",
        text: message,
        attachments,
      };
      const aTurn: AssistantTurn = {
        id: `a-${Date.now()}`,
        role: "assistant",
        text: "",
        statusMessage: "preparing request…",
        recommendations: [],
        pending: null,
        clarification: null,
        toolTraces: [],
        imageUrls: [],
        files: [],
        usage: null,
        status: "streaming",
      };
      setTurns((prev) => [...prev, userTurn, aTurn]);
      setBusy(true);

      const ac = new AbortController();
      abortRef.current = ac;
      const body: ChatStartBody = {
        session_id: sessionId,
        message,
        asset_ids: attachments.map((a) => a.asset_id),
        model,
        top_k: topK,
        num_choices: numChoices,
        seed_history: seedHistory ?? undefined,
      };
      try {
        await consumeStream(streamChat("/api/chat", body, ac.signal));
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        updateLast((t) => ({ ...t, status: "error", error: msg }));
        setBusy(false);
      }
    },
    [consumeStream, sessionId, updateLast]
  );

  const approve = useCallback(async () => {
    if (!sessionId) return;
    const aTurn: AssistantTurn = {
      id: `a-${Date.now()}`,
      role: "assistant",
      text: "",
      statusMessage: null,
      recommendations: [],
      pending: null,
      clarification: null,
      toolTraces: [],
      imageUrls: [],
      files: [],
      usage: null,
      status: "streaming",
    };
    setTurns((prev) => [...prev, aTurn]);
    setBusy(true);
    const ac = new AbortController();
    abortRef.current = ac;
    try {
      await consumeStream(
        streamChat(`/api/chat/${sessionId}/approve`, {}, ac.signal)
      );
    } catch (e) {
      updateLast((t) => ({ ...t, status: "error", error: String(e) }));
      setBusy(false);
    }
  }, [consumeStream, sessionId, updateLast]);

  const decline = useCallback(async () => {
    if (!sessionId) return;
    const aTurn: AssistantTurn = {
      id: `a-${Date.now()}`,
      role: "assistant",
      text: "",
      statusMessage: null,
      recommendations: [],
      pending: null,
      clarification: null,
      toolTraces: [],
      imageUrls: [],
      files: [],
      usage: null,
      status: "streaming",
    };
    setTurns((prev) => [...prev, aTurn]);
    setBusy(true);
    const ac = new AbortController();
    abortRef.current = ac;
    try {
      await consumeStream(
        streamChat(`/api/chat/${sessionId}/decline`, {}, ac.signal)
      );
    } catch (e) {
      updateLast((t) => ({ ...t, status: "error", error: String(e) }));
      setBusy(false);
    }
  }, [consumeStream, sessionId, updateLast]);

  const confirmDemo = useCallback(async () => {
    if (!sessionId) return;
    const aTurn: AssistantTurn = {
      id: `a-${Date.now()}`,
      role: "assistant",
      text: "",
      statusMessage: null,
      recommendations: [],
      pending: null,
      clarification: null,
      toolTraces: [],
      imageUrls: [],
      files: [],
      usage: null,
      status: "streaming",
    };
    setTurns((prev) => [...prev, aTurn]);
    setBusy(true);
    const ac = new AbortController();
    abortRef.current = ac;
    try {
      await consumeStream(
        streamChat(`/api/chat/${sessionId}/confirm-demo`, {}, ac.signal)
      );
    } catch (e) {
      updateLast((t) => ({ ...t, status: "error", error: String(e) }));
      setBusy(false);
    }
  }, [consumeStream, sessionId, updateLast]);

  const stop = useCallback(() => {
    abortRef.current?.abort();
    setBusy(false);
  }, []);

  const reset = useCallback((seedTurns: Turn[] = []) => {
    setTurns(seedTurns);
    setSessionId(null);
    setBusy(false);
    abortRef.current?.abort();
    abortRef.current = null;
  }, []);

  const pushTurn = useCallback((turn: Turn) => {
    setTurns((prev) => [...prev, turn]);
  }, []);

  return {
    turns,
    sessionId,
    setSessionId,
    busy,
    send,
    approve,
    decline,
    confirmDemo,
    stop,
    reset,
    pushTurn,
  };
}
