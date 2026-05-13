/** Local conversation history stored in browser localStorage.
 *
 * The user can re-open any prior conversation. Re-opening sends the stored
 * transcript back to the server as ``seed_history`` so the agent picks up
 * where it left off. Uploaded files are NOT stored in localStorage (too big,
 * and the server-side TTL would have evicted the asset anyway) — when a
 * resumed conversation references an attachment we render a cat-themed
 * "please re-upload" placeholder.
 */

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import type { Turn } from "./useChat";

const STORAGE_KEY = "ai_agent.conversations.v1";

export type StoredConversation = {
  id: string;
  title: string;
  createdAt: number;
  updatedAt: number;
  turns: Turn[];
  /** Last known server session id — informational; we always start a fresh
   * server session on resume because the original is likely TTL'd. */
  serverSessionId: string | null;
};

type ConversationsContextValue = {
  conversations: StoredConversation[];
  activeId: string | null;
  setActiveId: (id: string | null) => void;
  saveTurns: (id: string, turns: Turn[], serverSessionId: string | null) => void;
  deleteConversation: (id: string) => void;
  clearAll: () => void;
  startNew: () => string;
};

const ConversationsContext = createContext<ConversationsContextValue | null>(null);

function readAll(): StoredConversation[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed as StoredConversation[];
  } catch {
    return [];
  }
}

function writeAll(list: StoredConversation[]): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(list));
  } catch {
    // QuotaExceeded — drop oldest until it fits.
    const trimmed = [...list].sort((a, b) => b.updatedAt - a.updatedAt).slice(0, 20);
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(trimmed));
    } catch {
      // give up silently
    }
  }
}

function newId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID();
  }
  return `conv-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

function titleFromTurns(turns: Turn[]): string {
  for (const t of turns) {
    if (t.role === "user" && t.text.trim()) {
      const s = t.text.trim();
      return s.length > 60 ? s.slice(0, 57) + "…" : s;
    }
  }
  return "new conversation";
}

export function ConversationsProvider({ children }: { children: ReactNode }) {
  const [conversations, setConversations] = useState<StoredConversation[]>(() =>
    readAll()
  );
  const [activeId, setActiveId] = useState<string | null>(null);

  useEffect(() => {
    writeAll(conversations);
  }, [conversations]);

  const saveTurns = useCallback(
    (id: string, turns: Turn[], serverSessionId: string | null) => {
      // Strip non-serializable bits — none currently, but stay safe.
      const cleanTurns: Turn[] = turns.map((t) => {
        if (t.role === "assistant") {
          return { ...t, statusMessage: null };
        }
        return t;
      });

      setConversations((prev) => {
        const idx = prev.findIndex((c) => c.id === id);
        const now = Date.now();
        if (idx === -1) {
          const conv: StoredConversation = {
            id,
            title: titleFromTurns(cleanTurns),
            createdAt: now,
            updatedAt: now,
            turns: cleanTurns,
            serverSessionId,
          };
          return [conv, ...prev];
        }
        const next = [...prev];
        const existing = next[idx];
        next[idx] = {
          ...existing,
          turns: cleanTurns,
          title:
            existing.title === "new conversation"
              ? titleFromTurns(cleanTurns)
              : existing.title,
          updatedAt: now,
          serverSessionId,
        };
        // Bring the most recent to the top.
        next.sort((a, b) => b.updatedAt - a.updatedAt);
        return next;
      });
    },
    []
  );

  const deleteConversation = useCallback((id: string) => {
    setConversations((prev) => prev.filter((c) => c.id !== id));
    setActiveId((cur) => (cur === id ? null : cur));
  }, []);

  const clearAll = useCallback(() => {
    setConversations([]);
    setActiveId(null);
  }, []);

  const startNew = useCallback(() => {
    const id = newId();
    setActiveId(id);
    return id;
  }, []);

  const value = useMemo<ConversationsContextValue>(
    () => ({
      conversations,
      activeId,
      setActiveId,
      saveTurns,
      deleteConversation,
      clearAll,
      startNew,
    }),
    [conversations, activeId, saveTurns, deleteConversation, clearAll, startNew]
  );

  return (
    <ConversationsContext.Provider value={value}>
      {children}
    </ConversationsContext.Provider>
  );
}

export function useConversations(): ConversationsContextValue {
  const ctx = useContext(ConversationsContext);
  if (!ctx) {
    throw new Error(
      "useConversations must be used inside <ConversationsProvider>"
    );
  }
  return ctx;
}

/** Pull the agent-side transcript ("User: …" / "Assistant: …" lines) out of
 *  a stored conversation, so we can ship it back as ``seed_history``. */
export function turnsToHistory(turns: Turn[]): string[] {
  const out: string[] = [];
  for (const t of turns) {
    if (t.role === "user") out.push(`User: ${t.text}`);
    else if (t.role === "assistant" && t.text) out.push(`Assistant: ${t.text}`);
  }
  return out;
}

/** True if this conversation referenced any image attachments. We use it to
 *  decide whether to render the cat-please-re-upload placeholder when we
 *  resume the conversation (server-side asset ids are gone). */
export function conversationHadAttachments(turns: Turn[]): boolean {
  return turns.some((t) => t.role === "user" && t.attachments.length > 0);
}
