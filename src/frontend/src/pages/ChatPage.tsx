import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { Asset } from "../lib/api";
import { HELP_TEXT, type ParsedCommand } from "../lib/slashCommands";
import type { EmbedTurn } from "../hooks/useChat";
type ChatInputRefs = {
  openGallery: () => void;
  prefill: (text: string, attachments: Asset[]) => void;
} | null;
import ChatInput, { type StagedAsset } from "../components/ChatInput";
import MessageList from "../components/MessageList";
import Minimap from "../components/Minimap";
import ModelPicker from "../components/ModelPicker";
import QueueBanner, { type QueuedMessage } from "../components/QueueBanner";
import Sidebar from "../components/Sidebar";
import ThemeToggle from "../components/ThemeToggle";
import { useAuth } from "../hooks/useAuth";
import { useChat } from "../hooks/useChat";
import {
  conversationHadAttachments,
  turnsToHistory,
  useConversations,
  type StoredConversation,
} from "../hooks/useConversations";

export default function ChatPage() {
  const { logout } = useAuth();
  const chat = useChat();
  const conv = useConversations();

  const [model, setModel] = useState<string | null>(null);
  const [topK, setTopK] = useState(8);
  const [numChoices, setNumChoices] = useState(3);

  // Once the user has picked an example (or has resumed an old chat), the
  // example cards stop showing — they prefill the composer and the user is
  // expected to edit/send from there.
  const [examplesDismissed, setExamplesDismissed] = useState(false);

  // Title shown in the chat header. Prefer the title of the active stored
  // conversation; fall back to the first user message if a fresh chat is
  // mid-flight; otherwise show "new conversation".
  const activeConv = conv.conversations.find((c) => c.id === conv.activeId);
  const headerTitle = useMemo(() => {
    if (activeConv?.title) return activeConv.title;
    const firstUser = chat.turns.find((t) => t.role === "user");
    if (firstUser && firstUser.role === "user" && firstUser.text.trim()) {
      const s = firstUser.text.trim();
      return s.length > 60 ? s.slice(0, 57) + "…" : s;
    }
    return "new conversation";
  }, [activeConv, chat.turns]);
  const hasTurns = chat.turns.length > 0;

  // True only for the first send of a resumed conversation; toggles off after
  // we attach seedHistory to that request.
  const pendingHistoryRef = useRef<string[] | null>(null);
  const [restoredWithAttachments, setRestoredWithAttachments] = useState(false);

  // Lifted to the page so the minimap (rendered in the right gutter, outside
  // the centered chat shell) can target the same scroll container as the
  // message list.
  const scrollRef = useRef<HTMLDivElement>(null);

  // ChatInput exposes a small imperative surface so the sidebar can pop the
  // gallery modal without lifting all of ChatInput's state up here.
  const chatInputRefs = useRef<ChatInputRefs>(null);

  // Queue of follow-up messages submitted while the agent was still
  // generating. They drain one-at-a-time when the agent becomes idle.
  type PendingSend = QueuedMessage & {
    attachments: StagedAsset[];
    model: string | null;
    topK: number;
    numChoices: number;
  };
  const [queue, setQueue] = useState<PendingSend[]>([]);

  // Whenever the chat turns or session change, persist to localStorage under
  // the active conversation id. If there's no active id and the user just
  // started typing, mint a new id.
  useEffect(() => {
    if (chat.turns.length === 0) return;
    let id = conv.activeId;
    if (!id) {
      id = conv.startNew();
    }
    conv.saveTurns(id, chat.turns, chat.sessionId);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [chat.turns, chat.sessionId]);

  const onNewChat = useCallback(() => {
    chat.reset();
    conv.setActiveId(null);
    pendingHistoryRef.current = null;
    setRestoredWithAttachments(false);
    setExamplesDismissed(false);
  }, [chat, conv]);

  const onPickConversation = useCallback(
    (c: StoredConversation) => {
      chat.reset(c.turns);
      conv.setActiveId(c.id);
      pendingHistoryRef.current = turnsToHistory(c.turns);
      setRestoredWithAttachments(conversationHadAttachments(c.turns));
      setExamplesDismissed(true);
    },
    [chat, conv]
  );

  /** Look up an asset in the current session by id-prefix or name-substring. */
  async function findSessionAsset(
    query: string
  ): Promise<{ asset_id: string; display_name?: string; preview_url?: string | null } | null> {
    if (!chat.sessionId) return null;
    try {
      const r = await fetch(`/api/files/sessions/${chat.sessionId}`, {
        credentials: "include",
      });
      if (!r.ok) return null;
      const list: any[] = await r.json();
      const q = query.toLowerCase();
      // exact id, then id prefix, then name contains
      let hit =
        list.find((a) => a.asset_id?.toLowerCase() === q) ||
        list.find((a) => a.asset_id?.toLowerCase().startsWith(q)) ||
        list.find((a) =>
          (a.display_name || "").toLowerCase().includes(q)
        );
      return hit || null;
    } catch {
      return null;
    }
  }

  async function runCommand(cmd: ParsedCommand, raw: string) {
    function emit(turn: Omit<EmbedTurn, "id" | "role"> & { role?: "embed" }) {
      chat.pushTurn({
        id: `e-${Date.now()}`,
        role: "embed",
        ...turn,
      } as EmbedTurn);
    }
    switch (cmd.kind) {
      case "help":
        emit({
          kind: "info",
          command: raw,
          label: HELP_TEXT,
        });
        return;
      case "image": {
        const q = cmd.query.trim();
        if (!q) {
          emit({
            kind: "info",
            command: raw,
            label: "usage: /img <id | name-substring | url>",
          });
          return;
        }
        // URL?
        if (/^https?:\/\//i.test(q) || q.startsWith("/")) {
          emit({ kind: "image", command: raw, src: q, label: q });
          return;
        }
        // Session asset
        const asset = await findSessionAsset(q);
        if (!asset || !asset.preview_url) {
          emit({
            kind: "info",
            command: raw,
            label: `no session asset matched "${q}"`,
          });
          return;
        }
        emit({
          kind: "image",
          command: raw,
          src: asset.preview_url,
          label: asset.display_name || q,
        });
        return;
      }
      case "audio":
        if (!cmd.url) {
          emit({ kind: "info", command: raw, label: "usage: /audio <url>" });
          return;
        }
        emit({ kind: "audio", command: raw, src: cmd.url, label: cmd.url });
        return;
      case "video":
        if (!cmd.url) {
          emit({ kind: "info", command: raw, label: "usage: /video <url>" });
          return;
        }
        emit({ kind: "video", command: raw, src: cmd.url, label: cmd.url });
        return;
      case "youtube":
        if (!cmd.videoId) {
          emit({
            kind: "info",
            command: raw,
            label: "usage: /youtube <id | watch-url>",
          });
          return;
        }
        emit({
          kind: "youtube",
          command: raw,
          src: cmd.videoId,
          label: `youtu.be/${cmd.videoId}`,
        });
        return;
      case "iframe":
        if (!cmd.url) {
          emit({ kind: "info", command: raw, label: "usage: /embed <url>" });
          return;
        }
        emit({ kind: "iframe", command: raw, src: cmd.url, label: cmd.url });
        return;
      case "unknown":
        emit({
          kind: "info",
          command: raw,
          label: `unknown command — try /help`,
        });
        return;
    }
  }

  function dispatchSend(p: PendingSend) {
    const seedHistory = pendingHistoryRef.current;
    pendingHistoryRef.current = null;
    if (p.attachments.length > 0) setRestoredWithAttachments(false);

    void chat.send({
      message: p.text,
      attachments: p.attachments.map((a) => ({
        asset_id: a.asset_id,
        display_name: a.display_name,
        preview_url: a.preview_url ?? null,
      })),
      model: p.model,
      topK: p.topK,
      numChoices: p.numChoices,
      seedHistory,
    });
  }

  function onSend(message: string, attachments: StagedAsset[]) {
    const pending: PendingSend = {
      id: `q-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
      text: message,
      attachmentCount: attachments.length,
      attachments,
      model,
      topK,
      numChoices,
    };
    // If the agent is mid-reply or there's already a queue waiting, append
    // and let the effect below drain it. Otherwise dispatch immediately.
    if (chat.busy || queue.length > 0) {
      setQueue((prev) => [...prev, pending]);
    } else {
      dispatchSend(pending);
    }
  }

  // Drain the queue one message at a time as the agent finishes each reply.
  useEffect(() => {
    if (chat.busy) return;
    if (queue.length === 0) return;
    const [next, ...rest] = queue;
    setQueue(rest);
    dispatchSend(next);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [chat.busy, queue]);

  function cancelQueued(id: string) {
    setQueue((prev) => prev.filter((q) => q.id !== id));
  }

  return (
    <div className="app-shell">
      <Sidebar
        onPick={onPickConversation}
        onNewChat={onNewChat}
        onOpenGallery={() => chatInputRefs.current?.openGallery()}
      />

      <div className="app-main">
        <div className="chat-shell">
          <header className="chat-header">
            <span
              className={"conv-dot" + (hasTurns ? " is-active" : "")}
              aria-hidden
            />
            <div className="title-block">
              <div className="title">{headerTitle}</div>
              <div className="sub">
                {hasTurns
                  ? `${chat.turns.filter((t) => t.role === "user").length} turn${
                      chat.turns.filter((t) => t.role === "user").length === 1
                        ? ""
                        : "s"
                    } · ai_plaza`
                  : "ai_plaza · drop an image or just ask"}
              </div>
            </div>
            <div className="spacer" />
            <ModelPicker
              value={model}
              onChange={setModel}
              topK={topK}
              onTopK={setTopK}
              numChoices={numChoices}
              onNumChoices={setNumChoices}
              sessionId={chat.sessionId}
            />
            <div className="header-sep" aria-hidden />
            <ThemeToggle />
            <button className="btn-logout" onClick={() => void logout()}>
              [ sign_out ]
            </button>
          </header>

          <QueueBanner items={queue} onCancel={cancelQueued} />

          <MessageList
            turns={chat.turns}
            busy={chat.busy}
            restoredWithAttachments={restoredWithAttachments}
            sessionId={chat.sessionId}
            onSessionId={chat.setSessionId}
            scrollRef={scrollRef}
            showExamples={!examplesDismissed}
            onApprove={() => void chat.approve()}
            onDecline={() => void chat.decline()}
            onConfirmDemo={() => void chat.confirmDemo()}
            onExamplePick={(text, attachment) => {
              setExamplesDismissed(true);
              chatInputRefs.current?.prefill(
                text,
                attachment ? [attachment] : []
              );
            }}
            onClarificationPick={(option) => onSend(option, [])}
          />

          <ChatInput
            sessionId={chat.sessionId}
            onSessionId={chat.setSessionId}
            onSend={onSend}
            busy={chat.busy}
            externalRefs={chatInputRefs}
            onOpenConversation={onPickConversation}
            onSlashCommand={(cmd, raw) => void runCommand(cmd, raw)}
          />
        </div>

        <Minimap turns={chat.turns} scrollRef={scrollRef} />
      </div>
    </div>
  );
}
