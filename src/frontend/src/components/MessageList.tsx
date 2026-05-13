import { useEffect, useRef, useState } from "react";
import type { Turn } from "../hooks/useChat";
import AssetModal from "./AssetModal";
import CryingCat from "./CryingCat";
import EmbedCard from "./EmbedCard";
import ExamplePrompts from "./ExamplePrompts";
import Lightbox from "./Lightbox";
import OptionCard from "./OptionCard";
import PendingActionPanel from "./PendingActionPanel";
import RecommendationCard from "./RecommendationCard";

type Props = {
  turns: Turn[];
  busy: boolean;
  restoredWithAttachments?: boolean;
  sessionId: string | null;
  onSessionId: (id: string) => void;
  scrollRef: React.RefObject<HTMLDivElement>;
  /** Whether to render the example-prompts row on the empty state. Hidden
   *  after the user picks an example or resumes an old chat. */
  showExamples?: boolean;
  onApprove: () => void;
  onDecline: () => void;
  onConfirmDemo: () => void;
  onExamplePick?: (
    text: string,
    attachment: import("../lib/api").Asset | null
  ) => void;
  onClarificationPick?: (option: string) => void;
};

export default function MessageList({
  turns,
  busy,
  restoredWithAttachments = false,
  sessionId,
  onSessionId,
  scrollRef,
  showExamples = true,
  onApprove,
  onDecline,
  onConfirmDemo,
  onExamplePick,
  onClarificationPick,
}: Props) {
  const bottomRef = useRef<HTMLDivElement>(null);
  const [lightboxSrc, setLightboxSrc] = useState<string | null>(null);
  const [inspectAsset, setInspectAsset] = useState<{
    id: string;
    name?: string;
    previewUrl?: string | null;
  } | null>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [turns]);

  return (
    <div className="chat-body">
      <Lightbox
        src={lightboxSrc}
        alt="attached preview"
        onClose={() => setLightboxSrc(null)}
      />
      <AssetModal
        assetId={inspectAsset?.id ?? null}
        fallbackPreviewUrl={inspectAsset?.previewUrl ?? null}
        fallbackName={inspectAsset?.name ?? null}
        onClose={() => setInspectAsset(null)}
      />
      <div className="chat-messages" ref={scrollRef}>
        <div className="chat-messages-inner">
        {turns.length === 0 && (
          <div className="empty-state">
            <div className="glyph">▣</div>
            <div className="hint">ask anything · attach an image if you have one</div>
            <div className="body">
              text-only is fine — uploads sharpen the recommendation.
            </div>
            {onExamplePick && showExamples && (
              <ExamplePrompts
                sessionId={sessionId}
                onSessionId={onSessionId}
                onPick={onExamplePick}
              />
            )}
          </div>
        )}

        {restoredWithAttachments && turns.length > 0 && <CryingCat />}

        {turns.map((t, idx) => {
          if (t.role === "embed") {
            return (
              <div key={t.id} className="msg embed" data-turn-idx={idx}>
                <EmbedCard turn={t} onOpenImage={setLightboxSrc} />
              </div>
            );
          }
          if (t.role === "user") {
            return (
              <div key={t.id} className="msg user" data-turn-idx={idx}>
                <div className="avatar">YOU</div>
                <div className="bubble">
                  <div className="content">{t.text}</div>
                  {t.attachments.length > 0 && (
                    <div className="attachments large">
                      {t.attachments.map((a) => (
                        <button
                          key={a.asset_id}
                          type="button"
                          className="attachment-thumb"
                          onClick={() =>
                            setInspectAsset({
                              id: a.asset_id,
                              name: a.display_name,
                              previewUrl: a.preview_url,
                            })
                          }
                          title={a.display_name || "preview"}
                        >
                          {a.preview_url ? (
                            <img
                              src={a.preview_url}
                              alt={a.display_name || "asset"}
                            />
                          ) : (
                            <div
                              style={{
                                width: "100%",
                                height: "100%",
                                display: "grid",
                                placeItems: "center",
                                fontFamily: "var(--font-mono)",
                                fontSize: 10,
                                color: "var(--text-soft)",
                              }}
                            >
                              file
                            </div>
                          )}
                          <div className="label">
                            {(a.display_name || "").slice(-18)}
                          </div>
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            );
          }

          return (
            <div key={t.id} className="msg assistant" data-turn-idx={idx}>
              <div className="avatar">IP</div>
              <div className="bubble">
                {t.status === "streaming" && !t.text && t.statusMessage && (
                  <div className="status-line">
                    <span className="status-pulse" aria-hidden />
                    <span className="mono">{t.statusMessage}</span>
                  </div>
                )}
                {(t.text || t.status !== "streaming") && (
                  <div className="content md">
                    {renderMarkdownLite(t.text)}
                    {t.status === "streaming" && (
                      <span className="streaming-dot" aria-hidden />
                    )}
                  </div>
                )}

                {t.imageUrls.length > 0 && (
                  <div className="attachments large" style={{ marginTop: 12 }}>
                    {t.imageUrls.map((u) => (
                      <button
                        key={u}
                        type="button"
                        className="attachment-thumb"
                        onClick={() => setLightboxSrc(u)}
                        title="result"
                      >
                        <img src={u} alt="result" />
                      </button>
                    ))}
                  </div>
                )}

                {t.recommendations.length > 0 && (
                  <>
                    <div className="rec-divider">recommendations</div>
                    <div className="recs">
                      {t.recommendations.map((r) => (
                        <RecommendationCard key={r.rank} rec={r} />
                      ))}
                    </div>
                  </>
                )}

                {t.clarification && t.clarification.options.length > 0 && (
                  <div className="recs" style={{ marginTop: 12 }}>
                    <div className="rec-divider">options · click to reply</div>
                    {t.clarification.options.map((o) => (
                      <OptionCard
                        key={o}
                        option={o}
                        busy={busy}
                        onSend={(text) =>
                          onClarificationPick && onClarificationPick(text)
                        }
                      />
                    ))}
                  </div>
                )}

                {t.pending && !busy && (
                  <PendingActionPanel
                    pending={t.pending}
                    onApprove={onApprove}
                    onDecline={onDecline}
                    onConfirmDemo={onConfirmDemo}
                    busy={busy}
                  />
                )}

                {t.toolTraces.length > 0 && (
                  <div className="traces">
                    <details>
                      <summary>
                        tool_calls ({t.toolTraces.length})
                        {t.usage && (
                          <span style={{ marginLeft: 8, color: "var(--text-faint)" }}>
                            · {t.usage.total} tok
                          </span>
                        )}
                      </summary>
                      <ul>
                        {t.toolTraces.map((tc, i) => (
                          <li key={i}>
                            <code>{(tc as any).tool}</code>{" "}
                            {(tc as any).duration_ms !== undefined && (
                              <span>· {(tc as any).duration_ms} ms</span>
                            )}
                            {(tc as any).error && (
                              <span style={{ color: "var(--danger)" }}>
                                {" "}· {(tc as any).error}
                              </span>
                            )}
                          </li>
                        ))}
                      </ul>
                    </details>
                  </div>
                )}

                {t.status === "error" && (
                  <div className="error-banner" style={{ marginTop: 10 }}>
                    {t.error || "Something went wrong."}
                  </div>
                )}
              </div>
            </div>
          );
        })}
          <div ref={bottomRef} />
        </div>
      </div>
    </div>
  );
}

/** Tiny markdown-ish renderer: bold, italic, inline code, hrs, line breaks. */
function renderMarkdownLite(text: string) {
  if (!text) return null;
  const escaped = text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
  const html = escaped
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*([^*\n]+)\*(?!\*)/g, "$1<em>$2</em>")
    .replace(/`([^`\n]+)`/g, "<code>$1</code>")
    .replace(/^---$/gm, "<hr/>")
    .replace(/\n/g, "<br/>");
  // eslint-disable-next-line react/no-danger
  return <span dangerouslySetInnerHTML={{ __html: html }} />;
}
