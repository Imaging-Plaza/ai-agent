import { useEffect, useRef } from "react";
import type { Turn } from "../hooks/useChat";
import RecommendationCard from "./RecommendationCard";
import PendingActionPanel from "./PendingActionPanel";

type Props = {
  turns: Turn[];
  busy: boolean;
  onApprove: () => void;
  onDecline: () => void;
  onConfirmDemo: () => void;
};

export default function MessageList({
  turns,
  busy,
  onApprove,
  onDecline,
  onConfirmDemo,
}: Props) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [turns]);

  return (
    <div className="chat-messages">
      <div className="chat-messages-inner">
        {turns.length === 0 && (
          <div
            style={{
              textAlign: "center",
              color: "#6b727a",
              padding: "40px 20px",
            }}
          >
            <div style={{ fontSize: 36, marginBottom: 8 }}>🩻</div>
            <p style={{ margin: 0 }}>
              Upload an image and describe what you'd like to do —{" "}
              <em>e.g. "segment the lungs in this CT"</em>.
            </p>
          </div>
        )}

        {turns.map((t) => {
          if (t.role === "user") {
            return (
              <div key={t.id} className="msg user">
                <div className="avatar">You</div>
                <div className="bubble">
                  <div className="content">{t.text}</div>
                  {t.attachments.length > 0 && (
                    <div className="attachments">
                      {t.attachments.map((a) => (
                        <div key={a.asset_id} className="attachment-thumb">
                          {a.preview_url ? (
                            <img src={a.preview_url} alt={a.display_name || "asset"} />
                          ) : (
                            <div
                              style={{
                                width: "100%",
                                height: "100%",
                                display: "grid",
                                placeItems: "center",
                                fontSize: 11,
                                color: "#6b727a",
                              }}
                            >
                              file
                            </div>
                          )}
                          <div className="label">{a.display_name?.slice(-12)}</div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            );
          }

          // Assistant turn
          return (
            <div key={t.id} className="msg assistant">
              <div className="avatar">IP</div>
              <div className="bubble">
                <div className="content md">
                  {renderMarkdownLite(t.text)}
                  {t.status === "streaming" && (
                    <span className="streaming-dot" aria-hidden>
                      &nbsp;▍
                    </span>
                  )}
                </div>

                {t.imageUrls.length > 0 && (
                  <div className="attachments" style={{ marginTop: 12 }}>
                    {t.imageUrls.map((u) => (
                      <div key={u} className="attachment-thumb">
                        <img src={u} alt="result" />
                      </div>
                    ))}
                  </div>
                )}

                {t.recommendations.length > 0 && (
                  <div className="recs">
                    {t.recommendations.map((r) => (
                      <RecommendationCard key={r.rank} rec={r} />
                    ))}
                  </div>
                )}

                {t.clarification && t.clarification.options.length > 0 && (
                  <div className="recs">
                    <em style={{ display: "block", marginBottom: 6 }}>
                      Pick the option that fits:
                    </em>
                    {t.clarification.options.map((o) => (
                      <div key={o} className="rec-card">
                        {o}
                      </div>
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
                        🔧 Tool calls ({t.toolTraces.length})
                        {t.usage && (
                          <span style={{ marginLeft: 8 }}>
                            · {t.usage.total} tokens
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
                              <span style={{ color: "#a01a1a" }}>
                                {" "}
                                · {(tc as any).error}
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
  );
}

/** Tiny markdown-ish renderer: paragraphs, bold, italics, code, links.
 *
 * Intentionally minimalist — the backend usually returns short, structured
 * text. For full markdown we'd plug in `marked` or `react-markdown`.
 */
function renderMarkdownLite(text: string) {
  if (!text) return null;
  // Replace **bold** and *italic* with HTML; escape carefully.
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
