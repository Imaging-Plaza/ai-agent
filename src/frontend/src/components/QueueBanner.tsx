/** Banner shown above the message list whenever the user has queued
 *  follow-up messages while the agent was still replying. Each ticket can
 *  be cancelled before it gets sent.
 */

export type QueuedMessage = {
  id: string;
  text: string;
  attachmentCount: number;
};

type Props = {
  items: QueuedMessage[];
  onCancel: (id: string) => void;
};

export default function QueueBanner({ items, onCancel }: Props) {
  // We always render a slot — even empty — so the parent grid keeps its row
  // count stable. Otherwise the composer would slide into the `1fr` row and
  // float in the middle of the page.
  if (items.length === 0) {
    return <div className="queue-slot empty" aria-hidden />;
  }
  return (
    <div className="queue-banner queue-slot" role="status" aria-live="polite">
      <div className="queue-banner-head mono">
        queued · {items.length} message{items.length === 1 ? "" : "s"}
      </div>
      <ul className="queue-list">
        {items.map((q, i) => (
          <li key={q.id} className="queue-item">
            <span className="queue-pos mono">#{i + 1}</span>
            <span className="queue-text" title={q.text}>
              {q.text || (q.attachmentCount > 0
                ? `(${q.attachmentCount} attachment${q.attachmentCount === 1 ? "" : "s"})`
                : "(empty)")}
            </span>
            {q.attachmentCount > 0 && (
              <span className="queue-attach mono">📎 {q.attachmentCount}</span>
            )}
            <button
              type="button"
              className="queue-cancel"
              title="Remove from queue"
              aria-label="Remove from queue"
              onClick={() => onCancel(q.id)}
            >
              ×
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}
