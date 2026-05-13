import type { PendingAction } from "../hooks/useChat";

type Props = {
  pending: PendingAction;
  onApprove: () => void;
  onDecline: () => void;
  onConfirmDemo: () => void;
  busy: boolean;
};

export default function PendingActionPanel({
  pending,
  onApprove,
  onDecline,
  onConfirmDemo,
  busy,
}: Props) {
  if (pending.type === "tool_approval") {
    return (
      <div className="pending-panel">
        <div className="prompt">
          ⌁ {pending.prompt}
        </div>
        {pending.image_name && (
          <div className="detail">image: {pending.image_name}</div>
        )}
        {pending.demo_url && (
          <div className="detail">
            endpoint: <a href={pending.demo_url}>{pending.demo_url}</a>
          </div>
        )}
        <div className="actions">
          <button className="btn-approve" onClick={onApprove} disabled={busy}>
            ↳ run {pending.display_name || pending.tool_name}
          </button>
          <button className="btn-decline" onClick={onDecline} disabled={busy}>
            cancel
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="pending-panel">
      <div className="prompt">⌁ {pending.prompt}</div>
      {pending.demo_url && (
        <div className="detail">
          demo: <a href={pending.demo_url}>{pending.demo_url}</a>
        </div>
      )}
      <div className="actions">
        <button className="btn-approve" onClick={onConfirmDemo} disabled={busy}>
          ↳ run demo
        </button>
        <button className="btn-decline" onClick={onDecline} disabled={busy}>
          cancel
        </button>
      </div>
    </div>
  );
}
