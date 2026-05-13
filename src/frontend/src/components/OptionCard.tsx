import { useState } from "react";

type Props = {
  option: string;
  busy: boolean;
  onSend: (text: string) => void;
};

/** Heuristic: the agent commonly emits "Other", "Other (briefly specify)" or
 * "Other — please describe" as the last clarification option. We treat any
 * option whose first word is "other" as free-form. */
function isOtherOption(option: string): boolean {
  return /^\s*other\b/i.test(option);
}

export default function OptionCard({ option, busy, onSend }: Props) {
  const free = isOtherOption(option);
  const [expanded, setExpanded] = useState(false);
  const [text, setText] = useState("");

  function submit() {
    const trimmed = text.trim();
    if (!trimmed) return;
    onSend(trimmed);
    setText("");
    setExpanded(false);
  }

  if (!free) {
    return (
      <button
        type="button"
        className="rec-card option-card"
        onClick={() => onSend(option)}
        disabled={busy}
      >
        <span className="option-arrow">↳</span>
        <span>{option}</span>
      </button>
    );
  }

  if (!expanded) {
    return (
      <button
        type="button"
        className="rec-card option-card"
        onClick={() => setExpanded(true)}
        disabled={busy}
      >
        <span className="option-arrow">↳</span>
        <span>{option}</span>
        <span className="option-meta mono">type to specify</span>
      </button>
    );
  }

  return (
    <div className="rec-card option-card-expanded">
      <span className="option-arrow">↳</span>
      <input
        autoFocus
        type="text"
        value={text}
        placeholder="describe your option…"
        onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && text.trim()) {
            e.preventDefault();
            submit();
          } else if (e.key === "Escape") {
            setExpanded(false);
            setText("");
          }
        }}
        disabled={busy}
      />
      <button
        type="button"
        className="btn-approve"
        disabled={busy || !text.trim()}
        onClick={submit}
      >
        send
      </button>
      <button
        type="button"
        className="btn-decline"
        onClick={() => {
          setExpanded(false);
          setText("");
        }}
        disabled={busy}
      >
        cancel
      </button>
    </div>
  );
}
