/** Conversation minimap — minimalist rail in the right gutter.
 *
 * Lives outside the centered chat shell so it never crowds the conversation.
 * Default presentation is a single-pixel-thin rail with one tiny dash per
 * turn (gray for user, accent for assistant). A translucent green rectangle
 * tracks the scroll viewport in real time, and dragging anywhere on the rail
 * scrubs the chat proportionally.
 *
 * On hover the rail expands and reveals each turn's topic snippet.
 */

import { useEffect, useRef, useState } from "react";
import type { Turn } from "../hooks/useChat";

type Props = {
  turns: Turn[];
  scrollRef: React.RefObject<HTMLDivElement>;
};

type Topic = {
  role: "user" | "assistant";
  topic: string;
  detail: string;
};

function turnsToTopics(turns: Turn[]): Topic[] {
  return turns.map((t) => {
    if (t.role === "user") {
      const text = t.text.trim();
      return {
        role: "user",
        topic: text ? truncate(text, 60) : `[${t.attachments.length} file(s)]`,
        detail: text || `attached ${t.attachments.length} file(s)`,
      };
    }
    if (t.role === "embed") {
      return {
        role: "assistant",
        topic: `▷ ${truncate(t.command, 50)}`,
        detail: t.label || t.command,
      };
    }
    if (t.recommendations.length > 0) {
      const names = t.recommendations
        .map((r) => r.name)
        .slice(0, 3)
        .join(", ");
      return {
        role: "assistant",
        topic: `▷ ${truncate(t.recommendations[0].name, 50)}`,
        detail: `${t.recommendations.length} tool(s): ${names}`,
      };
    }
    if (t.clarification) {
      return {
        role: "assistant",
        topic: `▷ ${truncate(t.clarification.question, 50)}`,
        detail: t.clarification.question,
      };
    }
    if (t.pending) {
      return {
        role: "assistant",
        topic: `▷ ${truncate(t.pending.prompt, 50)}`,
        detail: t.pending.prompt,
      };
    }
    const txt = t.text.trim();
    return {
      role: "assistant",
      topic: txt ? `▷ ${truncate(txt, 50)}` : "▷ thinking…",
      detail: txt || "agent is thinking…",
    };
  });
}

function truncate(s: string, n: number): string {
  return s.length > n ? s.slice(0, n - 1) + "…" : s;
}

export default function Minimap({ turns, scrollRef }: Props) {
  const topics = turnsToTopics(turns);
  const railRef = useRef<HTMLDivElement>(null);
  const [activeIdx, setActiveIdx] = useState(0);
  const [scrollPct, setScrollPct] = useState(0);
  const [viewportPct, setViewportPct] = useState(1);
  const draggingRef = useRef(false);

  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    function onScroll() {
      if (!el) return;
      const denom = Math.max(1, el.scrollHeight - el.clientHeight);
      setScrollPct(Math.min(1, Math.max(0, el.scrollTop / denom)));
      setViewportPct(Math.min(1, el.clientHeight / el.scrollHeight));

      const turnEls = el.querySelectorAll<HTMLElement>("[data-turn-idx]");
      let best = 0;
      let bestDist = Infinity;
      const offset = el.getBoundingClientRect().top + 40;
      turnEls.forEach((t) => {
        const idx = parseInt(t.dataset.turnIdx || "0", 10);
        const top = t.getBoundingClientRect().top - offset;
        const d = Math.abs(top);
        if (d < bestDist) {
          bestDist = d;
          best = idx;
        }
      });
      setActiveIdx(best);
    }
    el.addEventListener("scroll", onScroll, { passive: true });
    onScroll();
    return () => el.removeEventListener("scroll", onScroll);
  }, [scrollRef, turns.length]);

  function scrollToTurn(idx: number) {
    const el = scrollRef.current;
    if (!el) return;
    const turnEl = el.querySelector<HTMLElement>(
      `[data-turn-idx="${idx}"]`
    );
    if (!turnEl) return;
    const top =
      turnEl.getBoundingClientRect().top -
      el.getBoundingClientRect().top +
      el.scrollTop -
      8;
    el.scrollTo({ top, behavior: "smooth" });
  }

  function scrubTo(clientY: number) {
    const rail = railRef.current;
    const sc = scrollRef.current;
    if (!rail || !sc) return;
    const rect = rail.getBoundingClientRect();
    const pct = Math.min(1, Math.max(0, (clientY - rect.top) / rect.height));
    const target = pct * (sc.scrollHeight - sc.clientHeight);
    sc.scrollTo({ top: target });
  }

  function onMouseDown(e: React.MouseEvent) {
    if ((e.target as HTMLElement).closest(".rail-item")) return;
    draggingRef.current = true;
    scrubTo(e.clientY);
    const onMove = (ev: MouseEvent) => {
      if (!draggingRef.current) return;
      scrubTo(ev.clientY);
    };
    const onUp = () => {
      draggingRef.current = false;
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
    };
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
  }

  if (topics.length === 0) {
    return null;
  }

  return (
    <div
      className="rail"
      ref={railRef}
      onMouseDown={onMouseDown}
      role="navigation"
      aria-label="Conversation minimap"
    >
      <div className="rail-items">
        {topics.map((tp, i) => (
          <div
            key={i}
            className={
              "rail-item " +
              (tp.role === "user" ? "is-user " : "is-assistant ") +
              (i === activeIdx ? "active" : "")
            }
            onClick={() => scrollToTurn(i)}
            title={tp.detail}
          >
            <span className="rail-dash" aria-hidden />
            <span className="rail-label mono">{tp.topic}</span>
          </div>
        ))}
      </div>
      <div
        className="rail-viewport"
        style={{
          top: `${scrollPct * (1 - viewportPct) * 100}%`,
          height: `${Math.max(6, viewportPct * 100)}%`,
        }}
        aria-hidden
      />
    </div>
  );
}
