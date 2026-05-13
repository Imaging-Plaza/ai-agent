import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import {
  useConversations,
  type StoredConversation,
} from "../hooks/useConversations";
import { exportLocal, importLocal } from "../lib/backup";
import { formatRelativeDate } from "../lib/dates";
import { SidebarPartners } from "./Logos";
import Marquee from "./Marquee";

type Props = {
  onPick: (conv: StoredConversation) => void;
  onNewChat: () => void;
  onOpenGallery: () => void;
};

export default function Sidebar({ onPick, onNewChat, onOpenGallery }: Props) {
  const { conversations, activeId, deleteConversation, clearAll } =
    useConversations();
  // Floating-card state. Tracks the row the user is hovering plus its
  // bounding rect so the popover (rendered through a portal at body level)
  // can fix itself on top of the row and extend to the right beyond the
  // sidebar's overflow context.
  const [hovered, setHovered] = useState<{
    conv: StoredConversation;
    rect: DOMRect;
  } | null>(null);
  const importInputRef = useRef<HTMLInputElement>(null);

  async function onImportFile(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    e.target.value = "";
    try {
      const result = await importLocal(file);
      const msg = result.exported_at
        ? `Restored ${result.restored_keys.length} key(s) from backup taken ${new Date(result.exported_at).toLocaleString()}.\nThe page will reload now.`
        : `Restored ${result.restored_keys.length} key(s).\nThe page will reload now.`;
      alert(msg);
      window.location.reload();
    } catch (err: any) {
      alert(`Import failed: ${err?.message || err}`);
    }
  }

  function pop(conv: StoredConversation, el: HTMLLIElement | null) {
    if (!el) return;
    setHovered({ conv, rect: el.getBoundingClientRect() });
  }

  return (
    <aside className="sidebar">
      <div className="sidebar-brand">
        <div className="sidebar-brand-logo">AI</div>
        <div className="sidebar-brand-text">
          <div className="sidebar-brand-title">ai_plaza</div>
          <div className="sidebar-brand-sub">imaging · v1</div>
        </div>
      </div>

      <div className="sidebar-head">
        <button className="btn-new-chat" onClick={onNewChat}>
          <span className="btn-new-chat-icon" aria-hidden>
            +
          </span>
          <span>new chat</span>
        </button>
        <button className="btn-sidebar-link" onClick={onOpenGallery}>
          <span className="btn-sidebar-link-icon" aria-hidden>
            ▦
          </span>
          <span>gallery</span>
        </button>
      </div>

      <div className="sidebar-section">
        <div className="sidebar-label">recent</div>
        {conversations.length === 0 && (
          <div className="sidebar-empty">no conversations yet</div>
        )}
        <ul className="sidebar-list">
          {conversations.map((c) => (
            <li
              key={c.id}
              className={
                "sidebar-item" + (c.id === activeId ? " active" : "")
              }
              onMouseEnter={(e) => pop(c, e.currentTarget)}
              onClick={() => onPick(c)}
            >
              <SidebarItemRow conv={c} />
            </li>
          ))}
        </ul>
      </div>

      <div className="sidebar-foot">
        <div className="sidebar-actions">
          <button
            className="sidebar-action"
            onClick={exportLocal}
            title="Download a .zip of your local conversations + theme"
          >
            <span className="sidebar-action-icon" aria-hidden>↓</span>
            <span>export</span>
          </button>
          <button
            className="sidebar-action"
            onClick={() => importInputRef.current?.click()}
            title="Restore from a previously exported .zip"
          >
            <span className="sidebar-action-icon" aria-hidden>↑</span>
            <span>import</span>
          </button>
          {conversations.length > 0 && (
            <button
              className="sidebar-action sidebar-action-danger"
              onClick={() => {
                if (confirm("Delete all stored conversations?")) clearAll();
              }}
              title="Wipe every stored conversation from this browser"
            >
              <span className="sidebar-action-icon" aria-hidden>×</span>
              <span>clear</span>
            </button>
          )}
          <input
            ref={importInputRef}
            type="file"
            accept=".zip,application/zip"
            style={{ display: "none" }}
            onChange={onImportFile}
          />
        </div>
        <SidebarPartners />
      </div>

      {hovered && (
        <ItemPopover
          conv={hovered.conv}
          rect={hovered.rect}
          isActive={hovered.conv.id === activeId}
          onPick={(c) => {
            setHovered(null);
            onPick(c);
          }}
          onDelete={(id) => {
            deleteConversation(id);
            setHovered(null);
          }}
          onLeave={() => setHovered(null)}
        />
      )}
    </aside>
  );
}

/** Inner content for both the compact list row and the floating popover.
 *
 *  Compact mode renders just the title (one tight line) so the sidebar can
 *  hold many conversations without each taking a lot of vertical real estate.
 *  Popover mode adds the marquee'd title + date + delete affordance. */
function SidebarItemRow({
  conv,
  fullTitle = false,
}: {
  conv: StoredConversation;
  fullTitle?: boolean;
}) {
  if (!fullTitle) {
    return (
      <div className="sidebar-title sidebar-title-compact">{conv.title}</div>
    );
  }
  return (
    <>
      <Marquee
        text={conv.title}
        className="sidebar-title"
        tooltip={false}
        alwaysOn
      />
      <div className="sidebar-meta">
        <span>{formatRelativeDate(conv.updatedAt)}</span>
      </div>
    </>
  );
}

type PopoverProps = {
  conv: StoredConversation;
  rect: DOMRect;
  isActive: boolean;
  onPick: (c: StoredConversation) => void;
  onDelete: (id: string) => void;
  onLeave: () => void;
};

function ItemPopover({
  conv,
  rect,
  isActive,
  onPick,
  onDelete,
  onLeave,
}: PopoverProps) {
  const ref = useRef<HTMLDivElement>(null);

  // Close the popover if the user scrolls (the captured rect would go stale)
  // or resizes the window.
  useEffect(() => {
    function close() {
      onLeave();
    }
    window.addEventListener("scroll", close, true);
    window.addEventListener("resize", close);
    return () => {
      window.removeEventListener("scroll", close, true);
      window.removeEventListener("resize", close);
    };
  }, [onLeave]);

  return createPortal(
    <div
      ref={ref}
      className={
        "sidebar-item sidebar-item-pop" + (isActive ? " active" : "")
      }
      style={{
        position: "fixed",
        top: rect.top,
        left: rect.left,
        width: 360,
        zIndex: 100,
      }}
      onMouseLeave={onLeave}
      onClick={() => onPick(conv)}
    >
      <SidebarItemRow conv={conv} fullTitle />
      <button
        className="sidebar-del"
        aria-label="Delete conversation"
        onClick={(e) => {
          e.stopPropagation();
          onDelete(conv.id);
        }}
      >
        <svg
          width="13"
          height="13"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          aria-hidden
        >
          <polyline points="3 6 5 6 21 6"></polyline>
          <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"></path>
          <path d="M10 11v6"></path>
          <path d="M14 11v6"></path>
          <path d="M9 6V4a2 2 0 0 1 2-2h2a2 2 0 0 1 2 2v2"></path>
        </svg>
      </button>
    </div>,
    document.body
  );
}
