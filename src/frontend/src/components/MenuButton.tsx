/** Small chip-style dropdown button used for the per-conversation controls
 * (model / top_k / num_choices) sitting in the header.
 *
 * Generic on the value type — the parent passes an array of `{value, label,
 * hint?}` and a setter. Only one menu opens at a time per instance; clicking
 * outside or pressing Escape closes it.
 */

import { useEffect, useRef, useState, type ReactNode } from "react";

export type MenuOption<T> = {
  value: T;
  label: string;
  hint?: string;
};

type Props<T> = {
  label: string;
  value: T;
  options: MenuOption<T>[];
  onChange: (v: T) => void;
  /** Optional renderer for the value display (defaults to the matching option's label). */
  formatValue?: (v: T) => ReactNode;
  /** Where to anchor the menu — defaults to "right" of the chip. */
  align?: "left" | "right";
};

export default function MenuButton<T>({
  label,
  value,
  options,
  onChange,
  formatValue,
  align = "right",
}: Props<T>) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    function onDocClick(e: MouseEvent) {
      if (!rootRef.current) return;
      if (!rootRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    }
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") setOpen(false);
    }
    document.addEventListener("mousedown", onDocClick);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDocClick);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  const current = options.find((o) => o.value === value);
  const display = formatValue
    ? formatValue(value)
    : current?.label ?? String(value ?? "");

  return (
    <div className="menu-button" ref={rootRef}>
      <button
        type="button"
        className={"chip" + (open ? " open" : "")}
        onClick={() => setOpen((v) => !v)}
        title={label}
      >
        <span className="chip-label">{label}</span>
        <span className="chip-value">{display}</span>
        <span className="chip-caret" aria-hidden>
          ▾
        </span>
      </button>

      {open && (
        <div
          className={"menu " + (align === "left" ? "menu-left" : "menu-right")}
          role="listbox"
        >
          {options.map((o) => {
            const active = o.value === value;
            return (
              <button
                key={String(o.value)}
                type="button"
                className={"menu-item" + (active ? " active" : "")}
                onClick={() => {
                  onChange(o.value);
                  setOpen(false);
                }}
                role="option"
                aria-selected={active}
              >
                <span className="menu-check" aria-hidden>
                  {active ? "●" : "○"}
                </span>
                <span className="menu-label">
                  {o.label}
                  {o.hint && <span className="menu-hint mono"> · {o.hint}</span>}
                </span>
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}
