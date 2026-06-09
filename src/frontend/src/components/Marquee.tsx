/** Hover-to-reveal text label.
 *
 *  When the user hovers, the parent container (the sidebar) widens and most
 *  titles will then fit fully. For titles that *still* overflow even at the
 *  expanded width, we run an infinite seamless marquee — two adjacent
 *  copies of the text translated continuously by one copy's width, with a
 *  small `·` separator so the loop is readable.
 *
 *  The overflow measurement happens *after* the sidebar's width transition
 *  completes (a small delay), so we don't trigger the animation on titles
 *  that comfortably fit the expanded layout.
 */

import { useEffect, useLayoutEffect, useRef, useState } from "react";

type Props = {
  text: string;
  className?: string;
  /** Px-per-second scroll speed during the loop. */
  speedPxPerSec?: number;
  /** ms to wait before measuring (must be ≥ the sidebar transition). */
  measureDelayMs?: number;
  /** When false, no native `title` attribute is attached (used inside the
   *  popover where the full text is already visible). */
  tooltip?: boolean;
  /** When true, the marquee behaves as if it were always being hovered —
   *  the parent component (e.g. a sidebar popover) is the real hover target,
   *  so the user shouldn't have to land the cursor on the text itself. */
  alwaysOn?: boolean;
};

export default function Marquee({
  text,
  className = "",
  speedPxPerSec = 45,
  measureDelayMs = 320,
  tooltip = true,
  alwaysOn = false,
}: Props) {
  const wrapRef = useRef<HTMLSpanElement>(null);
  const copyRef = useRef<HTMLSpanElement>(null);
  const [hovering, setHovering] = useState(alwaysOn);
  const [scrollState, setScrollState] = useState<{
    duration: number;
  } | null>(null);

  // Stay in sync if the parent flips the always-on flag mid-life.
  useEffect(() => {
    if (alwaysOn) setHovering(true);
  }, [alwaysOn]);

  useLayoutEffect(() => {
    if (!hovering) {
      setScrollState(null);
      return;
    }
    // Wait for the sidebar expansion to settle before deciding whether the
    // text still overflows.
    const t = window.setTimeout(() => {
      const w = wrapRef.current;
      const c = copyRef.current;
      if (!w || !c) return;
      const overflow = c.scrollWidth - w.clientWidth;
      if (overflow <= 4) {
        // Fits in the expanded layout — no animation.
        setScrollState(null);
        return;
      }
      // Loop one full copy width (c.scrollWidth) per cycle.
      setScrollState({
        duration: Math.max(3, c.scrollWidth / speedPxPerSec),
      });
    }, measureDelayMs);
    return () => clearTimeout(t);
  }, [hovering, text, speedPxPerSec, measureDelayMs]);

  const separator = "  ·  ";

  return (
    <span
      ref={wrapRef}
      className={
        "marquee " +
        className +
        (scrollState ? " is-scrolling" : "") +
        (hovering ? " is-hovering" : "")
      }
      title={tooltip ? text : undefined}
      onMouseEnter={alwaysOn ? undefined : () => setHovering(true)}
      onMouseLeave={alwaysOn ? undefined : () => setHovering(false)}
    >
      {scrollState ? (
        <span
          className="marquee-track"
          style={{ animationDuration: `${scrollState.duration}s` }}
        >
          <span ref={copyRef} className="marquee-copy">
            {text}
            {separator}
          </span>
          {/* Duplicate so the loop seam is invisible. aria-hidden because we
           * don't want screen readers to see "title  ·  title". */}
          <span className="marquee-copy" aria-hidden>
            {text}
            {separator}
          </span>
        </span>
      ) : (
        <span ref={copyRef} className="marquee-copy single">
          {text}
        </span>
      )}
    </span>
  );
}
