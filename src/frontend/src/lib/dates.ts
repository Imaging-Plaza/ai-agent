/** Friendly relative-date formatting shared by the sidebar and the gallery.
 *
 *  Behaviour:
 *    < 1 minute   → "just now"
 *    < 1 hour     → "Nm ago"
 *    < 24 hours   → "Nh ago"
 *    < 5 days     → "Nd ago"
 *    same year    → "May 8" (locale-aware)
 *    older        → "May 8, 2024"
 */

const MONTHS = [
  "Jan", "Feb", "Mar", "Apr", "May", "Jun",
  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];

export function formatRelativeDate(ts: number | null | undefined): string {
  if (!ts || !Number.isFinite(ts)) return "—";
  // Accept either seconds or milliseconds — the backend ships seconds, the
  // frontend's localStorage ships ms.
  let ms = ts > 1e12 ? ts : ts * 1000;
  const now = Date.now();
  const secs = Math.max(0, (now - ms) / 1000);

  if (secs < 60) return "just now";
  if (secs < 3600) return `${Math.floor(secs / 60)}m ago`;
  if (secs < 86400) return `${Math.floor(secs / 3600)}h ago`;
  if (secs < 5 * 86400) return `${Math.floor(secs / 86400)}d ago`;

  const d = new Date(ms);
  const sameYear = d.getFullYear() === new Date(now).getFullYear();
  const month = MONTHS[d.getMonth()];
  return sameYear
    ? `${month} ${d.getDate()}`
    : `${month} ${d.getDate()}, ${d.getFullYear()}`;
}
