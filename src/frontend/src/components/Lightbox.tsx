/** Lightweight image lightbox.
 *
 * Click outside or press Escape to close. Renders nothing when `src` is null.
 * The image is shown at its natural size, capped to 92vw / 92vh, so previews
 * stay sharp even when generated as small thumbnails.
 */

import { useEffect } from "react";

type Props = {
  src: string | null;
  alt?: string;
  onClose: () => void;
};

export default function Lightbox({ src, alt, onClose }: Props) {
  useEffect(() => {
    if (!src) return;
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
    }
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [src, onClose]);

  if (!src) return null;
  return (
    <div className="lightbox-backdrop" onClick={onClose}>
      <button
        type="button"
        className="lightbox-close"
        onClick={onClose}
        aria-label="Close preview"
        title="Close (Esc)"
      >
        ✕
      </button>
      <img
        src={src}
        alt={alt || ""}
        className="lightbox-img"
        onClick={(e) => e.stopPropagation()}
      />
    </div>
  );
}
