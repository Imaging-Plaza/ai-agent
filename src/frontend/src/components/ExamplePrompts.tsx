/** Empty-state example cards.
 *
 * Each example carries a real PNG bundled under /examples/*. Clicking a card
 * fetches the file, uploads it to /api/files for the current session, then
 * sends the prompt + the freshly minted asset_id — so the user immediately
 * sees how the agent reacts to a representative image without having to
 * upload their own.
 *
 * Falls back to a text-only send if the upload fails for any reason.
 */

import { useState } from "react";
import { api, type Asset } from "../lib/api";

type Props = {
  sessionId: string | null;
  onSessionId: (id: string) => void;
  onPick: (text: string, attachment: Asset | null) => void;
};

type Example = {
  tag: string;
  text: string;
  image: string;
  filename: string;
  /** Optional sprite for an animated cycle through volumetric slices. */
  sprite?: { url: string; frames: number; duration: number };
  badge?: string;
};

const EXAMPLES: Example[] = [
  {
    tag: "segmentation · 3D lungs",
    text: "Segment the lungs in this 3D CT volume (TIFF). Show me a per-slice mask.",
    image: "/examples/lungs-ct-3d.tif",
    filename: "lungs-ct.tif",
    sprite: { url: "/examples/lungs-ct-sprite.png", frames: 24, duration: 2.4 },
    badge: "3D",
  },
  {
    tag: "segmentation · CT",
    text: "I have a CT scan of a chest and I want to segment the lungs in it.",
    image: "/examples/ct-chest.png",
    filename: "ct-chest.png",
  },
  {
    tag: "classification · MRI",
    text: "Classify the organ visible in this MRI slice.",
    image: "/examples/brain-mri.png",
    filename: "brain-mri.png",
  },
  {
    tag: "registration · MRI",
    text: "Register these two brain MRI volumes from the same subject.",
    image: "/examples/registration.png",
    filename: "registration.png",
  },
  {
    tag: "detection · microscopy",
    text: "Detect cell nuclei in this fluorescence microscopy image (TIFF, 2D).",
    image: "/examples/nuclei.png",
    filename: "nuclei.png",
  },
  {
    tag: "denoising · CT",
    text: "Recommend a tool to denoise this low-dose CT slice.",
    image: "/examples/denoise.png",
    filename: "denoise.png",
  },
];

export default function ExamplePrompts({ sessionId, onSessionId, onPick }: Props) {
  const [loadingTag, setLoadingTag] = useState<string | null>(null);

  async function pick(ex: Example) {
    if (loadingTag) return;
    setLoadingTag(ex.tag);
    try {
      const res = await fetch(ex.image);
      if (!res.ok) throw new Error(`fetch ${ex.image} → ${res.status}`);
      const blob = await res.blob();
      const file = new File([blob], ex.filename, { type: blob.type || "image/png" });
      const uploaded = await api.uploadFiles([file], sessionId ?? undefined);
      if (!sessionId) onSessionId(uploaded.session_id);
      const asset = uploaded.assets[0] ?? null;
      onPick(ex.text, asset);
    } catch (e) {
      console.warn("example pick: upload failed, sending text-only", e);
      onPick(ex.text, null);
    } finally {
      setLoadingTag(null);
    }
  }

  return (
    <div className="examples">
      <div className="examples-label">try one · the agent runs on the attached sample</div>
      <div className="examples-grid">
        {EXAMPLES.map((e) => {
          const busy = loadingTag === e.tag;
          return (
            <button
              key={e.tag}
              type="button"
              className={"example-card" + (busy ? " is-busy" : "")}
              onClick={() => void pick(e)}
              disabled={Boolean(loadingTag)}
            >
              <div className="example-thumb">
                {e.sprite ? (
                  <div
                    className="example-sprite"
                    style={
                      {
                        backgroundImage: `url(${e.sprite.url})`,
                        backgroundSize: `100% ${e.sprite.frames * 100}%`,
                        animation: `sprite-cycle ${e.sprite.duration}s steps(${e.sprite.frames}) infinite`,
                      } as React.CSSProperties
                    }
                    aria-label={e.tag}
                  />
                ) : (
                  <img src={e.image} alt={e.tag} loading="lazy" />
                )}
                {e.badge && <span className="example-badge mono">{e.badge}</span>}
                {busy && <div className="example-spinner" aria-hidden />}
              </div>
              <span className="example-tag">{e.tag}</span>
              <span className="example-text">{e.text}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
