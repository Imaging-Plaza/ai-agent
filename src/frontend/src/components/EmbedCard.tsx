/** Renders one EmbedTurn — image / audio / video / youtube / iframe / info. */

import type { EmbedTurn } from "../hooks/useChat";

type Props = {
  turn: EmbedTurn;
  onOpenImage?: (src: string) => void;
};

export default function EmbedCard({ turn, onOpenImage }: Props) {
  return (
    <div className="embed-card">
      <div className="embed-head mono">
        <span className="embed-kind">{turn.kind}</span>
        <span className="embed-cmd" title={turn.command}>
          {turn.command}
        </span>
      </div>
      <div className="embed-body">{renderBody(turn, onOpenImage)}</div>
      {turn.label && <div className="embed-label">{turn.label}</div>}
    </div>
  );
}

function renderBody(turn: EmbedTurn, onOpenImage?: (src: string) => void) {
  switch (turn.kind) {
    case "image":
      if (!turn.src)
        return <div className="embed-empty mono">no image source</div>;
      return (
        <img
          className="embed-image"
          src={turn.src}
          alt={turn.label || "image"}
          onClick={() => onOpenImage && turn.src && onOpenImage(turn.src)}
        />
      );
    case "audio":
      if (!turn.src)
        return <div className="embed-empty mono">no audio source</div>;
      return <audio controls src={turn.src} className="embed-audio" />;
    case "video":
      if (!turn.src)
        return <div className="embed-empty mono">no video source</div>;
      return <video controls src={turn.src} className="embed-video" />;
    case "youtube":
      if (!turn.src)
        return (
          <div className="embed-empty mono">no youtube id / url provided</div>
        );
      return (
        <div className="embed-youtube">
          <iframe
            src={`https://www.youtube.com/embed/${turn.src}`}
            title="YouTube"
            frameBorder={0}
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
            referrerPolicy="strict-origin-when-cross-origin"
            allowFullScreen
          />
        </div>
      );
    case "iframe":
      if (!turn.src)
        return <div className="embed-empty mono">no url</div>;
      return (
        <iframe
          src={turn.src}
          className="embed-iframe"
          title="embedded content"
          referrerPolicy="strict-origin-when-cross-origin"
          sandbox="allow-scripts allow-same-origin allow-popups allow-forms"
        />
      );
    case "info":
      return (
        <pre className="embed-info mono">{turn.label || turn.command}</pre>
      );
  }
  return null;
}
