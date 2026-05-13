import type { Recommendation } from "../hooks/useChat";

export default function RecommendationCard({ rec }: { rec: Recommendation }) {
  const doc = rec.doc || {};
  const modality = Array.isArray(doc.modality) ? doc.modality.join(", ") : "";
  const license = (doc as any).license || "";
  const dims: number[] = Array.isArray(doc.dims) ? doc.dims : [];
  const keywords: string[] = Array.isArray(doc.keywords) ? doc.keywords : [];

  return (
    <div className="rec-card">
      <div>
        <span className="rank">{rec.rank}</span>
        <span className="name">{rec.name}</span>
        <span className="accuracy">{rec.accuracy.toFixed(1)}%</span>
      </div>
      {rec.why && <div className="why">{rec.why}</div>}
      <div className="meta">
        {modality && <span className="tag">📡 {modality}</span>}
        {dims.length > 0 && <span className="tag">{dims.map((d) => `${d}D`).join("/")}</span>}
        {license && <span className="tag">📜 {license}</span>}
        {keywords.slice(0, 4).map((k) => (
          <span key={k} className="tag">
            {k}
          </span>
        ))}
        {rec.demo_url && (
          <a className="tag" href={rec.demo_url} target="_blank" rel="noreferrer">
            🔗 Demo
          </a>
        )}
      </div>
    </div>
  );
}
