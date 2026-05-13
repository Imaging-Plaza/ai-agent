import { useEffect, useState } from "react";
import { api, type ModelOption } from "../lib/api";

type Props = {
  value: string | null;
  onChange: (display: string | null) => void;
  topK: number;
  onTopK: (k: number) => void;
  numChoices: number;
  onNumChoices: (n: number) => void;
};

export default function ModelPicker({
  value,
  onChange,
  topK,
  onTopK,
  numChoices,
  onNumChoices,
}: Props) {
  const [models, setModels] = useState<ModelOption[]>([]);

  useEffect(() => {
    api.models().then(setModels).catch(() => setModels([]));
  }, []);

  return (
    <div className="toolbar">
      <label>Model:</label>
      <select
        value={value ?? ""}
        onChange={(e) => onChange(e.target.value || null)}
      >
        <option value="">(default)</option>
        {models.map((m) => (
          <option key={m.display_name} value={m.display_name}>
            {m.display_name}
          </option>
        ))}
      </select>

      <label>top_k:</label>
      <select value={topK} onChange={(e) => onTopK(parseInt(e.target.value, 10))}>
        {[4, 6, 8, 12, 16, 24].map((n) => (
          <option key={n} value={n}>
            {n}
          </option>
        ))}
      </select>

      <label>choices:</label>
      <select
        value={numChoices}
        onChange={(e) => onNumChoices(parseInt(e.target.value, 10))}
      >
        {[1, 2, 3, 5].map((n) => (
          <option key={n} value={n}>
            {n}
          </option>
        ))}
      </select>

      <div className="spacer" />
    </div>
  );
}
