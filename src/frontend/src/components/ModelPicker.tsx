/** Three header chips: model selector, retrieval top_k, and the number of
 * recommendations the agent should produce. All three use the same
 * <MenuButton> dropdown for a consistent look.
 */

import { useEffect, useState } from "react";
import { api, type ModelOption } from "../lib/api";
import MenuButton, { type MenuOption } from "./MenuButton";

type Props = {
  value: string | null;
  onChange: (display: string | null) => void;
  topK: number;
  onTopK: (k: number) => void;
  numChoices: number;
  onNumChoices: (n: number) => void;
  sessionId: string | null;
};

const TOP_K_OPTIONS: MenuOption<number>[] = [4, 6, 8, 12, 16, 24].map((n) => ({
  value: n,
  label: String(n),
}));

const NUM_CHOICES_OPTIONS: MenuOption<number>[] = [1, 2, 3, 5].map((n) => ({
  value: n,
  label: String(n),
}));

export default function ModelPicker({
  value,
  onChange,
  topK,
  onTopK,
  numChoices,
  onNumChoices,
  sessionId,
}: Props) {
  const [models, setModels] = useState<ModelOption[]>([]);

  useEffect(() => {
    api.models().then(setModels).catch(() => setModels([]));
  }, []);

  const modelOptions: MenuOption<string>[] = [
    { value: "", label: "default", hint: "config.yaml" },
    ...models.map((m) => ({
      value: m.display_name,
      label: m.display_name,
      hint: m.provider ?? undefined,
    })),
  ];

  return (
    <div className="header-controls">
      <MenuButton<string>
        label="model"
        value={value ?? ""}
        options={modelOptions}
        onChange={(v) => onChange(v || null)}
        formatValue={(v) => {
          if (!v) return <em className="muted">default</em>;
          const trimmed = v.length > 22 ? v.slice(0, 21) + "…" : v;
          return trimmed;
        }}
      />
      <MenuButton<number>
        label="top_k"
        value={topK}
        options={TOP_K_OPTIONS}
        onChange={onTopK}
      />
      <MenuButton<number>
        label="choices"
        value={numChoices}
        options={NUM_CHOICES_OPTIONS}
        onChange={onNumChoices}
      />
      {sessionId && (
        <span className="session-pill" title="session id">
          sid:{sessionId.slice(0, 8)}
        </span>
      )}
    </div>
  );
}
