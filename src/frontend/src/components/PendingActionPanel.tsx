import { useEffect, useMemo, useState } from "react";
import type { PendingAction } from "../hooks/useChat";

type Props = {
  pending: PendingAction;
  onApprove: (params?: Record<string, unknown>, endpointId?: string | null) => void;
  onDecline: () => void;
  onConfirmDemo: () => void;
  busy: boolean;
};

export default function PendingActionPanel({
  pending,
  onApprove,
  onDecline,
  onConfirmDemo,
  busy,
}: Props) {
  const endpointOptions = pending.endpoint_options || [];
  const [selectedEndpointId, setSelectedEndpointId] = useState<string | null>(
    pending.endpoint_id || endpointOptions[0]?.endpoint_id || null
  );
  useEffect(() => {
    setSelectedEndpointId(pending.endpoint_id || endpointOptions[0]?.endpoint_id || null);
  }, [pending.endpoint_id, endpointOptions]);
  const activeEndpoint =
    endpointOptions.find((option) => option.endpoint_id === selectedEndpointId) ||
    endpointOptions[0] ||
    null;
  const runtimeParams = activeEndpoint?.runtime_parameters || pending.runtime_parameters || [];
  const initialValues = useMemo(() => {
    const values: Record<string, string> = {};
    for (const param of runtimeParams) {
      values[param.name] = param.default == null ? "" : String(param.default);
    }
    return values;
  }, [runtimeParams]);
  const [values, setValues] = useState<Record<string, string>>(initialValues);
  useEffect(() => {
    setValues(initialValues);
  }, [initialValues]);

  function submitTool(event: React.FormEvent) {
    event.preventDefault();
    const params: Record<string, unknown> = {};
    for (const param of runtimeParams) {
      const raw = values[param.name] ?? "";
      if (raw.trim() !== "") params[param.name] = coerceValue(raw);
    }
    onApprove(params, selectedEndpointId);
  }

  if (pending.type === "tool_approval") {
    const endpointLabel =
      activeEndpoint?.display_name ||
      pending.endpoint_display_name ||
      pending.display_name ||
      pending.tool_name;
    return (
      <form className="pending-panel" onSubmit={submitTool}>
        <div className="prompt">run {pending.display_name || pending.tool_name}?</div>
        {pending.image_name && (
          <div className="detail">image: {pending.image_name}</div>
        )}
        {endpointLabel && (
          <div className="detail">endpoint: {endpointLabel}</div>
        )}
        {pending.recommendation_rank && (
          <div className="detail">recommendation: #{pending.recommendation_rank} {pending.recommendation_name}</div>
        )}
        {pending.demo_url && (
          <div className="detail">
            space: <a href={pending.demo_url}>{pending.demo_url}</a>
          </div>
        )}
        {pending.prompt && <div className="detail">{pending.prompt}</div>}
        {endpointOptions.length > 1 && (
          <div className="pending-params">
            <label>
              <span>endpoint</span>
              <select
                value={selectedEndpointId || ""}
                onChange={(event) => setSelectedEndpointId(event.target.value || null)}
                disabled={busy}
              >
                {endpointOptions.map((option) => (
                  <option key={option.endpoint_id} value={option.endpoint_id}>
                    {option.display_name || option.endpoint_id}
                  </option>
                ))}
              </select>
            </label>
            {activeEndpoint?.description && (
              <div className="detail">{activeEndpoint.description}</div>
            )}
          </div>
        )}
        {runtimeParams.length > 0 && (
          <div className="pending-params">
            {runtimeParams.map((param) => (
              <label key={param.name}>
                <span>
                  {param.label || param.name}
                  {param.required ? "" : " (optional)"}
                </span>
                {(param.choices || []).length > 0 ? (
                  <select
                    value={values[param.name] ?? ""}
                    onChange={(event) =>
                      setValues((current) => ({ ...current, [param.name]: event.target.value }))
                    }
                    required={param.required}
                  >
                    {(param.choices || []).map((choice) => (
                      <option key={String(choice)} value={String(choice)}>
                        {String(choice)}
                      </option>
                    ))}
                  </select>
                ) : (
                  <input
                    value={values[param.name] ?? ""}
                    onChange={(event) =>
                      setValues((current) => ({ ...current, [param.name]: event.target.value }))
                    }
                    placeholder={param.description || param.name}
                    required={param.required}
                  />
                )}
              </label>
            ))}
          </div>
        )}
        <div className="actions">
          <button className="btn-approve" type="submit" disabled={busy}>
            run {endpointLabel}
          </button>
          <button className="btn-decline" type="button" onClick={onDecline} disabled={busy}>
            cancel
          </button>
        </div>
      </form>
    );
  }

  return (
    <div className="pending-panel">
      <div className="prompt">{pending.prompt}</div>
      {pending.demo_url && (
        <div className="detail">
          demo: <a href={pending.demo_url}>{pending.demo_url}</a>
        </div>
      )}
      <div className="actions">
        <button className="btn-approve" onClick={onConfirmDemo} disabled={busy}>
          run demo
        </button>
        <button className="btn-decline" onClick={onDecline} disabled={busy}>
          cancel
        </button>
      </div>
    </div>
  );
}

function coerceValue(value: string): unknown {
  const trimmed = value.trim();
  if (trimmed === "true") return true;
  if (trimmed === "false") return false;
  if (trimmed === "null") return null;
  const numeric = Number(trimmed);
  if (trimmed !== "" && !Number.isNaN(numeric)) return numeric;
  return value;
}
