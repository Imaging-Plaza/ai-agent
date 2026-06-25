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
  const workflowSteps = pending.workflow_steps || [];
  const initialWorkflowValues = useMemo(
    () => workflowInitialValues(workflowSteps),
    [workflowSteps]
  );
  const [workflowValues, setWorkflowValues] = useState<Record<string, string>>(
    initialWorkflowValues
  );
  useEffect(() => {
    setWorkflowValues(initialWorkflowValues);
  }, [initialWorkflowValues]);

  function submitTool(event: React.FormEvent) {
    event.preventDefault();
    const params: Record<string, unknown> = {};
    for (const param of runtimeParams) {
      const raw = values[param.name] ?? "";
      if (raw.trim() !== "") params[param.name] = coerceValue(raw);
    }
    onApprove(params, selectedEndpointId);
  }

  if (pending.type === "workflow_approval") {
    function submitWorkflow(event: React.FormEvent) {
      event.preventDefault();
      const workflowSteps: Record<string, Record<string, unknown>> = {};
      for (const step of pending.workflow_steps || []) {
        const params: Record<string, unknown> = {};
        for (const param of step.runtime_parameters || []) {
          const key = workflowFieldKey(step.id, param.name);
          const raw = workflowValues[key] ?? "";
          if (raw.trim() !== "") params[param.name] = coerceValue(raw);
        }
        if (Object.keys(params).length > 0) workflowSteps[step.id] = params;
      }
      onApprove({ workflow_steps: workflowSteps });
    }

    return (
      <form className="pending-panel" onSubmit={submitWorkflow}>
        <div className="prompt">{pending.prompt || `run ${workflowSteps.length}-step tool chain?`}</div>
        {pending.image_name && (
          <div className="detail">input: {pending.image_name}</div>
        )}
        {workflowSteps.length > 0 && (
          <ol className="workflow-steps">
            {workflowSteps.map((step, index) => (
              <li key={step.id || `${step.tool_name}-${step.endpoint_id}`}>
                <div className="workflow-step-title">
                  step {index + 1}: {step.endpoint_display_name || step.display_name}
                </div>
                <div className="detail">
                  {step.input_name} → {step.output_name}
                </div>
                {(step.runtime_parameters || []).length > 0 && (
                  <div className="pending-params workflow-step-params">
                    {(step.runtime_parameters || []).map((param) => {
                      const key = workflowFieldKey(step.id, param.name);
                      return (
                        <label key={key}>
                          <span>
                            {param.label || param.name}
                            {param.required ? "" : " (optional)"}
                          </span>
                          {(param.choices || []).length > 0 ? (
                            <select
                              value={workflowValues[key] ?? ""}
                              onChange={(event) =>
                                setWorkflowValues((current) => ({
                                  ...current,
                                  [key]: event.target.value,
                                }))
                              }
                              required={param.required}
                              disabled={busy}
                            >
                              {(param.choices || []).map((choice) => (
                                <option key={String(choice)} value={String(choice)}>
                                  {String(choice)}
                                </option>
                              ))}
                            </select>
                          ) : (
                            <input
                              value={workflowValues[key] ?? ""}
                              onChange={(event) =>
                                setWorkflowValues((current) => ({
                                  ...current,
                                  [key]: event.target.value,
                                }))
                              }
                              placeholder={param.description || param.name}
                              required={param.required}
                              disabled={busy}
                            />
                          )}
                        </label>
                      );
                    })}
                  </div>
                )}
              </li>
            ))}
          </ol>
        )}
        <div className="actions">
          <button className="btn-approve" type="submit" disabled={busy}>
            run chain
          </button>
          <button className="btn-decline" type="button" onClick={onDecline} disabled={busy}>
            cancel
          </button>
        </div>
      </form>
    );
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

function workflowInitialValues(
  steps: NonNullable<PendingAction["workflow_steps"]>
): Record<string, string> {
  const values: Record<string, string> = {};
  for (const step of steps) {
    for (const param of step.runtime_parameters || []) {
      values[workflowFieldKey(step.id, param.name)] =
        param.default == null ? "" : String(param.default);
    }
  }
  return values;
}

function workflowFieldKey(stepId: string, paramName: string): string {
  return `${stepId}.${paramName}`;
}
