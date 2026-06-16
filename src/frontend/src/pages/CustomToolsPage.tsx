import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { ApiError, api } from "../lib/api";

type InputParamDraft = {
  name: string;
  source: "session_file" | "image_path" | "description" | "literal" | "param";
  required: boolean;
  value: string;
  param: string;
  asGradioFile: boolean;
};

type EndpointDraft = {
  id: string;
  displayName: string;
  apiName: string;
  aliases: string;
  description: string;
  enabled: boolean;
  inputParameters: InputParamDraft[];
  callStyle: "keyword" | "positional";
  originalSelector: string;
  previewSelector: string;
  buildPreview: boolean;
  approvalMessage: string;
  demoAvailable: boolean;
};

type ToolDraft = {
  id: string;
  displayName: string;
  description: string;
  icon: string;
  enabled: boolean;
  gradioUrl: string;
  aliases: string;
  defaultEndpoint: string;
  timeoutSeconds: string;
  maxDownloadBytes: string;
  endpoints: EndpointDraft[];
};

const emptyInputParam = (name = "file_obj", source: InputParamDraft["source"] = "session_file"): InputParamDraft => ({
  name,
  source,
  required: source !== "literal",
  value: "",
  param: "",
  asGradioFile: source === "session_file" || source === "image_path",
});

const emptyEndpoint = (id = "endpoint") : EndpointDraft => ({
  id,
  displayName: id === "endpoint" ? "Endpoint" : id,
  apiName: "/predict",
  aliases: "",
  description: "",
  enabled: true,
  inputParameters: [emptyInputParam()],
  callStyle: "keyword",
  originalSelector: "first",
  previewSelector: "first",
  buildPreview: true,
  approvalMessage: "",
  demoAvailable: true,
});

const emptyTool = (): ToolDraft => ({
  id: "",
  displayName: "",
  description: "",
  icon: "T",
  enabled: true,
  gradioUrl: "",
  aliases: "",
  defaultEndpoint: "segment",
  timeoutSeconds: "300",
  maxDownloadBytes: "1073741824",
  endpoints: [emptyEndpoint("segment")],
});

function pretty(value: unknown): string {
  return JSON.stringify(value, null, 2);
}

function splitList(value: string): string[] {
  return value
    .split(/[\n,]/)
    .map((item) => item.trim())
    .filter(Boolean);
}


function readableError(error: unknown): string {
  const raw = error instanceof ApiError ? error.message : String(error);
  try {
    const parsed = JSON.parse(raw);
    if (typeof parsed?.detail === "string") return parsed.detail;
  } catch {
    // Keep the original message.
  }
  return raw;
}

function slug(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "");
}

function coerceLiteral(value: string): unknown {
  const trimmed = value.trim();
  if (trimmed === "") return "";
  if (trimmed === "true") return true;
  if (trimmed === "false") return false;
  if (trimmed === "null") return null;
  const numeric = Number(trimmed);
  if (!Number.isNaN(numeric) && trimmed !== "") return numeric;
  return trimmed;
}

function buildTool(draft: ToolDraft): Record<string, any> {
  const endpoints = draft.endpoints.map((endpoint) => ({
    id: endpoint.id.trim(),
    display_name: endpoint.displayName.trim() || endpoint.id.trim(),
    description: endpoint.description.trim() || undefined,
    api_name: endpoint.apiName.trim(),
    enabled: endpoint.enabled,
    catalog_aliases: splitList(endpoint.aliases),
    supported_input_types: ["image", "file"],
    input_mapping: {
      call_style: endpoint.callStyle,
      parameters: endpoint.inputParameters.map((parameter) => ({
        name: parameter.name.trim() || "value",
        source: parameter.source,
        required: parameter.required,
        value: parameter.source === "literal" ? coerceLiteral(parameter.value) : undefined,
        param: parameter.source === "param" ? parameter.param.trim() || parameter.name.trim() : undefined,
        as_gradio_file: parameter.asGradioFile,
      })),
    },
    output_mapping: {
      original: {
        selector: endpoint.originalSelector.trim() || "first",
        materialize: true,
      },
      preview: {
        selector: endpoint.previewSelector.trim() || "first",
        materialize: true,
        build_preview: endpoint.buildPreview,
      },
    },
    approval: {
      required: true,
      message: endpoint.approvalMessage.trim() || undefined,
    },
    demo: {
      available: endpoint.demoAvailable,
    },
  }));

  return {
    id: draft.id.trim(),
    display_name: draft.displayName.trim() || draft.id.trim(),
    description: draft.description.trim() || undefined,
    icon: draft.icon.trim() || "T",
    enabled: draft.enabled,
    gradio_url: draft.gradioUrl.trim(),
    catalog_aliases: splitList(draft.aliases),
    default_endpoint: draft.defaultEndpoint.trim() || endpoints[0]?.id,
    timeout_seconds: Number(draft.timeoutSeconds) || 300,
    max_download_bytes: Number(draft.maxDownloadBytes) || 1073741824,
    endpoints,
  };
}

export default function CustomToolsPage() {
  const [text, setText] = useState("");
  const [status, setStatus] = useState("loading...");
  const [errors, setErrors] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const [showJson, setShowJson] = useState(false);
  const [showForm, setShowForm] = useState(false);
  const [draft, setDraft] = useState<ToolDraft>(() => emptyTool());

  const parsed = useMemo(() => {
    try {
      return JSON.parse(text || "{}");
    } catch {
      return null;
    }
  }, [text]);

  const tools = Array.isArray(parsed?.tools) ? parsed.tools : [];

  useEffect(() => {
    void load();
  }, []);

  async function load() {
    setBusy(true);
    setErrors([]);
    try {
      const result = await api.gradioTools();
      setText(pretty(result.config));
      setStatus("active configuration loaded");
    } catch (err: any) {
      setErrors([err?.message || String(err)]);
      setStatus("load failed");
    } finally {
      setBusy(false);
    }
  }

  function parseConfig(): Record<string, any> | null {
    try {
      setErrors([]);
      return JSON.parse(text || "{}");
    } catch (err: any) {
      setErrors([`JSON parse error: ${err?.message || err}`]);
      return null;
    }
  }

  async function validate() {
    const config = parseConfig();
    if (!config) return;
    setBusy(true);
    try {
      const result = await api.validateGradioTools(config);
      setErrors(result.errors || []);
      setStatus(result.ok ? "configuration is valid" : "validation failed");
    } finally {
      setBusy(false);
    }
  }

  async function save() {
    const config = parseConfig();
    if (!config) return;
    setBusy(true);
    try {
      const result = await api.saveGradioTools(config);
      setErrors(result.errors || []);
      setStatus(result.ok && result.reloaded ? "saved and reloaded" : "saved, restart required");
    } catch (err) {
      setErrors([readableError(err)]);
      setStatus("save failed");
    } finally {
      setBusy(false);
    }
  }

  async function reload() {
    setBusy(true);
    try {
      const result = await api.reloadGradioTools();
      setErrors(result.errors || []);
      setStatus(result.ok ? "registry reloaded" : "reload failed");
    } finally {
      setBusy(false);
    }
  }

  async function copy() {
    await navigator.clipboard.writeText(text);
    setStatus("configuration copied");
  }

  function importJson(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;
    event.target.value = "";
    const reader = new FileReader();
    reader.onload = () => {
      setText(String(reader.result || ""));
      setStatus("json imported; validate before saving");
      setShowJson(true);
    };
    reader.readAsText(file);
  }

  function exportJson() {
    const blob = new Blob([text], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "gradio_tools.json";
    a.click();
    URL.revokeObjectURL(url);
  }

  function openAddForm() {
    setDraft(emptyTool());
    setShowForm(true);
  }

  function updateEndpoint(index: number, patch: Partial<EndpointDraft>) {
    setDraft((current) => {
      const oldEndpoint = current.endpoints[index];
      const defaultEndpoint = patch.id && oldEndpoint?.id === current.defaultEndpoint
        ? patch.id
        : current.defaultEndpoint;
      return {
        ...current,
        defaultEndpoint,
        endpoints: current.endpoints.map((endpoint, i) => i === index ? { ...endpoint, ...patch } : endpoint),
      };
    });
  }

  function updateInputParameter(endpointIndex: number, parameterIndex: number, patch: Partial<InputParamDraft>) {
    setDraft((current) => ({
      ...current,
      endpoints: current.endpoints.map((endpoint, i) => {
        if (i !== endpointIndex) return endpoint;
        return {
          ...endpoint,
          inputParameters: endpoint.inputParameters.map((parameter, j) => {
            if (j !== parameterIndex) return parameter;
            const next = { ...parameter, ...patch };
            if (patch.source) {
              next.required = patch.source !== "literal";
              next.asGradioFile = patch.source === "session_file" || patch.source === "image_path";
            }
            return next;
          }),
        };
      }),
    }));
  }

  function addInputParameter(endpointIndex: number) {
    setDraft((current) => ({
      ...current,
      endpoints: current.endpoints.map((endpoint, i) => i === endpointIndex
        ? { ...endpoint, inputParameters: [...endpoint.inputParameters, emptyInputParam(`param_${endpoint.inputParameters.length + 1}`, "literal")] }
        : endpoint
      ),
    }));
  }

  function removeInputParameter(endpointIndex: number, parameterIndex: number) {
    setDraft((current) => ({
      ...current,
      endpoints: current.endpoints.map((endpoint, i) => i === endpointIndex
        ? { ...endpoint, inputParameters: endpoint.inputParameters.filter((_, j) => j !== parameterIndex) }
        : endpoint
      ),
    }));
  }

  function addEndpoint() {
    setDraft((current) => {
      const nextId = `endpoint_${current.endpoints.length + 1}`;
      return {
        ...current,
        endpoints: [...current.endpoints, emptyEndpoint(nextId)],
      };
    });
  }

  function removeEndpoint(index: number) {
    setDraft((current) => ({
      ...current,
      endpoints: current.endpoints.filter((_, i) => i !== index),
    }));
  }

  function saveDraftAsJson(event: React.FormEvent) {
    event.preventDefault();
    const config = parseConfig();
    if (!config) return;
    const tool = buildTool(draft);
    if (!tool.id || !tool.gradio_url || !tool.endpoints?.[0]?.api_name) {
      setErrors(["Tool id, Gradio URL, and at least one endpoint API name are required."]);
      return;
    }
    const existingTools = Array.isArray(config.tools) ? config.tools : [];
    if (existingTools.some((item: any) => item.id === tool.id)) {
      setErrors([`A tool with id ${tool.id} already exists.`]);
      return;
    }
    const next = {
      ...config,
      version: config.version || 1,
      tools: [...existingTools, tool],
    };
    setText(pretty(next));
    setShowForm(false);
    setShowJson(true);
    setErrors([]);
    setStatus("tool added to json; validate and save to activate");
  }

  return (
    <main className="tools-page">
      <header className="tools-header">
        <div>
          <div className="mono">custom tools</div>
          <h1>Gradio Tools</h1>
        </div>
        <Link className="sidebar-action" to="/">back to chat</Link>
      </header>

      <section className="tools-command-bar" aria-label="Tool actions">
        <button className="tools-primary" onClick={openAddForm} disabled={busy}>add tool</button>
        <label className="tools-import">
          import json
          <input type="file" accept=".json,application/json" onChange={importJson} />
        </label>
        <button onClick={exportJson} disabled={busy}>export all</button>
        <button onClick={validate} disabled={busy}>validate</button>
        <button onClick={save} disabled={busy}>save changes</button>
        <button onClick={reload} disabled={busy}>reload</button>
        <button onClick={() => setShowJson((value) => !value)} disabled={busy}>
          {showJson ? "hide json" : "edit json"}
        </button>
      </section>

      <section className="tools-list" aria-label="Configured Gradio tools">
        {tools.length === 0 && <div className="tools-empty">no configured Gradio tools</div>}
        {tools.map((tool: any) => (
          <article className="tool-card" key={tool.id}>
            <div className="tool-card-main">
              <div className="tool-icon" aria-hidden>{tool.icon || "T"}</div>
              <div className="tool-card-copy">
                <div className="tool-card-title-row">
                  <h2>{tool.display_name || tool.id}</h2>
                  <span className={tool.enabled === false ? "tool-state off" : "tool-state"}>
                    {tool.enabled === false ? "disabled" : "enabled"}
                  </span>
                </div>
                {tool.description && <p>{tool.description}</p>}
                <div className="tool-meta mono">
                  <span>{tool.id}</span>
                  {tool.gradio_url && <span>{tool.gradio_url}</span>}
                </div>
              </div>
            </div>
            <div className="endpoint-list">
              {(tool.endpoints || []).map((endpoint: any) => (
                <div className="endpoint-row" key={endpoint.id}>
                  <div>
                    <strong>{endpoint.display_name || endpoint.id}</strong>
                    <span>{endpoint.api_name || "missing api_name"}</span>
                  </div>
                  <span className={endpoint.enabled === false ? "tool-state off" : "tool-state"}>
                    {endpoint.enabled === false ? "disabled" : "enabled"}
                  </span>
                </div>
              ))}
            </div>
          </article>
        ))}
      </section>

      {showJson && (
        <section className="tools-editor">
          <div className="tools-toolbar">
            <button onClick={copy} disabled={busy}>copy json</button>
            <button onClick={exportJson} disabled={busy}>export all</button>
          </div>
          <textarea
            value={text}
            onChange={(event) => setText(event.target.value)}
            spellCheck={false}
            aria-label="Gradio tools JSON configuration"
          />
        </section>
      )}

      <footer className="tools-status">
        <span>{status}</span>
        {errors.length > 0 && (
          <div className="tools-errors">
            {errors.map((error, i) => <pre key={i}>{error}</pre>)}
          </div>
        )}
      </footer>

      {showForm && (
        <div className="modal-backdrop" onClick={() => setShowForm(false)}>
          <form className="modal tools-form-modal" onSubmit={saveDraftAsJson} onClick={(event) => event.stopPropagation()}>
            <header className="modal-head">
              <div className="modal-tag mono">gradio tool</div>
              <button className="modal-x" type="button" onClick={() => setShowForm(false)} aria-label="Close">x</button>
            </header>
            <h2 className="modal-title">add a Gradio tool</h2>
            <p className="modal-sub">Create a JSON-backed Gradio app entry with one or more callable endpoints.</p>

            <div className="tools-form-grid">
              <label>
                <span>Tool name</span>
                <input value={draft.displayName} onChange={(event) => setDraft({ ...draft, displayName: event.target.value, id: draft.id || slug(event.target.value) })} placeholder="3D Lungs Segmentation" />
              </label>
              <label>
                <span>Tool id</span>
                <input value={draft.id} onChange={(event) => setDraft({ ...draft, id: slug(event.target.value) })} placeholder="lungs_segmentation" required />
              </label>
              <label className="wide">
                <span>Gradio URL</span>
                <input value={draft.gradioUrl} onChange={(event) => setDraft({ ...draft, gradioUrl: event.target.value })} placeholder="https://example.hf.space/" required />
              </label>
              <label>
                <span>Catalog aliases</span>
                <input value={draft.aliases} onChange={(event) => setDraft({ ...draft, aliases: event.target.value })} placeholder="lungs-segmentation, lung-mask" />
              </label>
              <label className="wide">
                <span>Description</span>
                <textarea value={draft.description} onChange={(event) => setDraft({ ...draft, description: event.target.value })} rows={3} placeholder="What this Gradio app does" />
              </label>
              <label>
                <span>Default endpoint</span>
                <select value={draft.defaultEndpoint} onChange={(event) => setDraft({ ...draft, defaultEndpoint: event.target.value })}>
                  {draft.endpoints.map((endpoint) => <option key={endpoint.id} value={endpoint.id}>{endpoint.id}</option>)}
                </select>
              </label>
              <label>
                <span>Enabled</span>
                <select value={draft.enabled ? "true" : "false"} onChange={(event) => setDraft({ ...draft, enabled: event.target.value === "true" })}>
                  <option value="true">enabled</option>
                  <option value="false">disabled</option>
                </select>
              </label>
            </div>

            <div className="tools-form-section-head">
              <h3>Endpoints</h3>
              <button type="button" onClick={addEndpoint}>add endpoint</button>
            </div>

            <div className="tools-endpoint-drafts">
              {draft.endpoints.map((endpoint, index) => (
                <fieldset className="endpoint-draft" key={`${endpoint.id}-${index}`}>
                  <legend>{endpoint.displayName || endpoint.id || `endpoint ${index + 1}`}</legend>
                  <div className="tools-form-grid">
                    <label>
                      <span>Endpoint name</span>
                      <input value={endpoint.displayName} onChange={(event) => updateEndpoint(index, { displayName: event.target.value, id: endpoint.id || slug(event.target.value) })} />
                    </label>
                    <label>
                      <span>Endpoint id</span>
                      <input value={endpoint.id} onChange={(event) => updateEndpoint(index, { id: slug(event.target.value) })} required />
                    </label>
                    <label>
                      <span>API name</span>
                      <input value={endpoint.apiName} onChange={(event) => updateEndpoint(index, { apiName: event.target.value })} placeholder="/predict" required />
                    </label>
                    <label>
                      <span>Endpoint aliases</span>
                      <input value={endpoint.aliases} onChange={(event) => updateEndpoint(index, { aliases: event.target.value })} placeholder="catalog-name" />
                    </label>
                    <label>
                      <span>Call style</span>
                      <select value={endpoint.callStyle} onChange={(event) => updateEndpoint(index, { callStyle: event.target.value as EndpointDraft["callStyle"] })}>
                        <option value="keyword">keyword</option>
                        <option value="positional">positional</option>
                      </select>
                    </label>
                    <label>
                      <span>Enabled</span>
                      <select value={endpoint.enabled ? "true" : "false"} onChange={(event) => updateEndpoint(index, { enabled: event.target.value === "true" })}>
                        <option value="true">enabled</option>
                        <option value="false">disabled</option>
                      </select>
                    </label>
                    <label>
                      <span>Original output selector</span>
                      <input value={endpoint.originalSelector} onChange={(event) => updateEndpoint(index, { originalSelector: event.target.value })} />
                    </label>
                    <label>
                      <span>Preview output selector</span>
                      <input value={endpoint.previewSelector} onChange={(event) => updateEndpoint(index, { previewSelector: event.target.value })} />
                    </label>
                    <label className="wide">
                      <span>Approval message</span>
                      <input value={endpoint.approvalMessage} onChange={(event) => updateEndpoint(index, { approvalMessage: event.target.value })} placeholder="Run this endpoint on your uploaded image?" />
                    </label>
                  </div>

                  <div className="parameter-section-head">
                    <h4>Parameters</h4>
                    <button type="button" onClick={() => addInputParameter(index)}>add parameter</button>
                  </div>
                  <div className="parameter-list">
                    {endpoint.inputParameters.map((parameter, parameterIndex) => (
                      <div className="parameter-row" key={`${parameter.name}-${parameterIndex}`}>
                        <label>
                          <span>Name</span>
                          <input value={parameter.name} onChange={(event) => updateInputParameter(index, parameterIndex, { name: event.target.value })} placeholder="mode" />
                        </label>
                        <label>
                          <span>Source</span>
                          <select value={parameter.source} onChange={(event) => updateInputParameter(index, parameterIndex, { source: event.target.value as InputParamDraft["source"] })}>
                            <option value="session_file">session file</option>
                            <option value="image_path">image path</option>
                            <option value="description">description</option>
                            <option value="literal">literal value</option>
                            <option value="param">runtime parameter</option>
                          </select>
                        </label>
                        {parameter.source === "literal" && (
                          <label>
                            <span>Value</span>
                            <input value={parameter.value} onChange={(event) => updateInputParameter(index, parameterIndex, { value: event.target.value })} placeholder="RIGID_BODY" />
                          </label>
                        )}
                        {parameter.source === "param" && (
                          <label>
                            <span>Parameter key</span>
                            <input value={parameter.param} onChange={(event) => updateInputParameter(index, parameterIndex, { param: event.target.value })} placeholder="mode" />
                          </label>
                        )}
                        <label className="inline-check parameter-check">
                          <input type="checkbox" checked={parameter.required} onChange={(event) => updateInputParameter(index, parameterIndex, { required: event.target.checked })} />
                          required
                        </label>
                        <label className="inline-check parameter-check">
                          <input type="checkbox" checked={parameter.asGradioFile} onChange={(event) => updateInputParameter(index, parameterIndex, { asGradioFile: event.target.checked })} />
                          file
                        </label>
                        {endpoint.inputParameters.length > 1 && <button type="button" onClick={() => removeInputParameter(index, parameterIndex)}>remove</button>}
                      </div>
                    ))}
                  </div>

                  <div className="endpoint-draft-actions">
                    <label className="inline-check">
                      <input type="checkbox" checked={endpoint.buildPreview} onChange={(event) => updateEndpoint(index, { buildPreview: event.target.checked })} />
                      build preview
                    </label>
                    <label className="inline-check">
                      <input type="checkbox" checked={endpoint.demoAvailable} onChange={(event) => updateEndpoint(index, { demoAvailable: event.target.checked })} />
                      runnable demo
                    </label>
                    {draft.endpoints.length > 1 && <button type="button" onClick={() => removeEndpoint(index)}>remove endpoint</button>}
                  </div>
                </fieldset>
              ))}
            </div>

            <div className="modal-actions tools-form-actions">
              <button type="button" onClick={() => setShowForm(false)}>cancel</button>
              <button className="tools-primary" type="submit">save form as json</button>
            </div>
          </form>
        </div>
      )}
    </main>
  );
}
