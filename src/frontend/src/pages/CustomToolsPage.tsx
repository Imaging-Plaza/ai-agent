import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { ApiError, api } from "../lib/api";

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

export default function CustomToolsPage() {
  const [config, setConfig] = useState<Record<string, any>>({ version: 1, tools: [] });
  const [status, setStatus] = useState("loading...");
  const [errors, setErrors] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const [showForm, setShowForm] = useState(false);
  const [spaceUrl, setSpaceUrl] = useState("");

  const tools = useMemo(() => (Array.isArray(config?.tools) ? config.tools : []), [config]);

  useEffect(() => {
    void load();
  }, []);

  async function load() {
    setBusy(true);
    setErrors([]);
    try {
      const result = await api.gradioTools();
      setConfig(result.config);
      setStatus("active configuration loaded");
    } catch (err: any) {
      setErrors([err?.message || String(err)]);
      setStatus("load failed");
    } finally {
      setBusy(false);
    }
  }

  function openAddForm() {
    setSpaceUrl("");
    setShowForm(true);
    setErrors([]);
  }

  async function importToolLink(event: React.FormEvent) {
    event.preventDefault();
    if (!spaceUrl.trim()) {
      setErrors(["Enter a Hugging Face Space URL."]);
      return;
    }
    setBusy(true);
    setErrors([]);
    setStatus("fetching tool metadata...");
    try {
      const result = await api.importGradioToolLink(spaceUrl.trim());
      setStatus(result.ok && result.reloaded ? "tool link added and reloaded" : "tool link saved, restart required");
      setShowForm(false);
      setSpaceUrl("");
      await load();
    } catch (err) {
      setErrors([readableError(err)]);
      setStatus("add tool link failed");
    } finally {
      setBusy(false);
    }
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
        <button className="tools-primary" onClick={openAddForm} disabled={busy}>add tool link</button>
        <button onClick={load} disabled={busy}>refresh</button>
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
                    {endpoint.description && <small>{endpoint.description}</small>}
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
          <form className="modal tools-link-modal" onSubmit={importToolLink} onClick={(event) => event.stopPropagation()}>
            <header className="modal-head">
              <div className="modal-tag mono">gradio link</div>
              <button className="modal-x" type="button" onClick={() => setShowForm(false)} aria-label="Close">x</button>
            </header>
            <h2 className="modal-title">add tool link</h2>
            <p className="modal-sub">Paste a Hugging Face Space URL. The app will fetch its Gradio endpoint metadata and register the runnable tool automatically.</p>
            <label className="tools-link-field">
              <span>HF Space URL</span>
              <input
                value={spaceUrl}
                onChange={(event) => setSpaceUrl(event.target.value)}
                placeholder="user-tool.hf.space or huggingface.co/user/tool"
                autoFocus
                required
              />
            </label>
            <div className="modal-actions tools-form-actions">
              <button type="button" onClick={() => setShowForm(false)}>cancel</button>
              <button className="tools-primary" type="submit" disabled={busy}>add link</button>
            </div>
          </form>
        </div>
      )}
    </main>
  );
}
