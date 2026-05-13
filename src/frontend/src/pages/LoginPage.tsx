import { useState } from "react";
import PartnerStrip from "../components/Logos";
import ThemeToggle from "../components/ThemeToggle";
import { useAuth } from "../hooks/useAuth";

export default function LoginPage() {
  const { login } = useAuth();
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setBusy(true);
    try {
      await login(password);
    } catch {
      setError("auth_failed — wrong password");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="login-shell">
      <div style={{ position: "absolute", top: 16, right: 16 }}>
        <ThemeToggle />
      </div>

      <div className="login-card">
        <div className="login-tag">ai_plaza / restricted</div>
        <h1 className="login-title">ai_plaza</h1>
        <p className="login-sub">
          Imaging assistant for the Imaging Plaza catalog. Enter the shared
          passphrase to continue — sessions stay local to this instance.
        </p>
        <form className="login-form" onSubmit={onSubmit}>
          <div className="field">
            <label htmlFor="pw">passphrase</label>
            <input
              id="pw"
              type="password"
              autoFocus
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              disabled={busy}
              placeholder="••••••••"
            />
          </div>
          {error && <div className="error-banner">{error}</div>}
          <button className="btn-primary" type="submit" disabled={busy || !password}>
            {busy ? "↳ signing in…" : "↳ sign in"}
          </button>
        </form>
        <div className="login-foot">v1.0 · sse + faiss + pydantic-ai</div>
      </div>

      <div style={{ position: "absolute", bottom: 18, left: 0, right: 0 }}>
        <PartnerStrip />
      </div>
    </div>
  );
}
