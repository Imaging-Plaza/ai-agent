import { useCallback, useEffect, useState } from "react";
import { api, ApiError } from "../lib/api";

type AuthState =
  | { kind: "loading" }
  | { kind: "anonymous"; required: boolean }
  | { kind: "authenticated" };

export function useAuth() {
  const [state, setState] = useState<AuthState>({ kind: "loading" });

  const refresh = useCallback(async () => {
    try {
      const status = await api.authStatus();
      if (!status.required) {
        setState({ kind: "authenticated" });
        return;
      }
      // Required: probe a protected endpoint to learn whether the cookie is
      // currently valid. Cheap because /api/models is small and cached.
      try {
        await api.models();
        setState({ kind: "authenticated" });
      } catch (e) {
        if (e instanceof ApiError && e.status === 401) {
          setState({ kind: "anonymous", required: true });
        } else {
          // Network/etc — treat as anonymous so the login screen shows up.
          setState({ kind: "anonymous", required: true });
        }
      }
    } catch {
      setState({ kind: "anonymous", required: true });
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const login = useCallback(
    async (password: string) => {
      await api.login(password);
      setState({ kind: "authenticated" });
    },
    [setState]
  );

  const logout = useCallback(async () => {
    try {
      await api.logout();
    } catch {
      // ignore
    }
    setState({ kind: "anonymous", required: true });
  }, [setState]);

  return { state, refresh, login, logout };
}
