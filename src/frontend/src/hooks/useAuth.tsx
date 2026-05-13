import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import { api, ApiError } from "../lib/api";

type AuthState =
  | { kind: "loading" }
  | { kind: "anonymous"; required: boolean }
  | { kind: "authenticated" };

type AuthContextValue = {
  state: AuthState;
  refresh: () => Promise<void>;
  login: (password: string) => Promise<void>;
  logout: () => Promise<void>;
};

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<AuthState>({ kind: "loading" });

  const refresh = useCallback(async () => {
    try {
      const status = await api.authStatus();
      if (!status.required) {
        setState({ kind: "authenticated" });
        return;
      }
      try {
        await api.models();
        setState({ kind: "authenticated" });
      } catch (e) {
        if (e instanceof ApiError && e.status === 401) {
          setState({ kind: "anonymous", required: true });
        } else {
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

  const login = useCallback(async (password: string) => {
    await api.login(password);
    setState({ kind: "authenticated" });
  }, []);

  const logout = useCallback(async () => {
    try {
      await api.logout();
    } catch {
      // ignore network errors on logout
    }
    setState({ kind: "anonymous", required: true });
  }, []);

  const value = useMemo<AuthContextValue>(
    () => ({ state, refresh, login, logout }),
    [state, refresh, login, logout]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) {
    throw new Error("useAuth must be used inside <AuthProvider>");
  }
  return ctx;
}
