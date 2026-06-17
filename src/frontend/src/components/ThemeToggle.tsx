import { useTheme } from "../hooks/useTheme";

export default function ThemeToggle() {
  const { theme, toggle } = useTheme();
  return (
    <button
      type="button"
      className="theme-toggle"
      onClick={toggle}
      title={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
      aria-label="Toggle theme"
    >
      <span className="mono">{theme === "dark" ? "[ dark ]" : "[ light ]"}</span>
    </button>
  );
}
