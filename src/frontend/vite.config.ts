import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "node:path";

// In dev, run `npm run dev` to get a hot-reloading SPA at http://localhost:5173.
// The /api/* proxy targets the FastAPI backend on :8000 — start it with
// `ai_agent serve` from the project root.
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: process.env.VITE_API_TARGET || "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: "dist",
    sourcemap: true,
    emptyOutDir: true,
  },
  worker: {
    // parakeet.js / onnxruntime-web rely on dynamic imports which the default
    // IIFE worker format can't emit. ES modules in workers are supported by
    // every browser we care about (Chrome 80+/Safari 15+/FF 114+).
    format: "es",
  },
  optimizeDeps: {
    exclude: ["onnxruntime-web", "parakeet.js"],
  },
});
