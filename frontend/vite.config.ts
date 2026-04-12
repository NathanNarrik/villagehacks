import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";
import path from "path";
import { componentTagger } from "lovable-tagger";

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => ({
  server: {
    // `::` can fail on Windows when IPv6 is disabled or restricted; `true` uses 0.0.0.0 (still reachable on LAN).
    host: true,
    port: 8080,
    strictPort: false,
    hmr: {
      overlay: false,
    },
  },
  plugins: [
    react(),
    // Set VITE_DISABLE_LOVABLE_TAGGER=1 if this plugin causes dev-server crashes on your machine.
    mode === "development" && process.env.VITE_DISABLE_LOVABLE_TAGGER !== "1" && componentTagger(),
  ].filter(Boolean),
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
    dedupe: ["react", "react-dom", "react/jsx-runtime", "react/jsx-dev-runtime", "@tanstack/react-query", "@tanstack/query-core"],
  },
}));
