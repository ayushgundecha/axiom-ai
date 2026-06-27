import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The built SPA is served by the Express server at the same origin, so the app
// uses relative /api/* fetches. In `vite dev` we proxy /api to the running
// AxiomChat server (default port 3100) so the dev UI talks to real data.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: process.env.AXIOMCHAT_PROXY ?? "http://localhost:3100",
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
});
