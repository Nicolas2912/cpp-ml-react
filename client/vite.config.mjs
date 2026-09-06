import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    host: "127.0.0.1",
    strictPort: true,
    proxy: {
      "/api": { target: "http://127.0.0.1:3001" },
      "/ws": { target: "ws://127.0.0.1:3001", ws: true },
    },
  },
  build: {
    outDir: "build",
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: "./src/setupTests.js",
  },
});
