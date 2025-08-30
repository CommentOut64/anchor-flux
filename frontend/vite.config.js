import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";
import path from "path";

const backendPort = process.env.BACKEND_PORT || 8000;
const backendHost = process.env.BACKEND_HOST || "127.0.0.1";

export default defineConfig({
  plugins: [vue()],
  resolve: {
    alias: {
      "@": path.resolve(process.cwd(), "src"),
    },
  },
  server: {
    port: 5174,
    proxy: {
      "/api": {
        target: `http://${backendHost}:${backendPort}`,
        changeOrigin: true,
        ws: true,
        secure: false,
        configure: (proxy) => {
          proxy.on("error", (err) => {
            console.error("[proxy error]", err.message);
          });
          proxy.on("proxyReq", (proxyReq, req, res) => {
            console.log(
              "[proxy request]",
              req.method,
              req.url,
              "->",
              proxyReq.path
            );
          });
        },
      },
    },
  },
});
