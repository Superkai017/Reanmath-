import {defineConfig, type ProxyOptions} from 'vite';

// Dev server for the plain HTML/JS UI. API calls go to FastAPI, which must be
// running separately:  uv run uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
const backend = process.env.BACKEND_URL ?? 'http://localhost:8000';

// Scanned PDFs are OCR'd during the upload request, which can take minutes.
const LONG_REQUEST_MS = 30 * 60 * 1000;

const proxied: ProxyOptions = {
  target: backend,
  changeOrigin: true,
  timeout: LONG_REQUEST_MS,
  proxyTimeout: LONG_REQUEST_MS,
  // When FastAPI is down, answer with a clear JSON error instead of a bare 500.
  configure: (proxy) => {
    proxy.on('error', (error, _req, res) => {
      if (!('writeHead' in res) || res.headersSent) return;
      // "Connection refused" arrives as an AggregateError with an empty message.
      const reason = (error as NodeJS.ErrnoException).code || error.message || 'connection failed';
      res.writeHead(502, {'Content-Type': 'application/json'});
      res.end(
        JSON.stringify({
          detail: `Backend not reachable at ${backend} (${reason}). Start it with: uv run uvicorn src.api:app --reload --host 0.0.0.0 --port 8000`,
        }),
      );
    });
  },
};

export default defineConfig({
  // `public/` holds the old prototype; never serve it over static/.
  publicDir: false,
  server: {
    port: 3000,
    strictPort: true,
    proxy: {
      '/api': proxied,
      '/health': proxied,
      '/docs': proxied,
      '/openapi.json': proxied,
    },
  },
});
