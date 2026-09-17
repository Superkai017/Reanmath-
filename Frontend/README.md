# bondus — web UI

A minimalist chat UI for the Reanmath RAG backend. It is plain HTML, CSS and
JavaScript, with no build step, and is served by the FastAPI app itself.

```bash
# from the project root
uv run uvicorn src.api:app --reload
# open http://localhost:8000
```

## Development with Vite (optional)

Vite serves the UI on port 3000 with live reload, and forwards `/api`,
`/health`, `/docs` and `/openapi.json` to FastAPI on port 8000. Start both:

```bash
# terminal 1 (project root)
uv run uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
# terminal 2
cd Frontend && npm install && npm run dev
# open http://localhost:3000
```

In VS Code, run **Terminal → Run Build Task** (`Ctrl+Shift+B`) to start the
"Dev: backend + frontend" task. You can also press `F5` and choose
**FastAPI (debug) + Vite UI** to debug the backend with breakpoints. Set
`BACKEND_URL` if FastAPI runs somewhere else.

"localhost refused to connect" means that server isn't running. Check the
FastAPI terminal for errors.

| File | Role |
|---|---|
| `index.html` | Page layout (served at `/`) |
| `static/style.css` | Styles (served at `/static/style.css`) |
| `static/app.js` | Chat, uploads, document library and status (served at `/static/app.js`) |

The page talks to the backend on the same origin:

- `POST /api/query` with `{prompt, history, sources?}`
- `POST /api/ingest` (multipart `file`): PDF, PNG/JPG/WebP, Markdown or text
- `GET /api/documents` and `DELETE /api/documents/{source}`
- `GET /health`

Markdown is rendered with marked and sanitised with DOMPurify. Math is
rendered with KaTeX. All three load from jsdelivr at pinned versions.
Conversations are stored only in the browser's `localStorage`.
