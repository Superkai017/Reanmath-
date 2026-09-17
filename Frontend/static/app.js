"use strict";

/*
 * bondus — chat UI for the Reanmath RAG backend (src/api.py).
 *
 *   POST   /api/query            ask a question (grounded in ingested documents)
 *   POST   /api/ingest           upload a PDF / image / Markdown / text file
 *   GET    /api/documents        list the knowledge base
 *   DELETE /api/documents/{name} remove a document
 *   GET    /health               backend status
 *
 * Conversations are kept in this browser's localStorage only.
 */
(() => {
  const $ = (selector) => document.querySelector(selector);

  const els = {
    app: $("#app"),
    sidebar: $("#sidebar"),
    scrim: $("#scrim"),
    brand: $("#brand"),
    closeSidebar: $("#close-sidebar"),
    openSidebar: $("#open-sidebar"),
    newChat: $("#new-chat"),
    topbarNewChat: $("#topbar-new-chat"),
    openLibrary: $("#open-library"),
    docCount: $("#doc-count"),
    recents: $("#recents"),
    recentsEmpty: $("#recents-empty"),
    statusDot: $("#status-dot"),
    statusText: $("#status-text"),
    status: $("#status"),
    chatTitle: $("#chat-title"),
    thread: $("#thread"),
    messages: $("#messages"),
    composer: $("#composer"),
    prompt: $("#prompt"),
    send: $("#send"),
    attach: $("#attach"),
    attachments: $("#attachments"),
    fileInput: $("#file-input"),
    toggleSymbols: $("#toggle-symbols"),
    symbols: $("#symbols"),
    charCount: $("#char-count"),
    suggestions: $("#suggestions"),
    dropOverlay: $("#drop-overlay"),
    library: $("#library"),
    closeLibrary: $("#close-library"),
    libraryUpload: $("#library-upload"),
    librarySub: $("#library-sub"),
    docList: $("#doc-list"),
    docEmpty: $("#doc-empty"),
    toasts: $("#toasts"),
  };

  const STORAGE_KEY = "bondus.conversations.v1";
  const SIDEBAR_KEY = "bondus.sidebar";
  const MAX_CONVERSATIONS = 50;
  const MAX_HISTORY_MESSAGES = 20; // the API accepts up to 40 turns
  const MAX_TURN_CHARS = 20000;
  const MAX_PROMPT_CHARS = 4000;
  const MAX_UPLOAD_BYTES = 25 * 1024 * 1024; // server default (MAX_UPLOAD_MB)
  const UPLOAD_EXTENSIONS = [".pdf", ".md", ".markdown", ".txt", ".png", ".jpg", ".jpeg", ".webp"];
  const MOBILE = window.matchMedia("(max-width: 820px)");

  const state = {
    conversations: [],
    currentId: null,
    busy: false,
    controller: null,
    attachments: [], // {id, file, status: "uploading"|"done"|"error", result, error}
    health: null,
    maxUploadBytes: MAX_UPLOAD_BYTES,
  };

  // ----------------------------------------------------------------- utils

  const escapeHtml = (value) =>
    String(value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");

  const uid = () => `${Date.now().toString(36)}${Math.random().toString(36).slice(2, 8)}`;

  const icon = (name, extra = "") => `<svg class="icon ${extra}"><use href="#i-${name}"/></svg>`;

  const formatNumber = (value) => Number(value || 0).toLocaleString("en-US");

  const extensionOf = (name) => {
    const index = name.lastIndexOf(".");
    return index >= 0 ? name.slice(index).toLowerCase() : "";
  };

  function storageGet(key) {
    try {
      return window.localStorage.getItem(key);
    } catch {
      return null;
    }
  }

  function storageSet(key, value) {
    try {
      window.localStorage.setItem(key, value);
      return true;
    } catch {
      return false;
    }
  }

  function toast(message, kind = "") {
    const item = document.createElement("div");
    item.className = `toast ${kind}`;
    item.textContent = message;
    els.toasts.appendChild(item);
    setTimeout(() => item.remove(), kind === "error" ? 7000 : 4000);
  }

  // ------------------------------------------------------------------ math

  // Mirrors src/ingestion/latex_guard.py so the UI and the ingestion
  // pipeline agree on what counts as a formula.
  const MATH_RE = new RegExp(
    [
      String.raw`(?<!\\)\$\$((?:\\[\s\S]|[^$\\]|\$(?!\$))+?)\$\$`,
      String.raw`\\\[([\s\S]+?)\\\]`,
      String.raw`\\begin\{((?:equation|align|alignat|gather|multline|flalign|eqnarray|displaymath|cases|array|[pbvBV]?matrix)\*?)\}[\s\S]*?\\end\{\3\}`,
      String.raw`\\\(([\s\S]+?)\\\)`,
      String.raw`(?<![\\$])\$(?!\$)((?:\\[\s\S]|[^$\\\n])+?)\$(?!\d)`,
    ].join("|"),
    "g",
  );
  const PLACEHOLDER_RE = /MATHPH(\d+)XEND/g;

  const librariesReady = () =>
    typeof window.katex !== "undefined" &&
    typeof window.marked !== "undefined" &&
    typeof window.DOMPurify !== "undefined";

  function renderMath({ tex, display }) {
    try {
      const html = window.katex.renderToString(tex, {
        displayMode: display,
        throwOnError: false,
        strict: "ignore",
        trust: false,
        output: "htmlAndMathml",
      });
      return display ? `<span class="math-display">${html}</span>` : html;
    } catch (error) {
      return `<span class="math-error" title="${escapeHtml(error.message)}">${escapeHtml(tex)}</span>`;
    }
  }

  // Markdown with LaTeX: formulas are pulled out before Markdown parsing (so
  // `_`, `*` and `\\` inside them survive), the HTML is sanitised, and KaTeX
  // output is spliced back in. With `figures`, ```geogebra blocks become
  // placeholders that mountFigures() turns into interactive graphs.
  function renderRich(text, { figures = false } = {}) {
    if (!librariesReady()) {
      return `<p style="white-space: pre-wrap">${escapeHtml(text)}</p>`;
    }
    const blocks = [];
    let source = String(text);
    if (figures) {
      source = source.replace(FIGURE_RE, (_, kind, body) => {
        blocks.push({ is3d: kind.toLowerCase() === "geogebra-3d", commands: body });
        return `\n\nGGBFIG${blocks.length - 1}XEND\n\n`;
      });
    }
    const formulas = [];
    const masked = source.replace(MATH_RE, (match, display, bracket, env, paren, inline) => {
      let formula;
      if (display !== undefined) formula = { tex: display, display: true };
      else if (bracket !== undefined) formula = { tex: bracket, display: true };
      else if (env !== undefined) formula = { tex: match, display: true };
      else if (paren !== undefined) formula = { tex: paren, display: false };
      else formula = { tex: inline, display: false };
      formulas.push(formula);
      return `MATHPH${formulas.length - 1}XEND`;
    });
    const html = window.DOMPurify.sanitize(window.marked.parse(masked, { gfm: true, breaks: true }));
    return html
      .replace(PLACEHOLDER_RE, (_, index) => {
        const formula = formulas[Number(index)];
        return formula ? renderMath(formula) : "";
      })
      .replace(FIGURE_PLACEHOLDER_RE, (_, index) => {
        const block = blocks[Number(index)];
        if (!block) return "";
        // Added after sanitising; the commands are only ever read back as text.
        const data = escapeHtml(JSON.stringify(block));
        return `<div class="ggb-figure" data-figure="${data}"></div>`;
      });
  }

  // ------------------------------------------------------ GeoGebra figures

  // A fenced block with info string `geogebra` (2D) or `geogebra-3d`.
  const FIGURE_RE = /^[ \t]*```[ \t]*(geogebra(?:-3d)?)[ \t]*\r?\n([\s\S]*?)^[ \t]*```[ \t]*$/gim;
  const FIGURE_PLACEHOLDER_RE = /(?:<p>)?GGBFIG(\d+)XEND(?:<\/p>)?/g;
  const GGB_SCRIPT = "https://www.geogebra.org/apps/deployggb.js";
  const FIGURE_WIDTH = 720;
  const FIGURE_HEIGHT = 420;
  // Keep x and y at the same scale (circles stay round) unless that would
  // stretch a range by more than this factor.
  const MAX_EQUAL_AXES_STRETCH = 3;
  const MAX_FIGURE_LINES = 40;
  const MAX_FIGURE_LINE_CHARS = 400;
  // GeoGebra script commands can run other commands or play media; never
  // execute them from model output.
  const BLOCKED_COMMANDS =
    /\b(Execute|SetClickScript|SetUpdateScript|RunClickScript|RunUpdateScript|PlaySound|ReadText|UpdateConstruction)\s*[(\[]/i;

  let ggbLoader = null;
  let figureCounter = 0;
  const mountedApplets = new Map(); // element id -> GeoGebra api

  function loadGeoGebra() {
    if (window.GGBApplet) return Promise.resolve();
    if (!ggbLoader) {
      ggbLoader = new Promise((resolve, reject) => {
        const script = document.createElement("script");
        script.src = GGB_SCRIPT;
        script.async = true;
        script.onload = () => (window.GGBApplet ? resolve() : reject(new Error("GeoGebra did not load")));
        script.onerror = () => {
          ggbLoader = null; // allow a retry later
          script.remove();
          reject(new Error("Could not load GeoGebra (are you offline?)"));
        };
        document.head.appendChild(script);
      });
    }
    return ggbLoader;
  }

  function parseFigureCommands(body) {
    const commands = [];
    const skipped = [];
    for (const raw of String(body).split(/\r?\n/)) {
      const line = raw.trim();
      if (!line || line.startsWith("#") || line.startsWith("//")) continue;
      if (BLOCKED_COMMANDS.test(line) || line.length > MAX_FIGURE_LINE_CHARS) {
        skipped.push(line);
      } else if (commands.length < MAX_FIGURE_LINES) {
        commands.push(line);
      } else {
        skipped.push(line);
      }
    }
    return { commands, skipped };
  }

  // ZoomIn(xmin, ymin, xmax, ymax) or the 3D form with six numbers. The
  // command zooms but reports failure, so the view is set through the API.
  const ZOOM_RE = /^ZoomIn\s*\(([^()]*)\)\s*;?$/i;

  function applyZoom(api, command, is3d) {
    const match = ZOOM_RE.exec(command);
    if (!match) return null;
    const values = match[1].split(",").map((part) => Number(part.trim()));
    if (values.some((value) => !Number.isFinite(value))) return null;
    try {
      if (values.length === 4) {
        let [xmin, ymin, xmax, ymax] = values;
        if (xmin >= xmax || ymin >= ymax) return false;
        const viewRatio = FIGURE_WIDTH / FIGURE_HEIGHT;
        const width = xmax - xmin;
        const height = ymax - ymin;
        const ratio = width / height;
        // How much one range must grow for x and y to share a scale.
        const stretch = ratio < viewRatio ? viewRatio / ratio : ratio / viewRatio;
        if (stretch <= MAX_EQUAL_AXES_STRETCH) {
          if (ratio < viewRatio) {
            const extra = (height * viewRatio - width) / 2;
            xmin -= extra;
            xmax += extra;
          } else {
            const extra = (width / viewRatio - height) / 2;
            ymin -= extra;
            ymax += extra;
          }
        }
        api.setCoordSystem(xmin, xmax, ymin, ymax);
        return true;
      }
      if (values.length === 6 && is3d) {
        const [xmin, ymin, zmin, xmax, ymax, zmax] = values;
        if (xmin >= xmax || ymin >= ymax || zmin >= zmax) return false;
        api.setCoordSystem(xmin, xmax, ymin, ymax, zmin, zmax, false);
        return true;
      }
    } catch {
      return false;
    }
    return null;
  }

  function figureFallback(figure, block, message) {
    figure.classList.add("failed");
    figure.replaceChildren();
    const note = document.createElement("div");
    note.className = "ggb-note";
    note.textContent = message;
    const pre = document.createElement("pre");
    const code = document.createElement("code");
    code.textContent = block.commands;
    pre.appendChild(code);
    figure.append(note, pre);
  }

  function downloadPng(api) {
    try {
      const base64 = api.getPNGBase64(2, false, 144);
      const link = document.createElement("a");
      link.href = `data:image/png;base64,${base64}`;
      link.download = "bondus-graph.png";
      document.body.appendChild(link);
      link.click();
      link.remove();
    } catch {
      toast("Could not export the graph as PNG.", "error");
    }
  }

  async function mountFigure(figure) {
    let block;
    try {
      block = JSON.parse(figure.dataset.figure || "{}");
    } catch {
      return;
    }
    const { commands, skipped } = parseFigureCommands(block.commands);
    if (!commands.length) {
      figureFallback(figure, block, "This figure has no drawable commands.");
      return;
    }

    figure.innerHTML = `
      <div class="ggb-frame"><div class="ggb-loading"><span class="spinner"></span>Drawing graph…</div></div>
      <div class="ggb-bar">
        <span class="ggb-label">${block.is3d ? "3D graph" : "Graph"} · GeoGebra</span>
        <span class="ggb-status"></span>
        <button type="button" class="ggb-download" disabled>${icon("download")}<span>PNG</span></button>
      </div>`;
    const frame = figure.querySelector(".ggb-frame");
    const status = figure.querySelector(".ggb-status");
    const download = figure.querySelector(".ggb-download");

    try {
      await loadGeoGebra();
    } catch (error) {
      figureFallback(figure, block, `${error.message} Showing the GeoGebra commands instead.`);
      return;
    }
    if (!figure.isConnected) return;

    figureCounter += 1;
    const id = `ggb-${figureCounter}`;
    const host = document.createElement("div");
    host.id = id;
    host.className = "ggb-host";
    frame.appendChild(host);

    const params = {
      id,
      // The classic app honours `perspective`: graphics view only ("G" 2D,
      // "T" 3D), since the equations are already in the answer text.
      appName: "classic",
      perspective: block.is3d ? "T" : "G",
      width: FIGURE_WIDTH,
      height: FIGURE_HEIGHT,
      scaleContainerClass: "ggb-frame",
      allowUpscale: false,
      showToolBar: false,
      showMenuBar: false,
      showAlgebraInput: false,
      showResetIcon: true,
      showZoomButtons: true,
      showFullscreenButton: true,
      enableRightClick: false,
      enableLabelDrags: false,
      enableShiftDragZoom: true,
      enableUndoRedo: false,
      errorDialogsActive: false,
      disableJavaScript: true,
      useBrowserForJS: true,
      preventFocus: true,
      borderColor: "#FFFFFF",
      appletOnLoad: (api) => {
        mountedApplets.set(id, api);
        frame.querySelector(".ggb-loading")?.remove();
        if (!figure.isConnected) {
          unmountApplet(id);
          return;
        }
        try {
          api.setErrorDialogsActive(false);
        } catch {
          /* older API */
        }
        const failed = [...skipped];
        for (const command of commands) {
          let ok = applyZoom(api, command, block.is3d);
          if (ok === null) {
            try {
              ok = api.evalCommand(command);
            } catch {
              ok = false;
            }
          }
          if (!ok) failed.push(command);
        }
        if (failed.length) {
          status.textContent = `${failed.length} command${failed.length === 1 ? "" : "s"} skipped`;
          status.title = failed.join("\n");
        }
        download.disabled = false;
        download.addEventListener("click", () => downloadPng(api));
      },
    };

    try {
      new window.GGBApplet(params, true).inject(id);
    } catch (error) {
      figureFallback(figure, block, `GeoGebra failed to start (${error.message}).`);
    }
  }

  // Graphs load only when scrolled into view.
  const figureObserver =
    "IntersectionObserver" in window
      ? new IntersectionObserver(
          (entries) => {
            for (const entry of entries) {
              if (entry.isIntersecting) {
                figureObserver.unobserve(entry.target);
                mountFigure(entry.target);
              }
            }
          },
          { rootMargin: "200px 0px" },
        )
      : null;

  // Call after the figures are attached to the document.
  function mountFigures(root) {
    root.querySelectorAll(".ggb-figure[data-figure]:not([data-queued])").forEach((figure) => {
      figure.dataset.queued = "1";
      if (figureObserver) figureObserver.observe(figure);
      else mountFigure(figure);
    });
  }

  function unmountApplet(id) {
    const api = mountedApplets.get(id);
    mountedApplets.delete(id);
    try {
      api?.remove();
    } catch {
      /* already gone */
    }
  }

  function unmountFigures(root) {
    root.querySelectorAll(".ggb-figure").forEach((figure) => {
      if (figureObserver) figureObserver.unobserve(figure);
      figure.querySelectorAll(".ggb-host").forEach((host) => unmountApplet(host.id));
    });
  }

  // ------------------------------------------------------------------- api

  class ApiError extends Error {
    constructor(message, status) {
      super(message);
      this.status = status;
    }
  }

  function formatError(body, status) {
    if (body && typeof body === "object" && "detail" in body) {
      const { detail } = body;
      if (Array.isArray(detail)) {
        return detail
          .map((item) => {
            const where = (item.loc || []).filter((part) => part !== "body").join(".");
            return where ? `${where}: ${item.msg}` : item.msg;
          })
          .join("; ");
      }
      return String(detail);
    }
    if (typeof body === "string" && body.trim() && body.length < 300) return body.trim();
    if (status === 502 || status === 503 || status === 504 || status === 500) {
      return `The backend did not respond (HTTP ${status}). Is FastAPI running on port 8000?`;
    }
    return `Request failed (HTTP ${status})`;
  }

  async function api(path, options = {}) {
    let response;
    try {
      response = await fetch(path, options);
    } catch (error) {
      if (error.name === "AbortError") throw error;
      throw new ApiError("Cannot reach the server. Is the backend running?", 0);
    }
    const type = response.headers.get("content-type") || "";
    let body;
    try {
      body = type.includes("application/json") ? await response.json() : await response.text();
    } catch (error) {
      if (error.name === "AbortError") throw error;
      body = null;
    }
    if (!response.ok) throw new ApiError(formatError(body, response.status), response.status);
    return body;
  }

  // --------------------------------------------------------- conversations

  function loadConversations() {
    try {
      const parsed = JSON.parse(storageGet(STORAGE_KEY) || "[]");
      if (Array.isArray(parsed)) {
        state.conversations = parsed.filter(
          (conv) => conv && typeof conv.id === "string" && Array.isArray(conv.messages),
        );
      }
    } catch {
      state.conversations = [];
    }
  }

  function saveConversations() {
    state.conversations.sort((a, b) => (b.updatedAt || 0) - (a.updatedAt || 0));
    state.conversations = state.conversations.slice(0, MAX_CONVERSATIONS);
    if (!storageSet(STORAGE_KEY, JSON.stringify(state.conversations))) {
      // Storage full: drop the oldest conversations and try once more.
      state.conversations = state.conversations.slice(0, Math.ceil(state.conversations.length / 2));
      storageSet(STORAGE_KEY, JSON.stringify(state.conversations));
    }
  }

  const currentConversation = () =>
    state.conversations.find((conv) => conv.id === state.currentId) || null;

  function makeTitle(text) {
    const flat = text.replace(/\s+/g, " ").trim();
    return flat.length > 60 ? `${flat.slice(0, 57)}…` : flat;
  }

  function ensureConversation(firstPrompt) {
    let conv = currentConversation();
    if (!conv) {
      conv = { id: uid(), title: makeTitle(firstPrompt), messages: [], updatedAt: Date.now() };
      state.conversations.unshift(conv);
      state.currentId = conv.id;
    }
    return conv;
  }

  function renderRecents() {
    els.recents.replaceChildren();
    const sorted = [...state.conversations].sort((a, b) => (b.updatedAt || 0) - (a.updatedAt || 0));
    for (const conv of sorted) {
      const item = document.createElement("li");
      item.className = `recent${conv.id === state.currentId ? " active" : ""}`;
      item.innerHTML = `
        <button class="recent-open" type="button"></button>
        <button class="recent-delete" type="button" aria-label="Delete conversation" title="Delete">${icon("trash")}</button>`;
      const open = item.querySelector(".recent-open");
      open.textContent = conv.title || "Untitled";
      open.title = conv.title || "";
      open.addEventListener("click", () => openConversation(conv.id));
      item.querySelector(".recent-delete").addEventListener("click", () => deleteConversation(conv.id));
      els.recents.appendChild(item);
    }
    els.recentsEmpty.hidden = sorted.length > 0;
  }

  function renderConversation() {
    const conv = currentConversation();
    unmountFigures(els.messages);
    els.messages.replaceChildren();
    const messages = conv ? conv.messages : [];
    messages.forEach((message, index) => {
      els.messages.appendChild(
        message.role === "user"
          ? buildUserMessage(message)
          : buildAssistantMessage(message, index === messages.length - 1),
      );
    });
    mountFigures(els.messages);
    showConversationChrome(conv);
  }

  function showConversationChrome(conv) {
    els.chatTitle.textContent = conv ? conv.title : "";
    document.title = conv ? `${conv.title} · bondus` : "bondus · Khmer Math Tutor";
    setEmpty(!conv || conv.messages.length === 0);
    renderRecents();
    scrollToBottom(true);
  }

  function setEmpty(empty) {
    els.app.classList.toggle("is-empty", empty);
  }

  function openConversation(id) {
    if (state.busy) stopGenerating();
    state.currentId = id;
    renderConversation();
    closeSidebarOnMobile();
    els.prompt.focus();
  }

  function newChat() {
    if (state.busy) stopGenerating();
    state.currentId = null;
    clearFinishedAttachments();
    renderConversation();
    closeSidebarOnMobile();
    els.prompt.focus();
  }

  function deleteConversation(id) {
    const conv = state.conversations.find((item) => item.id === id);
    if (!conv || !window.confirm(`Delete "${conv.title}"?`)) return;
    if (id === state.currentId && state.busy) stopGenerating();
    state.conversations = state.conversations.filter((item) => item.id !== id);
    saveConversations();
    if (id === state.currentId) {
      state.currentId = null;
      renderConversation();
    } else {
      renderRecents();
    }
  }

  // --------------------------------------------------------------- messages

  function isNearBottom() {
    const { scrollTop, scrollHeight, clientHeight } = els.thread;
    return scrollHeight - scrollTop - clientHeight < 120;
  }

  function scrollToBottom(force = false) {
    if (force || isNearBottom()) els.thread.scrollTop = els.thread.scrollHeight;
  }

  function buildUserMessage(message) {
    const node = document.createElement("div");
    node.className = "msg msg-user";
    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = message.content;
    node.appendChild(bubble);
    if (message.scope && message.scope.length) {
      const scope = document.createElement("div");
      scope.className = "msg-meta";
      scope.style.marginTop = "6px";
      scope.textContent = `Searching only: ${message.scope.join(", ")}`;
      node.appendChild(scope);
    }
    return node;
  }

  function describeModel(message) {
    if (message.provider === "none") return "Passages only · no LLM configured";
    const parts = [];
    if (message.model) parts.push(message.model);
    if (typeof message.latency_ms === "number") parts.push(`${(message.latency_ms / 1000).toFixed(1)}s`);
    return parts.join(" · ");
  }

  function buildSources(sources) {
    const list = document.createElement("div");
    list.className = "sources";
    list.hidden = true;
    sources.forEach((source, index) => {
      const card = document.createElement("div");
      card.className = "source";
      const page = source.page != null ? ` · p.${source.page}` : "";
      card.innerHTML = `
        <div class="source-head">
          <span class="source-index">[${index + 1}]</span>
          <span class="source-name"></span>
          <span class="source-meta">${Math.round((source.score || 0) * 100)}%${page}</span>
        </div>
        <div class="source-text content"></div>`;
      const name = card.querySelector(".source-name");
      name.textContent = source.source;
      name.title = source.source;
      const text = card.querySelector(".source-text");
      text.innerHTML = renderRich(source.text || "");
      if ((source.text || "").length > 360) {
        const more = document.createElement("button");
        more.type = "button";
        more.className = "source-more";
        more.textContent = "Show more";
        more.addEventListener("click", () => {
          const expanded = text.classList.toggle("expanded");
          more.textContent = expanded ? "Show less" : "Show more";
        });
        card.appendChild(more);
      }
      list.appendChild(card);
    });
    return list;
  }

  function buildAssistantMessage(message, isLast) {
    const node = document.createElement("div");
    node.className = "msg msg-assistant";

    const content = document.createElement("div");
    content.className = `content${message.error ? " error" : ""}`;
    if (message.error) {
      content.textContent = message.error;
    } else {
      content.innerHTML = renderRich(message.content || "_(empty answer)_", { figures: true });
    }
    node.appendChild(content);

    if (!message.error && message.grounded === false) {
      const notice = document.createElement("div");
      notice.className = "notice";
      notice.textContent =
        "No matching passage in your documents — this answer is not grounded in the curriculum.";
      node.appendChild(notice);
    }
    if (message.stop_reason === "max_tokens" || message.stop_reason === "MAX_TOKENS") {
      const notice = document.createElement("div");
      notice.className = "notice";
      notice.textContent = "The answer was cut off at the length limit.";
      node.appendChild(notice);
    }

    const actions = document.createElement("div");
    actions.className = "msg-actions";

    if (!message.error) {
      const copy = document.createElement("button");
      copy.type = "button";
      copy.className = "icon-btn";
      copy.title = "Copy";
      copy.setAttribute("aria-label", "Copy answer");
      copy.innerHTML = icon("copy");
      copy.addEventListener("click", async () => {
        try {
          await navigator.clipboard.writeText(message.content || "");
          copy.innerHTML = icon("check");
          setTimeout(() => (copy.innerHTML = icon("copy")), 1500);
        } catch {
          toast("Copy failed — your browser blocked clipboard access.", "error");
        }
      });
      actions.appendChild(copy);
    }

    if (isLast) {
      const retry = document.createElement("button");
      retry.type = "button";
      retry.className = "icon-btn regen";
      retry.title = "Regenerate";
      retry.setAttribute("aria-label", "Regenerate answer");
      retry.innerHTML = icon("refresh");
      retry.addEventListener("click", regenerate);
      actions.appendChild(retry);
    }

    let sourcesList = null;
    const sources = Array.isArray(message.sources) ? message.sources : [];
    if (sources.length) {
      const toggle = document.createElement("button");
      toggle.type = "button";
      toggle.className = "sources-toggle";
      toggle.setAttribute("aria-expanded", "false");
      toggle.innerHTML = `${sources.length} source${sources.length === 1 ? "" : "s"} ${icon("chevron")}`;
      sourcesList = buildSources(sources);
      toggle.addEventListener("click", () => {
        const open = sourcesList.hidden;
        sourcesList.hidden = !open;
        toggle.setAttribute("aria-expanded", String(open));
      });
      actions.appendChild(toggle);
    }

    const meta = message.error ? "" : describeModel(message);
    if (meta) {
      const span = document.createElement("span");
      span.className = "msg-meta";
      span.textContent = meta;
      actions.appendChild(span);
    }

    if (actions.childElementCount) node.appendChild(actions);
    if (sourcesList) node.appendChild(sourcesList);
    return node;
  }

  function buildThinking() {
    const node = document.createElement("div");
    node.className = "msg msg-assistant";
    node.innerHTML = `<div class="thinking"><span class="dots"><span></span><span></span><span></span></span><span>Thinking…</span></div>`;
    return node;
  }

  // ------------------------------------------------------------------- chat

  // Completed question/answer pairs only, so roles always alternate.
  function historyFor(messages) {
    const turns = [];
    for (let i = 0; i + 1 < messages.length; i += 1) {
      const question = messages[i];
      const answer = messages[i + 1];
      if (
        question.role === "user" &&
        answer.role === "assistant" &&
        !answer.error &&
        (question.content || "").trim() &&
        (answer.content || "").trim()
      ) {
        turns.push(
          { role: "user", content: question.content.slice(0, MAX_TURN_CHARS) },
          { role: "assistant", content: answer.content.slice(0, MAX_TURN_CHARS) },
        );
        i += 1;
      }
    }
    return turns.slice(-MAX_HISTORY_MESSAGES);
  }

  function setBusy(busy) {
    state.busy = busy;
    els.send.innerHTML = icon(busy ? "stop" : "arrow-up");
    els.send.title = busy ? "Stop" : "Send (Enter)";
    els.send.setAttribute("aria-label", busy ? "Stop generating" : "Send");
    updateComposer();
  }

  async function ask(conv, prompt, scope) {
    const history = historyFor(conv.messages);
    const question = { role: "user", content: prompt, scope: scope.length ? scope : undefined };
    conv.messages.push(question);
    conv.updatedAt = Date.now();
    saveConversations();
    // Append instead of re-rendering, so graphs in earlier answers stay loaded.
    els.messages.querySelectorAll(".regen").forEach((button) => button.remove());
    els.messages.appendChild(buildUserMessage(question));
    showConversationChrome(conv);

    const thinking = buildThinking();
    els.messages.appendChild(thinking);
    scrollToBottom(true);

    const controller = new AbortController();
    state.controller = controller;
    setBusy(true);

    const payload = { prompt, history };
    if (scope.length) payload.sources = scope;

    let reply;
    try {
      const result = await api("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        signal: controller.signal,
      });
      reply = {
        role: "assistant",
        content: result.answer || "",
        sources: result.sources || [],
        grounded: result.grounded,
        provider: result.provider,
        model: result.model,
        stop_reason: result.stop_reason,
        latency_ms: result.latency_ms,
      };
    } catch (error) {
      if (error.name === "AbortError") {
        reply = { role: "assistant", content: "", error: "Stopped." };
      } else {
        reply = { role: "assistant", content: "", error: error.message || "Something went wrong." };
      }
    } finally {
      if (state.controller === controller) state.controller = null;
    }

    thinking.remove();
    // The user may have switched or deleted the conversation meanwhile.
    if (!state.conversations.includes(conv)) {
      setBusy(false);
      return;
    }
    conv.messages.push(reply);
    conv.updatedAt = Date.now();
    saveConversations();
    setBusy(false);
    if (conv.id === state.currentId) {
      els.messages.appendChild(buildAssistantMessage(reply, true));
      mountFigures(els.messages);
      scrollToBottom();
    }
    renderRecents();
  }

  async function submit(text) {
    const prompt = (text ?? els.prompt.value).trim();
    if (!prompt || state.busy) return;
    if (prompt.length > MAX_PROMPT_CHARS) {
      toast(`Questions are limited to ${MAX_PROMPT_CHARS} characters.`, "error");
      return;
    }
    if (state.attachments.some((item) => item.status === "uploading")) {
      toast("Please wait until your upload finishes.");
      return;
    }
    const scope = state.attachments
      .filter((item) => item.status === "done")
      .map((item) => item.result.source);
    clearFinishedAttachments();
    els.prompt.value = "";
    autoResize();
    const conv = ensureConversation(prompt);
    await ask(conv, prompt, scope);
  }

  async function regenerate() {
    const conv = currentConversation();
    if (!conv || state.busy) return;
    // Drop the last answer and ask the last question again.
    while (conv.messages.length && conv.messages[conv.messages.length - 1].role === "assistant") {
      conv.messages.pop();
    }
    const last = conv.messages.pop();
    renderConversation();
    if (!last) return;
    await ask(conv, last.content, last.scope || []);
  }

  function stopGenerating() {
    if (state.controller) state.controller.abort();
  }

  // --------------------------------------------------------------- composer

  function autoResize() {
    const input = els.prompt;
    input.style.height = "auto";
    input.style.height = `${Math.min(input.scrollHeight, 220)}px`;
    updateComposer();
  }

  function updateComposer() {
    const length = els.prompt.value.trim().length;
    els.send.disabled = state.busy ? false : length === 0;
    const showCount = els.prompt.value.length > MAX_PROMPT_CHARS * 0.8;
    els.charCount.hidden = !showCount;
    if (showCount) {
      els.charCount.textContent = `${els.prompt.value.length}/${MAX_PROMPT_CHARS}`;
      els.charCount.classList.toggle("over", els.prompt.value.length >= MAX_PROMPT_CHARS);
    }
  }

  function insertAtCursor(text) {
    const input = els.prompt;
    const start = input.selectionStart ?? input.value.length;
    const end = input.selectionEnd ?? input.value.length;
    input.value = input.value.slice(0, start) + text + input.value.slice(end);
    const caret = start + text.length;
    input.setSelectionRange(caret, caret);
    input.focus();
    autoResize();
  }

  // ---------------------------------------------------------------- uploads

  function validateFile(file) {
    if (!UPLOAD_EXTENSIONS.includes(extensionOf(file.name))) {
      return `Unsupported file type. Use ${UPLOAD_EXTENSIONS.join(", ")}.`;
    }
    if (file.size === 0) return "The file is empty.";
    if (file.size > state.maxUploadBytes) {
      return `The file is larger than ${Math.round(state.maxUploadBytes / (1024 * 1024))} MB.`;
    }
    return null;
  }

  function describeIngest(result) {
    const parts = [`${formatNumber(result.chunks_added)} chunks`];
    if (result.pages) parts.push(`${result.pages} page${result.pages === 1 ? "" : "s"}`);
    if (result.ocr_pages) parts.push(`${result.ocr_pages} OCR`);
    if (result.formulas_protected) parts.push(`${formatNumber(result.formulas_protected)} formulas`);
    return parts.join(" · ");
  }

  async function ingestFile(file) {
    const form = new FormData();
    form.append("file", file, file.name);
    form.append("replace", "true");
    const result = await api("/api/ingest", { method: "POST", body: form });
    for (const warning of result.warnings || []) toast(`${result.source}: ${warning}`);
    loadHealth();
    if (els.library.open) loadDocuments();
    return result;
  }

  function renderAttachments() {
    els.attachments.replaceChildren();
    for (const item of state.attachments) {
      const chip = document.createElement("div");
      chip.className = `attachment ${item.status}`;
      const lead = item.status === "uploading" ? `<span class="spinner"></span>` : icon("file");
      let meta;
      if (item.status === "uploading") meta = "Indexing… scanned pages may take a few minutes";
      else if (item.status === "done") meta = `Indexed · ${describeIngest(item.result)}`;
      else meta = item.error;
      chip.innerHTML = `
        ${lead}
        <div class="attachment-body">
          <span class="attachment-name"></span>
          <span class="attachment-meta"></span>
        </div>
        <button class="attachment-close" type="button" aria-label="Remove">${icon("x")}</button>`;
      chip.querySelector(".attachment-name").textContent = item.file.name;
      chip.querySelector(".attachment-meta").textContent = meta;
      chip.title =
        item.status === "done"
          ? "Your next question will search only this document. Remove to search everything."
          : item.file.name;
      const close = chip.querySelector(".attachment-close");
      close.hidden = item.status === "uploading";
      close.addEventListener("click", () => {
        state.attachments = state.attachments.filter((other) => other !== item);
        renderAttachments();
      });
      els.attachments.appendChild(chip);
    }
  }

  function clearFinishedAttachments() {
    state.attachments = state.attachments.filter((item) => item.status === "uploading");
    renderAttachments();
  }

  async function uploadToComposer(files) {
    for (const file of files) {
      const problem = validateFile(file);
      if (problem) {
        toast(`${file.name}: ${problem}`, "error");
        continue;
      }
      const item = { id: uid(), file, status: "uploading", result: null, error: "" };
      state.attachments.push(item);
      renderAttachments();
      ingestFile(file)
        .then((result) => {
          item.status = "done";
          item.result = result;
        })
        .catch((error) => {
          item.status = "error";
          item.error = error.message;
        })
        .finally(() => {
          if (state.attachments.includes(item)) renderAttachments();
          updateComposer();
        });
    }
  }

  // ---------------------------------------------------------------- library

  const pendingLibraryUploads = new Set();

  function formatDate(value) {
    const date = new Date(value);
    return Number.isNaN(date.getTime())
      ? ""
      : date.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
  }

  function renderDocuments(documents) {
    els.docList.replaceChildren();
    for (const name of pendingLibraryUploads) {
      const row = document.createElement("li");
      row.className = "doc";
      row.innerHTML = `<span class="spinner"></span><div class="doc-body"><div class="doc-name"></div><div class="doc-meta">Indexing…</div></div>`;
      row.querySelector(".doc-name").textContent = name;
      els.docList.appendChild(row);
    }
    for (const doc of documents) {
      const row = document.createElement("li");
      row.className = "doc";
      row.innerHTML = `
        ${icon("file")}
        <div class="doc-body">
          <div class="doc-name"></div>
          <div class="doc-meta"></div>
        </div>
        <button class="icon-btn" type="button" title="Remove from knowledge base" aria-label="Delete document">${icon("trash")}</button>`;
      const name = row.querySelector(".doc-name");
      name.textContent = doc.source;
      name.title = doc.source;
      const meta = [doc.format.toUpperCase(), `${formatNumber(doc.chunks)} chunks`];
      if (doc.pages) meta.push(`${doc.pages} page${doc.pages === 1 ? "" : "s"}`);
      const date = formatDate(doc.ingested_at);
      if (date) meta.push(date);
      row.querySelector(".doc-meta").textContent = meta.join(" · ");
      row.querySelector("button").addEventListener("click", () => deleteDocument(doc.source));
      els.docList.appendChild(row);
    }
    els.docEmpty.hidden = documents.length > 0 || pendingLibraryUploads.size > 0;
  }

  async function loadDocuments() {
    try {
      const result = await api("/api/documents");
      renderDocuments(result.documents || []);
      els.librarySub.textContent = `${formatNumber((result.documents || []).length)} documents · ${formatNumber(result.total_chunks)} chunks`;
    } catch (error) {
      els.librarySub.textContent = error.message;
    }
  }

  async function deleteDocument(source) {
    if (!window.confirm(`Remove "${source}" from the knowledge base?`)) return;
    const path = source.split("/").map(encodeURIComponent).join("/");
    try {
      const result = await api(`/api/documents/${path}`, { method: "DELETE" });
      toast(`Removed ${result.source} (${formatNumber(result.chunks_removed)} chunks)`);
    } catch (error) {
      toast(error.message, "error");
    }
    loadDocuments();
    loadHealth();
  }

  function uploadToLibrary(files) {
    for (const file of files) {
      const problem = validateFile(file);
      if (problem) {
        toast(`${file.name}: ${problem}`, "error");
        continue;
      }
      pendingLibraryUploads.add(file.name);
      loadDocuments();
      ingestFile(file)
        .then((result) => toast(`Indexed ${result.source} · ${describeIngest(result)}`))
        .catch((error) => toast(`${file.name}: ${error.message}`, "error"))
        .finally(() => {
          pendingLibraryUploads.delete(file.name);
          if (els.library.open) loadDocuments();
        });
    }
  }

  function openLibrary() {
    closeSidebarOnMobile();
    if (typeof els.library.showModal === "function") els.library.showModal();
    else els.library.setAttribute("open", "");
    loadDocuments();
  }

  function closeLibrary() {
    if (typeof els.library.close === "function") els.library.close();
    else els.library.removeAttribute("open");
  }

  // ----------------------------------------------------------------- health

  let healthTimer = null;

  async function loadHealth() {
    clearTimeout(healthTimer);
    try {
      const health = await api("/health");
      state.health = health;
      if (health.max_upload_mb) state.maxUploadBytes = health.max_upload_mb * 1024 * 1024;
      const providerLabel =
        health.llm_provider === "none"
          ? "Passages only"
          : `${health.llm_provider === "anthropic" ? "Claude" : "Gemini"}`;
      els.statusText.textContent = `${providerLabel} · ${formatNumber(health.chunks)} chunks`;
      els.statusDot.className = `status-dot ${health.llm_provider === "none" ? "warn" : "ok"}`;
      const ocr = health.ocr_enabled ? `${health.ocr_engine || "on"} (${health.ocr_model})` : "off";
      els.status.title = [
        `Model: ${health.llm_model || "none (set ANTHROPIC_API_KEY or GEMINI_API_KEY)"}`,
        `Embeddings: ${health.embedding_model}`,
        `OCR: ${ocr}`,
        `Documents: ${health.documents}`,
        `Version: ${health.version}`,
      ].join("\n");
      els.docCount.hidden = !health.documents;
      els.docCount.textContent = formatNumber(health.documents);
      els.libraryUpload.querySelector(".dropzone-hint").textContent = health.ocr_enabled
        ? `PDF, images, Markdown or text. Scanned pages are read with ${health.ocr_engine === "kiri" ? "Kiri OCR (Khmer words only)" : "Gemini OCR"}.`
        : "PDF, Markdown or text. OCR is off, so scanned pages and images cannot be read.";
    } catch {
      state.health = null;
      els.statusText.textContent = "Backend offline";
      els.statusDot.className = "status-dot down";
      els.status.title = "Start it with: uv run uvicorn src.api:app --reload";
      healthTimer = setTimeout(loadHealth, 10000);
    }
  }

  // ---------------------------------------------------------------- sidebar

  function setSidebar(open, remember = true) {
    els.app.classList.toggle("sidebar-closed", !open);
    els.scrim.hidden = !open || !MOBILE.matches;
    if (remember && !MOBILE.matches) storageSet(SIDEBAR_KEY, open ? "open" : "closed");
  }

  function closeSidebarOnMobile() {
    if (MOBILE.matches) setSidebar(false, false);
  }

  // ----------------------------------------------------------------- events

  function hasFiles(event) {
    return Array.from(event.dataTransfer?.types || []).includes("Files");
  }

  function bindEvents() {
    els.composer.addEventListener("submit", (event) => {
      event.preventDefault();
      if (state.busy) stopGenerating();
      else submit();
    });

    els.prompt.addEventListener("input", autoResize);
    els.prompt.addEventListener("keydown", (event) => {
      // isComposing: do not send while a Khmer/IME composition is active.
      if (event.key === "Enter" && !event.shiftKey && !event.isComposing && event.keyCode !== 229) {
        event.preventDefault();
        if (!state.busy) submit();
      }
    });

    els.suggestions.addEventListener("click", (event) => {
      const chip = event.target.closest("[data-prompt]");
      if (chip) submit(chip.dataset.prompt);
    });

    els.toggleSymbols.addEventListener("click", () => {
      const open = els.symbols.hidden;
      els.symbols.hidden = !open;
      els.toggleSymbols.setAttribute("aria-pressed", String(open));
    });
    els.symbols.addEventListener("click", (event) => {
      const button = event.target.closest("[data-insert]");
      if (button) insertAtCursor(button.dataset.insert);
    });

    els.attach.addEventListener("click", () => {
      els.fileInput.dataset.target = "composer";
      els.fileInput.click();
    });
    els.libraryUpload.addEventListener("click", () => {
      els.fileInput.dataset.target = "library";
      els.fileInput.click();
    });
    els.fileInput.addEventListener("change", () => {
      const files = Array.from(els.fileInput.files || []);
      if (files.length) {
        if (els.fileInput.dataset.target === "library") uploadToLibrary(files);
        else uploadToComposer(files);
      }
      els.fileInput.value = "";
    });

    // Drag and drop anywhere on the page (or onto the Documents dialog).
    let dragDepth = 0;
    window.addEventListener("dragenter", (event) => {
      if (!hasFiles(event)) return;
      event.preventDefault();
      dragDepth += 1;
      if (els.library.open) els.libraryUpload.classList.add("over");
      else els.dropOverlay.hidden = false;
    });
    window.addEventListener("dragover", (event) => {
      if (hasFiles(event)) event.preventDefault();
    });
    window.addEventListener("dragleave", (event) => {
      if (!hasFiles(event)) return;
      dragDepth = Math.max(0, dragDepth - 1);
      if (dragDepth === 0) {
        els.dropOverlay.hidden = true;
        els.libraryUpload.classList.remove("over");
      }
    });
    window.addEventListener("drop", (event) => {
      if (!hasFiles(event)) return;
      event.preventDefault();
      dragDepth = 0;
      els.dropOverlay.hidden = true;
      els.libraryUpload.classList.remove("over");
      const files = Array.from(event.dataTransfer.files || []);
      if (!files.length) return;
      if (els.library.open) uploadToLibrary(files);
      else uploadToComposer(files);
    });

    els.brand.addEventListener("click", newChat);
    els.newChat.addEventListener("click", newChat);
    els.topbarNewChat.addEventListener("click", newChat);
    els.openLibrary.addEventListener("click", openLibrary);
    els.closeLibrary.addEventListener("click", closeLibrary);
    els.library.addEventListener("click", (event) => {
      if (event.target !== els.library) return;
      // Clicks on the dialog element itself are either on its backdrop or in its own padding.
      const box = els.library.getBoundingClientRect();
      const inside =
        event.clientX >= box.left && event.clientX <= box.right &&
        event.clientY >= box.top && event.clientY <= box.bottom;
      if (!inside) closeLibrary();
    });

    els.closeSidebar.addEventListener("click", () => setSidebar(false));
    els.openSidebar.addEventListener("click", () => setSidebar(true));
    els.scrim.addEventListener("click", () => setSidebar(false, false));
    MOBILE.addEventListener("change", () => {
      setSidebar(!MOBILE.matches && storageGet(SIDEBAR_KEY) !== "closed", false);
    });

    document.addEventListener("keydown", (event) => {
      const mod = event.ctrlKey || event.metaKey;
      if (mod && event.shiftKey && event.key.toLowerCase() === "o") {
        event.preventDefault();
        newChat();
      } else if (event.key === "Escape" && state.busy && !els.library.open) {
        stopGenerating();
      }
    });
  }

  function init() {
    loadConversations();
    bindEvents();
    setSidebar(!MOBILE.matches && storageGet(SIDEBAR_KEY) !== "closed", false);
    renderConversation();
    setBusy(false);
    autoResize();
    loadHealth();
    els.prompt.focus();
    if (!librariesReady()) {
      toast("Math rendering libraries did not load; formulas will show as plain text.", "error");
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
