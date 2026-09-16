"use strict";

(() => {
  const $ = (selector) => document.querySelector(selector);

  const els = {
    sidebar: $("#sidebar"),
    toggleSidebar: $("#toggle-sidebar"),
    chat: $("#chat"),
    composer: $("#composer"),
    prompt: $("#prompt"),
    send: $("#send"),
    clearChat: $("#clear-chat"),
    uploadForm: $("#upload-form"),
    fileInput: $("#file-input"),
    dropzone: $("#dropzone"),
    dropzoneLabel: $("#dropzone-label"),
    uploadBtn: $("#upload-btn"),
    pasteForm: $("#paste-form"),
    pasteName: $("#paste-name"),
    pasteText: $("#paste-text"),
    ingestStatus: $("#ingest-status"),
    docList: $("#doc-list"),
    refreshDocs: $("#refresh-docs"),
    topK: $("#top-k"),
    topKValue: $("#top-k-value"),
    threshold: $("#threshold"),
    thresholdValue: $("#threshold-value"),
    retrievalOnly: $("#retrieval-only"),
    healthPill: $("#health-pill"),
    sourcesTemplate: $("#sources-template"),
  };

  const MAX_HISTORY_TURNS = 20;
  const MAX_UPLOAD_BYTES = 25 * 1024 * 1024;
  const GREETING = [
    "សួស្តី! ខ្ញុំគឺជាជំនួយការគណិតវិទ្យាថ្នាក់ទី១២។ សួរខ្ញុំអំពីលំហាត់ ឬមេរៀនណាមួយ ដូចជា",
    "$\\lim_{x \\to 0} \\frac{\\sin x}{x}$ ឬ",
    "",
    "$$\\int_0^1 x^2\\,dx$$",
    "",
    "_Ask in Khmer or English. Answers are grounded in the documents you ingest._",
  ].join("\n");

  let history = [];
  let busy = false;

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

  const escapeHtml = (value) =>
    String(value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");

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

  // Markdown with LaTeX: formulas are pulled out before Markdown parsing
  // (so `_`, `*` and `\\` inside them survive), the HTML is sanitised, and
  // KaTeX output is spliced back in.
  function renderRich(text) {
    if (!librariesReady()) {
      return `<p style="white-space: pre-wrap">${escapeHtml(text)}</p>`;
    }
    const formulas = [];
    const masked = String(text).replace(MATH_RE, (match, display, bracket, env, paren, inline) => {
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
    return html.replace(PLACEHOLDER_RE, (_, index) => {
      const formula = formulas[Number(index)];
      return formula ? renderMath(formula) : "";
    });
  }

  // ------------------------------------------------------------------- api

  async function api(path, options = {}) {
    let response;
    try {
      response = await fetch(path, options);
    } catch (error) {
      throw new Error("Network error: cannot reach the server");
    }
    const contentType = response.headers.get("content-type") || "";
    const body = contentType.includes("application/json") ? await response.json() : await response.text();
    if (!response.ok) {
      throw new Error(formatError(body, response.status));
    }
    return body;
  }

  function formatError(body, statusCode) {
    if (body && typeof body === "object" && "detail" in body) {
      const { detail } = body;
      if (Array.isArray(detail)) {
        return detail
          .map((item) => `${(item.loc || []).filter((part) => part !== "body").join(".")}: ${item.msg}`)
          .join("; ");
      }
      return String(detail);
    }
    return `Request failed (${statusCode})${typeof body === "string" && body ? `: ${body.slice(0, 200)}` : ""}`;
  }

  // ------------------------------------------------------------------ chat

  function scrollToBottom() {
    els.chat.scrollTop = els.chat.scrollHeight;
  }

  function addMessage(role, { text = "", html = null, meta = "" } = {}) {
    const article = document.createElement("article");
    article.className = `msg msg--${role}`;
    const bubble = document.createElement("div");
    bubble.className = "msg__bubble";
    if (html !== null) bubble.innerHTML = html;
    else bubble.textContent = text;
    article.appendChild(bubble);
    if (meta) {
      const metaEl = document.createElement("div");
      metaEl.className = "msg__meta";
      metaEl.textContent = meta;
      article.appendChild(metaEl);
    }
    els.chat.appendChild(article);
    scrollToBottom();
    return article;
  }

  function addTypingIndicator() {
    return addMessage("bot", {
      html: '<span class="typing" aria-label="Thinking"><span></span><span></span><span></span></span>',
    });
  }

  function renderSources(sources) {
    const fragment = els.sourcesTemplate.content.cloneNode(true);
    const details = fragment.querySelector("details");
    details.querySelector("summary").textContent = `ឯកសារយោង · ${sources.length} source${sources.length === 1 ? "" : "s"}`;
    const list = details.querySelector("ol");
    sources.forEach((source, index) => {
      const item = document.createElement("li");
      item.className = "source";
      const page = source.page != null ? ` · p.${source.page}` : "";
      item.innerHTML = `
        <div class="source__head">
          <span class="source__name">[${index + 1}] ${escapeHtml(source.source)}${escapeHtml(page)}</span>
          <span class="source__score">${Number(source.score).toFixed(3)}</span>
        </div>
        <div class="source__body"></div>`;
      item.querySelector(".source__body").innerHTML = renderRich(source.text);
      list.appendChild(item);
    });
    return details;
  }

  function resetChat() {
    history = [];
    els.chat.replaceChildren();
    addMessage("bot", { html: renderRich(GREETING) });
  }

  async function sendPrompt(prompt) {
    if (busy) return;
    busy = true;
    els.send.disabled = true;
    addMessage("user", { text: prompt });
    const typing = addTypingIndicator();

    const payload = {
      prompt,
      top_k: Number(els.topK.value),
      score_threshold: Number(els.threshold.value),
      history: history.slice(-MAX_HISTORY_TURNS),
      generate: !els.retrievalOnly.checked,
    };

    try {
      const result = await api("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      typing.remove();

      const parts = [
        result.provider === "none" ? "retrieval only" : `${result.provider}${result.model ? ` · ${result.model}` : ""}`,
        result.grounded ? `${result.sources.length} source(s)` : "not grounded",
        `${Math.round(result.latency_ms)} ms`,
      ];
      if (result.stop_reason === "max_tokens" || result.stop_reason === "MAX_TOKENS") parts.push("truncated");

      let answer = result.answer;
      if (!payload.generate) {
        answer = result.sources.length
          ? `_Retrieved ${result.sources.length} passage(s); see sources below._`
          : "_No passage passed the score threshold._";
      }
      const message = addMessage("bot", { html: renderRich(answer), meta: parts.join(" · ") });
      if (result.sources.length) {
        message.insertBefore(renderSources(result.sources), message.querySelector(".msg__meta"));
      }

      if (payload.generate && result.answer) {
        history.push({ role: "user", content: prompt });
        history.push({ role: "assistant", content: result.answer });
        history = history.slice(-MAX_HISTORY_TURNS);
      }
      scrollToBottom();
    } catch (error) {
      typing.remove();
      addMessage("error", { text: error.message });
    } finally {
      busy = false;
      els.send.disabled = false;
      els.prompt.focus();
    }
  }

  function autoResize() {
    els.prompt.style.height = "auto";
    els.prompt.style.height = `${Math.min(els.prompt.scrollHeight, 220)}px`;
  }

  // ------------------------------------------------------------- ingestion

  function setIngestStatus(message, kind = "") {
    els.ingestStatus.textContent = message;
    els.ingestStatus.className = `status-line${kind ? ` is-${kind}` : ""}`;
  }

  function describeIngest(result) {
    const parts = [
      `✓ ${result.source}: ${result.chunks_added} chunks`,
      `${result.formulas_protected} formulas`,
    ];
    if (result.chunks_replaced) parts.push(`replaced ${result.chunks_replaced}`);
    let text = parts.join(" · ");
    if (result.warnings && result.warnings.length) text += ` — ${result.warnings.join(" ")}`;
    return text;
  }

  function selectFile(file) {
    if (!file) {
      els.uploadBtn.disabled = true;
      els.dropzoneLabel.innerHTML = "ទម្លាក់ឯកសារ ឬចុចជ្រើសរើស<br /><small>PDF · Markdown · TXT</small>";
      return;
    }
    els.dropzoneLabel.innerHTML = `${escapeHtml(file.name)}<br /><small>${(file.size / 1024).toFixed(1)} KB</small>`;
    els.uploadBtn.disabled = false;
  }

  async function uploadFile(event) {
    event.preventDefault();
    const file = els.fileInput.files[0];
    if (!file) return;
    if (file.size > MAX_UPLOAD_BYTES) {
      setIngestStatus("File is larger than 25 MB", "error");
      return;
    }
    const form = new FormData();
    form.append("file", file);
    els.uploadBtn.disabled = true;
    setIngestStatus(`Indexing ${file.name}… (large PDFs can take a while)`);
    try {
      const result = await api("/api/ingest", { method: "POST", body: form });
      setIngestStatus(describeIngest(result), "success");
      els.fileInput.value = "";
      selectFile(null);
      await Promise.all([loadDocuments(), loadHealth()]);
    } catch (error) {
      setIngestStatus(error.message, "error");
      els.uploadBtn.disabled = false;
    }
  }

  async function ingestPastedText(event) {
    event.preventDefault();
    const text = els.pasteText.value;
    let name = els.pasteName.value.trim();
    if (!text.trim() || !name) return;
    if (!/\.(md|markdown|txt)$/i.test(name)) name += ".md";
    const submit = els.pasteForm.querySelector("button");
    submit.disabled = true;
    setIngestStatus(`Indexing ${name}…`);
    try {
      const result = await api("/api/ingest/text", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text,
          source_name: name,
          format: /\.txt$/i.test(name) ? "text" : "markdown",
        }),
      });
      setIngestStatus(describeIngest(result), "success");
      els.pasteText.value = "";
      await Promise.all([loadDocuments(), loadHealth()]);
    } catch (error) {
      setIngestStatus(error.message, "error");
    } finally {
      submit.disabled = false;
    }
  }

  // ------------------------------------------------------------- documents

  async function loadDocuments() {
    try {
      const { documents } = await api("/api/documents");
      els.docList.replaceChildren();
      if (!documents.length) {
        const empty = document.createElement("li");
        empty.className = "muted";
        empty.textContent = "No documents indexed yet";
        els.docList.appendChild(empty);
        return;
      }
      for (const doc of documents) {
        const item = document.createElement("li");
        item.className = "doc";
        item.title = `${doc.source}\nIndexed ${doc.ingested_at}`;
        const pages = doc.pages ? ` · ${doc.pages}p` : "";
        item.innerHTML = `
          <span class="doc__name">${escapeHtml(doc.source)}</span>
          <span class="doc__meta">${doc.chunks}c${pages}</span>
          <button type="button" class="btn btn--ghost btn--small btn--danger" aria-label="Delete ${escapeHtml(doc.source)}">✕</button>`;
        item.querySelector("button").addEventListener("click", () => deleteDocument(doc.source));
        els.docList.appendChild(item);
      }
    } catch (error) {
      els.docList.innerHTML = `<li class="status-line is-error">${escapeHtml(error.message)}</li>`;
    }
  }

  async function deleteDocument(source) {
    if (!window.confirm(`Remove "${source}" from the index?`)) return;
    try {
      const encoded = source.split("/").map(encodeURIComponent).join("/");
      const result = await api(`/api/documents/${encoded}`, { method: "DELETE" });
      setIngestStatus(`Removed ${result.source} (${result.chunks_removed} chunks)`, "success");
      await Promise.all([loadDocuments(), loadHealth()]);
    } catch (error) {
      setIngestStatus(error.message, "error");
    }
  }

  // ---------------------------------------------------------------- health

  let defaultsApplied = false;

  async function loadHealth() {
    try {
      const health = await api("/health");
      const llm = health.llm_provider === "none" ? "no LLM" : health.llm_model || health.llm_provider;
      els.healthPill.textContent = `● ${llm} · ${health.chunks} chunks`;
      els.healthPill.title = `Embeddings: ${health.embedding_model} (${health.embedding_loaded ? "loaded" : "loads on first use"})\nVersion ${health.version}`;
      els.healthPill.className = "pill is-ok";
      if (!defaultsApplied) {
        els.topK.value = health.default_top_k;
        els.threshold.value = health.default_score_threshold;
        els.topKValue.textContent = els.topK.value;
        els.thresholdValue.textContent = Number(els.threshold.value).toFixed(2);
        defaultsApplied = true;
      }
    } catch (error) {
      els.healthPill.textContent = "● offline";
      els.healthPill.title = error.message;
      els.healthPill.className = "pill is-error";
    }
  }

  // ---------------------------------------------------------------- wiring

  function bindEvents() {
    els.composer.addEventListener("submit", (event) => {
      event.preventDefault();
      const prompt = els.prompt.value.trim();
      if (!prompt) return;
      els.prompt.value = "";
      autoResize();
      sendPrompt(prompt);
    });
    els.prompt.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
        event.preventDefault();
        els.composer.requestSubmit();
      }
    });
    els.prompt.addEventListener("input", autoResize);
    els.clearChat.addEventListener("click", resetChat);

    els.fileInput.addEventListener("change", () => selectFile(els.fileInput.files[0]));
    els.uploadForm.addEventListener("submit", uploadFile);
    els.pasteForm.addEventListener("submit", ingestPastedText);
    ["dragenter", "dragover"].forEach((type) =>
      els.dropzone.addEventListener(type, (event) => {
        event.preventDefault();
        els.dropzone.classList.add("is-dragover");
      }),
    );
    ["dragleave", "drop"].forEach((type) =>
      els.dropzone.addEventListener(type, (event) => {
        event.preventDefault();
        els.dropzone.classList.remove("is-dragover");
      }),
    );
    els.dropzone.addEventListener("drop", (event) => {
      const [file] = event.dataTransfer.files;
      if (!file) return;
      const transfer = new DataTransfer();
      transfer.items.add(file);
      els.fileInput.files = transfer.files;
      selectFile(file);
    });

    els.refreshDocs.addEventListener("click", loadDocuments);
    els.topK.addEventListener("input", () => {
      els.topKValue.textContent = els.topK.value;
    });
    els.threshold.addEventListener("input", () => {
      els.thresholdValue.textContent = Number(els.threshold.value).toFixed(2);
    });

    els.toggleSidebar.addEventListener("click", () => {
      const open = els.sidebar.classList.toggle("is-open");
      els.toggleSidebar.setAttribute("aria-expanded", String(open));
    });
    els.chat.addEventListener("click", () => {
      els.sidebar.classList.remove("is-open");
      els.toggleSidebar.setAttribute("aria-expanded", "false");
    });
  }

  function init() {
    bindEvents();
    resetChat();
    loadHealth();
    loadDocuments();
    setInterval(loadHealth, 30000);
    els.prompt.focus();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
