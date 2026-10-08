// Agentic RAG browser UI: a thin client over the /api/v1 endpoints.
//
// Security rules followed throughout:
// - Model output and document text are untrusted: Markdown is rendered with
//   marked and *always* sanitised with DOMPurify. Everything else is set
//   with textContent, never innerHTML.
// - The API key stays in this browser (localStorage) and is only sent to
//   this origin, in the Authorization header.

const API = "/api/v1";
const KEY_STORAGE = "agentic-rag.apiKey";
const COLLECTION_STORAGE = "agentic-rag.collection";

// Carry settings over from the project's previous name (simple-rag).
for (const name of ["apiKey", "collection"]) {
  const old = localStorage.getItem(`simple-rag.${name}`);
  if (old !== null && localStorage.getItem(`agentic-rag.${name}`) === null) {
    localStorage.setItem(`agentic-rag.${name}`, old);
  }
  localStorage.removeItem(`simple-rag.${name}`);
}

const state = {
  apiKey: localStorage.getItem(KEY_STORAGE) || "",
  collections: [],
  collectionId: localStorage.getItem(COLLECTION_STORAGE) || "",
  mode: "ask",
  busy: null, // AbortController of the running chat request
  pollTimer: null,
};

const $ = (selector) => document.querySelector(selector);

// --- HTTP -------------------------------------------------------------------

class ApiError extends Error {
  constructor(status, title, detail) {
    super(detail || title || `HTTP ${status}`);
    this.status = status;
  }
}

function authHeaders(extra = {}) {
  return state.apiKey ? { ...extra, Authorization: `Bearer ${state.apiKey}` } : extra;
}

async function raiseForStatus(response) {
  if (response.ok) return;
  let problem = {};
  try {
    problem = await response.json();
  } catch {
    // not a problem+json body
  }
  throw new ApiError(response.status, problem.title, problem.detail);
}

async function api(path, { method = "GET", json, form, signal } = {}) {
  const headers = authHeaders(json ? { "Content-Type": "application/json" } : {});
  const response = await fetch(API + path, {
    method,
    headers,
    body: json ? JSON.stringify(json) : form,
    signal,
  });
  await raiseForStatus(response);
  return response.status === 204 ? null : response.json();
}

// Server-Sent Events over POST (EventSource only supports GET).
async function* streamEvents(path, json, signal) {
  const response = await fetch(API + path, {
    method: "POST",
    headers: authHeaders({ "Content-Type": "application/json", Accept: "text/event-stream" }),
    body: JSON.stringify({ ...json, stream: true }),
    signal,
  });
  await raiseForStatus(response);
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += value;
    let boundary;
    while ((boundary = buffer.indexOf("\n\n")) >= 0) {
      const block = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      let event = "message";
      const data = [];
      for (const line of block.split("\n")) {
        if (line.startsWith("event: ")) event = line.slice(7);
        else if (line.startsWith("data: ")) data.push(line.slice(6));
      }
      if (data.length) yield { event, data: JSON.parse(data.join("\n")) };
    }
  }
}

// --- Small DOM helpers ------------------------------------------------------

function el(tag, { className, text, attrs } = {}, ...children) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  for (const [name, value] of Object.entries(attrs || {})) node.setAttribute(name, value);
  node.append(...children.filter(Boolean));
  return node;
}

function renderMarkdown(target, markdown) {
  target.innerHTML = DOMPurify.sanitize(marked.parse(markdown, { gfm: true, breaks: false }));
}

let toastTimer;
function toast(message, kind = "info") {
  const node = $("#toast");
  node.textContent = message;
  node.className = `toast ${kind}`;
  node.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (node.hidden = true), 5000);
}

function reportError(error) {
  if (error.name === "AbortError") return;
  if (error.status === 401) toast("Missing or invalid API key: set it in the sidebar.", "error");
  else toast(error.message || String(error), "error");
}

function locationOf(source) {
  const parts = [source.document_title, ...(source.heading_path || [])];
  const where = parts.join(" › ");
  return source.page ? `${where} (p. ${source.page})` : where;
}

// --- Health -----------------------------------------------------------------

async function refreshHealth() {
  const pill = $("#health");
  try {
    const response = await fetch("/health/ready");
    const report = await response.json();
    const down = report.dependencies.filter((d) => !d.healthy).map((d) => d.name);
    pill.textContent = down.length ? `degraded: ${down.join(", ")}` : "all systems ready";
    pill.className = `pill ${down.length ? "pill-warn" : "pill-ok"}`;
  } catch {
    pill.textContent = "API unreachable";
    pill.className = "pill pill-error";
  }
}

// --- API key ----------------------------------------------------------------

function showKeyStatus() {
  $("#key-input").value = state.apiKey;
  $("#key-status").textContent = state.apiKey
    ? `Saved in this browser (${state.apiKey.slice(0, 11)}…).`
    : "Create one with make api-key name=dev, then paste it here.";
}

$("#key-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  state.apiKey = $("#key-input").value.trim();
  localStorage.setItem(KEY_STORAGE, state.apiKey);
  showKeyStatus();
  await loadCollections();
});

// --- Collections ------------------------------------------------------------

async function loadCollections() {
  const select = $("#collection-select");
  select.replaceChildren();
  if (!state.apiKey) {
    select.append(el("option", { text: "Set your API key first" }));
    return;
  }
  try {
    const page = await api("/collections?limit=100");
    state.collections = page.items;
  } catch (error) {
    reportError(error);
    select.append(el("option", { text: "Could not load collections" }));
    return;
  }
  if (!state.collections.length) {
    select.append(el("option", { text: "No collections yet: create one" }));
    state.collectionId = "";
  }
  for (const c of state.collections) {
    select.append(el("option", { text: c.name, attrs: { value: c.id } }));
  }
  if (!state.collections.some((c) => c.id === state.collectionId)) {
    state.collectionId = state.collections[0]?.id || "";
  }
  select.value = state.collectionId;
  onCollectionChanged();
}

function onCollectionChanged() {
  localStorage.setItem(COLLECTION_STORAGE, state.collectionId);
  const c = state.collections.find((item) => item.id === state.collectionId);
  $("#collection-info").textContent = c
    ? `${c.description ? c.description + " · " : ""}embedded with ${c.embedding_model} (${c.embedding_dim} dims)`
    : "";
  loadDocuments();
}

$("#collection-select").addEventListener("change", (event) => {
  state.collectionId = event.target.value;
  onCollectionChanged();
});

$("#new-collection-toggle").addEventListener("click", () => {
  const form = $("#new-collection-form");
  form.hidden = !form.hidden;
  if (!form.hidden) $("#new-collection-name").focus();
});

$("#new-collection-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    const created = await api("/collections", {
      method: "POST",
      json: {
        name: $("#new-collection-name").value,
        description: $("#new-collection-description").value || null,
      },
    });
    event.target.reset();
    event.target.hidden = true;
    state.collectionId = created.id;
    toast(`Created collection “${created.name}”.`, "success");
    await loadCollections();
  } catch (error) {
    reportError(error);
  }
});

// --- Documents --------------------------------------------------------------

async function loadDocuments() {
  clearTimeout(state.pollTimer);
  const list = $("#document-list");
  if (!state.collectionId || !state.apiKey) {
    list.replaceChildren();
    return;
  }
  let documents;
  try {
    documents = (await api(`/collections/${state.collectionId}/documents?limit=100`)).items;
  } catch (error) {
    reportError(error);
    return;
  }
  list.replaceChildren(
    ...(documents.length
      ? documents.map(documentItem)
      : [el("li", { className: "hint", text: "No documents yet. Upload some to start." })]),
  );
  // While anything is being ingested, refresh to show progress.
  if (documents.some((d) => d.status === "pending" || d.status === "processing")) {
    state.pollTimer = setTimeout(loadDocuments, 2000);
  }
}

function documentItem(doc) {
  const remove = el("button", { className: "ghost small icon", text: "✕", attrs: { title: "Delete document" } });
  remove.addEventListener("click", async () => {
    if (!confirm(`Delete “${doc.title}” and its chunks?`)) return;
    try {
      await api(`/documents/${doc.id}`, { method: "DELETE" });
      loadDocuments();
    } catch (error) {
      reportError(error);
    }
  });
  // A job waits at most a couple of seconds for a running worker; much longer means none is running.
  const stuck = doc.status === "pending" && Date.now() - Date.parse(doc.created_at) > 30_000;
  const detail = doc.status === "ready"
    ? `${doc.chunk_count} chunks`
    : stuck
      ? "Still waiting: is the worker running? (make worker)"
      : doc.error || doc.source_filename;
  return el(
    "li",
    { className: `document${stuck ? " stuck" : ""}`, attrs: { title: doc.error || doc.source_filename } },
    el("div", { className: "doc-main" }, el("span", { className: "doc-title", text: doc.title }), el("span", { className: "hint", text: detail })),
    el("span", { className: `badge status-${doc.status}`, text: doc.status }),
    remove,
  );
}

$("#file-input").addEventListener("change", async (event) => {
  const files = [...event.target.files];
  event.target.value = "";
  if (!state.collectionId) return toast("Select or create a collection first.", "error");
  for (const file of files) {
    const form = new FormData();
    form.append("file", file);
    try {
      await api(`/collections/${state.collectionId}/documents`, { method: "POST", form });
      toast(`Uploaded ${file.name}: queued for ingestion.`, "success");
    } catch (error) {
      reportError(error);
    }
  }
  loadDocuments();
});

// --- Tabs -------------------------------------------------------------------

for (const tab of document.querySelectorAll(".tab")) {
  tab.addEventListener("click", () => {
    for (const other of document.querySelectorAll(".tab")) {
      const active = other === tab;
      other.classList.toggle("active", active);
      other.setAttribute("aria-selected", String(active));
      $(`#tab-${other.dataset.tab}`).hidden = !active;
    }
  });
}

// --- Chat -------------------------------------------------------------------

for (const button of document.querySelectorAll(".seg")) {
  button.addEventListener("click", () => {
    state.mode = button.dataset.mode;
    for (const other of document.querySelectorAll(".seg")) {
      other.classList.toggle("active", other === button);
      other.setAttribute("aria-checked", String(other === button));
    }
  });
}

$("#chat-input").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    $("#chat-form").requestSubmit();
  }
});

$("#chat-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (state.busy) {
    state.busy.abort(); // the Send button acts as Stop while streaming
    return;
  }
  const question = $("#chat-input").value.trim();
  if (!question) return;
  if (!state.collectionId) return toast("Select or create a collection first.", "error");
  $("#chat-input").value = "";
  $("#messages .empty")?.remove();
  await converse(question, state.mode);
});

async function converse(question, mode) {
  const messages = $("#messages");
  messages.append(el("div", { className: "message user", text: question }));
  const turn = new AssistantTurn(mode);
  messages.append(turn.node);
  turn.node.scrollIntoView({ block: "end" });

  state.busy = new AbortController();
  $("#chat-send").textContent = "Stop";
  const path = `/collections/${state.collectionId}/${mode === "agent" ? "agent/ask" : "ask"}`;
  try {
    for await (const { event, data } of streamEvents(path, { question }, state.busy.signal)) {
      turn.handle(event, data);
    }
  } catch (error) {
    turn.fail(error.name === "AbortError" ? "Stopped." : error.message);
    reportError(error);
  } finally {
    state.busy = null;
    $("#chat-send").textContent = "Send";
  }
}

class AssistantTurn {
  constructor(mode) {
    this.mode = mode;
    this.text = "";
    this.sources = new Map(); // number -> source (full for /ask; filled in at the end for the agent)
    this.timeline = el("ol", { className: "timeline" });
    this.answer = el("div", { className: "markdown answer" }, el("span", { className: "thinking", text: mode === "agent" ? "Planning searches…" : "Searching…" }));
    this.meta = el("div", { className: "meta" });
    this.sourceList = el("div", { className: "source-chips" });
    this.node = el(
      "div",
      { className: "message assistant" },
      el("div", { className: "message-head", text: mode === "agent" ? "Agent" : "Ask" }),
      mode === "agent" ? this.timeline : null,
      this.answer,
      this.sourceList,
      this.meta,
    );
    this.answer.addEventListener("click", (event) => this.openCitation(event));
    this.sourceList.addEventListener("click", (event) => this.openCitation(event));
    this.renderScheduled = false;
  }

  handle(event, data) {
    switch (event) {
      case "sources": // /ask: everything the model will see, before the answer starts
        for (const s of data.sources) this.sources.set(s.number, s);
        this.setThinking(data.sources.length ? `Found ${data.sources.length} sources. Writing…` : "No relevant sources.");
        break;
      case "tool_call":
        this.timeline.append(
          el("li", { className: "step", attrs: { id: this.stepId(data.step) } },
            el("span", { className: "step-tool", text: toolLabel(data.tool) }),
            el("span", { className: "step-args", text: describeArguments(data.arguments) }),
            el("span", { className: "step-result hint", text: "…" })),
        );
        this.setThinking("Searching…");
        break;
      case "tool_result": {
        const step = this.node.querySelector(`#${CSS.escape(this.stepId(data.step))} .step-result`);
        const found = data.result.sources || [];
        for (const s of found) if (!this.sources.has(s.number)) this.sources.set(s.number, { number: s.number, locationText: s.location });
        if (step) step.textContent = data.result.error ? `⚠ ${data.result.error}` : `${found.length} passages · ${Math.round(data.latency_ms)} ms`;
        this.setThinking("Reading results…");
        break;
      }
      case "token":
        this.text += data.text;
        this.scheduleRender();
        break;
      case "done":
        this.finish(data);
        break;
      case "error":
        this.fail(`${data.title} (${data.detail})`);
        break;
    }
  }

  stepId(step) {
    if (!this.id) this.id = `t${Math.random().toString(36).slice(2, 8)}`;
    return `${this.id}-step-${step}`;
  }

  setThinking(text) {
    const thinking = this.answer.querySelector(".thinking");
    if (thinking) thinking.textContent = text;
  }

  scheduleRender() {
    if (this.renderScheduled) return;
    this.renderScheduled = true;
    requestAnimationFrame(() => {
      this.renderScheduled = false;
      renderMarkdown(this.answer, this.text);
      linkCitations(this.answer);
    });
  }

  finish(done) {
    this.text = done.answer;
    renderMarkdown(this.answer, this.text);
    linkCitations(this.answer);
    if (Array.isArray(done.sources)) {
      // Agent: full sources (with text) arrive with the result.
      for (const s of done.sources) this.sources.set(s.number, s);
    }
    const cited = new Set(done.cited);
    const shown = [...this.sources.values()].filter((s) => cited.has(s.number));
    this.sourceList.replaceChildren(
      ...(shown.length ? [el("span", { className: "hint", text: "Cited:" })] : []),
      ...shown.map((s) => el("button", { className: "chip", text: `[${s.number}] ${s.locationText || locationOf(s)}`, attrs: { "data-source": String(s.number), type: "button" } })),
    );
    const parts = [];
    if (done.tool_calls !== undefined) parts.push(`${done.tool_calls} tool call${done.tool_calls === 1 ? "" : "s"}`);
    parts.push(`${(done.timings_ms.total / 1000).toFixed(1)} s`);
    if (done.usage) parts.push(`${done.usage.prompt_tokens + done.usage.completion_tokens} tokens`);
    if (done.model) parts.push(done.model);
    this.meta.textContent = parts.join(" · ");
    if (done.answer_withheld) {
      this.meta.append(el("span", { className: "warn", text: " · draft answer withheld: it cited no sources" }));
    } else if (done.invalid_citations.length) {
      this.meta.append(el("span", { className: "warn", text: ` · cites unknown sources ${done.invalid_citations.map((n) => `[${n}]`).join("")}` }));
    } else if (!cited.size && this.text && !/couldn.t find/i.test(this.text)) {
      this.meta.append(el("span", { className: "warn", text: " · no citations" }));
    }
    if (done.run_id) {
      this.meta.append(el("span", { className: "hint", text: ` · run ${done.run_id.slice(0, 8)}` }));
    }
  }

  fail(message) {
    this.setThinking("");
    this.meta.replaceChildren(el("span", { className: "warn", text: message }));
  }

  openCitation(event) {
    const target = event.target.closest("[data-source]");
    if (!target) return;
    const source = this.sources.get(Number(target.dataset.source));
    if (source) openDrawer(source);
    else toast(`Source [${target.dataset.source}] is not among the sources the model received.`, "error");
  }
}

function toolLabel(tool) {
  return { search_knowledge_base: "🔎 search", read_more_context: "📖 read more", list_documents: "📚 list documents" }[tool] || tool;
}

function describeArguments(args) {
  if (args.query) return `“${args.query}”`;
  if (args.source !== undefined) return `around [${args.source}]`;
  if (args.raw !== undefined) return "(invalid arguments)";
  return "";
}

// Turn [1] / [1, 3] in rendered answers into buttons, except inside code.
function linkCitations(root) {
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
    acceptNode: (node) => (node.parentElement.closest("code, pre, .cite") ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT),
  });
  const pattern = /\[(\d+(?:\s*,\s*\d+)*)\]/g;
  const nodes = [];
  while (walker.nextNode()) if (pattern.test(walker.currentNode.nodeValue)) nodes.push(walker.currentNode);
  for (const node of nodes) {
    const fragment = document.createDocumentFragment();
    let last = 0;
    const text = node.nodeValue;
    for (const match of text.matchAll(pattern)) {
      fragment.append(text.slice(last, match.index));
      for (const number of match[1].split(",").map((n) => n.trim())) {
        fragment.append(el("button", { className: "cite", text: number, attrs: { "data-source": number, type: "button", title: `Show source ${number}` } }));
      }
      last = match.index + match[0].length;
    }
    fragment.append(text.slice(last));
    node.replaceWith(fragment);
  }
}

// --- Source drawer ----------------------------------------------------------

function openDrawer(source) {
  $("#drawer-title").textContent = `Source [${source.number}]`;
  $("#drawer-location").textContent = source.locationText || locationOf(source);
  const body = $("#drawer-body");
  if (source.text) renderMarkdown(body, source.text);
  else body.replaceChildren(el("p", { className: "hint", text: "The full passage appears when the answer completes." }));
  $("#drawer").hidden = false;
}

$("#drawer-close").addEventListener("click", () => ($("#drawer").hidden = true));
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") $("#drawer").hidden = true;
});

// --- Search lab ---------------------------------------------------------------

const LAB_CONFIGS = [
  { label: "Vector", hint: "semantic similarity", body: { mode: "vector", rerank: false } },
  { label: "Keyword", hint: "full-text search", body: { mode: "keyword", rerank: false } },
  { label: "Hybrid", hint: "both, fused with RRF", body: { mode: "hybrid", rerank: false } },
  { label: "Hybrid + rerank", hint: "LLM grades the top candidates", body: { mode: "hybrid", rerank: true } },
];

$("#search-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!state.collectionId) return toast("Select or create a collection first.", "error");
  const query = $("#search-input").value.trim();
  const lab = $("#search-results");
  const columns = LAB_CONFIGS.map((config) => {
    const list = el("ol", { className: "results" }, el("li", { className: "hint", text: "searching…" }));
    const timing = el("span", { className: "hint" });
    const column = el("div", { className: "lab-column" }, el("div", { className: "lab-head" }, el("strong", { text: config.label }), el("span", { className: "hint", text: config.hint }), timing), list);
    return { config, list, timing, column };
  });
  lab.replaceChildren(...columns.map((c) => c.column));
  await Promise.all(columns.map((c) => runLabColumn(query, c)));
});

async function runLabColumn(query, { config, list, timing }) {
  try {
    const result = await api(`/collections/${state.collectionId}/search`, {
      method: "POST",
      json: { query, top_k: 5, ...config.body },
    });
    timing.textContent = `${Math.round(result.timings_ms.total)} ms`;
    if (result.rerank_error) timing.textContent += " · rerank failed";
    list.replaceChildren(
      ...(result.hits.length ? result.hits.map(labItem) : [el("li", { className: "hint", text: "no results" })]),
    );
  } catch (error) {
    list.replaceChildren(el("li", { className: "warn", text: error.message }));
  }
}

function labItem(hit, index) {
  const score = hit.rerank_score !== null && hit.rerank_score !== undefined ? `grade ${hit.rerank_score}` : hit.score.toFixed(3);
  const item = el(
    "li",
    { className: "result", attrs: { "data-chunk": hit.chunk_id, tabindex: "0" } },
    el("div", { className: "result-head" }, el("span", { className: "rank", text: String(index + 1) }), el("span", { className: "result-location", text: locationOf(hit) }), el("span", { className: "score", text: score })),
    el("p", { className: "snippet", text: hit.text.slice(0, 220) + (hit.text.length > 220 ? "…" : "") }),
  );
  const highlight = (on) => {
    for (const same of document.querySelectorAll(`.result[data-chunk="${CSS.escape(hit.chunk_id)}"]`)) same.classList.toggle("same", on);
  };
  item.addEventListener("mouseenter", () => highlight(true));
  item.addEventListener("mouseleave", () => highlight(false));
  item.addEventListener("click", () => openDrawer({ ...hit, number: index + 1 }));
  return item;
}

// --- Start --------------------------------------------------------------------

showKeyStatus();
refreshHealth();
setInterval(refreshHealth, 30000);
loadCollections();
