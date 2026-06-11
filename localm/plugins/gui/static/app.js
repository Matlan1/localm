/* localm GUI — vanilla JS, no build step.
   Talks to the localm FastAPI server: /v1 (OpenAI-compatible) + /api (GUI).
   All model/agent-originating strings go through textContent or DOMPurify —
   never raw innerHTML. pages.js builds on the helpers defined here. */

"use strict";

/* ================================================================ */
/*  Shared helpers                                                   */
/* ================================================================ */

const $ = (id) => document.getElementById(id);

const API_KEY = localStorage.getItem("localm.apiKey") || "";

function authHeaders(extra = {}) {
  const h = { "Content-Type": "application/json", ...extra };
  if (API_KEY) h["Authorization"] = "Bearer " + API_KEY;
  return h;
}

function toast(msg, isError = false) {
  const el = $("toast");
  el.textContent = msg;
  el.className = "show" + (isError ? " error" : "");
  clearTimeout(el._t);
  el._t = setTimeout(() => (el.className = ""), 3500);
}

marked.setOptions({ breaks: true, mangle: false, headerIds: false });

function renderMarkdown(target, text) {
  target.innerHTML = DOMPurify.sanitize(marked.parse(text || ""));
  // LaTeX math: $...$, $$...$$, \(...\), \[...\]. KaTeX only rewrites text
  // nodes after sanitisation, so this stays XSS-safe.
  if (typeof renderMathInElement !== "undefined") {
    try {
      renderMathInElement(target, {
        delimiters: [
          { left: "$$", right: "$$", display: true },
          { left: "$", right: "$", display: false },
          { left: "\\(", right: "\\)", display: false },
          { left: "\\[", right: "\\]", display: true },
        ],
        throwOnError: false,
        ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code"],
      });
    } catch (e) { /* malformed TeX mid-stream — final render fixes it */ }
  }
  target.querySelectorAll("pre code").forEach((block) => {
    try { hljs.highlightElement(block); } catch (e) { /* unknown lang */ }
  });
  target.querySelectorAll("pre").forEach((pre) => {
    if (pre.querySelector(".copy-btn")) return;
    const btn = document.createElement("button");
    btn.className = "copy-btn";
    btn.textContent = "copy";
    btn.onclick = () => {
      navigator.clipboard.writeText(pre.querySelector("code")?.innerText || pre.innerText);
      btn.textContent = "copied";
      setTimeout(() => (btn.textContent = "copy"), 1200);
    };
    pre.appendChild(btn);
  });
}

/** Create an element with class and (safe) text content. */
function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function autoGrow(textarea) {
  textarea.style.height = "auto";
  textarea.style.height = Math.min(textarea.scrollHeight, 220) + "px";
}

function nearBottom(elm) {
  return elm.scrollHeight - elm.scrollTop - elm.clientHeight < 80;
}

/** Parse an SSE byte stream from fetch(), invoking onData per `data:` payload. */
async function readSSE(response, onData) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let idx;
    while ((idx = buf.indexOf("\n\n")) !== -1) {
      const frame = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      for (const line of frame.split("\n")) {
        if (line.startsWith("data: ")) onData(line.slice(6));
      }
    }
  }
}

/** Stream a background job's events; onLine gets text lines; resolves with the end event. */
async function streamJob(jobId, onLine) {
  const r = await fetch(`/api/jobs/${jobId}/events`, { headers: authHeaders() });
  if (!r.ok) throw new Error(r.statusText);
  let endEvent = null;
  await readSSE(r, (payload) => {
    let ev;
    try { ev = JSON.parse(payload); } catch { return; }
    if (ev.type === "line" && onLine) onLine(ev.text);
    if (ev.type === "end") endEvent = ev;
  });
  return endEvent || { status: "failed" };
}

/** Fetch an auth-protected image into an object URL. */
async function fetchImageURL(path) {
  const r = await fetch(path, { headers: authHeaders() });
  if (!r.ok) throw new Error(r.statusText);
  return URL.createObjectURL(await r.blob());
}

/* ---- modal ---- */

function openModal(title, bodyBuilder) {
  $("modal-title").textContent = title;
  const body = $("modal-body");
  body.innerHTML = "";
  bodyBuilder(body);
  $("modal").style.display = "flex";
}
$("modal-close").onclick = () => ($("modal").style.display = "none");
$("modal").onclick = (e) => { if (e.target === $("modal")) $("modal").style.display = "none"; };

/* ================================================================ */
/*  Theme                                                            */
/* ================================================================ */

function applyTheme(name) {
  document.documentElement.dataset.theme = name;
  localStorage.setItem("localm.theme", name);
}
applyTheme(localStorage.getItem("localm.theme") || "dark");
$("theme-toggle").onclick = () =>
  applyTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark");

/* ================================================================ */
/*  Tabs                                                             */
/* ================================================================ */

const VIEWS = ["chat", "coder", "models", "images", "plugins", "settings"];

function showView(name) {
  for (const v of VIEWS) {
    $("view-" + v).classList.toggle("active", v === name);
    $("nav-" + v).classList.toggle("active", v === name);
  }
  // Lazy page refreshes live in pages.js
  if (window.onViewShown) window.onViewShown(name);
}
for (const v of VIEWS) $("nav-" + v).onclick = () => showView(v);

/* ================================================================ */
/*  Models (sidebar selector)                                        */
/* ================================================================ */

const modelSelect = $("model-select");

function setStatus(state, text) {
  $("status-dot").className = "dot " + state;
  $("status-text").textContent = text;
}

let modelCache = { models: [], active: "" };

async function refreshModels() {
  try {
    const r = await fetch("/api/models", { headers: authHeaders() });
    if (r.status === 401) {
      const key = prompt("This server requires an API key (LOCALM_API_KEY):");
      if (key) { localStorage.setItem("localm.apiKey", key); location.reload(); }
      return;
    }
    const data = await r.json();
    modelCache = data;
    // Don't rebuild the select while the user has it open
    if (document.activeElement !== modelSelect) {
      modelSelect.innerHTML = "";
      for (const m of data.models) {
        const opt = document.createElement("option");
        opt.value = m.name;
        const size = m.size_bytes ? ` (${(m.size_bytes / 1e9).toFixed(1)} GB)` : "";
        opt.textContent = m.name + size;
        if (m.active) opt.selected = true;
        modelSelect.appendChild(opt);
      }
    }
    setStatus("ok", data.active || "no model");
  } catch (e) {
    setStatus("err", "server unreachable");
  }
}

async function switchModel(model) {
  setStatus("busy", "loading " + model + "…");
  const r = await fetch("/api/models/load", {
    method: "POST",
    headers: authHeaders(),
    body: JSON.stringify({ model }),
  });
  if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
  setStatus("ok", model);
}

modelSelect.onchange = async () => {
  const model = modelSelect.value;
  try {
    await switchModel(model);
    toast("Model switched to " + model);
  } catch (e) {
    setStatus("err", "load failed");
    toast("Model load failed: " + e.message, true);
    refreshModels();
  }
};

/* ================================================================ */
/*  Chat                                                             */
/* ================================================================ */

const chat = {
  conversations: JSON.parse(localStorage.getItem("localm.conversations") || "[]"),
  activeId: null,
  abort: null,
  attachments: [],   // {name, dataUri}
  ctxMax: 16384,     // context ceiling — refreshed from /v1/config
};

// Conversation compaction mirrors localm/inference/compact.py:
// summarise older turns at 70% of the ceiling, keep the last 4 verbatim,
// hard-trim with a visible note when summarisation fails. Never blocks chat.
const COMPACT_RATIO = 0.7;
const COMPACT_KEEP = 4;

function estimateConvTokens(conv) {
  let total = Math.ceil(($("p-system").value || "").length / 4);
  for (const m of conv.messages) {
    total += Math.ceil(msgText(m).length / 4) + msgImages(m).length * 750;
  }
  return total;
}

async function compactConversation(conv) {
  if (conv.messages.length <= COMPACT_KEEP) return false;
  const older = conv.messages.slice(0, -COMPACT_KEEP);
  const recent = conv.messages.slice(-COMPACT_KEEP);
  const excerpt = older.map((m) =>
    `${m.role.toUpperCase()}: ${msgText(m).slice(0, 600)}`).join("\n\n");

  let summary = "";
  try {
    const r = await fetch("/v1/chat/completions", {
      method: "POST",
      headers: authHeaders(),
      body: JSON.stringify({
        model: modelSelect.value,
        messages: [{
          role: "user",
          content: "Summarise the following conversation in under 200 words. " +
            "Keep facts, names, decisions, and anything the user asked to " +
            "remember. Reply with the summary only.\n\n" + excerpt,
        }],
        max_tokens: 400,
        temperature: 0.3,
        stream: false,
      }),
    });
    if (r.ok) {
      const data = await r.json();
      summary = (data.choices?.[0]?.message?.content || "").trim();
    }
  } catch (e) { /* summarisation unavailable — hard trim below */ }

  const bridge = summary
    ? [{ role: "user", content: "[Conversation summary]\n" + summary },
       { role: "assistant", content: "Understood. Continuing from this summary." }]
    : [{ role: "user", content:
         "[Earlier conversation was removed to fit the context window.]" },
       { role: "assistant", content: "Understood." }];

  conv.messages = [...bridge, ...recent];
  saveConversations();
  renderChat();
  toast(summary ? "Older messages summarised to free context"
                : "Older messages trimmed (summarisation unavailable)");
  return true;
}

async function maybeCompactConversation(conv) {
  if (!chat.ctxMax || chat.ctxMax <= 0) return;
  if (estimateConvTokens(conv) >= COMPACT_RATIO * chat.ctxMax) {
    await compactConversation(conv);
  }
}

async function refreshCtxLimit() {
  try {
    const r = await fetch("/v1/config", { headers: authHeaders() });
    if (r.ok) {
      const cfg = await r.json();
      chat.ctxMax = cfg.n_ctx_max ?? 16384;
    }
  } catch (e) { /* keep default */ }
}

function saveConversations() {
  try {
    localStorage.setItem("localm.conversations",
      JSON.stringify(chat.conversations.slice(0, 50)));
  } catch (e) {
    // Quota: drop image-heavy older conversations and retry once
    const slim = chat.conversations.slice(0, 10);
    try { localStorage.setItem("localm.conversations", JSON.stringify(slim)); } catch {}
  }
}

function currentConv() {
  return chat.conversations.find((c) => c.id === chat.activeId) || null;
}

function newConversation() {
  const conv = { id: Date.now().toString(36), title: "New chat", messages: [] };
  chat.conversations.unshift(conv);
  chat.activeId = conv.id;
  saveConversations();
  renderConvList();
  renderChat();
}

/* message content helpers — content is a string or OpenAI multipart list */
function msgText(m) {
  if (typeof m.content === "string") return m.content;
  return (m.content || []).filter((p) => p.type === "text").map((p) => p.text).join("");
}
function msgImages(m) {
  if (typeof m.content === "string") return [];
  return (m.content || []).filter((p) => p.type === "image_url")
    .map((p) => p.image_url?.url).filter(Boolean);
}

function renderConvList() {
  const list = $("conv-list");
  list.innerHTML = "";
  for (const conv of chat.conversations) {
    const item = el("div", "conv-item" + (conv.id === chat.activeId ? " active" : ""));
    const title = el("span", "title", conv.title);
    title.ondblclick = (e) => {
      e.stopPropagation();
      const input = document.createElement("input");
      input.value = conv.title;
      const commit = () => {
        conv.title = input.value.trim() || conv.title;
        saveConversations();
        renderConvList();
      };
      input.onblur = commit;
      input.onkeydown = (ke) => { if (ke.key === "Enter") input.blur(); };
      item.replaceChild(input, title);
      input.focus();
      input.select();
    };
    item.appendChild(title);
    const del = el("button", "del", "×");
    del.onclick = (e) => {
      e.stopPropagation();
      chat.conversations = chat.conversations.filter((c) => c.id !== conv.id);
      if (chat.activeId === conv.id) chat.activeId = chat.conversations[0]?.id || null;
      saveConversations();
      renderConvList();
      renderChat();
    };
    item.appendChild(del);
    item.onclick = () => {
      chat.activeId = conv.id;
      renderConvList();
      renderChat();
      showView("chat");
    };
    list.appendChild(item);
  }
}

function addMessageRow(container, role, text, opts = {}) {
  const row = el("div", "msg-row " + role);
  row.appendChild(el("div", "msg-role", role === "user" ? "You" : "Model"));
  const body = el("div", "msg-body");
  renderMarkdown(body, text);
  for (const url of opts.images || []) {
    const img = document.createElement("img");
    img.className = "msg-img";
    img.src = url;   // data: URI from the user's own attachment
    body.appendChild(img);
  }
  row.appendChild(body);
  const meta = el("div", "msg-meta");
  const copy = el("button", "copy-btn", "copy");
  copy.onclick = () => {
    navigator.clipboard.writeText(text);
    copy.textContent = "copied";
    setTimeout(() => (copy.textContent = "copy"), 1200);
  };
  meta.appendChild(copy);
  for (const [label, fn] of opts.actions || []) {
    const btn = el("button", "action", label);
    btn.onclick = fn;
    meta.appendChild(btn);
  }
  row.appendChild(meta);
  container.appendChild(row);
  return { row, body, meta };
}

function buildEmptyHint() {
  const div = el("div", "empty-hint");
  const big = el("div", "big");
  big.appendChild(document.createTextNode("local"));
  const accent = el("span", "", "m");
  accent.style.color = "var(--accent)";
  big.appendChild(accent);
  div.appendChild(big);
  div.appendChild(document.createTextNode(
    "Chat with your local model. Everything stays on this machine."));
  return div;
}

function renderChat() {
  const box = $("chat-messages");
  box.innerHTML = "";
  const conv = currentConv();
  if (!conv || conv.messages.length === 0) {
    box.appendChild(buildEmptyHint());
    return;
  }
  conv.messages.forEach((m, i) => {
    const actions = [];
    if (m.role === "user") {
      actions.push(["edit", () => editMessage(conv, i)]);
    }
    if (m.role === "assistant" && i === conv.messages.length - 1 && !chat.abort) {
      actions.push(["regenerate", () => regenerate(conv)]);
    }
    addMessageRow(box, m.role, msgText(m), { images: msgImages(m), actions });
  });
  box.scrollTop = box.scrollHeight;
}

function editMessage(conv, index) {
  const m = conv.messages[index];
  $("chat-input").value = msgText(m);
  autoGrow($("chat-input"));
  conv.messages = conv.messages.slice(0, index);   // drop it and everything after
  saveConversations();
  renderChat();
  $("chat-input").focus();
}

function regenerate(conv) {
  if (chat.abort) return;
  // Drop the last assistant reply, keep the user message, stream again
  if (conv.messages[conv.messages.length - 1]?.role === "assistant") {
    conv.messages.pop();
    saveConversations();
    renderChat();
  }
  runCompletion(conv);
}

function chatParams() {
  const num = (id) => {
    const v = $(id).value.trim();
    return v === "" ? null : Number(v);
  };
  return {
    temperature: num("p-temperature"),
    top_p: num("p-top-p"),
    top_k: num("p-top-k"),
    repeat_penalty: num("p-repeat-penalty"),
    max_tokens: num("p-max-tokens"),
    seed: num("p-seed"),
    system: $("p-system").value.trim() || null,
    grammar: $("p-grammar").value.trim() || null,
  };
}

/* attachments */

function renderAttachChips() {
  const box = $("attach-chips");
  box.innerHTML = "";
  chat.attachments.forEach((att, i) => {
    const chip = el("span", "chip");
    const img = document.createElement("img");
    img.src = att.dataUri;
    chip.appendChild(img);
    chip.appendChild(el("span", "", att.name));
    const rm = el("button", "", "×");
    rm.onclick = () => { chat.attachments.splice(i, 1); renderAttachChips(); };
    chip.appendChild(rm);
    box.appendChild(chip);
  });
}

$("chat-attach").onclick = () => $("chat-file").click();
$("chat-file").addEventListener("change", (e) => {
  for (const file of e.target.files) {
    const reader = new FileReader();
    reader.onload = () => {
      chat.attachments.push({ name: file.name, dataUri: reader.result });
      renderAttachChips();
    };
    reader.readAsDataURL(file);
  }
  e.target.value = "";
});

/* sending */

async function runCompletion(conv) {
  await maybeCompactConversation(conv);
  const params = chatParams();
  const messages = [];
  if (params.system) messages.push({ role: "system", content: params.system });
  messages.push(...conv.messages.map((m) => ({ role: m.role, content: m.content })));

  const body = { model: modelSelect.value, messages, stream: true };
  for (const k of ["temperature", "top_p", "top_k", "repeat_penalty",
                   "max_tokens", "seed"]) {
    if (params[k] !== null && !Number.isNaN(params[k])) body[k] = params[k];
  }
  if (params.grammar) body.grammar = params.grammar;

  const box = $("chat-messages");
  const { body: liveBody } = addMessageRow(box, "assistant", "");
  box.scrollTop = box.scrollHeight;

  const sendBtn = $("chat-send");
  sendBtn.classList.add("stop");
  sendBtn.textContent = "■";
  chat.abort = new AbortController();

  let full = "";
  let usage = null;
  try {
    const r = await fetch("/v1/chat/completions", {
      method: "POST",
      headers: authHeaders(),
      body: JSON.stringify(body),
      signal: chat.abort.signal,
    });
    if (!r.ok) {
      const detail = await r.text();
      throw new Error(`${r.status}: ${detail.slice(0, 300)}`);
    }
    await readSSE(r, (payload) => {
      if (payload === "[DONE]") return;
      let chunk;
      try { chunk = JSON.parse(payload); } catch { return; }
      if (chunk.usage) usage = chunk.usage;
      const delta = chunk.choices?.[0]?.delta?.content || "";
      if (delta) {
        full += delta;
        const stick = nearBottom(box);
        renderMarkdown(liveBody, full);
        if (stick) box.scrollTop = box.scrollHeight;
      }
    });
  } catch (e) {
    if (e.name !== "AbortError") {
      renderMarkdown(liveBody, full + "\n\n*[error: " + e.message + "]*");
      toast("Chat request failed: " + e.message, true);
    }
  } finally {
    chat.abort = null;
    sendBtn.classList.remove("stop");
    sendBtn.textContent = "➤";
  }

  conv.messages.push({ role: "assistant", content: full });
  saveConversations();
  if (usage) {
    const bits = [`${usage.total_tokens} tok`];
    if (usage.ttft_ms != null) bits.push(`TTFT ${usage.ttft_ms} ms`);
    if (usage.tokens_per_sec != null) bits.push(`${usage.tokens_per_sec} tok/s`);
    $("chat-usage").textContent = bits.join(" · ");
  }
  renderChat();
}

async function sendChat() {
  const input = $("chat-input");
  const text = input.value.trim();
  if (!text && chat.attachments.length === 0) return;
  if (chat.abort) return;

  if (!currentConv()) newConversation();
  const conv = currentConv();

  let content;
  if (chat.attachments.length) {
    content = [{ type: "text", text }];
    for (const att of chat.attachments) {
      content.push({ type: "image_url", image_url: { url: att.dataUri } });
    }
  } else {
    content = text;
  }
  conv.messages.push({ role: "user", content });
  chat.attachments = [];
  renderAttachChips();

  if (conv.messages.length === 1) {
    conv.title = text.slice(0, 42) + (text.length > 42 ? "…" : "") || "Image chat";
    renderConvList();
  }
  input.value = "";
  autoGrow(input);
  renderChat();
  await runCompletion(conv);
}

function exportConversation() {
  const conv = currentConv();
  if (!conv || !conv.messages.length) { toast("Nothing to export", true); return; }
  const lines = [`# ${conv.title}`, ""];
  for (const m of conv.messages) {
    lines.push(`**${m.role === "user" ? "You" : "Model"}:**`, "", msgText(m), "");
    if (msgImages(m).length) lines.push(`*[${msgImages(m).length} image(s) attached]*`, "");
  }
  const blob = new Blob([lines.join("\n")], { type: "text/markdown" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = conv.title.replace(/[^\w\- ]+/g, "").trim().replace(/\s+/g, "_") + ".md";
  a.click();
  URL.revokeObjectURL(a.href);
}

$("chat-send").onclick = () => {
  if (chat.abort) { chat.abort.abort(); return; }
  sendChat();
};
$("chat-input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); sendChat(); }
});
$("chat-input").addEventListener("input", (e) => autoGrow(e.target));
$("toggle-params").onclick = () => $("params").classList.toggle("open");
$("export-conv").onclick = exportConversation;
$("compact-conv").onclick = async () => {
  const conv = currentConv();
  if (!conv || conv.messages.length <= COMPACT_KEEP) {
    toast("Nothing to compact yet", true);
    return;
  }
  if (chat.abort) { toast("Wait for the current reply to finish", true); return; }
  await compactConversation(conv);
};
$("new-conv").onclick = () => { newConversation(); showView("chat"); };

/* ================================================================ */
/*  Coder — multi-session                                            */
/* ================================================================ */

const coder = {
  sessions: new Map(),   // id → {info, feedEl, busy, liveBody, liveText, pendingCards, gen}
  activeId: null,
};

function activeSession() {
  return coder.activeId ? coder.sessions.get(coder.activeId) : null;
}

function sessionLabel(info) {
  const dir = info.cwd.split(/[\\/]/).filter(Boolean).pop() || info.cwd;
  return `${dir} (${info.id.slice(0, 6)})`;
}

function renderSessionSelect() {
  const sel = $("session-select");
  sel.innerHTML = "";
  for (const [id, s] of coder.sessions) {
    const opt = document.createElement("option");
    opt.value = id;
    opt.textContent = sessionLabel(s.info) + (s.busy ? " ⏳" : "");
    if (id === coder.activeId) opt.selected = true;
    sel.appendChild(opt);
  }
}

function showCoderUI(hasSession) {
  $("coder-setup").style.display = hasSession ? "none" : "block";
  $("coder-composer").style.display = hasSession ? "block" : "none";
  $("coder-bar").classList.toggle("open", hasSession);
}

function activateSession(id) {
  coder.activeId = id;
  for (const [sid, s] of coder.sessions) {
    s.feedEl.classList.toggle("active", sid === id);
  }
  const s = coder.sessions.get(id);
  if (s) {
    $("coder-cwd").textContent = s.info.cwd;
    $("coder-state").textContent = s.busy ? "working…" : "idle";
    $("coder-usage").textContent = s.info.total_tokens
      ? `${s.info.total_tokens} tok · turn ${s.info.turns}` : "";
  }
  renderSessionSelect();
  showCoderUI(!!s);
}

function registerSession(info, { replay }) {
  const feedEl = el("div", "coder-feed");
  $("coder-feeds").appendChild(feedEl);
  const s = {
    info,
    feedEl,
    busy: info.busy || false,
    liveBody: null,
    liveText: "",
    pendingCards: [],
    closed: false,
  };
  coder.sessions.set(info.id, s);
  streamSession(s, replay);
  return s;
}

/* per-session feed helpers */

function feedAppend(s, node) {
  const stick = nearBottom(s.feedEl);
  s.feedEl.appendChild(node);
  if (stick) s.feedEl.scrollTop = s.feedEl.scrollHeight;
}

function startAssistantBlock(s) {
  if (s.liveBody) return;
  const { body } = addMessageRow(s.feedEl, "assistant", "");
  s.liveBody = body;
  s.liveText = "";
}

function flushAssistantBlock(s) {
  s.liveBody = null;
  s.liveText = "";
}

function renderDiff(text) {
  const pre = el("pre", "diff");
  for (const line of (text || "").split("\n")) {
    let cls = "";
    if (line.startsWith("+") && !line.startsWith("+++")) cls = "add";
    else if (line.startsWith("-") && !line.startsWith("---")) cls = "del";
    else if (line.startsWith("@@")) cls = "hunk";
    pre.appendChild(el("span", cls, line + "\n"));
  }
  return pre;
}

function buildToolCard(ev) {
  const card = el("div", "tool-card");
  const inner = el("div", "inner");
  const head = el("div", "head");
  head.appendChild(el("span", "name", ev.tool));
  const hintVal = ev.args?.path || ev.args?.command || ev.args?.pattern || ev.args?.url || "";
  head.appendChild(el("span", "hint", String(hintVal).slice(0, 120)));
  head.appendChild(el("span", "state", "…"));
  const body = el("div", "body");
  if (ev.diff) {
    body.appendChild(renderDiff(ev.diff));
  } else {
    body.textContent = JSON.stringify(ev.args, null, 2);
  }
  head.onclick = () => card.classList.toggle("open");
  inner.appendChild(head);
  inner.appendChild(body);
  card.appendChild(inner);
  return card;
}

function buildConfirmCard(s, ev) {
  const card = el("div", "confirm-card");
  const inner = el("div", "inner");
  const title = el("div", "title");
  title.appendChild(document.createTextNode("Approve "));
  title.appendChild(el("span", "name", ev.tool));
  title.appendChild(document.createTextNode("?"));
  inner.appendChild(title);
  if (ev.diff) {
    inner.appendChild(renderDiff(ev.diff));
  } else {
    inner.appendChild(el("pre", "diff", JSON.stringify(ev.args, null, 2)));
  }
  const buttons = el("div", "buttons");
  const yes = el("button", "btn-approve", "Approve");
  const no = el("button", "btn-reject", "Reject");
  const answer = async (approved) => {
    try {
      await fetch(`/api/coder/sessions/${s.info.id}/confirm`, {
        method: "POST",
        headers: authHeaders(),
        body: JSON.stringify({ confirm_id: ev.confirm_id, approved }),
      });
      card.classList.add("answered");
      title.textContent = (approved ? "✓ Approved " : "✗ Rejected ") + ev.tool;
    } catch (e) {
      toast("Failed to answer confirmation: " + e.message, true);
    }
  };
  yes.onclick = () => answer(true);
  no.onclick = () => answer(false);
  buttons.appendChild(yes);
  buttons.appendChild(no);
  inner.appendChild(buttons);
  card.appendChild(inner);
  return card;
}

function handleCoderEvent(s, ev) {
  switch (ev.type) {
    case "token": {
      startAssistantBlock(s);
      s.liveText += ev.text;
      const stick = nearBottom(s.feedEl);
      renderMarkdown(s.liveBody, s.liveText);
      if (stick) s.feedEl.scrollTop = s.feedEl.scrollHeight;
      break;
    }
    case "turn": {
      flushAssistantBlock(s);
      s.busy = true;
      s.info.turns = ev.turn;
      s.info.total_tokens = ev.total_tokens;
      if (s.info.id === coder.activeId) {
        $("coder-state").textContent = "working…";
        if (ev.total_tokens)
          $("coder-usage").textContent = `${ev.total_tokens} tok · turn ${ev.turn}`;
      }
      break;
    }
    case "tool_call": {
      flushAssistantBlock(s);
      const card = buildToolCard(ev);
      feedAppend(s, card);
      s.pendingCards.push(card);
      break;
    }
    case "tool_result": {
      const card = s.pendingCards.shift();
      if (card) {
        const state = card.querySelector(".state");
        state.textContent = ev.summary || (ev.ok ? "ok" : "failed");
        state.className = "state " + (ev.ok ? "ok" : "fail");
        if (ev.output && !card.querySelector(".body .diff")) {
          card.querySelector(".body").textContent = ev.output;
        }
      }
      break;
    }
    case "confirm_request": {
      flushAssistantBlock(s);
      feedAppend(s, buildConfirmCard(s, ev));
      break;
    }
    case "user": {
      // replayed user message (emitted client-side on send; replay rebuilds it)
      flushAssistantBlock(s);
      addMessageRow(s.feedEl, "user", ev.text);
      break;
    }
    case "info": {
      flushAssistantBlock(s);
      feedAppend(s, el("div", "feed-info", ev.text));
      break;
    }
    case "replay_done": {
      flushAssistantBlock(s);
      s.feedEl.scrollTop = s.feedEl.scrollHeight;
      break;
    }
    case "final": {
      flushAssistantBlock(s);
      s.busy = false;
      s.info.turns = ev.turns;
      s.info.total_tokens = ev.total_tokens;
      if (s.info.id === coder.activeId) $("coder-state").textContent = "idle";
      renderSessionSelect();
      feedAppend(s, el("div", "feed-final",
        (ev.ok ? "Task finished" : "Task ended") +
        ` — ${ev.turns} turns, ${ev.total_tokens} tokens`));
      break;
    }
    case "error": {
      flushAssistantBlock(s);
      s.busy = false;
      if (s.info.id === coder.activeId) $("coder-state").textContent = "error";
      toast("Agent error: " + ev.text, true);
      break;
    }
    case "closed": {
      s.busy = false;
      s.closed = true;
      break;
    }
  }
}

async function streamSession(s, replay) {
  while (coder.sessions.has(s.info.id) && !s.closed) {
    try {
      const r = await fetch(
        `/api/coder/sessions/${s.info.id}/events${replay ? "?replay=true" : ""}`,
        { headers: authHeaders() });
      if (r.status === 404) { s.closed = true; break; }
      if (!r.ok) throw new Error(r.statusText);
      replay = false;   // only the first connection replays
      await readSSE(r, (payload) => {
        let ev;
        try { ev = JSON.parse(payload); } catch { return; }
        if (coder.sessions.has(s.info.id)) handleCoderEvent(s, ev);
      });
    } catch (e) {
      if (!coder.sessions.has(s.info.id) || s.closed) return;
      await new Promise((res) => setTimeout(res, 1500));
    }
  }
}

/* session lifecycle */

function populateSetupModels() {
  const sel = $("setup-model");
  sel.innerHTML = "";
  const current = document.createElement("option");
  current.value = "";
  current.textContent = "active model (" + (modelCache.active || "?") + ")";
  sel.appendChild(current);
  for (const m of modelCache.models) {
    if (m.active) continue;
    const opt = document.createElement("option");
    opt.value = m.name;
    opt.textContent = m.name;
    sel.appendChild(opt);
  }
}

async function startCoderSession() {
  const cwd = $("setup-cwd").value.trim();
  if (!cwd) { toast("Enter a project directory", true); return; }
  $("setup-start").disabled = true;
  try {
    const body = {
      cwd,
      auto_approve: $("setup-auto").checked,
      mode: $("setup-mode").value,
      max_turns: Number($("setup-max-turns").value) || 40,
    };
    const model = $("setup-model").value;
    if (model) body.model = model;
    const temp = $("setup-temperature").value.trim();
    if (temp !== "") body.temperature = Number(temp);
    const scope = $("setup-scope").value.trim();
    if (scope) body.scope = scope;

    const r = await fetch("/api/coder/sessions", {
      method: "POST",
      headers: authHeaders(),
      body: JSON.stringify(body),
    });
    if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
    const info = await r.json();
    localStorage.setItem("localm.coderCwd", cwd);
    registerSession(info, { replay: false });
    activateSession(info.id);
    refreshModels();
  } catch (e) {
    toast("Failed to start session: " + e.message, true);
  } finally {
    $("setup-start").disabled = false;
  }
}

async function reattachSessions() {
  try {
    const r = await fetch("/api/coder/sessions", { headers: authHeaders() });
    if (!r.ok) return;
    const data = await r.json();
    for (const info of data.sessions) {
      if (!coder.sessions.has(info.id)) {
        registerSession(info, { replay: true });
      }
    }
    if (!coder.activeId && data.sessions.length) {
      activateSession(data.sessions[data.sessions.length - 1].id);
      toast("Reattached to a running coder session");
    }
  } catch (e) { /* server unreachable; startup poller will retry models anyway */ }
}

async function sendCoderTask() {
  const s = activeSession();
  const input = $("coder-input");
  const text = input.value.trim();
  if (!text || !s) return;
  if (s.busy) { toast("Agent is still working — stop it first or wait", true); return; }
  try {
    const r = await fetch(`/api/coder/sessions/${s.info.id}/message`, {
      method: "POST",
      headers: authHeaders(),
      body: JSON.stringify({ text }),
    });
    if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
    s.busy = true;
    $("coder-state").textContent = "working…";
    // The user message arrives back through the event stream (so replay
    // works after a page reload) — no client-side row here.
    renderSessionSelect();
    input.value = "";
    autoGrow(input);
  } catch (e) {
    toast("Failed to send task: " + e.message, true);
  }
}

async function endCoderSession() {
  const s = activeSession();
  if (!s) return;
  try {
    await fetch(`/api/coder/sessions/${s.info.id}`, {
      method: "DELETE", headers: authHeaders() });
  } catch (e) { /* server may already be gone */ }
  s.closed = true;
  s.feedEl.remove();
  coder.sessions.delete(s.info.id);
  const remaining = [...coder.sessions.keys()];
  coder.activeId = remaining[remaining.length - 1] || null;
  activateSession(coder.activeId);
}

/* coder bar buttons */

$("session-select").onchange = () => activateSession($("session-select").value);
$("session-new").onclick = () => {
  populateSetupModels();
  showCoderUI(false);
  $("coder-bar").classList.add("open");   // keep the bar so sessions stay reachable
};
$("setup-start").onclick = startCoderSession;
$("coder-send").onclick = sendCoderTask;
$("coder-input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); sendCoderTask(); }
});
$("coder-input").addEventListener("input", (e) => autoGrow(e.target));

$("coder-stop").onclick = async () => {
  const s = activeSession();
  if (!s) return;
  await fetch(`/api/coder/sessions/${s.info.id}/stop`, {
    method: "POST", headers: authHeaders() });
  toast("Stop requested — agent halts at the next safe point");
};
$("coder-end").onclick = endCoderSession;

$("coder-undo").onclick = async () => {
  const s = activeSession();
  if (!s) return;
  const r = await fetch(`/api/coder/sessions/${s.info.id}/undo`, {
    method: "POST", headers: authHeaders() });
  const data = await r.json();
  if (r.ok) {
    toast(data.summary);
    feedAppend(s, el("div", "feed-info", data.summary));
  } else {
    toast(data.detail || "Nothing to undo", true);
  }
};

$("coder-compact").onclick = async () => {
  const s = activeSession();
  if (!s) return;
  const r = await fetch(`/api/coder/sessions/${s.info.id}/compact`, {
    method: "POST", headers: authHeaders() });
  const data = await r.json();
  if (r.ok) {
    toast("History compacted");
    feedAppend(s, el("div", "feed-info", "Conversation history compacted."));
  } else {
    toast(data.detail || "Nothing to compact", true);
  }
};

$("coder-log").onclick = async () => {
  const s = activeSession();
  if (!s) return;
  const r = await fetch(`/api/coder/sessions/${s.info.id}/log`, {
    headers: authHeaders() });
  const data = await r.json();
  if (!r.ok) { toast(data.detail || "No log available", true); return; }
  openModal("Audit log — " + sessionLabel(s.info), (body) => {
    body.appendChild(el("div", "sub", data.path));
    for (const entry of data.entries) {
      const row = el("div", "log-entry");
      const ts = new Date(entry.t).toLocaleTimeString();
      row.appendChild(el("span", "t", `${ts} #${entry.turn} ${entry.type}`));
      row.appendChild(document.createTextNode(JSON.stringify(entry.data)));
      body.appendChild(row);
    }
    if (!data.entries.length) body.appendChild(el("div", "sub", "(empty)"));
  });
};

/* ================================================================ */
/*  Init                                                             */
/* ================================================================ */

$("setup-cwd").value = localStorage.getItem("localm.coderCwd") || "";
refreshModels().then(() => populateSetupModels());
refreshCtxLimit();
setInterval(refreshModels, 30000);
renderConvList();
if (chat.conversations.length) {
  chat.activeId = chat.conversations[0].id;
  renderConvList();
}
renderChat();
reattachSessions();
