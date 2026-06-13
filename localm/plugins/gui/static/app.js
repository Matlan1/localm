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

/** Split leading <think>…</think> reasoning from the visible reply.
 *  Handles a still-open block during streaming. */
function splitThink(text) {
  const m = /<think>([\s\S]*?)(?:<\/think>([\s\S]*)|$)/.exec(text || "");
  if (!m) return { think: null, open: false, rest: text || "" };
  const closed = m[2] !== undefined;
  const before = text.slice(0, m.index);
  return {
    think: m[1].trim(),
    open: !closed,
    rest: before + (closed ? m[2] : ""),
  };
}

/** Reply text with reasoning blocks removed — what gets sent back to the
 *  model on later turns. */
function stripThink(text) {
  return (text || "").replace(/<think>[\s\S]*?(<\/think>|$)/g, "").trim();
}

/** Replace raw <tool_call> JSON blocks with a compact human-readable note —
 *  shown while the web-access loop executes the request. */
function formatToolCalls(text) {
  return (text || "").replace(
    /<tool_call>\s*([\s\S]*?)\s*<\/tool_call>/g,
    (m, body) => {
      try {
        const call = JSON.parse(body);
        const a = call.args || call.arguments || {};
        const what =
          call.name === "web_search" ? `web search: "${a.query || ""}"` :
          call.name === "fetch_url"  ? `read page: ${a.url || ""}` :
          String(call.name || "request");
        return `\n> 🌐 *${what}*\n`;
      } catch (e) {
        return "\n> 🌐 *web request*\n";
      }
    });
}

function renderMarkdown(target, text) {
  const { think, open, rest: rawRest } = splitThink(text);
  const rest = formatToolCalls(rawRest);
  target.innerHTML = "";
  if (think) {
    const det = document.createElement("details");
    det.className = "think-block";
    if (open) det.open = true;
    const sum = document.createElement("summary");
    sum.textContent = open ? "Thinking…" : "Thoughts";
    det.appendChild(sum);
    const body = document.createElement("div");
    body.innerHTML = DOMPurify.sanitize(marked.parse(think));
    det.appendChild(body);
    target.appendChild(det);
  }
  const main = document.createElement("div");
  main.innerHTML = DOMPurify.sanitize(marked.parse(rest || ""));
  while (main.firstChild) target.appendChild(main.firstChild);
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

/** Stream a background job's events. onLine gets text lines; the optional
 *  onProgress gets {downloaded,total,pct,phase} events. Resolves with end. */
async function streamJob(jobId, onLine, onProgress) {
  const r = await fetch(`/api/jobs/${jobId}/events`, { headers: authHeaders() });
  if (!r.ok) throw new Error(r.statusText);
  let endEvent = null;
  await readSSE(r, (payload) => {
    let ev;
    try { ev = JSON.parse(payload); } catch { return; }
    if (ev.type === "line" && onLine) onLine(ev.text);
    if (ev.type === "progress" && onProgress) onProgress(ev);
    if (ev.type === "end") endEvent = ev;
  });
  return endEvent || { status: "failed" };
}

// Sizes are shown in binary units (GiB/MiB/KiB) but labelled GB/MB/KB — the
// GPU/LLM convention: matches the VRAM printed on the card, llama.cpp's logs,
// and HuggingFace quant tables. (The driver reports e.g. 16 GiB; showing
// decimal GB would read a confusing 17.2 for the same card.)
const GIB = 1024 ** 3, MIB = 1024 ** 2, KIB = 1024;

function fmtBytes(n) {
  if (n == null) return "";
  if (n >= GIB) return (n / GIB).toFixed(2) + " GB";
  if (n >= MIB) return (n / MIB).toFixed(1) + " MB";
  if (n >= KIB) return (n / KIB).toFixed(0) + " KB";
  return n + " B";
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

const VIEWS = ["chat", "coder", "models", "images", "music", "video", "knowledge", "plugins", "settings"];

function showView(name) {
  if (!VIEWS.includes(name)) name = "chat";
  for (const v of VIEWS) {
    $("view-" + v).classList.toggle("active", v === name);
    $("nav-" + v).classList.toggle("active", v === name);
  }
  // Remembered across reloads — but never in privacy mode (no traces).
  if (!chat.privacy) localStorage.setItem("localm.activeView", name);
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
        const size = m.size_bytes ? ` (${(m.size_bytes / GIB).toFixed(1)} GB)` : "";
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
  attachments: [],   // image attachments: {name, dataUri}
  docs: [],          // document attachments: {name, text, chars, truncated}
  ctxMax: 16384,     // context ceiling — refreshed from /v1/config
  privacy: false,    // server in privacy mode → conversations not persisted
  persist: false,    // non-privacy: conversations sync to the server store
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
  pruneBranches(conv);   // forks anchored in the summarised-away region die
  saveConversations(conv);
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
      // Prefer the resolved ceiling (VRAM-derived under ctx_auto) over the
      // static config value — compaction should track what the model can
      // actually hold.
      chat.ctxMax = cfg.effective_ctx_max ?? cfg.n_ctx_max ?? 16384;
      // Privacy mode: conversations live in memory only — wipe anything a
      // previous non-privacy session left behind and show the hint.
      chat.privacy = cfg.effective_mode === "privacy";
      if (chat.privacy) {
        localStorage.removeItem("localm.conversations");
        localStorage.removeItem("localm.activeView");
        localStorage.removeItem("localm.coderCwd");
        localStorage.removeItem("localm.kbAddPath");
        localStorage.removeItem("localm.convCollapsed");
        localStorage.removeItem("localm.imgMoveDest");
        localStorage.removeItem("localm.musicMoveDest");
        localStorage.removeItem("localm.videoMoveDest");
        const h = document.querySelector("#conversations h3");
        if (h && !document.getElementById("privacy-hint")) {
          const hint = document.createElement("div");
          hint.id = "privacy-hint";
          hint.className = "privacy-hint";
          hint.textContent = "privacy mode — this session only";
          hint.title = "The server runs in privacy mode: conversations are " +
            "not saved (here or on disk) and vanish on reload. Export still works.";
          h.after(hint);
        }
      }
    }
  } catch (e) { /* keep default */ }
}

function saveConversations(changed) {
  if (chat.privacy) return;   // privacy mode: no traces, not even localStorage
  try {
    localStorage.setItem("localm.conversations",
      JSON.stringify(chat.conversations.slice(0, 50)));
  } catch (e) {
    // Quota: drop image-heavy older conversations and retry once
    const slim = chat.conversations.slice(0, 10);
    try { localStorage.setItem("localm.conversations", JSON.stringify(slim)); } catch {}
  }
  if (changed) pushConversation(changed);
}

/* ---- server-side conversation store (non-privacy modes only) ---- */

const _convPushTimers = new Map();

/** Debounced upsert of one conversation to the server store. */
function pushConversation(conv) {
  if (!chat.persist) return;
  // Brand-new conversations with nothing in them yet aren't worth a file.
  if (!conv.messages.length && conv.title === "New chat") return;
  conv.updated_at = Date.now();
  clearTimeout(_convPushTimers.get(conv.id));
  _convPushTimers.set(conv.id, setTimeout(async () => {
    _convPushTimers.delete(conv.id);
    try {
      await fetch("/api/conversations/" + encodeURIComponent(conv.id), {
        method: "PUT",
        headers: authHeaders(),
        body: JSON.stringify({ title: conv.title,
                               updated_at: conv.updated_at,
                               pinned: !!conv.pinned,
                               folder: conv.folder || null,
                               branches: conv.branches || [],
                               messages: conv.messages }),
      });
    } catch (e) { /* offline — localStorage still has the copy */ }
  }, 600));
}

function deleteConversationRemote(convId) {
  if (!chat.persist) return;
  clearTimeout(_convPushTimers.get(convId));
  _convPushTimers.delete(convId);
  fetch("/api/conversations/" + encodeURIComponent(convId), {
    method: "DELETE", headers: authHeaders(),
  }).catch(() => {});
}

/** Load the server store and merge with the localStorage cache: the newer
 *  copy of each conversation wins; local-only ones are uploaded. */
async function initServerConversations() {
  if (chat.privacy) return;
  try {
    const r = await fetch("/api/conversations", { headers: authHeaders() });
    if (!r.ok) return;
    const data = await r.json();
    if (!data.enabled) return;
    chat.persist = true;
    const byId = new Map(data.conversations.map((c) => [c.id, c]));
    for (const local of chat.conversations) {
      const remote = byId.get(local.id);
      if (!remote || (local.updated_at || 0) > (remote.updated_at || 0)) {
        byId.set(local.id, local);
        pushConversation(local);
      }
    }
    chat.conversations = [...byId.values()]
      .sort((a, b) => (b.updated_at || 0) - (a.updated_at || 0));
    if (!chat.activeId && chat.conversations.length) {
      chat.activeId = chat.conversations[0].id;
    }
    saveConversations();
    renderConvList();
    renderChat();
    const h = document.querySelector("#conversations h3");
    if (h && !document.getElementById("persist-hint")) {
      const hint = document.createElement("div");
      hint.id = "persist-hint";
      hint.className = "privacy-hint";
      hint.textContent = "history saved on this machine";
      hint.title = "Conversations are stored in the localm data directory " +
        "(chats/) because the server runs in log or full mode. They survive " +
        "browser reloads and profile wipes.";
      h.after(hint);
    }
  } catch (e) { /* store unavailable — localStorage keeps working */ }
}

function currentConv() {
  return chat.conversations.find((c) => c.id === chat.activeId) || null;
}

function newConversation() {
  const conv = { id: Date.now().toString(36), title: "New chat", messages: [] };
  chat.conversations.unshift(conv);
  chat.activeId = conv.id;
  saveConversations(conv);
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

/* ---- conversation list: search, pin, folders ---- */

const convUI = {
  search: "",
  collapsed: new Set(JSON.parse(
    localStorage.getItem("localm.convCollapsed") || "[]")),
};

function saveCollapsed() {
  if (chat.privacy) return;   // folder names are conversation-derived
  localStorage.setItem("localm.convCollapsed",
    JSON.stringify([...convUI.collapsed]));
}

/** Short excerpt around the first content match, or null (no match). */
function searchSnippet(conv, term) {
  for (const m of conv.messages) {
    const text = msgText(m);
    const idx = text.toLowerCase().indexOf(term);
    if (idx !== -1) {
      const start = Math.max(0, idx - 18);
      return (start > 0 ? "…" : "") +
        text.slice(start, idx + 50).replace(/\s+/g, " ");
    }
  }
  return null;
}

function buildConvItem(conv, snippet) {
  const item = el("div", "conv-item" + (conv.id === chat.activeId ? " active" : ""));
  const title = el("span", "title");
  title.appendChild(document.createTextNode(conv.title));
  if (snippet) title.appendChild(el("span", "snippet", snippet));
  title.ondblclick = (e) => {
    e.stopPropagation();
    const input = document.createElement("input");
    input.value = conv.title;
    const commit = () => {
      conv.title = input.value.trim() || conv.title;
      saveConversations(conv);
      renderConvList();
    };
    input.onblur = commit;
    input.onkeydown = (ke) => { if (ke.key === "Enter") input.blur(); };
    item.replaceChild(input, title);
    input.focus();
    input.select();
  };
  item.appendChild(title);

  const pin = el("button", "del" + (conv.pinned ? " pinned-btn" : ""), "📌");
  pin.title = conv.pinned ? "Unpin" : "Pin to top";
  pin.onclick = (e) => {
    e.stopPropagation();
    conv.pinned = !conv.pinned;
    if (!conv.pinned) delete conv.pinned;
    saveConversations(conv);
    renderConvList();
  };
  item.appendChild(pin);

  const fold = el("button", "del", "📁");
  fold.title = conv.folder ? `Folder: ${conv.folder} (click to change)`
                           : "Move to a folder";
  fold.onclick = (e) => {
    e.stopPropagation();
    const name = prompt("Folder name (empty removes it from the folder):",
                        conv.folder || "");
    if (name === null) return;
    if (name.trim()) conv.folder = name.trim();
    else delete conv.folder;
    saveConversations(conv);
    renderConvList();
  };
  item.appendChild(fold);

  const del = el("button", "del", "×");
  del.title = "Delete conversation";
  del.onclick = (e) => {
    e.stopPropagation();
    chat.conversations = chat.conversations.filter((c) => c.id !== conv.id);
    if (chat.activeId === conv.id) chat.activeId = chat.conversations[0]?.id || null;
    deleteConversationRemote(conv.id);
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
  return item;
}

function renderConvList() {
  const list = $("conv-list");
  list.replaceChildren();
  const term = convUI.search.trim().toLowerCase();

  // Filter: title match shows plain; content match shows a snippet
  const visible = [];
  for (const conv of chat.conversations) {
    if (!term) {
      visible.push({ conv, snippet: null });
    } else if ((conv.title || "").toLowerCase().includes(term)) {
      visible.push({ conv, snippet: null });
    } else {
      const snippet = searchSnippet(conv, term);
      if (snippet !== null) visible.push({ conv, snippet });
    }
  }

  // Group: pinned on top, then folders (A-Z), then the rest
  const pinned = visible.filter((v) => v.conv.pinned);
  const folders = new Map();
  const loose = [];
  for (const v of visible) {
    if (v.conv.pinned) continue;
    if (v.conv.folder) {
      if (!folders.has(v.conv.folder)) folders.set(v.conv.folder, []);
      folders.get(v.conv.folder).push(v);
    } else {
      loose.push(v);
    }
  }

  const addGroup = (label, key, items) => {
    if (!items.length) return;
    if (label) {
      // While searching, groups stay expanded so matches are never hidden
      const collapsed = !term && convUI.collapsed.has(key);
      const head = el("div", "conv-group", (collapsed ? "▸ " : "▾ ") + label);
      head.onclick = () => {
        if (convUI.collapsed.has(key)) convUI.collapsed.delete(key);
        else convUI.collapsed.add(key);
        saveCollapsed();
        renderConvList();
      };
      list.appendChild(head);
      if (collapsed) return;
    }
    for (const v of items) list.appendChild(buildConvItem(v.conv, v.snippet));
  };

  addGroup(pinned.length ? "📌 pinned" : "", "::pinned", pinned);
  for (const name of [...folders.keys()].sort()) {
    addGroup("📁 " + name, "f:" + name, folders.get(name));
  }
  addGroup((pinned.length || folders.size) && loose.length ? "chats" : "",
           "::chats", loose);

  if (term && !visible.length) {
    list.appendChild(el("div", "privacy-hint", "no matching chats"));
  }
}

$("conv-search").addEventListener("input", (e) => {
  convUI.search = e.target.value;
  renderConvList();
});

function addMessageRow(container, role, text, opts = {}) {
  const row = el("div", "msg-row " + role + (opts.cls ? " " + opts.cls : ""));
  row.appendChild(el("div", "msg-role",
    opts.label || (role === "user" ? "You" : "Model")));
  const body = el("div", "msg-body");
  renderMarkdown(body, text);
  for (const url of opts.images || []) {
    const img = document.createElement("img");
    img.className = "msg-img";
    if (url.startsWith("/api/")) {
      // server-side generated image — fetch with auth headers
      fetchImageURL(url).then((u) => (img.src = u)).catch(() => img.remove());
    } else {
      img.src = url;   // data: URI from the user's own attachment
    }
    body.appendChild(img);
  }
  for (const url of opts.audio || []) {
    const player = document.createElement("audio");
    player.controls = true;
    player.style.width = "100%";
    // bearer-protected file — fetch as a blob with auth headers
    fetchImageURL(url).then((u) => (player.src = u)).catch(() => player.remove());
    body.appendChild(player);
  }
  for (const url of opts.video || []) {
    const player = document.createElement("video");
    player.controls = true;
    player.style.width = "100%";
    player.style.borderRadius = "8px";
    fetchImageURL(url).then((u) => (player.src = u)).catch(() => player.remove());
    body.appendChild(player);
  }
  row.appendChild(body);
  const meta = el("div", "msg-meta");
  const copy = el("button", "copy-btn", "copy");
  copy.onclick = () => {
    navigator.clipboard.writeText(stripThink(text) || text);
    copy.textContent = "copied";
    setTimeout(() => (copy.textContent = "copy"), 1200);
  };
  meta.appendChild(copy);
  if (opts.variant) {
    const nav = el("span", "variant");
    const prev = el("button", "action", "‹");
    prev.title = "Previous variant";
    prev.onclick = opts.variant.prev;
    const next = el("button", "action", "›");
    next.title = "Next variant";
    next.onclick = opts.variant.next;
    nav.appendChild(prev);
    nav.appendChild(el("span", "k", `${opts.variant.k}/${opts.variant.n}`));
    nav.appendChild(next);
    meta.appendChild(nav);
  }
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
  const tip = el("div", "", "Type / for commands — /imagine generates images locally.");
  tip.style.marginTop = "10px";
  tip.style.fontSize = "13px";
  div.appendChild(tip);
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
  const NOTE_LABELS = { web: "Web", doc: "Doc", kb: "Sources" };
  conv.messages.forEach((m, i) => {
    const tag = m.tag || (m.web ? "web" : null);
    const actions = [];
    if (m.role === "user" && !tag) {
      actions.push(["edit", () => editMessage(conv, i)]);
    }
    if (m.role === "assistant" && !tag) {
      actions.push(["🔊", () => speak(msgText(m), { toggle: true })]);
    }
    if (m.role === "assistant" && i === conv.messages.length - 1 && !chat.abort) {
      actions.push(["regenerate", () => regenerate(conv)]);
    }
    // ‹ k/N › on the first message of a fork point with siblings
    let variant = null;
    const pid = i > 0 ? conv.messages[i - 1].id : "root";
    const rec = (conv.branches || []).find((b) => b.parent === pid);
    if (rec && rec.tails.length > 1) {
      variant = {
        k: rec.current + 1,
        n: rec.tails.length,
        prev: () => switchBranch(conv, i, -1),
        next: () => switchBranch(conv, i, +1),
      };
    }
    const noteSuffix = m.truncated
      ? "\n\n*[stopped at the max-tokens limit — raise “Max tokens” in ⚙ parameters, or reply “continue”]*"
      : "";
    addMessageRow(box, m.role, msgText(m) + noteSuffix, {
      images: msgImages(m),
      audio: m.audio ? [m.audio] : [],
      video: m.video ? [m.video] : [],
      actions,
      variant,
      cls: tag ? "web-note" : "",
      label: tag ? NOTE_LABELS[tag] : undefined,
    });
  });
  box.scrollTop = box.scrollHeight;
}

/* ---- message branching ----
   conv.messages is always the LIVE linear branch (so compaction, retrieval
   injection, export, and the API mapping stay untouched). Alternative
   timelines are parked at fork points:
     conv.branches = [{parent: <msg id or "root">, tails: [[msg…]…], current}]
   The slot at `current` belongs to the live tail and is only written back
   when switching away. Editing a message or regenerating a reply parks the
   old tail as a sibling instead of destroying it; ‹ k/N › in the message
   meta row navigates between siblings. */

let _msgIdCounter = 0;

function msgId(m) {
  if (!m.id) m.id = Date.now().toString(36) + "-" + (_msgIdCounter++);
  return m.id;
}

function parentIdAt(conv, index) {
  return index > 0 ? msgId(conv.messages[index - 1]) : "root";
}

function forkRecord(conv, parentId, create) {
  conv.branches = conv.branches || [];
  let rec = conv.branches.find((b) => b.parent === parentId);
  if (!rec && create) {
    rec = { parent: parentId, tails: [], current: 0 };
    conv.branches.push(rec);
  }
  return rec;
}

/** Park the live tail from *index* as a sibling and open a fresh timeline. */
function forkAt(conv, index) {
  const rec = forkRecord(conv, parentIdAt(conv, index), true);
  if (!rec.tails.length) {
    rec.tails.push(null);          // slot for the pre-existing timeline
    rec.current = 0;
  }
  rec.tails[rec.current] = conv.messages.slice(index);
  rec.tails.push(null);            // slot owned by the new live timeline
  rec.current = rec.tails.length - 1;
  conv.messages = conv.messages.slice(0, index);
}

/** Switch the fork at *index* one sibling left/right (dir = ±1). */
function switchBranch(conv, index, dir) {
  if (chat.abort) { toast("Wait for the current reply to finish", true); return; }
  const rec = forkRecord(conv, parentIdAt(conv, index), false);
  if (!rec || rec.tails.length < 2) return;
  const n = rec.tails.length;
  const next = (rec.current + dir + n) % n;
  if (next === rec.current) return;
  rec.tails[rec.current] = conv.messages.slice(index);   // park live tail
  conv.messages = conv.messages.slice(0, index).concat(rec.tails[next] || []);
  rec.current = next;
  saveConversations(conv);
  renderChat();
}

/** Drop fork records whose parent message no longer exists anywhere
 *  (active branch or any parked tail) — called after compaction rewrites
 *  old history. */
function pruneBranches(conv) {
  if (!conv.branches || !conv.branches.length) return;
  const ids = new Set(["root"]);
  for (const m of conv.messages) if (m.id) ids.add(m.id);
  for (const rec of conv.branches) {
    for (const tail of rec.tails) {
      for (const m of tail || []) if (m.id) ids.add(m.id);
    }
  }
  conv.branches = conv.branches.filter((b) => ids.has(b.parent));
}

function editMessage(conv, index) {
  const m = conv.messages[index];
  $("chat-input").value = msgText(m);
  autoGrow($("chat-input"));
  // Fork instead of destroy: the old timeline (this message and everything
  // after it) stays reachable via ‹ › once the edited version is sent.
  forkAt(conv, index);
  saveConversations(conv);
  renderChat();
  $("chat-input").focus();
}

function regenerate(conv) {
  if (chat.abort) return;
  const last = conv.messages.length - 1;
  if (conv.messages[last]?.role !== "assistant") return;
  forkAt(conv, last);                // park the old reply as a sibling
  saveConversations(conv);
  renderChat();
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
  box.replaceChildren();
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
  chat.docs.forEach((doc, i) => {
    const chip = el("span", "chip");
    chip.appendChild(el("span", "", "📄 " + doc.name +
      ` (${(doc.chars / 1000).toFixed(1)}k chars${doc.truncated ? ", trimmed" : ""})`));
    const rm = el("button", "", "×");
    rm.onclick = () => { chat.docs.splice(i, 1); renderAttachChips(); };
    chip.appendChild(rm);
    box.appendChild(chip);
  });
}

/** Document attachment → text via /api/rag/extract. The file is converted
 *  in memory on the server and never written to disk (privacy-clean). */
async function attachDocument(file) {
  const b64 = await new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result).split(",", 2)[1] || "");
    reader.onerror = () => reject(new Error("could not read file"));
    reader.readAsDataURL(file);
  });
  const r = await fetch("/api/rag/extract", {
    method: "POST", headers: authHeaders(),
    body: JSON.stringify({ filename: file.name, content_b64: b64 }),
  });
  const data = await r.json();
  if (!r.ok) throw new Error(data.detail || r.statusText);
  chat.docs.push({ name: data.filename, text: data.text,
                   chars: data.chars, truncated: data.truncated });
  renderAttachChips();
}

$("chat-attach").onclick = () => $("chat-file").click();
$("chat-file").addEventListener("change", (e) => {
  for (const file of e.target.files) {
    if (file.type.startsWith("image/")) {
      const reader = new FileReader();
      reader.onload = () => {
        chat.attachments.push({ name: file.name, dataUri: reader.result });
        renderAttachChips();
      };
      reader.readAsDataURL(file);
    } else {
      attachDocument(file).catch((err) =>
        toast(`${file.name}: ${err.message}`, true));
    }
  }
  e.target.value = "";
});

/* ---- web access (model-initiated, via the params-drawer toggle) ---- */

const WEB_MAX_ROUNDS = 3;

const WEB_TOOL_PROMPT =
  "You can access the internet. When the answer genuinely needs current " +
  "information from the web, reply with ONLY a tool call block:\n" +
  '<tool_call>{"name": "web_search", "args": {"query": "..."}}</tool_call>\n' +
  "To read a specific page:\n" +
  '<tool_call>{"name": "fetch_url", "args": {"url": "https://..."}}</tool_call>\n' +
  "The results arrive in the next user message. Then answer normally and " +
  "name the sources you used. Do not search for things you already know.";

/** First web tool call in a reply, or null. */
function parseWebCall(text) {
  const m = /<tool_call>\s*([\s\S]*?)\s*<\/tool_call>/.exec(stripThink(text));
  if (!m) return null;
  try {
    const call = JSON.parse(m[1]);
    if (call && (call.name === "web_search" || call.name === "fetch_url")) {
      return call;
    }
  } catch (e) { /* not valid JSON — treat as plain text */ }
  return null;
}

/** Run a web tool call through the policy-enforced server endpoints. */
async function requestWebTool(call) {
  const a = call.args || call.arguments || {};
  if (call.name === "web_search") {
    const r = await fetch("/api/web/search", {
      method: "POST", headers: authHeaders(),
      body: JSON.stringify({ query: a.query || "", max_results: 5 }),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || r.statusText);
    const lines = data.results.map((res, i) =>
      `${i + 1}. ${res.title}\n   ${res.url}` +
      (res.snippet ? `\n   ${res.snippet}` : ""));
    return `[Results of web_search "${a.query}"]\n` + lines.join("\n");
  }
  if (call.name === "fetch_url") {
    const r = await fetch("/api/web/fetch", {
      method: "POST", headers: authHeaders(),
      body: JSON.stringify({ url: a.url || "", max_chars: 6000 }),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || r.statusText);
    return `[Content of ${data.url}]` +
      (data.truncated ? " (truncated)" : "") + `\n${data.text}`;
  }
  throw new Error("Unknown web tool: " + call.name);
}

/** Run one model-requested web call, injecting the result (or the failure,
 *  so the model can adapt) as a dimmed "Web" message. */
async function runWebCall(conv, call) {
  let note;
  try {
    note = await requestWebTool(call);
  } catch (e) {
    note = `[Web request failed: ${e.message}] Answer without the web, ` +
           "and say that web access did not work.";
    toast("Web request failed: " + e.message, true);
  }
  conv.messages.push({ role: "user", content: note, web: true });
  saveConversations(conv);
  renderChat();
}

/* ---- voice: mic (Whisper STT) + read-aloud (browser TTS) ---- */

const voice = { rec: null, chunks: [], available: true, reason: "",
                modelCached: true, model: "" };

/** Grey out the mic up front when the server lacks the [voice] extra,
 *  instead of letting the user record and only then failing. */
async function refreshVoiceStatus() {
  try {
    const r = await fetch("/api/voice/status", { headers: authHeaders() });
    if (!r.ok) return;   // old server without the endpoint — leave enabled
    const data = await r.json();
    voice.available = data.available;
    voice.reason = data.reason || "";
    voice.modelCached = data.model_cached !== false;
    voice.model = data.model || "";
    const btn = $("chat-mic");
    btn.classList.toggle("unavailable", !data.available);
    if (!data.available) btn.title = data.reason;
  } catch (e) { /* server unreachable — status refreshes on next load */ }
}

function blobToB64(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result).split(",", 2)[1] || "");
    reader.onerror = () => reject(new Error("could not read recording"));
    reader.readAsDataURL(blob);
  });
}

async function toggleMic() {
  const btn = $("chat-mic");
  if (voice.rec) {           // second click stops and transcribes
    voice.rec.stop();
    return;
  }
  if (!voice.available) {
    toast(voice.reason || "Speech-to-text is not installed on the server", true);
    return;
  }
  if (!voice.modelCached) {
    // Transcription is fully local, but the FIRST use fetches the Whisper
    // model from HuggingFace — make that one network access explicit.
    if (!confirm(
        `First use downloads the Whisper "${voice.model}" speech model ` +
        "from HuggingFace (one-time). Transcription itself runs fully " +
        "offline afterwards. Download now?")) return;
    voice.modelCached = true;   // consent given — don't re-ask this session
  }
  if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
    toast("This browser does not support audio recording", true);
    return;
  }
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (e) {
    toast("Microphone unavailable: " + e.message, true);
    return;
  }
  voice.chunks = [];
  voice.rec = new MediaRecorder(stream);
  voice.rec.ondataavailable = (e) => { if (e.data.size) voice.chunks.push(e.data); };
  voice.rec.onstop = async () => {
    stream.getTracks().forEach((t) => t.stop());
    btn.classList.remove("recording");
    const blob = new Blob(voice.chunks, { type: voice.rec.mimeType || "audio/webm" });
    voice.rec = null;
    btn.disabled = true;
    btn.textContent = "…";
    try {
      const r = await fetch("/api/voice/transcribe", {
        method: "POST", headers: authHeaders(),
        body: JSON.stringify({ audio_b64: await blobToB64(blob) }),
      });
      const data = await r.json();
      if (!r.ok) throw new Error(data.detail || r.statusText);
      const input = $("chat-input");
      input.value = (input.value ? input.value.trimEnd() + " " : "") + data.text;
      autoGrow(input);
      input.focus();
    } catch (e) {
      toast("Transcription failed: " + e.message, true);
    } finally {
      btn.disabled = false;
      btn.textContent = "🎤";
    }
  };
  voice.rec.start();
  btn.classList.add("recording");
  toast("Recording — click 🎤 again to stop");
}

$("chat-mic").onclick = toggleMic;

/** Read text aloud with the browser's offline voices. With toggle: true
 *  (the 🔊 button) a second call stops instead; auto-speak replaces. */
function speak(text, opts = {}) {
  if (!window.speechSynthesis) {
    toast("This browser has no speech synthesis", true);
    return;
  }
  if (speechSynthesis.speaking) {
    speechSynthesis.cancel();
    if (opts.toggle) return;
  }
  const clean = stripThink(text).replace(/[*_`#>\[\]()]/g, " ");
  if (clean.trim()) speechSynthesis.speak(new SpeechSynthesisUtterance(clean));
}

/* ---- assistant memory ---- */

const memory = { text: "", writable: false };

async function refreshMemory() {
  try {
    const r = await fetch("/api/memory", { headers: authHeaders() });
    if (!r.ok) return;
    const data = await r.json();
    memory.text = data.text || "";
    memory.writable = !!data.writable;
  } catch (e) { /* server unreachable */ }
}

async function rememberFact(fact) {
  if (!fact) { toast("Usage: /remember <fact>", true); return; }
  try {
    const r = await fetch("/api/memory/append", {
      method: "POST", headers: authHeaders(),
      body: JSON.stringify({ text: fact }),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || r.statusText);
    await refreshMemory();
    toast("Remembered ✓");
  } catch (e) {
    toast("Could not save: " + e.message, true);
  }
}

function openMemoryModal() {
  openModal("Memory — what the model knows about you", (body) => {
    body.appendChild(el("div", "sub", memory.writable
      ? "Injected into the system prompt while the 🧠 toggle is on. " +
        "Edit freely — it's a plain markdown file in the localm data directory."
      : "Read-only: privacy mode blocks memory writes (no new traces). " +
        "Existing memory is still injected while the 🧠 toggle is on."));
    const ta = document.createElement("textarea");
    ta.value = memory.text;
    ta.rows = 14;
    ta.style.width = "100%";
    ta.readOnly = !memory.writable;
    body.appendChild(ta);
    if (memory.writable) {
      const save = el("button", "btn-primary", "Save");
      save.style.marginTop = "10px";
      save.onclick = async () => {
        try {
          const r = await fetch("/api/memory", {
            method: "PUT", headers: authHeaders(),
            body: JSON.stringify({ text: ta.value }),
          });
          const data = await r.json();
          if (!r.ok) throw new Error(data.detail || r.statusText);
          await refreshMemory();
          toast("Memory saved");
          $("modal").style.display = "none";
        } catch (e) {
          toast("Save failed: " + e.message, true);
        }
      };
      body.appendChild(save);
    }
  });
}

/* ---- prompt library (personas) ---- */

const PERSONA_PARAM_IDS = {
  temperature: "p-temperature",
  top_p: "p-top-p",
  top_k: "p-top-k",
  repeat_penalty: "p-repeat-penalty",
  max_tokens: "p-max-tokens",
};

let personaCache = [];

async function refreshPersonas() {
  try {
    const r = await fetch("/api/prompts", { headers: authHeaders() });
    if (!r.ok) return;
    personaCache = (await r.json()).prompts;
    const sel = $("p-persona");
    const current = sel.value;
    sel.replaceChildren();
    const none = document.createElement("option");
    none.value = "";
    none.textContent = "(none)";
    sel.appendChild(none);
    for (const p of personaCache) {
      const opt = document.createElement("option");
      opt.value = p.name;
      opt.textContent = p.name;
      sel.appendChild(opt);
    }
    if ([...sel.options].some((o) => o.value === current)) sel.value = current;
  } catch (e) { /* server unreachable */ }
}

function applyPersona(name) {
  const p = personaCache.find((x) => x.name === name);
  if (!p) { toast("No such persona: " + name, true); return false; }
  $("p-system").value = p.system || "";
  for (const [key, id] of Object.entries(PERSONA_PARAM_IDS)) {
    $(id).value = p.params?.[key] ?? "";
  }
  $("p-persona").value = name;
  toast(`Persona '${name}' applied`);
  return true;
}

$("p-persona").onchange = () => {
  const name = $("p-persona").value;
  if (name) applyPersona(name);
};

$("persona-save").onclick = async () => {
  const name = prompt("Persona name:", $("p-persona").value || "");
  if (!name || !name.trim()) return;
  const params = {};
  for (const [key, id] of Object.entries(PERSONA_PARAM_IDS)) {
    const v = $(id).value.trim();
    if (v !== "" && !Number.isNaN(Number(v))) params[key] = Number(v);
  }
  try {
    const r = await fetch("/api/prompts/" + encodeURIComponent(name.trim()), {
      method: "PUT", headers: authHeaders(),
      body: JSON.stringify({ system: $("p-system").value, params }),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || r.statusText);
    await refreshPersonas();
    $("p-persona").value = name.trim();
    toast(`Persona '${name.trim()}' saved`);
  } catch (e) {
    toast("Save failed: " + e.message, true);
  }
};

$("persona-delete").onclick = async () => {
  const name = $("p-persona").value;
  if (!name) { toast("Select a persona first", true); return; }
  if (!confirm(`Delete persona '${name}'? The drawer values stay as they are.`)) return;
  try {
    const r = await fetch("/api/prompts/" + encodeURIComponent(name), {
      method: "DELETE", headers: authHeaders() });
    if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
    await refreshPersonas();
    $("p-persona").value = "";
    toast(`Persona '${name}' deleted`);
  } catch (e) {
    toast("Delete failed: " + e.message, true);
  }
};

/* sending */

async function runCompletion(conv, webDepth = 0) {
  await maybeCompactConversation(conv);
  const params = chatParams();
  const webEnabled = $("p-web").checked;
  const messages = [];
  let sysText = params.system || "";
  if ($("p-memory").checked && memory.text.trim()) {
    sysText = (sysText ? sysText + "\n\n" : "") +
      "Long-term memory — things to remember about the user:\n" +
      memory.text.trim();
  }
  if (webEnabled) {
    sysText = (sysText ? sysText + "\n\n" : "") + WEB_TOOL_PROMPT;
  }
  if (sysText) messages.push({ role: "system", content: sysText });
  // Server-generated images (/api/ URLs from /imagine) must not be sent to
  // the model as image parts — replace those messages with a text note.
  const mapped = conv.messages.map((m) => {
    if (Array.isArray(m.content) &&
        m.content.some((p) => p.type === "image_url" &&
                              p.image_url?.url?.startsWith("/api/"))) {
      return { role: m.role,
               content: msgText(m) + "\n[An image was generated and shown to the user.]" };
    }
    // Reasoning blocks are display-only — never resend them as context.
    if (m.role === "assistant" && typeof m.content === "string") {
      return { role: m.role, content: stripThink(m.content) };
    }
    return { role: m.role, content: m.content };
  });
  // Attached documents and knowledge excerpts are stored as separate user
  // rows; some chat templates require strict user/assistant alternation,
  // so consecutive plain-text same-role messages are merged before sending.
  for (const m of mapped) {
    const prev = messages[messages.length - 1];
    if (prev && prev.role === m.role && prev.role !== "system" &&
        typeof prev.content === "string" && typeof m.content === "string") {
      prev.content += "\n\n" + m.content;
    } else {
      messages.push({ role: m.role, content: m.content });
    }
  }

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
  let finishReason = null;
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
      if (chunk.choices?.[0]?.finish_reason) finishReason = chunk.choices[0].finish_reason;
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

  const reply = { role: "assistant", content: full };
  if (finishReason === "length") {
    // The reply was cut by the max-tokens budget, not finished by the model.
    reply.truncated = true;
    toast("Reply hit the max-tokens limit — raise “Max tokens” in ⚙ parameters, or reply “continue”", true);
  }
  conv.messages.push(reply);
  saveConversations(conv);
  if (usage) {
    const bits = [`${usage.total_tokens} tok`];
    if (usage.ttft_ms != null) bits.push(`TTFT ${usage.ttft_ms} ms`);
    if (usage.tokens_per_sec != null) bits.push(`${usage.tokens_per_sec} tok/s`);
    $("chat-usage").textContent = bits.join(" · ");
  }
  renderChat();

  // Web-access loop: when the model requested a search/page and the toggle
  // is on, run it and let the model continue — bounded rounds per send.
  const nextCall = (webEnabled && webDepth < WEB_MAX_ROUNDS)
    ? parseWebCall(full) : null;
  if (nextCall) {
    await runWebCall(conv, nextCall);
    await runCompletion(conv, webDepth + 1);
  } else if ($("p-speak").checked && full) {
    speak(full);   // read the finished reply aloud (offline browser voices)
  }
}

/** Query the selected knowledge collection and inject cited excerpts. */
async function retrieveKnowledge(conv, query) {
  const kb = $("p-kb").value;
  if (!kb || !query) return;
  try {
    const r = await fetch(
      `/api/rag/collections/${encodeURIComponent(kb)}/query`, {
        method: "POST", headers: authHeaders(),
        body: JSON.stringify({ query, k: 4 }),
      });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || r.statusText);
    if (!data.hits.length) return;
    const basename = (p) => p.split(/[\\/]/).pop();
    const lines = data.hits.map((h, i) =>
      `[${i + 1}] ${basename(h.source)}:${h.pos}\n${h.text.slice(0, 900)}`);
    conv.messages.push({
      role: "user", tag: "kb",
      content:
        `[Excerpts from the "${kb}" collection relevant to: ` +
        `${query.slice(0, 120)}]\n\n` + lines.join("\n\n") +
        "\n\nUse these excerpts where relevant and cite them as [1], [2]… " +
        "If they don't answer the question, say so before answering from " +
        "general knowledge.",
    });
    saveConversations(conv);
    renderChat();
  } catch (e) {
    toast("Knowledge retrieval failed: " + e.message, true);
  }
}

/** Populate the params-drawer knowledge selector. pages.js calls this after
 *  collections change on the Knowledge page. */
async function refreshKbSelect() {
  try {
    const r = await fetch("/api/rag/collections", { headers: authHeaders() });
    if (!r.ok) return;
    const data = await r.json();
    const sel = $("p-kb");
    const current = sel.value;
    sel.replaceChildren();
    const none = document.createElement("option");
    none.value = "";
    none.textContent = "(none)";
    sel.appendChild(none);
    for (const c of data.collections) {
      const opt = document.createElement("option");
      opt.value = c.name;
      opt.textContent = `${c.name} (${c.n_docs} docs, ${c.n_chunks} chunks)`;
      sel.appendChild(opt);
    }
    if ([...sel.options].some((o) => o.value === current)) sel.value = current;
  } catch (e) { /* server unreachable — selector stays as-is */ }
}

async function sendChat() {
  const input = $("chat-input");
  const text = input.value.trim();
  if (!text && chat.attachments.length === 0 && chat.docs.length === 0) return;
  if (chat.abort) return;

  if (text.startsWith("/")) {
    input.value = "";
    autoGrow(input);
    handleSlashSubmit(text, execChatCommand);
    return;
  }

  if (!currentConv()) newConversation();
  const conv = currentConv();
  const isFirstMessage = conv.messages.length === 0;

  // Attached documents come first so the model reads them before the question
  for (const doc of chat.docs) {
    conv.messages.push({
      role: "user", tag: "doc",
      content: `[Attached document: ${doc.name}` +
        (doc.truncated ? " (truncated)" : "") + `]\n${doc.text}`,
    });
  }
  chat.docs = [];

  let content;
  if (chat.attachments.length) {
    content = [{ type: "text", text }];
    for (const att of chat.attachments) {
      content.push({ type: "image_url", image_url: { url: att.dataUri } });
    }
  } else {
    content = text || "Please read the attached document(s).";
  }
  conv.messages.push({ role: "user", content });
  chat.attachments = [];
  renderAttachChips();

  if (isFirstMessage) {
    conv.title = text.slice(0, 42) + (text.length > 42 ? "…" : "") || "Document chat";
    renderConvList();
  }
  saveConversations(conv);   // user message persists even if the reply dies
  input.value = "";
  autoGrow(input);
  renderChat();
  await retrieveKnowledge(conv, text);
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
/** Enter sends, Shift+Enter inserts a newline, Ctrl/Cmd+Enter also sends.
 *  Skipped while an IME composition or the slash-command menu is active
 *  (the menu's own keydown handler picks the highlighted command). */
function composerEnterToSend(e, send) {
  if (e.key !== "Enter" || e.isComposing) return;
  if (e.shiftKey) return;   // newline — the textarea's default behaviour
  const menu = e.target.closest(".composer-wrap")?.querySelector(".slash-menu");
  if (menu && menu.style.display !== "none") return;
  e.preventDefault();
  send();
}

$("chat-input").addEventListener("keydown", (e) => composerEnterToSend(e, sendChat));
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
  lastActiveId: null,    // session to return to when leaving setup mode
  docs: [],              // file attachments: {name, text, chars, truncated}
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
  sel.replaceChildren();
  // In setup mode (no active session) a placeholder holds the selection, so
  // picking any real session fires onchange — even when only one exists.
  if (!coder.activeId && coder.sessions.size) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "(new session)";
    opt.selected = true;
    sel.appendChild(opt);
  }
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
  // Keep the bar while other sessions exist so they stay reachable
  $("coder-bar").classList.toggle("open", hasSession || coder.sessions.size > 0);
  if (!hasSession) {
    // Setup mode: park every session feed and clear the session labels —
    // the form must not render on top of a previous session's transcript.
    // Remember where we came from so "back to session" can return there.
    if (coder.activeId && coder.sessions.has(coder.activeId)) {
      coder.lastActiveId = coder.activeId;
    }
    for (const [, s] of coder.sessions) s.feedEl.classList.remove("active");
    coder.activeId = null;
    $("coder-cwd").textContent = "";
    $("coder-state").textContent = "";
    $("coder-usage").textContent = "";
    renderSessionSelect();
  }
  $("setup-cancel").style.display =
    !hasSession && coder.sessions.size > 0 ? "" : "none";
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
    confirmCards: new Map(),   // confirm_id → {card, title, buttons, tool}
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

/** Args worth showing next to a diff — the bulky text fields ARE the diff. */
function slimArgs(args) {
  const slim = {};
  for (const [k, v] of Object.entries(args || {})) {
    if (k === "content" || k === "old" || k === "new" || k === "diff") continue;
    slim[k] = v;
  }
  return slim;
}

function buildToolCard(ev) {
  const card = el("div", "tool-card");
  card.dataset.t0 = String(Date.now());
  const inner = el("div", "inner");
  const head = el("div", "head");
  head.appendChild(el("span", "name", ev.tool));
  const hintVal = ev.args?.path || ev.args?.command || ev.args?.pattern || ev.args?.url || "";
  head.appendChild(el("span", "hint", String(hintVal).slice(0, 120)));
  head.appendChild(el("span", "state", "…"));
  const body = el("div", "body");
  if (ev.diff) {
    const rest = slimArgs(ev.args);
    if (Object.keys(rest).length) {
      body.appendChild(el("pre", "args", JSON.stringify(rest, null, 2)));
    }
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

/** Mark a confirm card as resolved. Idempotent — fed both by the local
 *  button click and by the confirm_resolved event from the server (which is
 *  also what replay sends for already-answered confirmations). */
function resolveConfirmCard(s, confirmId, approved, timedOut) {
  const entry = s.confirmCards.get(confirmId);
  if (!entry || entry.card.classList.contains("answered")) return;
  entry.card.classList.add("answered");
  entry.title.textContent = timedOut
    ? "✗ Timed out — rejected " + entry.tool
    : (approved ? "✓ Approved " : "✗ Rejected ") + entry.tool;
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
  // "always allow" lives inside .buttons so the answered-state CSS hides it
  const allowCb = document.createElement("input");
  allowCb.type = "checkbox";
  const allowLabel = el("label", "always-allow");
  allowLabel.appendChild(allowCb);
  allowLabel.appendChild(document.createTextNode(
    ` always allow ${ev.tool} this session`));
  const answer = async (approved) => {
    try {
      const r = await fetch(`/api/coder/sessions/${s.info.id}/confirm`, {
        method: "POST",
        headers: authHeaders(),
        body: JSON.stringify({
          confirm_id: ev.confirm_id, approved,
          always_allow: approved && allowCb.checked,
        }),
      });
      if (!r.ok) {
        // Already answered elsewhere (another tab) or timed out server-side —
        // the confirm_resolved event carries the real outcome.
        toast("Confirmation was no longer pending", true);
        return;
      }
      if (approved && allowCb.checked) {
        toast(`${ev.tool} auto-approved for the rest of this session`);
      }
      resolveConfirmCard(s, ev.confirm_id, approved, false);
    } catch (e) {
      toast("Failed to answer confirmation: " + e.message, true);
    }
  };
  yes.onclick = () => answer(true);
  no.onclick = () => answer(false);
  buttons.appendChild(yes);
  buttons.appendChild(no);
  buttons.appendChild(allowLabel);
  inner.appendChild(buttons);
  card.appendChild(inner);
  s.confirmCards.set(ev.confirm_id, { card, title, tool: ev.tool });
  return card;
}

function handleCoderEvent(s, ev) {
  // Keep a light event log (no token spam) so "export" can rebuild the
  // session as markdown without another server round-trip.
  if (ev.type !== "token") {
    (s.eventLog = s.eventLog || []).push(ev);
  }
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
        const ctx = ev.ctx_ratio ? ` · ctx ${Math.round(ev.ctx_ratio * 100)}%` : "";
        if (ev.total_tokens)
          $("coder-usage").textContent = `${ev.total_tokens} tok · turn ${ev.turn}${ctx}`;
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
        const t0 = Number(card.dataset.t0 || 0);
        const took = t0 ? ` · ${((Date.now() - t0) / 1000).toFixed(1)}s` : "";
        state.textContent = (ev.summary || (ev.ok ? "ok" : "failed")) + took;
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
    case "confirm_resolved": {
      resolveConfirmCard(s, ev.confirm_id, ev.approved, ev.timed_out);
      break;
    }
    case "user": {
      // replayed user message (emitted client-side on send; replay rebuilds it)
      flushAssistantBlock(s);
      addMessageRow(s.feedEl, "user", ev.text,
        ev.queued ? { cls: "web-note", label: "Queued" } : {});
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
      let finalLine = (ev.ok ? "Task finished" : "Task ended") +
        ` — ${ev.turns} turns, ${ev.total_tokens} tokens`;
      if (ev.changed_files?.length) {
        finalLine += ` · ${ev.changed_files.length} file(s) changed (see "files")`;
      }
      feedAppend(s, el("div", "feed-final", finalLine));
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
      dry_run: $("setup-dry").checked,
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
    if (!chat.privacy) localStorage.setItem("localm.coderCwd", cwd);
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

/* coder file attachments — extracted to text server-side (same in-memory
 * /api/rag/extract path as chat docs) and prepended to the task message,
 * so the agent sees the content without needing the file inside cwd. */

function renderCoderAttachChips() {
  const box = $("coder-attach-chips");
  box.replaceChildren();
  coder.docs.forEach((doc, i) => {
    const chip = el("span", "chip");
    chip.appendChild(el("span", "", "📄 " + doc.name +
      ` (${(doc.chars / 1000).toFixed(1)}k chars${doc.truncated ? ", trimmed" : ""})`));
    const rm = el("button", "", "×");
    rm.onclick = () => { coder.docs.splice(i, 1); renderCoderAttachChips(); };
    chip.appendChild(rm);
    box.appendChild(chip);
  });
}

async function attachCoderDocument(file) {
  const b64 = await new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result).split(",", 2)[1] || "");
    reader.onerror = () => reject(new Error("could not read file"));
    reader.readAsDataURL(file);
  });
  const r = await fetch("/api/rag/extract", {
    method: "POST", headers: authHeaders(),
    body: JSON.stringify({ filename: file.name, content_b64: b64 }),
  });
  const data = await r.json();
  if (!r.ok) throw new Error(data.detail || r.statusText);
  coder.docs.push({ name: data.filename, text: data.text,
                    chars: data.chars, truncated: data.truncated });
  renderCoderAttachChips();
}

$("coder-attach").onclick = () => $("coder-file").click();
$("coder-file").addEventListener("change", (e) => {
  for (const file of e.target.files) {
    attachCoderDocument(file).catch((err) =>
      toast(`${file.name}: ${err.message}`, true));
  }
  e.target.value = "";
});

async function sendCoderTask() {
  const s = activeSession();
  const input = $("coder-input");
  const text = input.value.trim();
  if ((!text && coder.docs.length === 0) || !s) return;

  if (text.startsWith("/")) {
    input.value = "";
    autoGrow(input);
    handleSlashSubmit(text, (c) => execCoderCommand(c));
    return;
  }

  // Attached file contents go first so the agent reads them before the task
  let payload = text || "Read the attached file(s).";
  if (coder.docs.length) {
    const blocks = coder.docs.map((d) =>
      `[Attached file: ${d.name}${d.truncated ? " (truncated)" : ""}]\n${d.text}`);
    payload = blocks.join("\n\n") + "\n\n" + payload;
  }

  try {
    const r = await fetch(`/api/coder/sessions/${s.info.id}/message`, {
      method: "POST",
      headers: authHeaders(),
      body: JSON.stringify({ text: payload }),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || r.statusText);
    if (data.status === "queued") {
      // Mid-task steering: the agent reads it at the next turn boundary
      toast("Queued — the agent picks it up at the next turn");
    } else {
      s.busy = true;
      $("coder-state").textContent = "working…";
      renderSessionSelect();
    }
    // The user message arrives back through the event stream (so replay
    // works after a page reload) — no client-side row here.
    input.value = "";
    autoGrow(input);
    coder.docs = [];
    renderCoderAttachChips();
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
$("setup-cancel").onclick = () => {
  // Return to the session we left (or any remaining one) without starting
  const id = coder.sessions.has(coder.lastActiveId)
    ? coder.lastActiveId
    : [...coder.sessions.keys()].pop();
  if (id) activateSession(id);
};

/* ---- directory picker (browse… on the setup form) ---- */

async function fetchDirs(path) {
  const r = await fetch("/api/fs/dirs?path=" + encodeURIComponent(path || ""),
                        { headers: authHeaders() });
  const data = await r.json();
  if (!r.ok) throw new Error(data.detail || r.statusText);
  return data;
}

/** Modal directory browser. Resolves with the chosen path, or null when the
 *  modal is dismissed. Used by the coder setup form and the media pages. */
function pickDirectory(title, startPath = "") {
  return new Promise((resolve) => {
    let settled = false;
    const finish = (value) => {
      if (settled) return;
      settled = true;
      $("modal").style.display = "none";
      resolve(value);
    };
    openModal(title || "Pick a directory", (body) => {
      const pathEl = el("div", "dir-picker-path");
      const listEl = el("div", "dir-picker-list");
      const actions = el("div", "actions");
      const useBtn = el("button", "btn-primary", "Use this directory");
      actions.appendChild(useBtn);
      body.append(pathEl, listEl, actions);
      let current = "";

      useBtn.onclick = () => { if (current) finish(current); };
      // Dismissing the modal (×, backdrop) resolves null — poll visibility
      // since the close handlers are owned by the shared modal chrome.
      const watch = setInterval(() => {
        if ($("modal").style.display === "none") {
          clearInterval(watch);
          finish(null);
        }
      }, 200);

      async function show(path) {
        let data;
        try {
          data = await fetchDirs(path);
        } catch (e) {
          toast("Cannot open: " + e.message, true);
          if (path) { show(""); return; }   // fall back to the drive list
          throw e;
        }
        current = data.path;
        pathEl.textContent = current || "Drives";
        useBtn.disabled = !current;
        listEl.replaceChildren();
        if (data.parent !== null && current) {
          const up = el("div", "dir-picker-item up", "↑ ..");
          up.onclick = () => show(data.parent);
          listEl.appendChild(up);
        }
        for (const name of data.dirs) {
          const item = el("div", "dir-picker-item", "📁 " + name);
          // "/" joins fine on Windows too (Python Path accepts both
          // separators); the server resolves and echoes the native form.
          item.onclick = () =>
            show(current ? current.replace(/[\\/]+$/, "") + "/" + name : name);
          listEl.appendChild(item);
        }
        if (!data.dirs.length) {
          listEl.appendChild(el("div", "dir-picker-empty", "no subdirectories"));
        }
      }
      show(startPath).catch(() => {});
    });
  });
}

$("setup-browse").onclick = async () => {
  const dir = await pickDirectory("Pick a project directory",
                                  $("setup-cwd").value.trim());
  if (dir) {
    $("setup-cwd").value = dir;
    localStorage.setItem("localm.coderCwd", dir);
  }
};
$("coder-send").onclick = sendCoderTask;
$("coder-input").addEventListener("keydown", (e) => composerEnterToSend(e, sendCoderTask));
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

/** Audit-entry modal shared by the live-session log and past-session history.
 *  A filter box narrows entries by substring (type, turn, or payload). */
function showAuditModal(title, data) {
  openModal(title, (body) => {
    body.appendChild(el("div", "sub", data.path));
    const filter = document.createElement("input");
    filter.type = "text";
    filter.placeholder = "filter entries… (tool name, text, type)";
    filter.className = "log-filter";
    filter.spellcheck = false;
    body.appendChild(filter);
    const rows = [];
    for (const entry of data.entries) {
      const row = el("div", "log-entry");
      const ts = new Date(entry.t).toLocaleTimeString();
      const label = `${ts} #${entry.turn} ${entry.type}`;
      const payload = JSON.stringify(entry.data);
      row.appendChild(el("span", "t", label));
      row.appendChild(document.createTextNode(payload));
      body.appendChild(row);
      rows.push({ row, text: (label + " " + payload).toLowerCase() });
    }
    filter.addEventListener("input", () => {
      const q = filter.value.trim().toLowerCase();
      for (const r of rows) {
        r.row.style.display = !q || r.text.includes(q) ? "" : "none";
      }
    });
    if (!data.entries.length) body.appendChild(el("div", "sub", "(empty)"));
  });
}

/** Files the agent changed this session, with per-file and full-session diffs. */
async function openFilesModal() {
  const s = activeSession();
  if (!s) return;
  let data;
  try {
    const r = await fetch(`/api/coder/sessions/${s.info.id}/files`,
                          { headers: authHeaders() });
    data = await r.json();
    if (!r.ok) throw new Error(data.detail || r.statusText);
  } catch (e) {
    toast("Could not load changed files: " + e.message, true);
    return;
  }
  openModal("Files changed — " + sessionLabel(s.info), (body) => {
    if (!data.files.length) {
      body.appendChild(el("div", "sub", "No files changed this session."));
      return;
    }
    const diffBox = el("div", "files-diff");
    const showDiff = async (path) => {
      diffBox.replaceChildren(el("div", "sub", "loading diff…"));
      try {
        const r = await fetch(
          `/api/coder/sessions/${s.info.id}/files/diff?path=` +
          encodeURIComponent(path || ""), { headers: authHeaders() });
        const d = await r.json();
        if (!r.ok) throw new Error(d.detail || r.statusText);
        diffBox.replaceChildren(
          d.diff ? renderDiff(d.diff) : el("div", "sub", "(no difference)"));
      } catch (e) {
        diffBox.replaceChildren(el("div", "sub", "diff failed: " + e.message));
      }
    };
    for (const f of data.files) {
      const row = el("div", "log-entry clickable");
      row.appendChild(el("span", "t", f.created ? "new" : "edit"));
      row.appendChild(document.createTextNode(
        `${f.path} — ${f.writes} write(s)` + (f.exists ? "" : " (deleted since)")));
      row.onclick = () => showDiff(f.path);
      body.appendChild(row);
    }
    const all = el("button", "btn-quiet", "full session diff");
    all.onclick = () => showDiff("");
    body.appendChild(all);
    body.appendChild(diffBox);
  });
}

/** Download the active session's feed as markdown (explicit user action —
 *  works in privacy mode too, same contract as chat /export). */
function exportCoderSession() {
  const s = activeSession();
  const log = s?.eventLog || [];
  if (!log.length) { toast("Nothing to export yet", true); return; }
  const lines = [`# Coder session — ${sessionLabel(s.info)}`, ""];
  for (const ev of log) {
    if (ev.type === "user") {
      lines.push(`**You${ev.queued ? " (queued)" : ""}**: ${ev.text}`, "");
    } else if (ev.type === "tool_call") {
      lines.push(`- \`${ev.tool}\` ` +
        JSON.stringify(slimArgs(ev.args)).slice(0, 200));
    } else if (ev.type === "tool_result") {
      lines.push(`  - ${ev.ok ? "ok" : "FAILED"}` +
        (ev.summary ? `: ${ev.summary}` : ""));
    } else if (ev.type === "info") {
      lines.push(`> ${ev.text}`, "");
    } else if (ev.type === "final") {
      lines.push("", `**Agent**: ${ev.text}`, "");
    }
  }
  const blob = new Blob([lines.join("\n")], { type: "text/markdown" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `coder-session-${s.info.id}.md`;
  a.click();
  URL.revokeObjectURL(a.href);
}

$("coder-files").onclick = openFilesModal;
$("coder-export").onclick = exportCoderSession;

$("coder-log").onclick = async () => {
  const s = activeSession();
  if (!s) return;
  const r = await fetch(`/api/coder/sessions/${s.info.id}/log`, {
    headers: authHeaders() });
  const data = await r.json();
  if (!r.ok) { toast(data.detail || "No log available", true); return; }
  showAuditModal("Audit log — " + sessionLabel(s.info), data);
};

/** Past coder sessions: audit logs left behind by log/full-mode sessions,
 *  including ones from before a server restart. */
async function openSessionHistory() {
  let data = null;
  try {
    const r = await fetch("/api/coder/history", { headers: authHeaders() });
    if (r.ok) data = await r.json();
  } catch (e) { /* handled below */ }
  if (!data) { toast("Could not load session history", true); return; }
  openModal("Past coder sessions", (body) => {
    if (!data.enabled) {
      body.appendChild(el("div", "sub",
        "New sessions are not being recorded (privacy mode). Anything below " +
        "is from earlier log/full-mode sessions."));
    }
    if (!data.logs.length) {
      body.appendChild(el("div", "sub",
        "No session logs yet — start a session with persistence set to " +
        "log or full, and its audit trail will appear here."));
      return;
    }
    for (const item of data.logs) {
      const row = el("div", "log-entry clickable");
      const when = new Date(item.mtime * 1000).toLocaleString();
      const kb = (item.size_bytes / 1024).toFixed(1);
      row.appendChild(el("span", "t", when));
      row.appendChild(document.createTextNode(`${item.name} (${kb} KB)`));
      row.onclick = async () => {
        try {
          const r = await fetch(
            "/api/coder/history/" + encodeURIComponent(item.name),
            { headers: authHeaders() });
          const entries = await r.json();
          if (!r.ok) throw new Error(entries.detail || r.statusText);
          showAuditModal("Session — " + item.name, entries);
        } catch (e) {
          toast("Could not open log: " + e.message, true);
        }
      };
      body.appendChild(row);
    }
  });
}

$("coder-history").onclick = openSessionHistory;
$("setup-history").onclick = openSessionHistory;

/* ================================================================ */
/*  Slash commands                                                   */
/* ================================================================ */

const CHAT_COMMANDS = [
  { cmd: "imagine", hint: "generate an image with FLUX", args: "<prompt>" },
  { cmd: "music", hint: "generate a music track (ACE-Step, 120s instrumental)", args: "<style tags>" },
  { cmd: "video", hint: "generate a short video clip (Wan, ~5s — slow)", args: "<prompt>" },
  { cmd: "web", hint: "search the web, then answer with sources", args: "<query>" },
  { cmd: "clear", hint: "clear this conversation" },
  { cmd: "compact", hint: "summarise older messages to free context" },
  { cmd: "export", hint: "download this conversation as markdown" },
  { cmd: "rename", hint: "rename this conversation", args: "<title>" },
  { cmd: "persona", hint: "apply a saved persona (system prompt + params)", args: "<name>" },
  { cmd: "remember", hint: "add a fact to the model's long-term memory", args: "<fact>" },
  { cmd: "memory", hint: "view or edit the memory file" },
  { cmd: "pin", hint: "pin/unpin this conversation" },
  { cmd: "folder", hint: "move this conversation to a folder (empty = remove)", args: "<name>" },
  { cmd: "system", hint: "edit the system prompt" },
  { cmd: "new", hint: "start a new conversation" },
];

const CODER_COMMANDS = [
  { cmd: "undo", hint: "revert the last file write" },
  { cmd: "files", hint: "files changed this session, with diffs" },
  { cmd: "compact", hint: "summarise older turns" },
  { cmd: "export", hint: "download this session's feed as markdown" },
  { cmd: "log", hint: "open the audit log" },
  { cmd: "stop", hint: "interrupt the current task" },
  { cmd: "end", hint: "end this session" },
  { cmd: "help", hint: "list available commands" },
];

async function runImagineInChat(promptText) {
  if (!promptText) { toast("Usage: /imagine <prompt>", true); return; }
  if (!currentConv()) newConversation();
  const conv = currentConv();
  conv.messages.push({ role: "user", content: "/imagine " + promptText });
  saveConversations(conv);
  renderChat();
  const box = $("chat-messages");
  const { body } = addMessageRow(box, "assistant", "");
  body.textContent = "Generating image…";
  box.scrollTop = box.scrollHeight;
  try {
    const r = await fetch("/api/imagine", {
      method: "POST", headers: authHeaders(),
      body: JSON.stringify({ prompt: promptText }),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || r.statusText);
    const end = await streamJob(data.job_id, (line) => {
      body.textContent = line;
      if (nearBottom(box)) box.scrollTop = box.scrollHeight;
    });
    if (end.status === "done" && end.result) {
      conv.messages.push({
        role: "assistant",
        content: [
          { type: "text", text: "Here is the generated image:" },
          { type: "image_url",
            image_url: { url: "/api/imagine/file/" + encodeURIComponent(end.result) } },
        ],
      });
      saveConversations(conv);
      renderChat();
    } else {
      body.textContent = "Image generation " + end.status +
        " — see the Images page for details.";
    }
  } catch (e) {
    body.textContent = "Image generation failed: " + e.message;
    toast(e.message, true);
  }
}

/** /web <query> — explicit, user-initiated web grounding: search, inject the
 *  results into the conversation, and let the model answer from them. */
async function runWebInChat(query) {
  if (!query) { toast("Usage: /web <query>", true); return; }
  if (chat.abort) { toast("Wait for the current reply to finish", true); return; }
  if (!currentConv()) newConversation();
  const conv = currentConv();
  conv.messages.push({ role: "user", content: "/web " + query });
  if (conv.messages.length === 1) {
    conv.title = query.slice(0, 42) + (query.length > 42 ? "…" : "");
    renderConvList();
  }
  saveConversations(conv);
  renderChat();
  let note;
  try {
    note = await requestWebTool({ name: "web_search", args: { query } });
    note += `\n\nUsing these results, answer: ${query}\nName the sources you used.`;
  } catch (e) {
    toast("Web search failed: " + e.message, true);
    note = `[Web search failed: ${e.message}] Tell the user, and answer ` +
           "from your own knowledge if you can.";
  }
  conv.messages.push({ role: "user", content: note, web: true });
  saveConversations(conv);
  renderChat();
  await runCompletion(conv);
}

/** /music <tags> — generate a default-length instrumental inline; the Music
 *  page has the full form (lyrics, duration, seed…). */
async function runMusicInChat(tags) {
  if (!tags) { toast("Usage: /music <style tags>", true); return; }
  if (!currentConv()) newConversation();
  const conv = currentConv();
  conv.messages.push({ role: "user", content: "/music " + tags });
  saveConversations(conv);
  renderChat();
  const box = $("chat-messages");
  const { body } = addMessageRow(box, "assistant", "");
  body.textContent = "Generating track… (long tracks take a while)";
  box.scrollTop = box.scrollHeight;
  try {
    const r = await fetch("/api/music", {
      method: "POST", headers: authHeaders(),
      body: JSON.stringify({ tags }),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || r.statusText);
    const end = await streamJob(data.job_id, (line) => {
      body.textContent = line;
      if (nearBottom(box)) box.scrollTop = box.scrollHeight;
    });
    if (end.status === "done" && end.result) {
      conv.messages.push({
        role: "assistant",
        content: "Here is the generated track:",
        audio: "/api/music/file/" + encodeURIComponent(end.result),
      });
      saveConversations(conv);
      renderChat();
    } else {
      body.textContent = "Music generation " + end.status +
        " — see the Music page for details.";
    }
  } catch (e) {
    body.textContent = "Music generation failed: " + e.message;
    toast(e.message, true);
  }
}

/** /video <prompt> — generate a default-length (~5s) clip inline; the Video
 *  page has the full form (negative, duration, size, start image…). */
async function runVideoInChat(promptText) {
  if (!promptText) { toast("Usage: /video <prompt>", true); return; }
  if (!currentConv()) newConversation();
  const conv = currentConv();
  conv.messages.push({ role: "user", content: "/video " + promptText });
  saveConversations(conv);
  renderChat();
  const box = $("chat-messages");
  const { body } = addMessageRow(box, "assistant", "");
  body.textContent = "Generating clip… (video is slow — expect several minutes)";
  box.scrollTop = box.scrollHeight;
  try {
    const r = await fetch("/api/video", {
      method: "POST", headers: authHeaders(),
      body: JSON.stringify({ prompt: promptText }),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || r.statusText);
    const end = await streamJob(data.job_id, (line) => {
      body.textContent = line;
      if (nearBottom(box)) box.scrollTop = box.scrollHeight;
    });
    if (end.status === "done" && end.result) {
      conv.messages.push({
        role: "assistant",
        content: "Here is the generated clip:",
        video: "/api/video/file/" + encodeURIComponent(end.result),
      });
      saveConversations(conv);
      renderChat();
    } else {
      body.textContent = "Video generation " + end.status +
        " — see the Video page for details.";
    }
  } catch (e) {
    body.textContent = "Video generation failed: " + e.message;
    toast(e.message, true);
  }
}

function execChatCommand(cmd, arg) {
  switch (cmd) {
    case "imagine": runImagineInChat(arg); return true;
    case "music": runMusicInChat(arg); return true;
    case "video": runVideoInChat(arg); return true;
    case "web": runWebInChat(arg); return true;
    case "clear": {
      const conv = currentConv();
      if (conv) {
        conv.messages = [];
        conv.branches = [];
        saveConversations(conv);
        renderChat();
      }
      return true;
    }
    case "compact": $("compact-conv").onclick(); return true;
    case "export": exportConversation(); return true;
    case "rename": {
      const conv = currentConv();
      if (conv && arg) { conv.title = arg; saveConversations(conv); renderConvList(); }
      else toast("Usage: /rename <title>", true);
      return true;
    }
    case "remember": rememberFact(arg); return true;
    case "memory": openMemoryModal(); return true;
    case "persona": {
      if (!arg) {
        const names = personaCache.map((p) => p.name);
        toast(names.length ? "Personas: " + names.join(", ")
                           : "No personas saved yet — use the drawer's save…",
              !names.length);
        return true;
      }
      // case-insensitive match for typing convenience
      const hit = personaCache.find(
        (p) => p.name.toLowerCase() === arg.toLowerCase());
      applyPersona(hit ? hit.name : arg);
      return true;
    }
    case "pin": {
      const conv = currentConv();
      if (!conv) return true;
      conv.pinned = !conv.pinned;
      if (!conv.pinned) delete conv.pinned;
      saveConversations(conv);
      renderConvList();
      toast(conv.pinned ? "Pinned" : "Unpinned");
      return true;
    }
    case "folder": {
      const conv = currentConv();
      if (!conv) return true;
      if (arg) conv.folder = arg;
      else delete conv.folder;
      saveConversations(conv);
      renderConvList();
      toast(arg ? `Moved to folder '${arg}'` : "Removed from its folder");
      return true;
    }
    case "system":
      $("params").classList.add("open");
      $("p-system").focus();
      return true;
    case "new": newConversation(); return true;
  }
  return false;
}

function execCoderCommand(cmd) {
  switch (cmd) {
    case "undo": $("coder-undo").onclick(); return true;
    case "files": openFilesModal(); return true;
    case "compact": $("coder-compact").onclick(); return true;
    case "export": exportCoderSession(); return true;
    case "log": $("coder-log").onclick(); return true;
    case "stop": $("coder-stop").onclick(); return true;
    case "end": $("coder-end").onclick(); return true;
    case "help":
      openModal("Coder commands", (body) => {
        for (const c of CODER_COMMANDS) {
          const row = el("div", "log-entry");
          row.appendChild(el("span", "t", "/" + c.cmd));
          row.appendChild(document.createTextNode(c.hint));
          body.appendChild(row);
        }
        body.appendChild(el("div", "sub",
          "Anything not starting with / is sent to the agent as a task."));
      });
      return true;
  }
  return false;
}

/** Attach a slash-command dropdown to a composer textarea. */
function attachSlashMenu(textarea, commands, execute) {
  const menu = el("div", "slash-menu");
  menu.style.display = "none";
  textarea.closest(".composer-wrap").appendChild(menu);
  let selected = 0;
  let visible = [];

  function close() { menu.style.display = "none"; visible = []; }

  function render() {
    const value = textarea.value;
    if (!value.startsWith("/") || value.includes("\n")) { close(); return; }
    const typed = value.slice(1).split(" ")[0].toLowerCase();
    visible = commands.filter((c) => c.cmd.startsWith(typed));
    if (!visible.length) { close(); return; }
    selected = Math.min(selected, visible.length - 1);
    menu.replaceChildren();
    visible.forEach((c, i) => {
      const row = el("div", "slash-item" + (i === selected ? " selected" : ""));
      row.appendChild(el("span", "cmd", "/" + c.cmd + (c.args ? " " + c.args : "")));
      row.appendChild(el("span", "hint", c.hint));
      row.onmousedown = (e) => { e.preventDefault(); pick(c); };
      menu.appendChild(row);
    });
    menu.style.display = "block";
  }

  function pick(c) {
    if (c.args) {
      textarea.value = "/" + c.cmd + " ";
      textarea.focus();
      close();
    } else {
      textarea.value = "";
      autoGrow(textarea);
      close();
      execute(c.cmd, "");
    }
  }

  textarea.addEventListener("input", () => { selected = 0; render(); });
  textarea.addEventListener("blur", () => setTimeout(close, 150));
  textarea.addEventListener("keydown", (e) => {
    if (menu.style.display === "none") return;
    if (e.key === "ArrowDown") {
      e.preventDefault(); selected = (selected + 1) % visible.length; render();
    } else if (e.key === "ArrowUp") {
      e.preventDefault(); selected = (selected - 1 + visible.length) % visible.length; render();
    } else if (e.key === "Enter" && !e.ctrlKey && !e.metaKey && !e.shiftKey) {
      e.preventDefault(); pick(visible[selected]);
    } else if (e.key === "Escape") {
      close();
    }
  });
}

/** Intercept "/cmd arg" on submit. Returns true when handled (not for the model). */
function handleSlashSubmit(text, execute) {
  if (!text.startsWith("/")) return false;
  const space = text.indexOf(" ");
  const cmd = (space === -1 ? text.slice(1) : text.slice(1, space)).toLowerCase();
  const arg = space === -1 ? "" : text.slice(space + 1).trim();
  if (!execute(cmd, arg)) {
    toast(`Unknown command: /${cmd}`, true);
  }
  return true;   // never send slash input to the model
}

attachSlashMenu($("chat-input"), CHAT_COMMANDS, execChatCommand);
attachSlashMenu($("coder-input"), CODER_COMMANDS, (c) => execCoderCommand(c));

/* ================================================================ */
/*  Init                                                             */
/* ================================================================ */

$("setup-cwd").value = localStorage.getItem("localm.coderCwd") || "";
refreshModels().then(() => populateSetupModels());
// Server persistence depends on knowing the privacy state first.
refreshCtxLimit().then(initServerConversations);
refreshKbSelect();
refreshPersonas();
refreshMemory();
refreshVoiceStatus();
setInterval(refreshModels, 30000);
// The resolved ctx ceiling only exists once a model has loaded — keep the
// compaction threshold in sync as models load or switch.
setInterval(refreshCtxLimit, 30000);
renderConvList();
if (chat.conversations.length) {
  chat.activeId = chat.conversations[0].id;
  renderConvList();
}
renderChat();
reattachSessions();
// Deep links + restore. Deferred a tick so pages.js has installed
// window.onViewShown and the #pull-start handler.
{
  const params = new URLSearchParams(location.search);
  const pullSpec = params.get("pull");      // from `localm gui --pull SPEC`
  const viewParam = params.get("view");
  if (pullSpec || viewParam) {
    // Strip the query so a reload doesn't restart the download.
    history.replaceState(null, "", location.pathname);
    setTimeout(() => {
      showView(VIEWS.includes(viewParam) ? viewParam : "models");
      if (pullSpec) {
        const specInput = $("pull-spec");
        if (specInput) {
          specInput.value = pullSpec;
          $("pull-start").click();   // kick off the pull with progress
        }
      }
    }, 0);
  } else {
    // Restore the last active page (set in non-privacy mode only).
    const savedView = localStorage.getItem("localm.activeView");
    if (savedView && savedView !== "chat") setTimeout(() => showView(savedView), 0);
  }
}
