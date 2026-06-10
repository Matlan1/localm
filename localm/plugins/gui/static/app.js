/* localm GUI — vanilla JS, no build step.
   Talks to the localm FastAPI server: /v1 (OpenAI-compatible) + /api (GUI).
   All model/agent-originating strings go through textContent or DOMPurify —
   never raw innerHTML. */

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

/* ================================================================ */
/*  Tabs                                                             */
/* ================================================================ */

function showView(name) {
  for (const v of ["chat", "coder"]) {
    $("view-" + v).classList.toggle("active", v === name);
    $("nav-" + v).classList.toggle("active", v === name);
  }
}
$("nav-chat").onclick = () => showView("chat");
$("nav-coder").onclick = () => showView("coder");

/* ================================================================ */
/*  Models                                                           */
/* ================================================================ */

const modelSelect = $("model-select");

function setStatus(state, text) {
  $("status-dot").className = "dot " + state;
  $("status-text").textContent = text;
}

async function refreshModels() {
  try {
    const r = await fetch("/api/models", { headers: authHeaders() });
    if (r.status === 401) {
      const key = prompt("This server requires an API key (LOCALM_API_KEY):");
      if (key) { localStorage.setItem("localm.apiKey", key); location.reload(); }
      return;
    }
    const data = await r.json();
    modelSelect.innerHTML = "";
    for (const m of data.models) {
      const opt = document.createElement("option");
      opt.value = m.name;
      const size = m.size_bytes ? ` (${(m.size_bytes / 1e9).toFixed(1)} GB)` : "";
      opt.textContent = m.name + size;
      if (m.active) opt.selected = true;
      modelSelect.appendChild(opt);
    }
    setStatus("ok", data.active || "no model");
  } catch (e) {
    setStatus("err", "server unreachable");
  }
}

modelSelect.onchange = async () => {
  const model = modelSelect.value;
  setStatus("busy", "loading " + model + "…");
  try {
    const r = await fetch("/api/models/load", {
      method: "POST",
      headers: authHeaders(),
      body: JSON.stringify({ model }),
    });
    if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
    setStatus("ok", model);
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
};

function saveConversations() {
  // Cap stored history so localStorage never overflows
  const slim = chat.conversations.slice(0, 50);
  localStorage.setItem("localm.conversations", JSON.stringify(slim));
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

function renderConvList() {
  const list = $("conv-list");
  list.innerHTML = "";
  for (const conv of chat.conversations) {
    const item = el("div", "conv-item" + (conv.id === chat.activeId ? " active" : ""));
    item.appendChild(el("span", "title", conv.title));
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

function addMessageRow(container, role, text) {
  const row = el("div", "msg-row " + role);
  row.appendChild(el("div", "msg-role", role === "user" ? "You" : "Model"));
  const body = el("div", "msg-body");
  renderMarkdown(body, text);
  row.appendChild(body);
  const meta = el("div", "msg-meta");
  const copy = el("button", "copy-btn", "copy");
  copy.onclick = () => {
    navigator.clipboard.writeText(text);
    copy.textContent = "copied";
    setTimeout(() => (copy.textContent = "copy"), 1200);
  };
  meta.appendChild(copy);
  row.appendChild(meta);
  container.appendChild(row);
  return { row, body, meta };
}

function buildEmptyHint() {
  const div = el("div", "empty-hint");
  div.id = "chat-empty";
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
  for (const m of conv.messages) addMessageRow(box, m.role, m.content);
  box.scrollTop = box.scrollHeight;
}

function chatParams() {
  const num = (id) => {
    const v = $(id).value.trim();
    return v === "" ? null : Number(v);
  };
  return {
    temperature: num("p-temperature"),
    top_p: num("p-top-p"),
    max_tokens: num("p-max-tokens"),
    seed: num("p-seed"),
    system: $("p-system").value.trim() || null,
  };
}

async function sendChat() {
  const input = $("chat-input");
  const text = input.value.trim();
  if (!text) return;
  if (chat.abort) return; // already streaming

  if (!currentConv()) newConversation();
  const conv = currentConv();

  conv.messages.push({ role: "user", content: text });
  if (conv.messages.length === 1) {
    conv.title = text.slice(0, 42) + (text.length > 42 ? "…" : "");
    renderConvList();
  }
  input.value = "";
  autoGrow(input);
  renderChat();

  const params = chatParams();
  const messages = [];
  if (params.system) messages.push({ role: "system", content: params.system });
  messages.push(...conv.messages.map((m) => ({ role: m.role, content: m.content })));

  const body = { model: modelSelect.value, messages, stream: true };
  for (const k of ["temperature", "top_p", "max_tokens", "seed"]) {
    if (params[k] !== null && !Number.isNaN(params[k])) body[k] = params[k];
  }

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

$("chat-send").onclick = () => {
  if (chat.abort) { chat.abort.abort(); return; }
  sendChat();
};
$("chat-input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); sendChat(); }
});
$("chat-input").addEventListener("input", (e) => autoGrow(e.target));
$("toggle-params").onclick = () => $("params").classList.toggle("open");
$("new-conv").onclick = () => { newConversation(); showView("chat"); };

/* ================================================================ */
/*  Coder                                                            */
/* ================================================================ */

const coder = {
  sessionId: null,
  busy: false,
  liveBody: null,   // current streaming assistant element
  liveText: "",
  pendingCards: [], // FIFO of tool cards awaiting their result event
};

function coderFeed() { return $("coder-feed"); }

function feedAppend(node) {
  const feed = coderFeed();
  const stick = nearBottom(feed);
  feed.appendChild(node);
  if (stick) feed.scrollTop = feed.scrollHeight;
}

function startAssistantBlock() {
  if (coder.liveBody) return;
  const { body } = addMessageRow(coderFeed(), "assistant", "");
  coder.liveBody = body;
  coder.liveText = "";
}

function flushAssistantBlock() {
  coder.liveBody = null;
  coder.liveText = "";
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
  const body = el("div", "body", JSON.stringify(ev.args, null, 2));
  head.onclick = () => card.classList.toggle("open");
  inner.appendChild(head);
  inner.appendChild(body);
  card.appendChild(inner);
  return card;
}

function buildConfirmCard(ev) {
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
      await fetch(`/api/coder/sessions/${coder.sessionId}/confirm`, {
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

function handleCoderEvent(ev) {
  switch (ev.type) {
    case "token": {
      startAssistantBlock();
      coder.liveText += ev.text;
      const feed = coderFeed();
      const stick = nearBottom(feed);
      renderMarkdown(coder.liveBody, coder.liveText);
      if (stick) feed.scrollTop = feed.scrollHeight;
      break;
    }
    case "turn": {
      flushAssistantBlock();
      if (ev.total_tokens) $("coder-usage").textContent = `${ev.total_tokens} tok · turn ${ev.turn}`;
      break;
    }
    case "tool_call": {
      flushAssistantBlock();
      const card = buildToolCard(ev);
      feedAppend(card);
      coder.pendingCards.push(card);
      break;
    }
    case "tool_result": {
      const card = coder.pendingCards.shift();
      if (card) {
        const state = card.querySelector(".state");
        state.textContent = ev.summary || (ev.ok ? "ok" : "failed");
        state.className = "state " + (ev.ok ? "ok" : "fail");
        if (ev.output) card.querySelector(".body").textContent = ev.output;
      }
      break;
    }
    case "confirm_request": {
      flushAssistantBlock();
      feedAppend(buildConfirmCard(ev));
      break;
    }
    case "info": {
      flushAssistantBlock();
      feedAppend(el("div", "feed-info", ev.text));
      break;
    }
    case "final": {
      flushAssistantBlock();
      coder.busy = false;
      $("coder-state").textContent = "idle";
      feedAppend(el("div", "feed-final",
        (ev.ok ? "Task finished" : "Task ended") +
        ` — ${ev.turns} turns, ${ev.total_tokens} tokens`));
      break;
    }
    case "error": {
      flushAssistantBlock();
      coder.busy = false;
      $("coder-state").textContent = "error";
      toast("Agent error: " + ev.text, true);
      break;
    }
    case "closed": {
      coder.busy = false;
      break;
    }
  }
}

async function streamCoderEvents() {
  while (coder.sessionId) {
    const id = coder.sessionId;
    try {
      const r = await fetch(`/api/coder/sessions/${id}/events`, {
        headers: authHeaders(),
      });
      if (!r.ok) throw new Error(r.statusText);
      await readSSE(r, (payload) => {
        let ev;
        try { ev = JSON.parse(payload); } catch { return; }
        if (coder.sessionId === id) handleCoderEvent(ev);
      });
    } catch (e) {
      if (coder.sessionId !== id) return;
      // transient drop — retry shortly
      await new Promise((res) => setTimeout(res, 1500));
    }
    if (coder.sessionId !== id) return;
  }
}

async function startCoderSession() {
  const cwd = $("setup-cwd").value.trim();
  if (!cwd) { toast("Enter a project directory", true); return; }
  $("setup-start").disabled = true;
  try {
    const r = await fetch("/api/coder/sessions", {
      method: "POST",
      headers: authHeaders(),
      body: JSON.stringify({
        cwd,
        auto_approve: $("setup-auto").checked,
        mode: $("setup-mode").value,
      }),
    });
    if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
    const data = await r.json();
    coder.sessionId = data.id;
    localStorage.setItem("localm.coderCwd", cwd);
    $("coder-setup").style.display = "none";
    $("coder-feed").style.display = "block";
    $("coder-composer").style.display = "block";
    $("coder-bar").classList.add("open");
    $("coder-cwd").textContent = data.cwd;
    $("coder-state").textContent = "idle";
    coderFeed().innerHTML = "";
    streamCoderEvents();
  } catch (e) {
    toast("Failed to start session: " + e.message, true);
  } finally {
    $("setup-start").disabled = false;
  }
}

async function sendCoderTask() {
  const input = $("coder-input");
  const text = input.value.trim();
  if (!text || !coder.sessionId) return;
  if (coder.busy) { toast("Agent is still working — stop it first or wait", true); return; }
  try {
    const r = await fetch(`/api/coder/sessions/${coder.sessionId}/message`, {
      method: "POST",
      headers: authHeaders(),
      body: JSON.stringify({ text }),
    });
    if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
    coder.busy = true;
    $("coder-state").textContent = "working…";
    addMessageRow(coderFeed(), "user", text);
    coderFeed().scrollTop = coderFeed().scrollHeight;
    input.value = "";
    autoGrow(input);
  } catch (e) {
    toast("Failed to send task: " + e.message, true);
  }
}

async function endCoderSession() {
  if (!coder.sessionId) return;
  const id = coder.sessionId;
  coder.sessionId = null;
  try {
    await fetch(`/api/coder/sessions/${id}`, { method: "DELETE", headers: authHeaders() });
  } catch (e) { /* server may already be gone */ }
  $("coder-setup").style.display = "block";
  $("coder-feed").style.display = "none";
  $("coder-composer").style.display = "none";
  $("coder-bar").classList.remove("open");
  coder.busy = false;
  flushAssistantBlock();
  coder.pendingCards = [];
}

$("setup-start").onclick = startCoderSession;
$("coder-send").onclick = sendCoderTask;
$("coder-input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); sendCoderTask(); }
});
$("coder-input").addEventListener("input", (e) => autoGrow(e.target));
$("coder-stop").onclick = async () => {
  if (!coder.sessionId) return;
  await fetch(`/api/coder/sessions/${coder.sessionId}/stop`, {
    method: "POST", headers: authHeaders(),
  });
  toast("Stop requested — agent halts at the next safe point");
};
$("coder-end").onclick = endCoderSession;

/* ================================================================ */
/*  Init                                                             */
/* ================================================================ */

$("setup-cwd").value = localStorage.getItem("localm.coderCwd") || "";
refreshModels();
setInterval(refreshModels, 30000);
renderConvList();
if (chat.conversations.length) {
  chat.activeId = chat.conversations[0].id;
  renderConvList();
  renderChat();
}
