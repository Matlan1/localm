// SPDX-License-Identifier: AGPL-3.0-or-later
// The chat turn in settings-perf.js: the Send/Stop button, the transcript
// controls during a reply, the message queue (double sends, Stop, failures,
// a model still loading, conversations that are gone), the chat title, saved
// failure notes, a web lookup stopped mid-flight, the memory checkbox and the
// image refusal toast.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp, runScript } from "./harness.mjs";

function abortError() {
  const e = new Error("The operation was aborted");
  e.name = "AbortError";
  return e;
}

function gated(gate, signal) {
  return new Promise((resolve, reject) => {
    if (signal && signal.aborted) { reject(abortError()); return; }
    if (signal) signal.addEventListener("abort", () => reject(abortError()), { once: true });
    gate.then(resolve);
  });
}

function makeGate() {
  let open;
  const promise = new Promise((r) => { open = r; });
  return { promise, open };
}

const jsonResp = (obj, status = 200) => ({
  ok: status < 400, status, statusText: "", headers: { get: () => null },
  json: async () => obj, text: async () => JSON.stringify(obj),
});

/** A jsdom app whose fetch records every call. `ctl.chat` decides the answer
 *  to each streamed completion ("ok", or a {status, detail, refusal} error);
 *  `ctl.holdStream` holds the stream until `ctl.streamGate` opens;
 *  `ctl.models` holds GET /api/models until `ctl.modelsGate` opens when set;
 *  `ctl.web` holds /api/web/retrieve until `ctl.webGate` opens. */
function setupApp() {
  const calls = [];
  const ctl = { chat: [], holdStream: false, streamGate: makeGate(), modelsGate: null,
                webGate: null, patch: null, routes: {} };
  const impl = async (url, opts = {}) => {
    const u = String(url);
    let body;
    try { body = opts.body ? JSON.parse(opts.body) : null; } catch { body = null; }
    calls.push({ url: u, opts, body });
    if (ctl.routes[u]) return ctl.routes[u](opts, body);
    if (u === "/v1/chat/completions" && body && body.stream === true) {
      const plan = ctl.chat.length ? ctl.chat.shift() : "ok";
      if (plan !== "ok") {
        return { ok: false, status: plan.status, statusText: "",
          headers: { get: (k) => (k === "X-Localm-Refusal" ? (plan.refusal || "") : null) },
          json: async () => ({ detail: plan.detail }), text: async () => plan.detail };
      }
      return { ok: true, status: 200, body: null, headers: { get: () => null },
        json: async () => ({}), __signal: opts.signal,
        __hold: ctl.holdStream ? ctl.streamGate.promise : null };
    }
    if (u.startsWith("/api/models")) {
      if (ctl.modelsGate) await ctl.modelsGate.promise;
      return jsonResp({ models: [{ name: "test-model", active: true }], active: "test-model" });
    }
    if (u === "/api/web/retrieve") {
      if (ctl.webGate) await gated(ctl.webGate.promise, opts.signal);
      return jsonResp({ query: "q", search_status: "ok", grounding: "failed", sources: [], chunks: [] });
    }
    if (u === "/v1/config" && opts.method === "PATCH" && ctl.patch) return ctl.patch(body);
    return jsonResp({});
  };
  const { window } = loadApp({ fetchImpl: impl });
  window.maybeCompactConversation = async () => {};
  window.refreshWebPolicy = async () => {};
  window.readSSE = async (r, onData) => {
    if (r.__hold) await gated(r.__hold, r.__signal);
    if (ctl.failStream) { ctl.failStream = false; throw new Error("connection reset"); }
    onData(JSON.stringify({ choices: [{ delta: { content: "reply" } }] }));
    onData(JSON.stringify({ choices: [{ delta: {}, finish_reason: "stop" }] }));
  };
  window.document.getElementById("p-web").checked = false;
  window.document.getElementById("p-speak").checked = false;
  runScript(window, "modelCache.active = 'test-model';");
  return { window, calls, ctl };
}

function activateConv(window, conv) {
  window.__testConv = conv;
  runScript(window, "chat.conversations = [window.__testConv]; chat.activeId = window.__testConv.id;");
  return conv;
}

function queue(window) {
  runScript(window, "window.__q = chat.queue;");
  return window.__q;
}

function busy(window) {
  runScript(window, "window.__b = chatBusy();");
  return window.__b;
}

const streamed = (calls) => calls.filter((c) => c.url === "/v1/chat/completions" && c.body && c.body.stream === true);

function lastUserText(call) {
  const users = call.body.messages.filter((m) => m.role === "user");
  const last = users[users.length - 1];
  return typeof last.content === "string" ? last.content : "";
}

async function waitFor(cond, what, ms = 3000) {
  const end = Date.now() + ms;
  while (Date.now() < end) {
    if (cond()) return;
    await new Promise((r) => setTimeout(r, 5));
  }
  assert.fail("timed out waiting for " + what);
}

async function settle(window) {
  await waitFor(() => !busy(window), "the chat to go idle");
  await new Promise((r) => setTimeout(r, 20));
  await waitFor(() => !busy(window), "the chat to stay idle");
}

async function send(window, text) {
  window.document.getElementById("chat-input").value = text;
  return window.sendChat();
}

const toastText = (window) => window.document.getElementById("toast").textContent;

// --------------------------------------------------------------------------
//  Send button and transcript controls while a reply runs
// --------------------------------------------------------------------------

test("while a reply streams the Send button is titled as a stop button, and as Send again after", async () => {
  const { window, calls, ctl } = setupApp();
  activateConv(window, { id: "c1", title: "t", messages: [] });
  const btn = window.document.getElementById("chat-send");
  ctl.holdStream = true;
  const turn = send(window, "hello");
  await waitFor(() => streamed(calls).length === 1, "the completion");
  assert.equal(btn.title, window.t("chat.stopReply"));
  assert.equal(btn.dataset.i18nTitle, "chat.stopReply", "a language switch keeps the stop title");
  ctl.streamGate.open();
  await turn;
  await settle(window);
  assert.equal(btn.title, window.t("chat.send"));
  assert.equal(btn.dataset.i18nTitle, "chat.send");
});

test("while a reply streams the transcript offers no edit or revert and its branch arrows are disabled", async () => {
  const { window, calls, ctl } = setupApp();
  activateConv(window, { id: "c1", title: "t", messages: [
    { id: "a", role: "user", content: "first" },
    { id: "b", role: "assistant", content: "answer" },
  ], branches: [{ parent: "root", tails: [null, [{ id: "z", role: "user", content: "other" }]], current: 0 }] });
  window.renderChat();
  const doc = window.document;
  const labels = () => [...doc.querySelectorAll("#chat-messages .msg-row:not(.queued) .msg-meta button.action")]
    .map((b) => b.title || b.textContent.trim());
  const arrows = () => [...doc.querySelectorAll("#chat-messages .msg-meta .variant button")];
  assert.ok(labels().includes(window.t("chat.action.edit")), "edit is offered while idle");
  assert.ok(arrows().length === 2 && arrows().every((b) => !b.disabled));

  ctl.holdStream = true;
  const turn = send(window, "second");
  await waitFor(() => streamed(calls).length === 1, "the completion");
  assert.ok(!labels().includes(window.t("chat.action.edit")), "no edit while the reply streams");
  assert.ok(!labels().includes(window.t("chat.action.revert")), "no revert while the reply streams");
  assert.ok(arrows().length === 2 && arrows().every((b) => b.disabled), "branch arrows are disabled");

  ctl.streamGate.open();
  await turn;
  await settle(window);
  assert.ok(labels().includes(window.t("chat.action.edit")), "edit is back after the reply");
  assert.ok(arrows().every((b) => !b.disabled), "branch arrows are enabled again");
});

// --------------------------------------------------------------------------
//  Title
// --------------------------------------------------------------------------

test("the first message keeps a title the person set and replaces only an automatic one", async () => {
  const { window } = setupApp();
  const typed = activateConv(window, { id: "c1", title: "My trip plans", messages: [] });
  await send(window, "where should I go in May?");
  await settle(window);
  assert.equal(typed.title, "My trip plans");

  const fresh = activateConv(window, { id: "c2", title: "New chat", messages: [] });
  await send(window, "first question");
  await settle(window);
  assert.equal(fresh.title, "first question");
  // The chat is empty again (an edit of its first message): the automatic
  // title follows the new first message.
  fresh.messages = [];
  await send(window, "second try");
  await settle(window);
  assert.equal(fresh.title, "second try");
  // A rename after that is kept.
  fresh.title = "Renamed";
  fresh.messages = [];
  await send(window, "third try");
  await settle(window);
  assert.equal(fresh.title, "Renamed");
});

// --------------------------------------------------------------------------
//  Queue: double sends, Stop, failures, gone conversations, loading models
// --------------------------------------------------------------------------

test("two quick sends in a pinned chat send each message once, in order", async () => {
  const { window, calls, ctl } = setupApp();
  activateConv(window, { id: "c1", title: "t", messages: [], pinnedModel: "test-model" });
  ctl.modelsGate = makeGate();
  const first = send(window, "first");
  // A second Enter with the box already emptied sends nothing.
  const again = window.sendChat();
  const second = send(window, "second");
  ctl.modelsGate.open();
  await Promise.all([first, again, second]);
  await waitFor(() => streamed(calls).length === 2, "both completions");
  await settle(window);
  assert.deepEqual(streamed(calls).map(lastUserText), ["first", "second"]);
  assert.equal(queue(window).length, 0);
});

test("a refused pinned-model check gives the message back to the composer", async () => {
  const { window, calls } = setupApp();
  activateConv(window, { id: "c1", title: "t", messages: [], pinnedModel: "gone-model" });
  runScript(window, "modelCache.models = [];");
  window.resolvePinnedModel = async () => false;
  window.showPinnedGone = () => null;
  await send(window, "keep me");
  assert.equal(window.document.getElementById("chat-input").value, "keep me");
  assert.equal(streamed(calls).length, 0);
});

test("a double click on Stop does not send the queued message", async () => {
  const { window, calls, ctl } = setupApp();
  activateConv(window, { id: "c1", title: "t", messages: [] });
  ctl.holdStream = true;
  const first = send(window, "first");
  await waitFor(() => streamed(calls).length === 1, "the first completion");
  await send(window, "second");
  const btn = window.document.getElementById("chat-send");
  btn.dispatchEvent(new window.MouseEvent("click", { bubbles: true, detail: 1 }));
  await first;
  await settle(window);
  btn.dispatchEvent(new window.MouseEvent("click", { bubbles: true, detail: 2 }));
  await new Promise((r) => setTimeout(r, 30));
  assert.equal(streamed(calls).length, 1, "the second click of the double click sends nothing");
  assert.equal(queue(window).length, 1, "the message stays queued");
  // A separate click on Send dispatches it.
  ctl.holdStream = false;
  btn.dispatchEvent(new window.MouseEvent("click", { bubbles: true, detail: 1 }));
  await waitFor(() => streamed(calls).length === 2, "the queued completion");
  await settle(window);
});

test("a failed reply leaves the queued message queued", async () => {
  const { window, calls, ctl } = setupApp();
  activateConv(window, { id: "c1", title: "t", messages: [] });
  ctl.holdStream = true;
  const first = send(window, "first");
  await waitFor(() => streamed(calls).length === 1, "the first completion");
  await send(window, "second");
  // The model is unloaded under the running reply: the stream breaks.
  ctl.failStream = true;
  ctl.streamGate.open();
  await first;
  await settle(window);
  assert.equal(streamed(calls).length, 1, "the queued message is not sent after the failure");
  assert.equal(queue(window).length, 1);
  assert.ok(window.document.querySelector(".msg-row.queued"), "it is still shown as queued");
});

test("a queued message of a conversation that is gone is dropped, never drawn in another chat", () => {
  const { window } = setupApp();
  activateConv(window, { id: "c1", title: "t", messages: [] });
  runScript(window, `chat.queue.push({ text: "orphan", attachments: [], docs: [], clips: [], convId: "gone" });
                     chat.conversations = []; chat.activeId = null;`);
  window.renderQueuedIndicator();
  assert.equal(window.document.querySelectorAll(".msg-row.queued").length, 0);
  assert.equal(queue(window).length, 0, "the orphan is dropped from the queue");

  activateConv(window, { id: "c2", title: "t", messages: [] });
  runScript(window, `chat.queue.push({ text: "mine", attachments: [], docs: [], clips: [], convId: "c2" });`);
  runScript(window, "chat.activeId = null;");
  window.renderQueuedIndicator();
  assert.equal(window.document.querySelectorAll(".msg-row.queued").length, 0,
    "with no chat open nothing queued is drawn");
  assert.equal(queue(window).length, 1, "a queued message of a live chat is kept");
});

test("a send while the model loads is queued and sent once it has loaded", async () => {
  const { window, calls } = setupApp();
  activateConv(window, { id: "c1", title: "t", messages: [] });
  runScript(window, "modelCache.active = ''; modelCache.resumable = ''; modelCache.loading = 'big-model';");
  await send(window, "while loading");
  assert.equal(streamed(calls).length, 0, "nothing is sent during the load");
  assert.equal(queue(window).length, 1);
  assert.match(toastText(window), /big-model/);
  assert.ok(window.document.querySelector(".msg-row.queued"), "the chat shows the message as queued");
  assert.doesNotMatch(window.document.getElementById("toast").className, /error/);

  runScript(window, "modelCache.active = 'big-model'; modelCache.loading = '';");
  window.dispatchEvent(new window.CustomEvent("localm:active-model-changed", { detail: { model: "big-model" } }));
  await waitFor(() => streamed(calls).length === 1, "the completion after the load");
  await settle(window);
  assert.equal(lastUserText(streamed(calls)[0]), "while loading");
});

test("a model change does not send a message left queued after a Stop", async () => {
  const { window, calls } = setupApp();
  activateConv(window, { id: "c1", title: "t", messages: [] });
  runScript(window, `chat.queue.push({ text: "left", attachments: [], docs: [], clips: [], convId: "c1" });`);
  window.dispatchEvent(new window.CustomEvent("localm:active-model-changed", { detail: { model: "test-model" } }));
  await new Promise((r) => setTimeout(r, 30));
  assert.equal(streamed(calls).length, 0);
  assert.equal(queue(window).length, 1);
});

// --------------------------------------------------------------------------
//  Failure notes
// --------------------------------------------------------------------------

test("a failed request is saved as a failure note that is shown after a reload and never sent", async () => {
  const { window, calls, ctl } = setupApp();
  const conv = activateConv(window, { id: "c1", title: "t", messages: [] });
  ctl.chat.push({ status: 500, detail: "the backend crashed" });
  await send(window, "hello");
  await settle(window);
  const note = conv.messages[conv.messages.length - 1];
  assert.equal(note.failureNote, true);
  assert.match(note.content, /the backend crashed/);
  runScript(window, "renderChat();");
  assert.match(window.document.getElementById("chat-messages").textContent, /the backend crashed/,
    "a redraw from the saved conversation still shows the error");

  await send(window, "again");
  await settle(window);
  const wire = streamed(calls)[1].body.messages;
  assert.ok(!wire.some((m) => /the backend crashed/.test(String(m.content))),
    "the failure note is not sent to the model");
  assert.equal(window.lastReplyIndex({ messages: [{ role: "user", content: "x" }, note] }), -1,
    "a continue never continues a failure note");
});

// --------------------------------------------------------------------------
//  Web lookup stopped
// --------------------------------------------------------------------------

test("Stop during a model-requested web lookup aborts it and ends the turn", async () => {
  const { window, calls, ctl } = setupApp();
  const conv = activateConv(window, { id: "c1", title: "t", messages: [] });
  window.document.getElementById("p-web").checked = true;
  runScript(window, "chat.netOff = false;");
  window.webModeIsAsk = async () => false;
  let round = 0;
  window.readSSE = async (_r, onData) => {
    round += 1;
    const text = round === 1
      ? '<tool_call>{"name": "web_search", "args": {"query": "q"}}</tool_call>' : "answer";
    onData(JSON.stringify({ choices: [{ delta: { content: text } }] }));
    onData(JSON.stringify({ choices: [{ delta: {}, finish_reason: "stop" }] }));
  };
  ctl.webGate = makeGate();
  const turn = send(window, "look it up");
  await waitFor(() => calls.some((c) => c.url === "/api/web/retrieve"), "the web lookup");
  const lookup = calls.find((c) => c.url === "/api/web/retrieve");
  assert.ok(lookup.opts.signal, "the lookup carries the turn's signal");
  window.document.getElementById("chat-send").click();
  assert.equal(lookup.opts.signal.aborted, true, "Stop aborts the lookup");
  await turn;
  await settle(window);
  assert.equal(streamed(calls).length, 1, "no completion follows a stopped lookup");
  const ev = conv.messages.find((m) => m.kind === "tool");
  assert.equal(ev.status, "failed");
  runScript(window, "window.__stopErr = WEB_STOPPED_ERROR;");
  assert.equal(ev.error, window.__stopErr);
  assert.ok(ev.error);
});

// --------------------------------------------------------------------------
//  Memory checkbox
// --------------------------------------------------------------------------

test("a refused memory checkbox save puts the box back and says why", async () => {
  const { window, ctl } = setupApp();
  ctl.patch = () => jsonResp({ detail: "This key lacks the config:write scope" }, 403);
  const box = window.document.getElementById("p-memory");
  box.checked = false;
  box.checked = true;
  const ok = await window.saveMemoryToggle(box);
  assert.equal(ok, false);
  assert.equal(box.checked, false, "the box shows the setting that is still in force");
  assert.match(toastText(window), /config:write/);

  ctl.patch = () => jsonResp({});
  box.checked = true;
  assert.equal(await window.saveMemoryToggle(box), true);
  assert.equal(box.checked, true);
});

// --------------------------------------------------------------------------
//  Image refusal
// --------------------------------------------------------------------------

test("an image refused by a text-only model shows the server's sentence about the vision models", async () => {
  const { window, ctl } = setupApp();
  activateConv(window, { id: "c1", title: "t", messages: [] });
  const detail = "This model cannot accept image input. Vision models in your library: llava-7b.";
  ctl.chat.push({ status: 400, detail, refusal: "image" });
  runScript(window, `chat.attachments = [{ name: "a.png", dataUri: "data:image/png;base64,iVBORw0KGgo" }];`);
  await send(window, "what is this?");
  await settle(window);
  assert.match(toastText(window), /llava-7b/);
  assert.match(toastText(window), /cannot read images/);
});
