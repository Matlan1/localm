// SPDX-License-Identifier: AGPL-3.0-or-later
// The chat request lets the server answer with a model that has what the
// request needs (capability routing) unless the conversation is pinned to a
// model; a routed reply records and shows which model answered; a
// conversation that outgrows the selected model's trained window asks for a
// roomier installed model instead of being compacted.
import { test } from "node:test";
import assert from "node:assert/strict";

import { loadApp, runScript } from "./harness.mjs";

function activateConv(window, conv) {
  window.__testConv = conv;
  runScript(window, "chat.conversations = [window.__testConv]; chat.activeId = window.__testConv.id;");
}

function setup({ routingHeader = null, models = null, active = "plain" } = {}) {
  const calls = [];
  const impl = async (url, opts = {}) => {
    let body;
    try { body = opts.body ? JSON.parse(opts.body) : null; } catch { body = opts.body; }
    calls.push({ url: String(url), body });
    const headers = {
      get: (k) => (k === "X-Localm-Model-Routing" && routingHeader
        ? JSON.stringify(routingHeader) : null),
    };
    return { ok: true, status: 200, headers, json: async () => ({}), text: async () => "" };
  };
  const { window } = loadApp({ fetchImpl: impl });
  window.readSSE = async (_r, onData) => {
    onData(JSON.stringify({ choices: [{ delta: { content: "an answer" } }] }));
    onData(JSON.stringify({ choices: [{ delta: {}, finish_reason: "stop" }] }));
  };
  const doc = window.document;
  doc.getElementById("p-speak").checked = false;
  doc.getElementById("p-memory").checked = false;
  doc.getElementById("p-web").checked = false;
  const select = doc.getElementById("model-select");
  for (const name of ["plain", "seer", "roomy"]) {
    const opt = doc.createElement("option");
    opt.value = name;
    select.appendChild(opt);
  }
  select.value = active;
  window.__models = models || [];
  runScript(window, `modelCache.active = ${JSON.stringify(active)}; modelCache.models = window.__models;`);
  return { window, calls, doc };
}

const chatPosts = (calls) => calls.filter((c) => c.url === "/v1/chat/completions");

test("an unpinned chat sends the selected model as a preference, not a pin", async () => {
  const { window, calls } = setup();
  const conv = { id: "c1", title: "t", messages: [{ role: "user", content: "hi" }] };
  activateConv(window, conv);
  await window.runCompletion(conv);
  const [post] = chatPosts(calls);
  assert.equal(post.body.model, "plain");
  assert.equal(post.body.pin_model, false,
    "a request that is not pinned must let the server route it");
});

test("a pinned chat always names its pinned model and pins it", async () => {
  const { window, calls } = setup({ active: "seer" });
  const conv = { id: "c1", title: "t", pinnedModel: "plain",
                 messages: [{ role: "user", content: "hi" }] };
  activateConv(window, conv);
  await window.runCompletion(conv);
  const [post] = chatPosts(calls);
  assert.equal(post.body.model, "plain", "the pinned model, not the sidebar's selection");
  assert.equal(post.body.pin_model, true);
});

test("web access asks for a model that can emit structured tool calls", async () => {
  const { window, calls, doc } = setup();
  doc.getElementById("p-web").checked = true;
  const conv = { id: "c1", title: "t", messages: [{ role: "user", content: "hi" }] };
  activateConv(window, conv);
  await window.runCompletion(conv);
  const [post] = chatPosts(calls);
  assert.deepEqual(post.body.required_capabilities, ["tool_use"]);
});

test("without web access no capability is requested", async () => {
  const { window, calls } = setup();
  const conv = { id: "c1", title: "t", messages: [{ role: "user", content: "hi" }] };
  activateConv(window, conv);
  await window.runCompletion(conv);
  const [post] = chatPosts(calls);
  assert.equal(post.body.required_capabilities, undefined);
});

test("a routed reply records the model that answered and why, and shows it", async () => {
  const { window, doc } = setup({ routingHeader: {
    resolved: "seer", requested: "plain", routed: true, pinned: false,
    gaps: { vision: "absent" }, unmet: [] } });
  const conv = { id: "c1", title: "t", messages: [{ role: "user", content: "what is this?" }] };
  activateConv(window, conv);
  await window.runCompletion(conv);
  const reply = conv.messages[conv.messages.length - 1];
  assert.equal(reply.model, "seer", "the transcript names the model that actually answered");
  assert.deepEqual(JSON.parse(JSON.stringify(reply.routed)),
                   { from: "plain", gaps: ["vision"] });
  const chip = doc.querySelector(".routed-chip");
  assert.ok(chip, "the routed reply carries a chip");
  assert.match(chip.textContent, /plain/);
  assert.match(chip.title, /reading images/);
});

test("a reply that was not routed records the selected model and no chip", async () => {
  const { window, doc } = setup({ routingHeader: {
    resolved: "plain", requested: "plain", routed: false, pinned: true,
    gaps: { tool_use: "absent" }, unmet: [] } });
  const conv = { id: "c1", title: "t", messages: [{ role: "user", content: "hi" }] };
  activateConv(window, conv);
  await window.runCompletion(conv);
  const reply = conv.messages[conv.messages.length - 1];
  assert.equal(reply.model, "plain");
  assert.equal(reply.routed, undefined);
  assert.equal(doc.querySelector(".routed-chip"), null);
});

test("parseRoutingHeader reads the header and survives junk", () => {
  const { window } = setup();
  const resp = (raw) => ({ headers: { get: () => raw } });
  const r = window.parseRoutingHeader(resp(JSON.stringify({
    resolved: "b", requested: "a", routed: true, pinned: false,
    gaps: { vision: "absent", tool_use: "unknown" } })));
  assert.deepEqual(JSON.parse(JSON.stringify(r)),
                   { resolved: "b", requested: "a", routed: true, pinned: false,
                     gaps: ["vision", "tool_use"], unmet: [] });
  const partial = window.parseRoutingHeader(resp(JSON.stringify({
    resolved: "seer", requested: "smol", routed: true, pinned: false,
    gaps: { vision: "absent", tool_use: "unknown" }, unmet: ["tool_use"] })));
  assert.deepEqual(JSON.parse(JSON.stringify(partial.gaps)), ["vision"],
                   "the chip names only what the answering model provides");
  assert.deepEqual(JSON.parse(JSON.stringify(partial.unmet)), ["tool_use"]);
  assert.equal(window.parseRoutingHeader(resp(null)), null);
  assert.equal(window.parseRoutingHeader(resp("not json")), null);
  assert.equal(window.parseRoutingHeader(null), null);
});

// ---- the per-chat pin ----

test("pinning pins the conversation to the selected model and saves it", async () => {
  const { window, calls, doc } = setup({ active: "seer" });
  runScript(window, "chat.persist = true;");
  const conv = { id: "c1", title: "t", messages: [{ role: "user", content: "hi" }] };
  activateConv(window, conv);
  const box = doc.getElementById("p-pin-model");
  box.checked = true;
  box.dispatchEvent(new window.Event("change"));
  assert.equal(conv.pinnedModel, "seer");
  assert.equal(doc.getElementById("p-pin-model-name").textContent, "seer");
  await new Promise((r) => setTimeout(r, 700));
  const put = calls.find((c) => c.url === "/api/conversations/c1");
  assert.ok(put, "the pin is pushed to the server store");
  assert.equal(put.body.pinned_model, "seer");

  box.checked = false;
  box.dispatchEvent(new window.Event("change"));
  assert.equal(conv.pinnedModel, undefined, "unpinning removes it");
});

test("the pin checkbox follows the conversation on screen", () => {
  const { window, doc } = setup({ active: "seer" });
  activateConv(window, { id: "a", title: "t", pinnedModel: "plain", messages: [] });
  runScript(window, "renderChat();");
  assert.equal(doc.getElementById("p-pin-model").checked, true);
  assert.equal(doc.getElementById("p-pin-model-name").textContent, "plain");
  activateConv(window, { id: "b", title: "t", messages: [] });
  runScript(window, "renderChat();");
  assert.equal(doc.getElementById("p-pin-model").checked, false);
  assert.equal(doc.getElementById("p-pin-model-name").textContent, "seer");
});

// ---- a longer conversation than the selected model was trained for ----

const LONG = "word ".repeat(4000);   // about 5000 estimated tokens

function longConv(extra = {}) {
  return {
    id: "c1", title: "t", ...extra,
    messages: [
      { role: "user", content: LONG }, { role: "assistant", content: "ok" },
      { role: "user", content: "a" }, { role: "assistant", content: "b" },
      { role: "user", content: "c" }, { role: "assistant", content: "d" },
      { role: "user", content: "and now?" },
    ],
  };
}

test("a conversation that outgrows the selected model asks for a roomier one instead of compacting", async () => {
  const { window, calls } = setup({ models: [
    { name: "plain", model_type: "llm", context_length: 4096 },
    { name: "roomy", model_type: "llm", context_length: 32768 },
  ] });
  runScript(window, "chat.ctxMax = 4096;");
  const conv = longConv();
  activateConv(window, conv);
  await window.runCompletion(conv);
  const posts = chatPosts(calls);
  assert.equal(posts.length, 1, "no summarisation request: the conversation was not compacted");
  assert.ok(posts[0].body.min_context > 4096,
    "the request states the window it needs, above the selected model's");
  assert.equal(posts[0].body.pin_model, false);
  assert.equal(conv.messages[0].content, LONG, "the history is intact");
});

test("with no roomier model installed it still compacts", async () => {
  const { window, calls } = setup({ models: [
    { name: "plain", model_type: "llm", context_length: 4096 },
    { name: "embedder", model_type: "embedding", context_length: 32768 },
  ] });
  runScript(window, "chat.ctxMax = 4096;");
  const conv = longConv();
  activateConv(window, conv);
  await window.runCompletion(conv);
  const posts = chatPosts(calls);
  assert.equal(posts.length, 2, "one summarisation request, then the reply");
  assert.equal(posts[1].body.min_context, undefined);
});

test("a pinned conversation compacts rather than asking for another model", async () => {
  const { window, calls } = setup({ models: [
    { name: "plain", model_type: "llm", context_length: 4096 },
    { name: "roomy", model_type: "llm", context_length: 32768 },
  ] });
  runScript(window, "chat.ctxMax = 4096;");
  const conv = longConv({ pinnedModel: "plain" });
  activateConv(window, conv);
  await window.runCompletion(conv);
  const posts = chatPosts(calls);
  assert.equal(posts.length, 2, "compacted: the pin is honored");
  assert.equal(posts[1].body.min_context, undefined);
  assert.equal(posts[1].body.pin_model, true);
});

test("a selected model whose window is unknown compacts as before", async () => {
  const { window, calls } = setup({ models: [
    { name: "plain", model_type: "llm" },
    { name: "roomy", model_type: "llm", context_length: 32768 },
  ] });
  runScript(window, "chat.ctxMax = 4096;");
  const conv = longConv();
  activateConv(window, conv);
  await window.runCompletion(conv);
  assert.equal(chatPosts(calls).length, 2);
});
