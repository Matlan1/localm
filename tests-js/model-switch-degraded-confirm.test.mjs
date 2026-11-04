// SPDX-License-Identifier: AGPL-3.0-or-later
// A "degraded_load" confirm_required answer is worded in the GUI's language
// from its figures (plan_mb, full_mb, free_mb, moe, unloaded), and the dialog
// the sidebar switch opens shows that text.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

import { loadApp, runScript } from "./harness.mjs";

const STATIC = join(dirname(fileURLToPath(import.meta.url)), "..",
  "localm", "plugins", "gui", "static");
const DE = JSON.parse(readFileSync(join(STATIC, "i18n", "de.json"), "utf-8"));

const MOE = {
  status: "confirm_required", model: "coder", reason: "degraded_load",
  plan_mb: 4200, full_mb: 25570, free_mb: 3000, moe: true,
  unloaded: ["chat-model"], unloads_shared: true, unloads_peers: false,
  detail: "server text",
};

test("a MoE degraded load names both figures and what would be unloaded", () => {
  const { window } = loadApp();
  window.__data = MOE;
  runScript(window, "window.__text = confirmLoadText(window.__data);");
  assert.equal(window.__text,
    "'coder' does not fit: even with its experts in system RAM, every layer on the GPU "
    + "needs about 4200 MB of free VRAM (about 25570 MB with every expert on the GPU "
    + "too), and about 3000 MB would be free after unloading 'chat-model', the shared "
    + "embedding, reranker and speech models. Loading it anyway runs some of its "
    + "layers on the CPU, which is slower.");
  assert.ok(!/localm config/.test(window.__text));
});

test("a dense degraded load with nothing to unload names the free VRAM now", () => {
  const { window } = loadApp();
  window.__data = { ...MOE, moe: false, plan_mb: 9000, full_mb: 9000, unloaded: [],
                    unloads_shared: false };
  runScript(window, "window.__text = confirmLoadText(window.__data);");
  assert.equal(window.__text,
    "'coder' does not fit: every layer on the GPU needs about 9000 MB of free VRAM, and "
    + "about 3000 MB is free. Loading it anyway runs some of its layers on the CPU, "
    + "which is slower.");
});

test("any other confirm_required answer shows the server's detail", () => {
  const { window } = loadApp();
  window.__data = { status: "confirm_required", model: "m", detail: "busy model text" };
  runScript(window, "window.__text = confirmLoadText(window.__data);");
  assert.equal(window.__text, "busy model text");
});

test("the question is German when the GUI is", async () => {
  const fetchImpl = async (url) => {
    const u = String(url);
    if (u.includes("/i18n/de.json")) return { ok: true, status: 200, json: async () => DE };
    if (u.includes("/i18n/")) return { ok: false, status: 404, json: async () => ({}) };
    return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
  };
  const { window } = loadApp({ fetchImpl });
  runScript(window, 'window.__p = applyLanguage("de");');
  await window.__p;
  window.__data = MOE;
  runScript(window, "window.__text = confirmLoadText(window.__data);");
  assert.match(window.__text, /^'coder' passt nicht: Selbst mit seinen Experten im System-RAM/);
  assert.match(window.__text, /nach dem Entladen von 'chat-model', den gemeinsamen/);
});

test("the sidebar switch dialog shows the worded question", async () => {
  const fetchImpl = async (url) => {
    const u = String(url);
    if (u.startsWith("/api/models/load")) {
      return { ok: true, status: 200, json: async () => MOE, text: async () => "" };
    }
    return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
  };
  const { window } = loadApp({ fetchImpl });
  runScript(window, `
    window.__asked = [];
    confirmDangerAsync = async (title, text) => { window.__asked.push(text); return false; };
    window.__p = switchModel("coder");`);
  const res = await window.__p;
  assert.equal(res.status, "cancelled");
  assert.equal(window.__asked.length, 1);
  assert.match(window.__asked[0], /even with its experts in system RAM/);
});

const CHOICE = {
  status: "confirm_required", model: "coder", reason: "unload_choice",
  unloaded: ["chat-model"], plan_mb: 9000, free_mb: 3000, moe: false,
  detail: "server text",
};

async function switchWithChoice(choiceKey) {
  const posts = [];
  const fetchImpl = async (url, opts = {}) => {
    const u = String(url);
    if (u.startsWith("/api/models/load")) {
      const body = JSON.parse(opts.body);
      posts.push(body);
      const answer = posts.length === 1 ? CHOICE : { status: "loaded", model: "coder" };
      return { ok: true, status: 200, json: async () => answer, text: async () => "" };
    }
    return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
  };
  const { window } = loadApp({ fetchImpl });
  window.__choice = choiceKey;
  runScript(window, `
    window.__asked = [];
    chooseAsync = async (title, text, choices) => {
      window.__asked.push({ text, keys: choices.map((c) => c.key),
                            labels: choices.map((c) => c.label) });
      return window.__choice;
    };
    window.__p = switchModel("coder");`);
  const res = await window.__p;
  return { res, posts, asked: window.__asked };
}

test("the never policy's choice offers loading beside or unloading", async () => {
  const { asked } = await switchWithChoice(null);
  assert.equal(asked.length, 1);
  assert.deepEqual([...asked[0].keys], ["beside", "unload"]);
  assert.match(asked[0].labels[1], /Unload 'chat-model' for this load/);
  assert.match(asked[0].text, /about 3000 MB is free beside 'chat-model'/);
});

test("choosing to load beside retries with keep_others", async () => {
  const { res, posts } = await switchWithChoice("beside");
  assert.equal(res.status, "loaded");
  assert.deepEqual(posts[1], { model: "coder", keep_others: true });
});

test("choosing to unload retries with force", async () => {
  const { posts } = await switchWithChoice("unload");
  assert.deepEqual(posts[1], { model: "coder", force: true });
});

test("cancelling the choice loads nothing", async () => {
  const { res, posts } = await switchWithChoice(null);
  assert.equal(res.status, "cancelled");
  assert.equal(posts.length, 1);
});

test("the ask policy's question names what it unloads and what it frees", () => {
  const { window } = loadApp();
  window.__data = { status: "confirm_required", model: "coder", reason: "unload_ask",
                    unloaded: ["chat-model"], unloads_shared: false,
                    peers: ["'big' (port 9000)"], freed_mb: 4000 };
  runScript(window, "window.__text = confirmLoadText(window.__data);");
  assert.equal(window.__text,
    "Loading 'coder' unloads 'chat-model', 'big' (port 9000) of another localm "
    + "instance, which frees about 4000 MB of VRAM.");
});
