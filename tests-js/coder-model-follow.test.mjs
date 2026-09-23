// SPDX-License-Identifier: AGPL-3.0-or-later
// A coder session started without a model follows the model loaded in the
// sidebar, as its preferred (unpinned) model; a session whose model was chosen
// is pinned to it and does not follow.
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp, runScript } from "./harness.mjs";

function setup(sessionInfo) {
  const calls = [];
  const fetchImpl = async (url, opts = {}) => {
    let body = null;
    try { body = opts.body ? JSON.parse(opts.body) : null; } catch { body = opts.body; }
    calls.push({ url: String(url), body });
    if (/\/api\/coder\/sessions\/[^/]+\/model$/.test(String(url))) {
      return { ok: true, status: 200,
               json: async () => ({ ...sessionInfo, model: body.model,
                                    model_pinned: body.pin }) };
    }
    return { ok: true, status: 200, json: async () => ({}) };
  };
  const { window } = loadApp({ fetchImpl });
  window.__info = sessionInfo;
  runScript(window, `
    coder.sessions.set("sid1", { info: window.__info, busy: false });
    coder.activeId = "sid1";
  `);
  return { window, calls };
}

async function flush() {
  for (let i = 0; i < 5; i++) await new Promise((r) => setTimeout(r, 10));
}

test("an unpinned session follows a sidebar model switch, still unpinned", async () => {
  const { window, calls } = setup({ id: "sid1", model: "old", model_pinned: false });
  window.dispatchEvent(new window.CustomEvent("localm:model-switched",
    { detail: { model: "new" } }));
  await flush();
  const posts = calls.filter((c) => c.url === "/api/coder/sessions/sid1/model");
  assert.equal(posts.length, 1);
  assert.equal(posts[0].body.model, "new");
  assert.equal(posts[0].body.pin, false, "following the loaded model is not a pin");
});

test("a pinned session does not follow a sidebar model switch", async () => {
  const { window, calls } = setup({ id: "sid1", model: "chosen", model_pinned: true });
  window.dispatchEvent(new window.CustomEvent("localm:model-switched",
    { detail: { model: "new" } }));
  await flush();
  assert.equal(calls.filter((c) => c.url === "/api/coder/sessions/sid1/model").length, 0);
});

test("choosing a model for the session pins it", async () => {
  const { window, calls } = setup({ id: "sid1", model: "old", model_pinned: false });
  await window.switchActiveSessionModel("picked");
  const post = calls.find((c) => c.url === "/api/coder/sessions/sid1/model");
  assert.equal(post.body.pin, true);
});
