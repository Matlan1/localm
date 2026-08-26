// SPDX-License-Identifier: AGPL-3.0-or-later
// images.js's img-generate handler forwards the CFG field into the
// /api/imagine request body, as music.js and video.js do for their own.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages } from "./harness.mjs";

const tick = () => new Promise((r) => setTimeout(r, 0));

function setup() {
  const calls = [];
  const fetchImpl = async (url, opts = {}) => {
    calls.push({ url: String(url), opts });
    if (String(url) === "/api/imagine") {
      return { ok: true, status: 200, json: async () => ({ job_id: "j1" }) };
    }
    return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
  };
  const { window } = loadAppWithPages({ fetchImpl });
  return { window, calls };
}

test("a CFG value is forwarded to POST /api/imagine", async () => {
  const { window, calls } = setup();
  window.document.getElementById("img-prompt").value = "a fox in snow";
  window.document.getElementById("img-negative").value = "blurry";
  window.document.getElementById("img-cfg").value = "4.2";

  window.document.getElementById("img-generate").onclick();
  await tick(); await tick();

  const imagine = calls.find((c) => c.url === "/api/imagine");
  assert.ok(imagine, "/api/imagine was called");
  const body = JSON.parse(imagine.opts.body);
  assert.equal(body.cfg, 4.2);
});

test("CFG left blank sends null, matching guidance/seed/denoise", async () => {
  const { window, calls } = setup();
  window.document.getElementById("img-prompt").value = "a fox in snow";
  // img-cfg stays at its default "" (blank) - no value typed.

  window.document.getElementById("img-generate").onclick();
  await tick(); await tick();

  const imagine = calls.find((c) => c.url === "/api/imagine");
  assert.ok(imagine, "/api/imagine was called");
  const body = JSON.parse(imagine.opts.body);
  assert.equal(body.cfg, null, "a blank field sends null, the same as guidance/seed/denoise");
});
