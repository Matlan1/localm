// SPDX-License-Identifier: AGPL-3.0-or-later
// jsdom tests for the logo-style picker (LOGO_STYLES / applyLogoStyle /
// setLogoStyle / syncLogoStyleFromConfig / renderLogoPicker in
// localm/plugins/gui/static/app.js). The sidebar wordmark has three styles; the
// choice is stored in server config (logo_style) with localStorage as a cache.
// The blue half lives in a <span> so #logo span (var(--accent)) tints it. The
// default is the single-blue-M treatment (LocaL white + M blue).

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp } from "./harness.mjs";

const settle = () => new Promise((r) => setTimeout(r, 0));
const ok = (j) => ({ ok: true, status: 200, json: async () => j });

// Well-formed responses for the endpoints app.js's bootstrap touches on load.
// Optionally records calls and serves a given /v1/config body.
function goodFetch({ calls = null, config = {} } = {}) {
  return async (url, opts = {}) => {
    const u = String(url);
    if (calls) calls.push({ url: u, method: opts.method || "GET", body: opts.body });
    if (u.includes("/api/models")) return ok({ models: [], active: "" });
    if (u.includes("/api/plugins")) return ok({ plugins: [] });
    if (u.includes("/v1/config")) return ok(config);
    return ok({});
  };
}

test("default wordmark is LocaLM with a single blue M in the accent span", () => {
  const { window } = loadApp({ fetchImpl: goodFetch() });
  const logo = window.document.getElementById("logo");
  assert.equal(logo.textContent, "LocaLM");
  assert.equal(logo.querySelector("span").textContent, "M");
});

test("the picker renders one tile per style, default first and active", () => {
  const { window } = loadApp({ fetchImpl: goodFetch() });
  const tiles = [...window.document.querySelectorAll("#logo-style-picker .logo-tile")];
  assert.equal(tiles.length, 3, "three tiles");
  assert.deepEqual(tiles.map((t) => t.dataset.style), ["local-m", "loca-lm", "localm"]);
  // each tile is itself a wordmark: white text + a blue <span>
  const lower = tiles.find((t) => t.dataset.style === "localm");
  assert.equal(lower.textContent, "localm");
  assert.equal(lower.querySelector("span").textContent, "m");
  // the default (single blue M) is active out of the box
  const active = window.document.querySelector("#logo-style-picker .logo-tile.active");
  assert.equal(active.dataset.style, "local-m");
});

test("clicking a tile applies live, caches locally, and PATCHes the shared config", async () => {
  const calls = [];
  const { window } = loadApp({ fetchImpl: goodFetch({ calls }) });
  const tile = window.document.querySelector('#logo-style-picker .logo-tile[data-style="loca-lm"]');
  tile.click();
  // the local apply (render + cache + active marker) is synchronous
  const logo = window.document.getElementById("logo");
  assert.equal(logo.textContent, "LocaLM");
  assert.equal(logo.querySelector("span").textContent, "LM");
  assert.equal(window.localStorage.getItem("localm.logoStyle"), "loca-lm");
  assert.equal(
    window.document.querySelector("#logo-style-picker .logo-tile.active").dataset.style,
    "loca-lm");
  // and it persists to the shared server config
  await settle();
  const patch = calls.find((c) => c.method === "PATCH" && c.url.includes("/v1/config"));
  assert.ok(patch, "a PATCH /v1/config was sent");
  assert.match(String(patch.body), /"logo_style":"loca-lm"/);
});

test("syncLogoStyleFromConfig adopts the shared server choice", async () => {
  const { window } = loadApp({ fetchImpl: goodFetch({ config: { logo_style: "loca-lm" } }) });
  window.applyLogoStyle("local-m");            // start from the default
  await window.syncLogoStyleFromConfig();
  const logo = window.document.getElementById("logo");
  assert.equal(logo.textContent, "LocaLM");
  assert.equal(logo.querySelector("span").textContent, "LM");   // loca-lm -> LM blue
});

test("applyLogoStyle honours each known style and falls back to the default", () => {
  const { window } = loadApp({ fetchImpl: goodFetch() });
  const logo = window.document.getElementById("logo");

  window.applyLogoStyle("localm");
  assert.equal(logo.textContent, "localm");
  assert.equal(logo.querySelector("span").textContent, "m");
  assert.equal(window.localStorage.getItem("localm.logoStyle"), "localm");

  window.applyLogoStyle("nonsense");          // unknown -> default treatment
  assert.equal(logo.textContent, "LocaLM");
  assert.equal(logo.querySelector("span").textContent, "M");
  assert.equal(window.localStorage.getItem("localm.logoStyle"), "local-m");
});
