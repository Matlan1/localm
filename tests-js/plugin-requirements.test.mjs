// SPDX-License-Identifier: AGPL-3.0-or-later
// B15: the plugin catalog now warns when an installed plugin's required plugins
// are missing ("requires alpha (missing: alpha)") and offers a one-click
// "Install requirements" button that installs each missing dependency.
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

function pluginsPayload() {
  return {
    plugins: [
      { name: "beta", label: "Beta", builtin: true, installed: true,
        enabled: false, active: false, protected: false, description: "needs alpha",
        requires: ["alpha"], missing_requires: ["alpha"] },
      { name: "gamma", label: "Gamma", builtin: true, installed: true,
        enabled: true, active: true, protected: false, description: "self-sufficient",
        requires: [], missing_requires: [] },
      { name: "alpha", label: "Alpha", builtin: true, installed: false,
        enabled: false, active: false, protected: false, description: "a dep",
        requires: [], missing_requires: [] },
    ],
  };
}

function makeFetch(calls) {
  return async (url, opts = {}) => {
    if (url === "/api/plugins") {
      return { ok: true, status: 200, json: async () => pluginsPayload(), text: async () => "" };
    }
    if (typeof url === "string" && url.includes("/install")) {
      calls.push(url);
      return { ok: true, status: 200, json: async () => ({ status: "installed" }), text: async () => "" };
    }
    return {
      ok: true, status: 200, text: async () => "",
      json: async () => ({ models: [], active: "", conversations: [], plugins: [], commands: [] }),
    };
  };
}

async function renderCatalog(win) {
  runScript(win, "renderCatalogPlugins();");
  await new Promise((r) => setTimeout(r, 0));
}

test("catalog warns about missing required plugins; others show no warning", async () => {
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch([]) });
  await renderCatalog(win);
  const box = win.document.getElementById("catalog-table");
  const text = box.textContent;
  assert.match(text, /requires alpha/, "names the requirement");
  assert.match(text, /missing: alpha/, "flags it as missing");

  // The self-sufficient plugin must not get a missing-requirements warning or button.
  assert.equal(box.querySelector('[data-reqfor="gamma"]'), null,
    "no requirements button for a plugin with nothing missing");
});

test("Install requirements installs each missing dependency", async () => {
  const calls = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(calls) });
  await renderCatalog(win);
  const box = win.document.getElementById("catalog-table");

  const btn = box.querySelector('[data-reqfor="beta"]');
  assert.ok(btn, "beta has an Install requirements button");
  btn.click();
  await new Promise((r) => setTimeout(r, 0));
  await new Promise((r) => setTimeout(r, 0));

  assert.ok(calls.some((u) => u.includes("/api/plugins/alpha/install")),
    "installs the missing dependency alpha");
});
