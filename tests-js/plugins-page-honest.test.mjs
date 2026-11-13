// SPDX-License-Identifier: AGPL-3.0-or-later
// The Plugins page names plugins as people and the CLI know them, marks a
// plugin that did not load, gives external plugins a state and an on/off
// switch, and reports every refusal or failed request.
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

const tick = () => new Promise((r) => setTimeout(r, 0));
const settle = async () => { for (let i = 0; i < 30; i++) await tick(); };
const ok = (body, status = 200) => ({ ok: status < 400, status, text: async () => "",
                                      json: async () => body });

const row = (o) => ({ builtin: true, installed: true, enabled: true, active: true,
  protected: false, loaded: true, description: "d", requires: [], missing_requires: [],
  requires_extras: [], missing_deps: [], tool_exports: [], ...o });

function setup(plugins, extra = {}) {
  const calls = [];
  const fetchImpl = async (url, opts = {}) => {
    const u = String(url);
    calls.push({ url: u, opts });
    for (const [part, fn] of Object.entries(extra)) if (u.includes(part)) return fn(u, opts);
    if (u.endsWith("/api/plugins")) return ok({ plugins, errors: {} });
    return ok({});
  };
  const { window } = loadAppWithPages({ fetchImpl });
  runScript(window, "_catalogStaggerMs = 0;");
  return { window, calls };
}

test("a plugin is shown by its label and its plugin name when they differ", () => {
  const { window: win } = setup([]);
  assert.equal(win.pluginTitle({ name: "rag", label: "Knowledge" }), "Knowledge (rag)");
  assert.equal(win.pluginTitle({ name: "chat", label: "Chat" }), "Chat");
  assert.equal(win.pluginTitle({ name: "mcp" }), "mcp");
});

test("an installed, enabled plugin that did not load is marked failed and not counted active", async () => {
  const { window: win } = setup([
    row({ name: "chat", label: "Chat", protected: true }),
    row({ name: "rag", label: "Knowledge", loaded: false, error: "load failed: boom" }),
  ]);
  runScript(win, "renderCatalogPlugins();");
  await settle();
  const box = win.document.getElementById("catalog-table");
  const rag = [...box.querySelectorAll("tbody tr")].find((tr) => tr.textContent.includes("Knowledge (rag)"));
  assert.ok(rag, "the row names the label and the plugin name");
  assert.equal(rag.querySelector(".job-state").textContent, win.t("plugins.pill.failed"));
  assert.match(box.querySelector(".catalog-status").textContent, /1\/2 plugins active/);
});

test("first-party rows: protected first, then installed, then available, each by shown name", async () => {
  const { window: win } = setup([
    row({ name: "voice", label: "Voice", installed: false, enabled: false, active: false }),
    row({ name: "tts", label: "Speech" }),
    row({ name: "rag", label: "Knowledge" }),
    row({ name: "chat", label: "Chat", protected: true }),
    row({ name: "image", label: "Images", installed: false, enabled: false, active: false }),
  ]);
  runScript(win, "renderCatalogPlugins();");
  await settle();
  const names = [...win.document.querySelectorAll("#catalog-table tbody tr .name")].map((n) => n.textContent);
  assert.deepEqual(names, ["Chat", "Knowledge (rag)", "Speech (tts)", "Images (image)", "Voice"]);
});

test("an external plugin shows its state and can be switched on", async () => {
  const { window: win, calls } = setup([
    row({ name: "mine", label: "Mine", builtin: false, enabled: false, active: false, version: "1.0" }),
  ], { "/api/plugins/mine/enable": () => ok({ status: "enabled", name: "mine" }) });
  runScript(win, "refreshPluginsPage();");
  await settle();
  const box = win.document.getElementById("plugins-table");
  assert.equal(box.querySelector(".job-state").textContent, win.t("plugins.pill.installedOff"));
  const on = [...box.querySelectorAll("button")].find((b) => b.textContent === win.t("plugins.enable"));
  assert.ok(on, "an Enable button");
  on.click();
  await settle();
  assert.equal(calls.filter((c) => c.url.endsWith("/api/plugins/mine/enable")).length, 1);
});

test("a 403 from install-deps shows the server's reason (a missing scope is not called a host problem)", async () => {
  const { window: win } = setup([row({ name: "beta", label: "Beta", requires_extras: ["x"], missing_deps: ["x>=1"] })], {
    "/install-deps": () => ok({ detail: "Key lacks required scope: plugins:admin" }, 403),
  });
  runScript(win, "renderCatalogPlugins();");
  await settle();
  win.document.querySelector('[data-depsfor="beta"]').click();
  await settle();
  const toasts = win.document.getElementById("toast-errors").textContent;
  assert.match(toasts, /Key lacks required scope: plugins:admin/);
  assert.doesNotMatch(toasts, /on the host/);
});

test("an external install that cannot reach the server says so", async () => {
  const { window: win } = setup([], {
    "/api/plugins/install-external": () => { throw new TypeError("Failed to fetch"); },
  });
  win.document.getElementById("plugin-source").value = "/some/folder";
  win.document.getElementById("plugin-install").click();
  await settle();
  assert.match(win.document.getElementById("toast-errors").textContent, /Install failed: Failed to fetch/);
});
