// SPDX-License-Identifier: AGPL-3.0-or-later
// A plugin tab's nav entry is rendered only once the plugin's client module has
// loaded and created its view, so clicking it right after an install opens the
// tab instead of falling back to Chat.
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp, runScript } from "./harness.mjs";

const PLUGIN = { name: "zed", scope: "zed", active: true, tab: "zed", label: "Zed",
                 icon: "plugins", client_entry: "z.js", assets_base: "/plugins/zed" };

function capabilitiesFetch(plugins) {
  return async (url) => {
    const u = String(url);
    const body = u.startsWith("/api/capabilities")
      ? { plugins, core: {}, suggest_plugins: true } : {};
    return { ok: true, status: 200, json: async () => body, text: async () => "" };
  };
}

const tick = (ms) => new Promise((r) => setTimeout(r, ms));

test("the nav entry of a newly active plugin appears only after its view exists", async () => {
  const state = { plugins: [] };
  const { window: win } = loadApp({ fetchImpl: async (url) => capabilitiesFetch(state.plugins)(url) });
  await tick(200);
  runScript(win, `
    window.__loads = 0;
    _loadClientPlugin = (p) => new Promise((resolve) => {
      window.__loads++;
      setTimeout(() => {
        const v = document.createElement("section");
        v.id = "view-" + p.tab; v.className = "view";
        document.getElementById("app").appendChild(v);
        resolve();
      }, 150);
    });`);
  state.plugins = [PLUGIN];
  const pending = win.refreshPluginCommands();
  await tick(40);
  assert.equal(win.document.getElementById("nav-zed"), null,
    "the nav entry must not exist while the view's module is still loading");
  await pending;
  const nav = win.document.getElementById("nav-zed");
  assert.ok(nav, "the nav entry exists once the module has loaded");
  nav.click();
  assert.equal((win.document.querySelector(".view.active") || {}).id, "view-zed",
    "clicking the entry opens the plugin's view, not Chat");
  assert.equal(win.__loads, 1);
});

test("a plugin's client module is loaded once however often the list is refreshed", async () => {
  const { window: win } = loadApp({ fetchImpl: capabilitiesFetch([PLUGIN]) });
  await tick(300);
  const failures = [];
  win.console.error = (...a) => failures.push(String(a[0]));
  const other = { ...PLUGIN, name: "once", tab: "once", client_entry: "o.js", assets_base: "/plugins/once" };
  runScript(win, `window.__p = Promise.all([loadClientPlugins([${JSON.stringify(other)}]),
                                            loadClientPlugins([${JSON.stringify(other)}])]);`);
  await win.__p;
  runScript(win, `window.__p = loadClientPlugins([${JSON.stringify(other)}]);`);
  await win.__p;
  assert.equal(failures.filter((m) => m.includes("client plugin once")).length, 1,
    "the import was attempted exactly once");
});
