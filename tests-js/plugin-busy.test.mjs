// SPDX-License-Identifier: AGPL-3.0-or-later
// The Plugins page says what it is doing while a request is in flight: a
// loading line while the lists load, and a disabled, relabelled button while an
// install / enable / disable / refresh / uninstall / remove runs.
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

const plugin = (over) => ({
  name: "image", label: "Images", builtin: true, installed: true,
  enabled: true, active: true, protected: false, description: "image gen",
  requires: [], missing_requires: [], ...over,
});

const ok = (body) => ({ ok: true, status: 200, json: async () => body, text: async () => "" });

/** A fetch stub whose /api/plugins answer and plugin POSTs wait for the test to
 *  release them, so the page can be inspected mid-request. */
function gatedFetch({ plugins, external = [] }) {
  const gates = { list: [], post: [] };
  const posts = [];
  const impl = async (url, opts = {}) => {
    if (url === "/api/plugins" && !opts.method) {
      await new Promise((r) => gates.list.push(r));
      return ok({ plugins: [...plugins, ...external], errors: {} });
    }
    if (typeof url === "string" && url.startsWith("/api/plugins/") && opts.method === "POST") {
      posts.push(url);
      await new Promise((r) => gates.post.push(r));
      return ok({ status: "done", name: "x", version: "1" });
    }
    return ok({ models: [], active: "", conversations: [], plugins: [], commands: [] });
  };
  return { impl, gates, posts };
}

const tick = async (n = 5) => { for (let i = 0; i < n; i++) await new Promise((r) => setTimeout(r, 0)); };
const rowButton = (win, label, rowText) => [...win.document.querySelectorAll("#catalog-table button")]
  .find((b) => b.textContent === label && b.closest("tr").textContent.includes(rowText));

async function renderedCatalog(plugins) {
  const g = gatedFetch({ plugins });
  const { window: win } = loadAppWithPages({ fetchImpl: g.impl });
  runScript(win, "_catalogStaggerMs = 0; renderCatalogPlugins();");
  await tick();
  g.gates.list.splice(0).forEach((r) => r());
  await tick(20);
  return { win, g };
}

test("the catalog shows a loading line until /api/plugins answers, then removes it", async () => {
  const g = gatedFetch({ plugins: [plugin()] });
  const { window: win } = loadAppWithPages({ fetchImpl: g.impl });
  runScript(win, "_catalogStaggerMs = 0; renderCatalogPlugins();");
  await tick();
  const box = win.document.getElementById("catalog-table");
  assert.match(box.textContent, /Loading plugins/);
  assert.equal(box.querySelectorAll("tr").length, 0);
  g.gates.list.splice(0).forEach((r) => r());
  await tick(20);
  assert.doesNotMatch(box.textContent, /Loading plugins…$/m);
  assert.ok(rowButton(win, "Disable", "Images"), "the rows replaced the loading line");
});

test("a failed catalog load replaces the loading line with the error", async () => {
  const fetchImpl = async (url) => {
    if (url === "/api/plugins") return { ok: false, status: 500, statusText: "Server Error",
                                         json: async () => ({}), text: async () => "" };
    return ok({ models: [], active: "", conversations: [], plugins: [], commands: [] });
  };
  const { window: win } = loadAppWithPages({ fetchImpl });
  runScript(win, "_catalogStaggerMs = 0; renderCatalogPlugins();");
  await tick(20);
  const text = win.document.getElementById("catalog-table").textContent;
  assert.match(text, /Could not load plugins/);
  assert.doesNotMatch(text, /Loading plugins/);
});

for (const [label, workingLabel, cls] of [
  ["Disable", "Disabling…"],
  ["Refresh", "Refreshing…"],
]) {
  test(`${label} on an installed plugin disables the button and shows "${workingLabel}" while the request runs`, async () => {
    const { win, g } = await renderedCatalog([plugin()]);
    const btn = rowButton(win, label, "Images");
    assert.ok(btn);
    btn.click();
    await tick();
    assert.equal(g.posts.length, 1);
    assert.equal(btn.disabled, true);
    assert.equal(btn.textContent, workingLabel);
    g.gates.post.splice(0).forEach((r) => r());
    await tick();
    g.gates.list.splice(0).forEach((r) => r());
    await tick(20);
    assert.equal(win.document.getElementById("toast").textContent, "image: done");
  });
}

test("Install on an available plugin shows the working label, and a failed request restores the button", async () => {
  const g = gatedFetch({ plugins: [plugin({ name: "voice", label: "Voice", installed: false,
                                            enabled: false, active: false })] });
  const orig = g.impl;
  g.impl = async (url, opts = {}) => {
    if (typeof url === "string" && url.endsWith("/install") && opts.method === "POST") {
      g.posts.push(url);
      await new Promise((r) => g.gates.post.push(r));
      return { ok: false, status: 500, json: async () => ({ detail: "disk full" }), text: async () => "" };
    }
    return orig(url, opts);
  };
  const { window: win } = loadAppWithPages({ fetchImpl: g.impl });
  runScript(win, "_catalogStaggerMs = 0; renderCatalogPlugins();");
  await tick();
  g.gates.list.splice(0).forEach((r) => r());
  await tick(20);
  const btn = rowButton(win, "Install", "Voice");
  btn.click();
  await tick();
  assert.equal(btn.disabled, true);
  assert.equal(btn.textContent, "Installing…");
  g.gates.post.splice(0).forEach((r) => r());
  await tick(10);
  assert.equal(btn.disabled, false);
  assert.equal(btn.textContent, "Install");
  assert.match(win.document.getElementById("toast").textContent, /disk full/);
});

test("Install requirements counts through the missing plugins on its button", async () => {
  const needy = plugin({ name: "video", label: "Video", requires: ["image", "voice"],
                         missing_requires: ["image", "voice"] });
  const { win, g } = await renderedCatalog([needy]);
  const btn = rowButton(win, "Install requirements", "Video");
  btn.click();
  await tick();
  assert.equal(btn.disabled, true);
  assert.equal(btn.textContent, "Installing image (1/2)…");
  g.gates.post.splice(0).forEach((r) => r());
  await tick(10);
  assert.equal(btn.textContent, "Installing voice (2/2)…");
  g.gates.post.splice(0).forEach((r) => r());
  await tick(10);
  assert.deepEqual(g.posts, ["/api/plugins/image/install", "/api/plugins/voice/install"]);
});

test("the external plugins list shows a loading line, and Install / remove show working labels", async () => {
  const ext = plugin({ name: "mine", label: "Mine", builtin: false, version: "1.0",
                       description: "custom", tool_exports: [] });
  const g = gatedFetch({ plugins: [], external: [ext] });
  const { window: win } = loadAppWithPages({ fetchImpl: g.impl });
  runScript(win, "refreshPluginsPage();");
  await tick();
  const box = win.document.getElementById("plugins-table");
  assert.match(box.textContent, /Loading plugins/);
  g.gates.list.splice(0).forEach((r) => r());
  await tick(20);
  assert.doesNotMatch(box.textContent, /Loading plugins/);

  const rm = [...box.querySelectorAll("button")].find((b) => b.textContent === "remove");
  assert.ok(rm);
  rm.click();
  await tick();
  const confirm = [...win.document.querySelectorAll("button")].find((b) => b.textContent === "Remove");
  assert.ok(confirm, "the confirm dialog opened");
  confirm.click();
  await tick();
  assert.equal(rm.disabled, true);
  assert.equal(rm.textContent, "Removing…");
  g.gates.post.splice(0).forEach((r) => r());
  await tick(10);
  g.gates.list.splice(0).forEach((r) => r());
  await tick(10);

  win.document.getElementById("plugin-source").value = "some/folder";
  const install = win.document.getElementById("plugin-install");
  install.click();
  await tick();
  assert.equal(install.disabled, true);
  assert.equal(install.textContent, "Installing…");
  const label = install.textContent;
  g.gates.post.splice(0).forEach((r) => r());
  await tick(10);
  assert.equal(install.disabled, false);
  assert.notEqual(install.textContent, label);
});
