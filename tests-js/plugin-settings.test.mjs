// SPDX-License-Identifier: AGPL-3.0-or-later
// The generic plugin-contributed settings section: buildPluginSettingsSections
// renders whatever active plugin(s) called host.add_settings() (GET
// /v1/plugins/settings), and savePluginSettings POSTs one plugin's changed
// fields to /v1/plugins/<name>/settings. Unlike the tts/media sections (a
// fixed, known field list), the sections here are entirely server-driven -
// the test fixtures below stand in for whatever fields a real plugin
// registered.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

const SCHEMA = { fields: [
  { key: "host", widget: "text", label: "Host", help: "", group: "Server",
    owner: "core", default: "127.0.0.1" },
]};

const MEDIA = { plugins: [] };
const TTS = { plugin: "tts", active: false, fields: [] };

function pluginField(key, extra = {}) {
  return { key, widget: "text", label: key, help: "", value: "",
          is_override: false, default: "", admin_only: false, ...extra };
}

function pluginsPayload({ plugins = [] } = {}) {
  return { plugins };
}

function makeFetch({ plugins = pluginsPayload(), plugsFail = false, posts = [] } = {}) {
  return async (url, opts = {}) => {
    const method = opts.method || "GET";
    const json = (body) => ({ ok: true, status: 200, json: async () => body,
                              text: async () => "" });
    if (url === "/v1/config/schema") return json(SCHEMA);
    if (url === "/v1/media/config") return json(MEDIA);
    if (url === "/v1/comfy/status")
      return json({ alive: false, launched_by_localm: false });
    if (url === "/v1/tts/config") return json(TTS);
    if (url === "/v1/plugins/settings") {
      if (plugsFail) return { ok: false, status: 503, statusText: "Service Unavailable",
                              json: async () => ({}), text: async () => "" };
      return json(plugins);
    }
    if (url.startsWith("/v1/plugins/") && url.endsWith("/settings") && method === "POST") {
      const name = decodeURIComponent(url.slice("/v1/plugins/".length, -"/settings".length));
      const body = JSON.parse(opts.body);
      posts.push({ name, body });
      const sec = (plugins.plugins || []).find((s) => s.plugin === name) || { fields: [] };
      const nextFields = sec.fields.map((f) => (f.key in body
        ? { ...f, value: body[f.key], is_override: body[f.key] !== null && body[f.key] !== "" }
        : f));
      return json({ plugin: name, fields: nextFields });
    }
    return json({ models: [], active: "", conversations: [], plugins: [] });
  };
}

async function render(win) {
  runScript(win, "refreshSettingsPage();");
  for (let i = 0; i < 16; i++) await new Promise((r) => setTimeout(r, 0));
}

const section = (win, plugin) => win.document.getElementById("settings-sec-plugin-" + plugin);
const ctrl = (sec, key) => sec.querySelector(`[data-field-key="${key}"]`);

// ---- rendering ------------------------------------------------------------ //

test("no section at all when no plugin contributed settings", async () => {
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch() });
  await render(win);
  assert.equal(win.document.querySelector('[id^="settings-sec-plugin-"]'), null);
});

test("one section per plugin, each in the Plugins nav group", async () => {
  const payload = pluginsPayload({ plugins: [
    { plugin: "myplug", label: "My Plugin", fields: [
      pluginField("greeting", { label: "Greeting", value: "hi", default: "hi" }),
      pluginField("count", { widget: "number", label: "Count", value: 3, default: 3 }),
    ] },
    { plugin: "other", label: "Other Plugin", fields: [
      pluginField("mode", { widget: "select", label: "Mode", value: "fast",
                           options: ["fast", "accurate"], default: "fast" }),
    ] },
  ] });
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch({ plugins: payload }) });
  await render(win);

  const sec1 = section(win, "myplug");
  assert.ok(sec1, "myplug section rendered");
  assert.equal(sec1.dataset.group, "plugins");
  assert.ok(ctrl(sec1, "greeting"));
  assert.ok(ctrl(sec1, "count"));

  const sec2 = section(win, "other");
  assert.ok(sec2, "other section rendered");
  assert.ok(ctrl(sec2, "mode"));
});

test("a HIDDEN field never renders a control", async () => {
  const payload = pluginsPayload({ plugins: [
    { plugin: "myplug", label: "My Plugin", fields: [
      pluginField("visible", { value: "x", default: "x" }),
      pluginField("secret_state", { widget: "hidden", value: "y", default: "y" }),
    ] },
  ] });
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch({ plugins: payload }) });
  await render(win);
  const sec = section(win, "myplug");
  assert.ok(ctrl(sec, "visible"));
  assert.equal(ctrl(sec, "secret_state"), null);
});

test("a failed fetch SHOWS the failure instead of silently dropping the section", async () => {
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch({ plugsFail: true }) });
  await render(win);
  const fail = win.document.getElementById("settings-sec-plugin-settings");
  assert.ok(fail, "a failure must still render a section (binding rule 5)");
  assert.match(fail.textContent, /Could not load plugin-contributed settings/i);
});

// ---- saving ---------------------------------------------------------------- //

test("saving POSTs only the fields the user actually changed, to that plugin's own endpoint", async () => {
  const payload = pluginsPayload({ plugins: [
    { plugin: "myplug", label: "My Plugin", fields: [
      pluginField("greeting", { value: "hi", default: "hi" }),
      pluginField("count", { widget: "number", value: 3, default: 3, min: 0, max: 10 }),
    ] },
  ] });
  const posts = [];
  const { window: win } = loadAppWithPages({
    fetchImpl: makeFetch({ plugins: payload, posts }) });
  await render(win);
  const sec = section(win, "myplug");
  ctrl(sec, "greeting").querySelector("input").value = "hello there";
  sec.querySelector(".settings-section-save").click();
  for (let i = 0; i < 16; i++) await new Promise((r) => setTimeout(r, 0));
  assert.equal(posts.length, 1);
  assert.equal(posts[0].name, "myplug");
  assert.deepEqual(posts[0].body, { greeting: "hello there" },
    "the untouched count field must not be sent (it would pin an override)");
});

test("saving nothing does not POST", async () => {
  const payload = pluginsPayload({ plugins: [
    { plugin: "myplug", label: "My Plugin", fields: [
      pluginField("greeting", { value: "hi", default: "hi" }),
    ] },
  ] });
  const posts = [];
  const { window: win } = loadAppWithPages({
    fetchImpl: makeFetch({ plugins: payload, posts }) });
  await render(win);
  section(win, "myplug").querySelector(".settings-section-save").click();
  for (let i = 0; i < 8; i++) await new Promise((r) => setTimeout(r, 0));
  assert.equal(posts.length, 0);
});

test("two plugin sections save independently", async () => {
  const payload = pluginsPayload({ plugins: [
    { plugin: "alpha", label: "Alpha", fields: [pluginField("x", { value: "1", default: "1" })] },
    { plugin: "beta", label: "Beta", fields: [pluginField("y", { value: "2", default: "2" })] },
  ] });
  const posts = [];
  const { window: win } = loadAppWithPages({
    fetchImpl: makeFetch({ plugins: payload, posts }) });
  await render(win);
  ctrl(section(win, "alpha"), "x").querySelector("input").value = "changed";
  section(win, "alpha").querySelector(".settings-section-save").click();
  for (let i = 0; i < 16; i++) await new Promise((r) => setTimeout(r, 0));
  assert.deepEqual(posts, [{ name: "alpha", body: { x: "changed" } }],
    "only alpha's endpoint was called, and beta's field never appeared in it");
});

test("a select clears back to inherit as an empty string, like the tts section", async () => {
  const payload = pluginsPayload({ plugins: [
    { plugin: "myplug", label: "My Plugin", fields: [
      pluginField("mode", { widget: "select", value: "accurate", default: "fast",
                           options: ["fast", "accurate"], is_override: true }),
    ] },
  ] });
  const posts = [];
  const { window: win } = loadAppWithPages({
    fetchImpl: makeFetch({ plugins: payload, posts }) });
  await render(win);
  const sel = ctrl(section(win, "myplug"), "mode").querySelector("select");
  const inherit = [...sel.options].find((o) => o.value === "");
  assert.ok(inherit, "an overridden select must offer a way back to the default");
  sel.value = "";
  section(win, "myplug").querySelector(".settings-section-save").click();
  for (let i = 0; i < 16; i++) await new Promise((r) => setTimeout(r, 0));
  assert.deepEqual(posts, [{ name: "myplug", body: { mode: null } }]);
});
