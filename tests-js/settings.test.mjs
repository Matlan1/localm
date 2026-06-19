// SPDX-License-Identifier: AGPL-3.0-or-later
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

// A canned /v1/config/schema response covering every control type the form
// must render. Mirrors the shape settings_schema.schema_json() emits: a flat
// field list, each non-secret field carrying its current value as `default`.
const SCHEMA = {
  fields: [
    { key: "mode", widget: "select", label: "Session persistence",
      help: "how much is saved", group: "Privacy", owner: "core",
      options: ["privacy", "log", "full"], default: "log" },
    { key: "n_ctx", widget: "number", label: "Context window", help: "",
      group: "Engine", owner: "core", min: 512, step: 512, default: 4096 },
    { key: "require_auth", widget: "toggle", label: "Require an API key",
      help: "", group: "Security", owner: "core", default: false },
    { key: "net_allow", widget: "list", label: "Allowed domains", help: "",
      group: "Network", owner: "web", default: ["a.com", "b.com"] },
    { key: "fake_secret", widget: "secret", label: "Fake secret", help: "",
      group: "Security", owner: "core", secret: true },   // NO default
    { key: "plugins_enabled", widget: "hidden", label: "Enabled plugins",
      help: "", group: "Plugins", owner: "core", default: [] },
  ],
};

/** Build a fetch stub that serves the schema and records PATCH calls. The
 *  default branch returns a model-list shape so app.js's init block
 *  (refreshModels -> populateSetupModels) does not throw while our awaits drain. */
function makeFetch(patches) {
  return async (url, opts = {}) => {
    if (url === "/v1/config/schema") {
      return { ok: true, status: 200, json: async () => SCHEMA, text: async () => "" };
    }
    if (url === "/v1/config" && (opts.method || "GET") === "PATCH") {
      patches.push(JSON.parse(opts.body));
      return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
    }
    return {
      ok: true, status: 200, text: async () => "",
      json: async () => ({ models: [], active: "", conversations: [], plugins: [] }),
    };
  };
}

// refreshSettingsPage is async; run it and let the awaited fetch microtasks
// drain before asserting on the rendered DOM.
async function render(win) {
  runScript(win, "refreshSettingsPage();");
  await new Promise((r) => setTimeout(r, 0));
}

test("settings form renders a typed control per schema field", async () => {
  const patches = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(patches) });
  await render(win);
  const form = win.document.getElementById("config-form");

  // SELECT -> <select> with the field's options, current value selected.
  const modeSel = form.querySelector('select[data-key="mode"]');
  assert.ok(modeSel, "mode renders as a <select>");
  const opts = [...modeSel.options].map((o) => o.value);
  assert.deepEqual(opts, ["privacy", "log", "full"], "select carries its options");
  assert.equal(modeSel.value, "log", "current value is selected");

  // NUMBER -> <input type=number> with min/step.
  const nctx = form.querySelector('input[data-key="n_ctx"]');
  assert.equal(nctx.type, "number");
  assert.equal(nctx.min, "512");
  assert.equal(nctx.value, "4096");

  // SECRET -> <input type=password>, never prefilled.
  const secret = form.querySelector('input[data-key="fake_secret"]');
  assert.equal(secret.type, "password");
  assert.equal(secret.value, "", "secret input is never prefilled");

  // LIST -> text input edited as a comma list.
  const list = form.querySelector('input[data-key="net_allow"]');
  assert.equal(list.type, "text");
  assert.equal(list.value, "a.com, b.com");

  // HIDDEN -> not rendered.
  assert.equal(form.querySelector('[data-key="plugins_enabled"]'), null,
    "hidden fields are not rendered");

  // Plugin-owned (owner != core) field grouped under a per-plugin heading.
  const heads = [...form.querySelectorAll(".settings-group-head")].map((h) => h.textContent);
  assert.ok(heads.some((h) => h.includes("web")), "web-owned section has a plugin heading");
});

test("save PATCHes native types (number for n_ctx, array for a LIST key)", async () => {
  const patches = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(patches) });
  await render(win);
  const form = win.document.getElementById("config-form");

  // User edits n_ctx and net_allow.
  form.querySelector('input[data-key="n_ctx"]').value = "8192";
  form.querySelector('input[data-key="net_allow"]').value = "x.com, y.com ,";

  win.document.getElementById("config-save").click();
  await new Promise((r) => setTimeout(r, 0));

  assert.equal(patches.length, 1, "exactly one PATCH sent");
  const body = patches[0];
  assert.equal(typeof body.n_ctx, "number", "n_ctx sent as a number");
  assert.equal(body.n_ctx, 8192);
  assert.ok(Array.isArray(body.net_allow), "net_allow sent as an array");
  assert.deepEqual(body.net_allow, ["x.com", "y.com"], "trimmed, blanks dropped");
  // The untouched secret must NOT be sent (no real value to send).
  assert.equal("fake_secret" in body, false, "untouched secret is omitted");
});
