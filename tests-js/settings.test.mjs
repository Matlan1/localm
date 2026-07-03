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

async function drain(times = 6) {
  for (let i = 0; i < times; i++) await new Promise((r) => setTimeout(r, 0));
}

test("settings renders a typed control per schema field, split into sections", async () => {
  const patches = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(patches) });
  await render(win);
  const doc = win.document;

  // SELECT -> <select> with the field's options, current value selected.
  const modeSel = doc.querySelector('select[data-key="mode"]');
  assert.ok(modeSel, "mode renders as a <select>");
  const opts = [...modeSel.options].map((o) => o.value);
  assert.deepEqual(opts, ["privacy", "log", "full"], "select carries its options");
  assert.equal(modeSel.value, "log", "current value is selected");

  // NUMBER -> <input type=number> with min/step.
  const nctx = doc.querySelector('input[data-key="n_ctx"]');
  assert.equal(nctx.type, "number");
  assert.equal(nctx.min, "512");
  assert.equal(nctx.value, "4096");

  // SECRET -> <input type=password>, never prefilled.
  const secret = doc.querySelector('input[data-key="fake_secret"]');
  assert.equal(secret.type, "password");
  assert.equal(secret.value, "", "secret input is never prefilled");

  // LIST -> text input edited as a comma list.
  const list = doc.querySelector('input[data-key="net_allow"]');
  assert.equal(list.type, "text");
  assert.equal(list.value, "a.com, b.com");

  // HIDDEN -> not rendered.
  assert.equal(doc.querySelector('[data-key="plugins_enabled"]'), null,
    "hidden fields are not rendered");

  // The web-owned field is in its OWN plugin section, each section has a Save.
  const webInput = doc.querySelector('input[data-key="net_allow"]');
  const webSec = webInput.closest(".settings-section");
  assert.ok(webSec.querySelector(".settings-section-save"),
    "each section has its own Save button");
  assert.ok(/web/i.test(webSec.querySelector(".settings-section-head").textContent),
    "web plugin has its own section heading");
  // n_ctx (core Engine) is a DIFFERENT section from net_allow (web plugin).
  assert.notEqual(nctx.closest(".settings-section"), webSec,
    "core and plugin settings live in separate sections");

  // Left nav lists the 6 top-level GROUPS (not one link per section); the web
  // plugin's section lives INSIDE the Plugins group.
  const navLabels = [...doc.querySelectorAll("#settings-nav .settings-nav-link")]
    .map((l) => l.textContent);
  assert.deepEqual(navLabels,
    ["Model", "Server & network", "Security", "Plugins", "Privacy & data", "System"],
    "nav shows one link per top-level group");
  assert.equal(webSec.dataset.group, "plugins", "the web plugin section is in the Plugins group");
  // A group shows all its sections stacked; only ONE group is active at a time.
  const active = [...doc.querySelectorAll("#settings-content .settings-section.active")];
  assert.ok(active.length >= 1, "the default group shows its section(s)");
  const activeGroups = new Set(active.map((s) => s.dataset.group));
  assert.equal(activeGroups.size, 1, "exactly one group is shown at a time");
  assert.ok(activeGroups.has("model"), "Model is the default (first) group");
});

test("group nav shows all of a group's sections; require_auth + keys live in Security", async () => {
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch([]) });
  await render(win);
  await drain();
  const doc = win.document;

  // Click the Plugins group -> the web plugin section becomes visible.
  const links = [...doc.querySelectorAll("#settings-nav .settings-nav-link")];
  const plugins = links.find((l) => l.textContent === "Plugins");
  assert.ok(plugins, "Plugins group link exists");
  plugins.click();
  const webSec = doc.querySelector('input[data-key="net_allow"]').closest(".settings-section");
  assert.ok(webSec.classList.contains("active"), "clicking Plugins shows the web section");
  // The require_auth toggle (core Security) and the keys card share the Security group.
  const reqSec = doc.querySelector('input[data-key="require_auth"]').closest(".settings-section");
  assert.equal(reqSec.dataset.group, "security", "require_auth is in the Security group");
  assert.equal(doc.getElementById("keys-card").dataset.group, "security",
    "the keys workbench is in the Security group");
  // Jumping to the keys card (command palette) activates the Security group.
  win.gotoSettingsSection("keys-card");
  assert.ok(reqSec.classList.contains("active"), "gotoSettingsSection(keys-card) shows Security");
  assert.ok(!webSec.classList.contains("active"), "and hides the Plugins group");
});

test("each section saves only its own keys (per-section PATCH)", async () => {
  const patches = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(patches) });
  await render(win);
  const doc = win.document;

  // Edit n_ctx (core Engine) and save THAT section.
  const nctx = doc.querySelector('input[data-key="n_ctx"]');
  nctx.value = "8192";
  nctx.closest(".settings-section").querySelector(".settings-section-save").click();
  await drain();

  assert.equal(patches.length, 1, "one PATCH for the Engine section");
  const body = patches[0];
  assert.equal(typeof body.n_ctx, "number", "n_ctx sent as a number");
  assert.equal(body.n_ctx, 8192);
  assert.equal("net_allow" in body, false, "another section's key is NOT sent");
  assert.equal("fake_secret" in body, false, "untouched secret is omitted");

  // Now edit net_allow (web plugin) and save its section -> array round-trip.
  const list = doc.querySelector('input[data-key="net_allow"]');
  list.value = "x.com, y.com ,";
  list.closest(".settings-section").querySelector(".settings-section-save").click();
  await drain();

  assert.equal(patches.length, 2, "a second PATCH for the web section");
  assert.ok(Array.isArray(patches[1].net_allow), "net_allow sent as an array");
  assert.deepEqual(patches[1].net_allow, ["x.com", "y.com"], "trimmed, blanks dropped");
  assert.equal("n_ctx" in patches[1], false, "Engine key not resent from the web section");
});
