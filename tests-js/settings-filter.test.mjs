// SPDX-License-Identifier: AGPL-3.0-or-later
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

// The Settings page filter box: one search across EVERY group at once. The form
// is schema-driven (~80 fields behind a 7-group nav), so a setting you cannot
// name the group of used to mean clicking through all seven. These tests drive
// the REAL input element in index.html (value + "input" event), not an internal
// helper, so the wiring is covered too.

// Fields spread over four top-level groups. "context window" deliberately
// appears in BOTH a core Engine field (Model group) and a plugin field (Plugins
// group) so the cross-group behaviour is a real assertion, not a claim.
const SCHEMA = {
  fields: [
    { key: "n_ctx", widget: "number", label: "Context window",
      help: "How many tokens the model can attend to at once.",
      group: "Engine", owner: "core", min: 512, step: 512, default: 4096 },
    // A second Engine field, so "hides the non-matching fields of a section it
    // DID match" is a real assertion and not two separate sections.
    { key: "n_gpu_layers", widget: "number", label: "GPU layers",
      help: "How much of the model to offload.",
      group: "Engine", owner: "core", min: 0, step: 1, default: 999 },
    { key: "temperature", widget: "number", label: "Temperature",
      help: "Higher is more random.", group: "Sampling", owner: "core",
      min: 0, step: 0.05, default: 0.8 },
    { key: "host", widget: "text", label: "Bind address",
      help: "Which interface the server listens on.",
      group: "Server", owner: "core", default: "127.0.0.1" },
    { key: "require_auth", widget: "toggle", label: "Require an API key",
      help: "", group: "Security", owner: "core", default: false },
    { key: "net_allow", widget: "list", label: "Allowed domains",
      help: "Domains the web plugin may fetch.",
      group: "Network", owner: "web", default: ["a.com"] },
    { key: "coder_n_ctx", widget: "number", label: "Coder context window",
      help: "", group: "Coder", owner: "coder", default: 8192 },
    { key: "plugins_enabled", widget: "hidden", label: "Enabled plugins",
      help: "", group: "Plugins", owner: "core", default: [] },
  ],
};

const MEDIA = {
  plugins: [
    { plugin: "image", label: "Image", fields: [
      { key: "fast_dequant", widget: "toggle", label: "Fast GGUF dequant",
        help: "", value: true, is_override: false },
    ] },
  ],
};

// A keys envelope with a preset, so the owner-only Keys & devices card is shown
// and its preset buttons (a custom row with no schema widget) are in the DOM.
const KEYS = { keys: [], is_owner: true, presets: [{ name: "Phone", scopes: ["chat"] }] };

// PATCH bodies from the current test (reset by setup()).
let PATCHES = [];

function makeFetch() {
  return async (url, opts = {}) => {
    const method = opts.method || "GET";
    if (url === "/v1/config/schema") {
      return { ok: true, status: 200, json: async () => SCHEMA, text: async () => "" };
    }
    if (url === "/v1/config" && method === "PATCH") {
      PATCHES.push(JSON.parse(opts.body));
      return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
    }
    if (url === "/v1/media/config" && method === "GET") {
      return { ok: true, status: 200, json: async () => MEDIA, text: async () => "" };
    }
    if (url === "/v1/keys" && method === "GET") {
      return { ok: true, status: 200, json: async () => KEYS, text: async () => "" };
    }
    if (url === "/api/comfy/managed-status") {
      return { ok: true, status: 200, json: async () => ({ installed: false, state: "absent" }),
               text: async () => "" };
    }
    return {
      ok: true, status: 200, text: async () => "",
      json: async () => ({ models: [], active: "", conversations: [], plugins: [] }),
    };
  };
}

async function drain(times = 8) {
  for (let i = 0; i < times; i++) await new Promise((r) => setTimeout(r, 0));
}

/** Load the app, render the settings form, and return the window + helpers. */
async function setup() {
  PATCHES = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch() });
  await drain();
  runScript(win, "refreshSettingsPage();");
  await drain();
  const doc = win.document;
  const box = doc.getElementById("settings-filter");
  assert.ok(box, "the Settings page has a filter box");
  const type = (q) => {
    box.value = q;
    box.dispatchEvent(new win.Event("input", { bubbles: true }));
  };
  const activeSections = () =>
    [...doc.querySelectorAll("#settings-content .settings-section.active")];
  const activeIds = () => activeSections().map((s) => s.id);
  const unit = (key) => doc.querySelector(`[data-field-key="${key}"]`);
  const shown = (key) => {
    const u = unit(key);
    return !!u && !u.classList.contains("filter-hidden")
      && !!u.closest(".settings-section").classList.contains("active");
  };
  return { win, doc, box, type, activeSections, activeIds, unit, shown };
}

test("filter matches a field by LABEL and shows it across groups at once", async () => {
  const { type, activeSections, shown, unit } = await setup();

  type("context window");

  // Both the core Engine field (Model group) and the coder plugin field
  // (Plugins group) match, and BOTH are shown together - the whole point.
  assert.ok(shown("n_ctx"), "core Context window matches");
  assert.ok(shown("coder_n_ctx"), "the coder plugin's context window matches too");
  const groups = new Set(activeSections().map((s) => s.dataset.group));
  assert.ok(groups.size >= 2,
    `results span groups, got ${[...groups].join(",")}`);
  assert.ok(groups.has("model") && groups.has("plugins"),
    "the Model and Plugins groups are both shown");
  // A non-matching field of a section that DID match is hidden, not left there
  // beside the hit (n_gpu_layers shares the Engine section with n_ctx).
  const sibling = unit("n_gpu_layers");
  assert.ok(sibling.closest(".settings-section").classList.contains("active"),
    "its section is shown");
  assert.ok(sibling.classList.contains("filter-hidden"),
    "but the non-matching field inside it is hidden");
  assert.equal(shown("host"), false, "a non-matching other-group field is hidden");
});

test("a filtered field still saves its own section", async () => {
  const { type, unit } = await setup();

  type("context window");
  const nctx = unit("n_ctx").querySelector("input");
  nctx.value = "8192";
  const sec = unit("n_ctx").closest(".settings-section");
  sec.querySelector(".settings-section-save").click();
  await drain();

  assert.equal(PATCHES.length, 1, "the section saved while filtered");
  assert.equal(PATCHES[0].n_ctx, 8192, "the edited value was sent");
  // The filter hides fields, it never removes them, so the section's other
  // values are still read back rather than silently dropped from the PATCH.
  assert.equal(PATCHES[0].n_gpu_layers, 999,
    "a filter-hidden field in the same section is still saved, not dropped");
});

test("filter matches by config KEY and by HELP text", async () => {
  const { type, shown } = await setup();

  // KEY: "n_ctx" appears in no label or help text, only the key itself.
  type("n_ctx");
  assert.ok(shown("n_ctx"), "matches on the raw config key");
  assert.equal(shown("temperature"), false, "and only that field");

  // HELP: "interface" appears only in host's help text.
  type("interface");
  assert.ok(shown("host"), "matches on help text");
  assert.equal(shown("n_ctx"), false, "and only that field");

  // Case-insensitive, and multi-term is AND across the field's text.
  type("ALLOWED DOMAINS");
  assert.ok(shown("net_allow"), "matching is case-insensitive");
  type("domains fetch");
  assert.ok(shown("net_allow"), "all terms may come from label + help together");
  type("domains temperature");
  assert.equal(shown("net_allow"), false, "terms are ANDed, not ORed");
});

test("custom rows with no schema widget are findable by their row text", async () => {
  const { doc, type, activeIds } = await setup();

  // The logo picker (Appearance card) has no schema field behind it.
  type("logo style");
  assert.ok(activeIds().includes("sec-appearance"),
    "the logo picker row surfaces its card");

  // A key preset button (rendered by buildKeyPresets, no schema widget).
  assert.ok(!doc.getElementById("keys-card").classList.contains("sec-hidden"),
    "the owner keys card is shown for this fixture");
  type("phone");
  assert.ok(activeIds().includes("keys-card"), "a key preset name is findable");

  // A static card's own controls (Server controls card).
  type("shut down server");
  assert.ok(activeIds().includes("sec-server"), "static card buttons are findable");
});

test("a control that is not offered is not findable, and becomes findable when it is", async () => {
  const { doc, type, activeIds } = await setup();

  // Main GPU / Split across GPUs live in rows that stay [hidden] until more
  // than one GPU is detected. A search must not surface a control the user
  // cannot use (and would then hunt for in a card that shows nothing).
  const splitRow = doc.getElementById("perf-gpu-split-row");
  assert.ok(splitRow.hidden, "the split row starts hidden (single GPU)");
  type("split across gpus");
  assert.equal(activeIds().includes("sec-performance"), false,
    "a hidden row does not match");
  assert.ok(doc.getElementById("settings-filter-empty").hidden === false,
    "and the no-match note is shown");

  // Reveal it the way settings-perf.js does on a multi-GPU box.
  splitRow.hidden = false;
  type("split across gpus");
  assert.ok(activeIds().includes("sec-performance"),
    "once offered, the row is findable");
});

test("gate-hidden sub-content never matches, but the card's always-shown content still does", async () => {
  const { doc, type, activeIds } = await setup();

  // The app-update sub-block is [hidden] unless the updater proxy is
  // configured; the "Updates" card ITSELF is never gated - the runtime-update
  // block and the update-behavior toggles it also holds must stay findable
  // regardless (see index.html's #sec-updates comment; S1 merged what used to
  // be three separate panels into this one card).
  assert.equal(doc.getElementById("sec-updates").hidden, false,
    "the Updates card itself is never gated");
  assert.ok(doc.getElementById("app-update-block").hidden,
    "the app-update sub-block starts gated off");

  type("check for a newer localm build");
  assert.equal(activeIds().includes("sec-updates"), false,
    "text scoped to the gated-off app-update sub-block does not match");

  type("check for a newer llama.cpp runtime build");
  assert.ok(activeIds().includes("sec-updates"),
    "the always-shown runtime-update text still matches the same card");

  // Same for a .sec-hidden card (the non-owner keys card).
  doc.getElementById("keys-card").classList.add("sec-hidden");
  type("phone");
  assert.equal(activeIds().includes("keys-card"), false,
    "a .sec-hidden card does not match");
});

test("no match shows a note naming the query, and nothing else", async () => {
  const { doc, type, activeSections } = await setup();

  type("zzz-no-such-setting");
  const note = doc.getElementById("settings-filter-empty");
  assert.equal(note.hidden, false, "the no-match note is shown");
  assert.ok(note.textContent.includes("zzz-no-such-setting"),
    "the note names what was searched");
  assert.equal(activeSections().length, 0, "no section is shown");
});

test("clearing the query restores the normal grouped view", async () => {
  const { doc, box, type, activeSections } = await setup();

  type("context");
  assert.ok(activeSections().length >= 1, "filtering shows matches");

  type("");
  const groups = new Set(activeSections().map((s) => s.dataset.group));
  assert.equal(groups.size, 1, "exactly one group is shown again");
  assert.equal(doc.querySelectorAll("#settings-content .filter-hidden").length, 0,
    "no field is left hidden");
  assert.equal(doc.getElementById("settings-filter-empty").hidden, true,
    "the no-match note is hidden");
  const activeLinks = [...doc.querySelectorAll("#settings-nav .settings-nav-link.active")];
  assert.equal(activeLinks.length, 1, "the group nav has its selection back");
  assert.equal(box.value, "", "the box is empty");
});

test("picking a group, or a deep link, leaves the filter", async () => {
  const { win, doc, box, type, activeSections } = await setup();

  type("context");
  assert.equal(
    [...doc.querySelectorAll("#settings-nav .settings-nav-link.active")].length, 0,
    "no single group is selected while filtering");

  const security = [...doc.querySelectorAll("#settings-nav .settings-nav-link")]
    .find((l) => l.textContent.includes("Security"));
  security.click();
  assert.equal(box.value, "", "clicking a group clears the query");
  const groups = new Set(activeSections().map((s) => s.dataset.group));
  assert.deepEqual([...groups], ["security"], "and shows that group only");

  // The command palette's deep link does the same.
  type("context");
  win.gotoSettingsSection("keys-card");
  assert.equal(box.value, "", "a deep link clears the query");
  assert.ok(doc.getElementById("keys-card").classList.contains("active"),
    "and lands on the target section");
});

test("Escape in the box clears the filter", async () => {
  const { win, doc, box, type, activeSections } = await setup();

  type("context");
  box.dispatchEvent(new win.KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
  assert.equal(box.value, "", "Escape empties the box");
  const groups = new Set(activeSections().map((s) => s.dataset.group));
  assert.equal(groups.size, 1, "and the grouped view is back");
  assert.equal(doc.querySelectorAll("#settings-content .filter-hidden").length, 0,
    "with nothing left hidden");
});

test("an async rebuild does not drop the filtered view", async () => {
  const { win, type, shown, activeSections } = await setup();

  type("context window");
  const before = activeSections().length;
  // The owner-only keys panel resolving rebuilds the nav mid-filter; so does a
  // section save re-render. Neither may bounce the user back to one group.
  runScript(win, "buildSettingsNav();");
  assert.ok(shown("n_ctx"), "the matched field is still shown");
  assert.equal(activeSections().length, before, "the same sections are still shown");
});

test("media per-plugin fields are filterable too", async () => {
  const { win, doc, type, activeIds } = await setup();
  // Media subsection fields need host FS access for the folder controls; this
  // fixture's field is a toggle, but pin it anyway to mirror the real owner GUI.
  runScript(win, `caps.fsAccess = "host"; refreshSettingsPage();`);
  await drain();

  type("dequant");
  assert.ok(activeIds().includes("settings-sec-media"),
    "a per-plugin media field surfaces the Media section");
  assert.ok(doc.querySelector('[data-field-key="fast_dequant"]'),
    "the media control itself is rendered");
  assert.equal(
    doc.querySelector('[data-field-key="fast_dequant"]').classList.contains("filter-hidden"),
    false, "and is not hidden by the filter");
});
