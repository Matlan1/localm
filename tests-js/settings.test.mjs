// SPDX-License-Identifier: AGPL-3.0-or-later
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

// A canned /v1/config/schema response covering every control type the form
// renders, in the shape settings_schema.schema_json() emits: a flat field list,
// each non-secret field carrying its current value as `default`.
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
      group: "Security", owner: "core", secret: true },   // no default
    { key: "plugins_enabled", widget: "hidden", label: "Enabled plugins",
      help: "", group: "Plugins", owner: "core", default: [] },
    { key: "chat_system_prompt", widget: "textarea", label: "Default system prompt",
      help: "Empty = no default system prompt.", group: "Chat", owner: "chat",
      default: "You are a helpful assistant." },
  ],
};

/** Build a fetch stub that serves the schema and records PATCH calls. The
 *  default branch returns a model-list shape so app.js's init block
 *  (refreshModels -> populateSetupModels) does not throw while the awaits drain. */
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

  // Placement is by `group`, not by `owner`: net_allow is owner="web" but
  // group="Network", so it files under Server & network > Outbound access.
  const webInput = doc.querySelector('input[data-key="net_allow"]');
  const webSec = webInput.closest(".settings-section");
  assert.ok(webSec.querySelector(".settings-section-save"),
    "each section has its own Save button");
  assert.match(webSec.querySelector(".settings-section-head").textContent,
    /outbound/i, "a Network-group field is headed by its GROUP, not its owner");
  assert.equal(webSec.dataset.group, "server",
    "group=Network routes to Server & network, not to Plugins");
  // n_ctx (Engine) is still a DIFFERENT section from net_allow (Network).
  assert.notEqual(nctx.closest(".settings-section"), webSec,
    "different groups are different sections");

  // Same rule for chat: owner="chat" but group="Chat", so the generation
  // defaults sit under Model.
  const chatSec = doc.querySelector('[data-key="chat_system_prompt"]')
    .closest(".settings-section");
  assert.equal(chatSec.dataset.group, "model",
    "chat generation defaults belong to Model, not Plugins");
  assert.notEqual(chatSec, nctx.closest(".settings-section"),
    "Generation defaults is its own section within Model");

  // Nav lists one link per top-level group that has a section. This schema has
  // nothing owned by a plugin-nav group, so Plugins is absent.
  const navLabels = [...doc.querySelectorAll("#settings-nav .settings-nav-link")]
    .map((l) => l.textContent);
  assert.ok(navLabels.includes("Server & network"), "Server & network is listed");
  assert.ok(navLabels.includes("Privacy & data"), "Privacy & data is listed");
  assert.equal(new Set(navLabels).size, navLabels.length, "no duplicate nav links");
  // A group shows all its sections stacked; only one group is active at a time.
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

  // Click Server & network -> the Outbound access section (net_allow) becomes
  // visible.
  const links = [...doc.querySelectorAll("#settings-nav .settings-nav-link")];
  const server = links.find((l) => l.textContent === "Server & network");
  assert.ok(server, "Server & network group link exists");
  server.click();
  const webSec = doc.querySelector('input[data-key="net_allow"]').closest(".settings-section");
  assert.ok(webSec.classList.contains("active"),
    "clicking Server & network shows the Outbound access section");
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

test("pathlist renders a folder-row editor and reads back an array", async () => {
  // The owner-only rag_allowed_roots field renders a stack of folder rows, each
  // with a Browse and a remove button, rather than the flat comma list. It needs
  // host FS access, granted below before rendering.
  const PATHLIST_SCHEMA = { fields: [
    { key: "rag_allowed_roots", widget: "pathlist",
      label: "Folders allowed for indexing", help: "extra folders",
      group: "Knowledge", owner: "rag", admin_only: true,
      default: ["F:\\docs", "F:\\more"] },
  ]};
  const patches = [];
  const fetchImpl = async (url, opts = {}) => {
    if (url === "/v1/config/schema")
      return { ok: true, status: 200, json: async () => PATHLIST_SCHEMA, text: async () => "" };
    if (url === "/v1/config" && (opts.method || "GET") === "PATCH") {
      patches.push(JSON.parse(opts.body));
      return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
    }
    // Everything else, including /api/capabilities, reports host FS, so
    // caps.fsAccess resolves to "host" and the folder editor renders.
    return { ok: true, status: 200, text: async () => "",
             json: async () => ({ models: [], active: "", conversations: [],
                                  plugins: [], fs_access: "host" }) };
  };
  const { window: win } = loadAppWithPages({ fetchImpl });
  await drain();          // let init's /api/capabilities set caps.fsAccess = host
  await render(win);
  await drain();
  const doc = win.document;

  const rows = [...doc.querySelectorAll(".pathlist-row input")];
  assert.equal(rows.length, 2, "one row per saved folder");
  assert.deepEqual(rows.map((i) => i.value), ["F:\\docs", "F:\\more"]);
  const firstRow = doc.querySelector(".pathlist-row");
  assert.ok(firstRow.querySelector(".dir-picker-btn"), "each row has a Browse button");
  assert.ok(firstRow.querySelector(".pathlist-rm"), "each row has a remove button");

  // Remove the first folder, then save -> the PATCH carries only what remains.
  firstRow.querySelector(".pathlist-rm").click();
  const save = doc.querySelector(".settings-section .settings-section-save");
  save.click();
  await drain();
  assert.equal(patches.length, 1, "one PATCH for the Knowledge section");
  assert.ok(Array.isArray(patches[0].rag_allowed_roots), "sent as an array");
  assert.deepEqual(patches[0].rag_allowed_roots, ["F:\\more"], "removed folder dropped");
});

test("pathlist is hidden without host filesystem access", async () => {
  // A non-host caller gets no folder editor.
  const PATHLIST_SCHEMA = { fields: [
    { key: "rag_allowed_roots", widget: "pathlist", label: "Folders", help: "h",
      group: "Knowledge", owner: "rag", admin_only: true, default: ["F:\\docs"] },
  ]};
  const fetchImpl = async (url) => {
    if (url === "/v1/config/schema")
      return { ok: true, status: 200, json: async () => PATHLIST_SCHEMA, text: async () => "" };
    return { ok: true, status: 200, text: async () => "",
             json: async () => ({ models: [], active: "", conversations: [], plugins: [] }) };
  };
  const { window: win } = loadAppWithPages({ fetchImpl });
  // caps.fsAccess defaults to "none" - do not grant host access.
  await render(win);
  assert.equal(win.document.querySelector(".pathlist-row"), null,
    "no folder rows render without host FS access");
});

test("indexing mode marks the active list; both stay visible", async () => {
  const SCHEMA = { fields: [
    { key: "rag_indexing_mode", widget: "select", label: "Indexing folder rule",
      help: "h", group: "Knowledge", owner: "rag", admin_only: true,
      options: ["whitelist", "blacklist"], default: "whitelist" },
    { key: "rag_allowed_roots", widget: "pathlist", label: "Allowed folders",
      help: "h", group: "Knowledge", owner: "rag", admin_only: true, default: [] },
    { key: "rag_denied_roots", widget: "pathlist", label: "Denied folders",
      help: "h", group: "Knowledge", owner: "rag", admin_only: true, default: [] },
  ]};
  const fetchImpl = async (url) => {
    if (url === "/v1/config/schema")
      return { ok: true, status: 200, json: async () => SCHEMA, text: async () => "" };
    return { ok: true, status: 200, text: async () => "",
             json: async () => ({ models: [], active: "", conversations: [],
                                  plugins: [], fs_access: "host" }) };
  };
  const { window: win } = loadAppWithPages({ fetchImpl });
  await drain();
  await render(win);
  await drain();
  const doc = win.document;
  const allow = doc.querySelector('[data-field-key="rag_allowed_roots"]');
  const deny = doc.querySelector('[data-field-key="rag_denied_roots"]');
  const sel = doc.querySelector('select[data-key="rag_indexing_mode"]');
  assert.ok(allow && deny && sel, "all three RAG controls render for the owner");
  // Both lists stay visible and editable in every mode.
  assert.notEqual(allow.style.display, "none", "Allowed visible");
  assert.notEqual(deny.style.display, "none", "Denied visible");
  // whitelist (default): the "in use" tag is on Allowed, not Denied.
  assert.ok(allow.querySelector(".rag-inuse"), "Allowed marked in use (whitelist)");
  assert.equal(deny.querySelector(".rag-inuse"), null, "Denied not marked");
  // flip to blacklist -> the tag moves to Denied; both still visible.
  sel.value = "blacklist";
  sel.dispatchEvent(new win.Event("change", { bubbles: true }));
  assert.equal(allow.querySelector(".rag-inuse"), null, "Allowed no longer marked");
  assert.ok(deny.querySelector(".rag-inuse"), "Denied marked in use (blacklist)");
  assert.notEqual(allow.style.display, "none", "Allowed still visible");
  assert.notEqual(deny.style.display, "none", "Denied still visible");
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

test("blanking the default system prompt textarea saves an empty string, not null", async () => {
  const patches = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(patches) });
  await render(win);
  const doc = win.document;

  const ta = doc.querySelector('textarea[data-key="chat_system_prompt"]');
  assert.ok(ta, "chat_system_prompt renders as a textarea");
  assert.equal(ta.value, "You are a helpful assistant.");

  ta.value = "";
  ta.closest(".settings-section").querySelector(".settings-section-save").click();
  await drain();

  assert.equal(patches.length, 1, "blanking it still triggers a PATCH");
  assert.equal(patches[0].chat_system_prompt, "",
    "blank saves as an empty string, not null - null would 400 on this widget");
});
