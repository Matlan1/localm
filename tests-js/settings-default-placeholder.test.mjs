// SPDX-License-Identifier: AGPL-3.0-or-later
// NEW-DEFAULT-VALUE-PLACEHOLDER: schema-driven Settings fields used to write
// every field's shipped default straight in as a solid value, indistinguishable
// from something the user actually typed. buildSettingControl now compares the
// current value against a separate `shipped_default` (from
// tests/test_settings_schema.py::TestShippedDefaultAnnotation on the server
// side) and renders NUMBER/TEXT fields blank+placeholder and SELECT fields
// dimmed, mirroring the chat-drawer/image-gen precedent - while a REAL
// override still renders solid, and clearing a real override still sends an
// explicit null (the regression this whole fix is careful not to introduce).
// Same harness/fetch-mock style as settings-save-dirty.test.mjs.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

const SCHEMA_FRESH = {
  fields: [
    { key: "n_ctx", widget: "number", label: "Context window", help: "",
      group: "Engine", owner: "core", min: 512, step: 512,
      default: 4096, shipped_default: 4096 },
    { key: "net_mode", widget: "select", label: "Network access", help: "",
      group: "Server", owner: "core", options: ["off", "ask", "on"],
      default: "ask", shipped_default: "ask" },
    { key: "mdns_name", widget: "text", label: "Network name (mDNS)", help: "",
      group: "Server", owner: "core", default: "localm", shipped_default: "localm" },
  ],
};

// Same three fields, each carrying a REAL user override (current != shipped).
const SCHEMA_OVERRIDDEN = {
  fields: [
    { key: "n_ctx", widget: "number", label: "Context window", help: "",
      group: "Engine", owner: "core", min: 512, step: 512,
      default: 8192, shipped_default: 4096 },
    { key: "net_mode", widget: "select", label: "Network access", help: "",
      group: "Server", owner: "core", options: ["off", "ask", "on"],
      default: "off", shipped_default: "ask" },
    { key: "mdns_name", widget: "text", label: "Network name (mDNS)", help: "",
      group: "Server", owner: "core", default: "my-box", shipped_default: "localm" },
  ],
};

function makeFetch(schema, patches) {
  return async (url, opts = {}) => {
    if (url === "/v1/config/schema") {
      return { ok: true, status: 200, json: async () => schema, text: async () => "" };
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

async function render(win) {
  runScript(win, "refreshSettingsPage();");
  await new Promise((r) => setTimeout(r, 0));
}

// ---------------------------------------------------------------- //
//  Rendering: still-default fields are blank/dimmed                //
// ---------------------------------------------------------------- //

test("a NUMBER field still at its shipped default renders BLANK with the value in the placeholder", async () => {
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(SCHEMA_FRESH, []) });
  await render(win);
  const el = win.document.querySelector('input[data-key="n_ctx"]');
  assert.equal(el.value, "", "no solid value");
  assert.match(el.placeholder, /default \(4096\)/);
});

test("a SELECT field still at its shipped default gets the dimmed .auto-detected class, keeps its real selected value", async () => {
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(SCHEMA_FRESH, []) });
  await render(win);
  const el = win.document.querySelector('select[data-key="net_mode"]');
  assert.equal(el.value, "ask", "the option shown IS the default - a dropdown can't be blank");
  assert.ok(el.classList.contains("auto-detected"), "dimmed via the shared grey-default class");
});

test("a TEXT field still at its shipped default renders BLANK with the value in the placeholder", async () => {
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(SCHEMA_FRESH, []) });
  await render(win);
  const el = win.document.querySelector('input[data-key="mdns_name"]');
  assert.equal(el.value, "", "no solid 'localm' value pre-filled");
  assert.match(el.placeholder, /default \(localm\)/);
});

// ---------------------------------------------------------------- //
//  Rendering: a REAL override still renders solid, unstyled         //
// ---------------------------------------------------------------- //

test("a NUMBER field with a real override renders its SOLID value, no placeholder trick", async () => {
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(SCHEMA_OVERRIDDEN, []) });
  await render(win);
  const el = win.document.querySelector('input[data-key="n_ctx"]');
  assert.equal(el.value, "8192", "the user's real value shows solid");
});

test("a SELECT field with a real override does NOT get the dimmed class", async () => {
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(SCHEMA_OVERRIDDEN, []) });
  await render(win);
  const el = win.document.querySelector('select[data-key="net_mode"]');
  assert.equal(el.value, "off");
  assert.ok(!el.classList.contains("auto-detected"), "a real choice must never read as 'default'");
});

test("a TEXT field with a real override renders its SOLID value, no placeholder trick", async () => {
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(SCHEMA_OVERRIDDEN, []) });
  await render(win);
  const el = win.document.querySelector('input[data-key="mdns_name"]');
  assert.equal(el.value, "my-box");
});

// ---------------------------------------------------------------- //
//  Save payload: an untouched default is OMITTED, never sent as a  //
//  value indistinguishable from a real choice                      //
// ---------------------------------------------------------------- //

test("saving without touching a still-default NUMBER/TEXT field omits both keys from the PATCH", async () => {
  const patches = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(SCHEMA_FRESH, patches) });
  await render(win);
  // Touch ONLY the select (a real change), leave n_ctx/mdns_name untouched.
  const sel = win.document.querySelector('select[data-key="net_mode"]');
  sel.value = "off";
  sel.dispatchEvent(new win.Event("change"));
  runScript(win, 'saveSettingsSection([...document.querySelectorAll(".settings-section")]' +
    '.find((s) => s.querySelector(\'[data-key="net_mode"]\')).dataset.sec);');
  await new Promise((r) => setTimeout(r, 0));
  assert.equal(patches.length, 1);
  assert.equal(patches[0].net_mode, "off", "the actually-changed field is sent");
  assert.ok(!("n_ctx" in patches[0]), "an untouched default NUMBER field must be omitted");
  assert.ok(!("mdns_name" in patches[0]), "an untouched default TEXT field must be omitted");
});

test("typing a NUMBER field back to blank after focusing it still omits it (still reads as unchanged)", async () => {
  const patches = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(SCHEMA_FRESH, patches) });
  await render(win);
  const el = win.document.querySelector('input[data-key="n_ctx"]');
  el.value = "9999";
  el.dispatchEvent(new win.Event("input"));
  el.value = "";
  el.dispatchEvent(new win.Event("input"));
  const secId = el.closest(".settings-section").dataset.sec;
  runScript(win, `saveSettingsSection(${JSON.stringify(secId)});`);
  await new Promise((r) => setTimeout(r, 0));
  if (patches.length) assert.ok(!("n_ctx" in patches[0]));
});

// ---------------------------------------------------------------- //
//  THE REGRESSION GUARD: clearing a REAL override still sends an   //
//  explicit null - this fix must never turn "I cleared my custom   //
//  value" into a silent no-op.                                     //
// ---------------------------------------------------------------- //

test("clearing a TEXT field that had a real override still sends an explicit null (real clear, not omitted)", async () => {
  const patches = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(SCHEMA_OVERRIDDEN, patches) });
  await render(win);
  const el = win.document.querySelector('input[data-key="mdns_name"]');
  assert.equal(el.value, "my-box", "starts as the real override, solid");
  el.value = "";
  el.dispatchEvent(new win.Event("input"));
  const secId = el.closest(".settings-section").dataset.sec;
  runScript(win, `saveSettingsSection(${JSON.stringify(secId)});`);
  await new Promise((r) => setTimeout(r, 0));
  assert.equal(patches.length, 1);
  assert.equal(patches[0].mdns_name, null,
    "an explicit clear of a REAL customization must still reach the server as null, " +
    "never silently dropped - dropping it would strand the old override forever");
});

test("legacy schema rows with no shipped_default key behave exactly as before (solid, no crash)", async () => {
  // A media-per-plugin control (renderMediaSubsection) never sets
  // shipped_default at all - confirm buildSettingControl tolerates its total
  // absence rather than crashing on `field.shipped_default` access.
  const schema = { fields: [
    { key: "n_ctx", widget: "number", label: "Context window", help: "",
      group: "Engine", owner: "core", default: 4096 },   // no shipped_default at all
  ] };
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(schema, []) });
  await render(win);
  const el = win.document.querySelector('input[data-key="n_ctx"]');
  assert.equal(el.value, "4096", "no shipped_default -> always renders solid, old behavior");
});
