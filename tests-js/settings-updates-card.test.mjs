// SPDX-License-Identifier: AGPL-3.0-or-later
// The Settings "Updates" card (S1): app update + runtime update + the two
// update-behavior toggles, merged into ONE card that used to be three separate
// panels (two of them both headed "Updates"). The merge has two INDEPENDENT
// visibility gates that must not collapse into each other:
//   #app-update-block   proxy-gated - hidden until capabilities.update_available
//   the rest of the card (runtime-update block, the toggles) - ALWAYS shown
// These tests pin both directions, plus the Ctrl+S save-target hazard the merge
// introduced (multiple .btn-primary buttons now share one card).

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

const SCHEMA = {
  fields: [
    { key: "update_allow_prerelease", widget: "toggle", label: "Offer prerelease updates",
      help: "Also offer release-candidate builds when checking for updates.",
      group: "Updates", owner: "core", admin_only: true, default: false },
    { key: "update_ignore_net_policy", widget: "toggle",
      label: "Check for updates even when network access is off",
      help: "The update check normally obeys Network access.",
      group: "Updates", owner: "core", admin_only: true, default: false },
  ],
};

function makeFetch({ updateAvailable, patches }) {
  return async (url, opts = {}) => {
    const u = String(url);
    const method = opts.method || "GET";
    if (u.includes("/api/capabilities")) {
      return { ok: true, status: 200, text: async () => "", json: async () => ({
        scopes: [], open: false,
        core: { chat: true, models: true, plugins: true, settings: true },
        plugins: [], suggest_plugins: true,
        update_available: updateAvailable, issues_available: false,
        bugreport_upload: false,
      }) };
    }
    if (u === "/v1/config/schema") {
      return { ok: true, status: 200, text: async () => "", json: async () => SCHEMA };
    }
    if (u === "/v1/config" && method === "PATCH") {
      patches.push(JSON.parse(opts.body));
      return { ok: true, status: 200, text: async () => "", json: async () => ({}) };
    }
    return { ok: true, status: 200, text: async () => "",
      json: async () => ({ models: [], active: "", conversations: [], plugins: [] }) };
  };
}

async function drain(times = 8) {
  for (let i = 0; i < times; i++) await new Promise((r) => setTimeout(r, 0));
}

/** Boot the app with capabilities + schema resolved, and the settings form
 *  rendered (mirrors settings-filter.test.mjs's setup()). */
async function setup({ updateAvailable = false } = {}) {
  const patches = [];
  const { window } = loadAppWithPages({ fetchImpl: makeFetch({ updateAvailable, patches }) });
  await drain();                              // let boot's own /api/capabilities settle
  runScript(window, "refreshSettingsPage();"); // render the schema-driven toggles
  await drain();
  return { window, doc: window.document, patches };
}

test("capabilities-absent: app-update is hidden, runtime-update and the toggles are not", async () => {
  const { doc } = await setup({ updateAvailable: false });

  assert.equal(doc.getElementById("sec-updates").hidden, false,
    "the merged card itself is never hidden");
  assert.equal(doc.getElementById("app-update-block").hidden, true,
    "app-update stays gated off when the proxy has no update to offer");

  // Always shown, regardless of update_available.
  assert.equal(doc.getElementById("runtime-update-check").closest(".update-subsection").hidden,
    false, "the runtime-update block is never gated");
  assert.equal(doc.getElementById("app-launcher-block").hidden, false,
    "App launcher is local disk state too - same rule as runtime-update");
  assert.equal(doc.getElementById("update-toggles-block").hidden, false,
    "the update-behavior toggles render regardless of update_available");
  assert.ok(doc.querySelector('#settings-sec-core-Updates [data-field-key="update_allow_prerelease"]'),
    "the prerelease toggle is actually rendered, not just the empty wrapper");
  assert.ok(doc.querySelector('#settings-sec-core-Updates [data-field-key="update_ignore_net_policy"]'),
    "the net-policy toggle is actually rendered too");
});

test("capabilities-present: app-update, runtime-update and the toggles are all shown", async () => {
  const { doc } = await setup({ updateAvailable: true });

  assert.equal(doc.getElementById("sec-updates").hidden, false);
  assert.equal(doc.getElementById("app-update-block").hidden, false,
    "app-update reveals once the proxy advertises update_available");
  assert.equal(doc.getElementById("runtime-update-check").closest(".update-subsection").hidden,
    false, "runtime-update is unaffected by the app-update gate flipping on");
  assert.equal(doc.getElementById("app-launcher-block").hidden, false,
    "App launcher is likewise unaffected by the app-update gate flipping on");
  assert.equal(doc.getElementById("update-toggles-block").hidden, false);
});

test("the toggles save through the merged card's own Save button", async () => {
  const { doc, patches } = await setup({ updateAvailable: false });

  const toggle = doc.querySelector(
    '#settings-sec-core-Updates [data-field-key="update_allow_prerelease"] input');
  assert.ok(toggle, "the prerelease checkbox exists");
  toggle.checked = true;

  doc.getElementById("update-toggles-save").click();
  await drain();

  assert.equal(patches.length, 1, "exactly one PATCH was sent");
  assert.equal(patches[0].update_allow_prerelease, true,
    "the toggle's value reached the PATCH body");
});

test("Ctrl+S from inside the merged card saves the toggles, never the pending app update", async () => {
  // updateAvailable:true so #update-apply ("Update now") is a live, visible
  // .btn-primary candidate in the SAME card as the toggles' Save button - the
  // exact configuration that would trip saveActiveSettingsSection()'s
  // `.settings-section-save || .actions .btn-primary` fallback onto the wrong
  // button if #update-toggles-save were missing its .settings-section-save class.
  const { window, doc, patches } = await setup({ updateAvailable: true });

  // The "Updates" card lives in the "system" nav group, which is not the
  // default active group (SETTINGS_GROUPS puts "model" first) - without this,
  // saveActiveSettingsSection()'s savable() check would reject #sec-updates for
  // lacking .active and silently fall back to whatever group IS active,
  // making the test pass or fail for a reason that has nothing to do with the
  // hazard it exists to catch.
  runScript(window, 'showSettingsGroup("system");');
  assert.ok(doc.getElementById("sec-updates").classList.contains("active"),
    "precondition: the Updates card is the active section under test");

  let applyClicked = false;
  doc.getElementById("update-apply").onclick = () => { applyClicked = true; };

  const toggle = doc.querySelector(
    '#settings-sec-core-Updates [data-field-key="update_ignore_net_policy"] input');
  toggle.checked = true;
  toggle.focus();
  const saved = window.saveActiveSettingsSection();
  await drain();

  assert.equal(saved, true, "a save target was found");
  assert.equal(applyClicked, false,
    "Ctrl+S must never trigger the app-update apply button as a side effect");
  assert.equal(patches.length, 1, "exactly one PATCH was sent");
  assert.equal(patches[0].update_ignore_net_policy, true,
    "and it is the toggles' own save (the field the user actually touched)");
});
