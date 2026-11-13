// SPDX-License-Identifier: AGPL-3.0-or-later
// R09: Ctrl+S on the Settings page saves the active section (instead of the
// browser's "Save page as"). R10: editing a setting marks the page dirty and
// leaving Settings without saving warns first.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

const SCHEMA = {
  fields: [
    { key: "n_ctx", widget: "number", label: "Context window", help: "",
      group: "Engine", owner: "core", min: 512, step: 512, default: 4096 },
    { key: "require_auth", widget: "toggle", label: "Require an API key",
      help: "", group: "Security", owner: "core", default: false },
  ],
};

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

async function render(win) {
  runScript(win, "refreshSettingsPage();");
  await new Promise((r) => setTimeout(r, 0));
}
async function drain(times = 6) {
  for (let i = 0; i < times; i++) await new Promise((r) => setTimeout(r, 0));
}
function makeSettingsActive(win) {
  // Mark the Settings VIEW active without going through showView (which would
  // re-render and reset the dirty flag we want to test).
  runScript(win, 'document.querySelectorAll(".view").forEach((v) =>' +
    ' v.classList.toggle("active", v.id === "view-settings"));');
}
function ctrlS(win) {
  win.document.dispatchEvent(new win.KeyboardEvent("keydown",
    { key: "s", ctrlKey: true, bubbles: true, cancelable: true }));
}

test("R09: Ctrl+S on the Settings page saves the active section", async () => {
  const patches = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(patches) });
  await render(win);
  makeSettingsActive(win);
  assert.ok(win.document.querySelector("#settings-content .settings-section.active"),
    "a settings section is active");
  // A save sends only edited values, so give every section an edit.
  const nctx = win.document.querySelector('input[data-key="n_ctx"]');
  nctx.value = "8192";
  nctx.dispatchEvent(new win.Event("input", { bubbles: true }));
  const auth = win.document.querySelector('input[data-key="require_auth"]');
  auth.checked = true;
  auth.dispatchEvent(new win.Event("change", { bubbles: true }));
  ctrlS(win);
  await drain();
  assert.ok(patches.length >= 1, "Ctrl+S saved the active section (a PATCH was sent)");
});

test("R09: Ctrl+S off the Settings page does not save", async () => {
  const patches = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(patches) });
  await render(win);
  runScript(win, 'document.querySelectorAll(".view").forEach((v) =>' +
    ' v.classList.toggle("active", v.id === "view-chat"));');
  ctrlS(win);
  await drain();
  assert.equal(patches.length, 0, "Ctrl+S on a non-Settings view saves nothing");
});

test("R10: editing a setting marks the page dirty; a re-render keeps the unsaved edit", async () => {
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch([]) });
  await render(win);
  assert.equal(win.settingsDirty(), false, "clean right after render");
  const nctx = win.document.querySelector('input[data-key="n_ctx"]');
  nctx.value = "8192";
  nctx.dispatchEvent(new win.Event("input", { bubbles: true }));
  assert.equal(win.settingsDirty(), true, "editing a control marks it dirty");
  await render(win);
  await drain();
  assert.equal(win.document.querySelector('input[data-key="n_ctx"]').value, "8192",
    "the redrawn control still holds the unsaved value");
  assert.equal(win.settingsDirty(), true, "and the page still has an unsaved edit");
});

const modalOpen = (win) => win.document.getElementById("modal").style.display !== "none";
const modalButton = (win, label) => [...win.document.querySelectorAll("#modal-body button")]
  .find((b) => b.textContent === label);
const settle = async () => { for (let i = 0; i < 4; i++) await new Promise((r) => setTimeout(r, 0)); };

test("R10: leaving Settings with unsaved edits asks in the page dialog; Cancel stays put", async () => {
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch([]) });
  await render(win);
  makeSettingsActive(win);
  const nctx = win.document.querySelector('input[data-key="n_ctx"]');
  nctx.value = "8192";
  nctx.dispatchEvent(new win.Event("input", { bubbles: true }));

  win.confirm = () => { throw new Error("the browser's own confirm must not be used"); };
  runScript(win, 'showView("chat");');
  assert.ok(modalOpen(win), "the unsaved-changes dialog opened");
  assert.match(win.document.getElementById("modal-body").textContent, /unsaved settings changes/);
  assert.equal(win.isSettingsView(), true, "nothing moves while the question is open");
  runScript(win, 'showView("chat");');
  assert.equal(win.document.querySelectorAll("#modal-body button").length, 2,
    "a second navigation while asking opens no second question");
  modalButton(win, "Cancel").click();
  await settle();
  assert.equal(win.isSettingsView(), true, "Cancel keeps you on Settings");

  runScript(win, 'showView("chat");');
  modalButton(win, "Leave without saving").click();
  await settle();
  assert.equal(win.isSettingsView(), false, "choosing to leave leaves Settings");
});

test("R10: leaving a clean Settings page does not prompt", async () => {
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch([]) });
  await render(win);
  makeSettingsActive(win);
  runScript(win, 'showView("chat");');
  assert.ok(!modalOpen(win), "no prompt when there are no unsaved edits");
  assert.equal(win.isSettingsView(), false, "navigation proceeds");
});
