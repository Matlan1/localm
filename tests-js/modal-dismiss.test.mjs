// SPDX-License-Identifier: AGPL-3.0-or-later
// The shared modal: Escape cancels it; a click outside the dialog does nothing
// (it neither cancels, dismisses nor accepts).

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp, runScript } from "./harness.mjs";

const tick = (ms = 260) => new Promise((r) => setTimeout(r, ms));

function race(p, ms = 600) {
  return Promise.race([p.then((v) => ({ v })), new Promise((r) => setTimeout(() => r(null), ms))]);
}

function setup() {
  const { window } = loadApp();
  runScript(window, "window.__cda = confirmDangerAsync; window.__cd = confirmDanger;");
  return { window, modal: window.document.getElementById("modal") };
}

function clickBackdrop(window, modal) {
  modal.dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
}

function pressEscape(window, target) {
  const ev = new window.KeyboardEvent("keydown", { key: "Escape", bubbles: true, cancelable: true });
  (target || window.document.body).dispatchEvent(ev);
}

test("a click outside a confirm dialog leaves it open and decides nothing", async () => {
  const { window, modal } = setup();
  const p = window.__cda("Delete?", "Delete this chat?", "Delete");
  clickBackdrop(window, modal);
  await tick();
  assert.notEqual(modal.style.display, "none", "the dialog stays open");
  assert.equal(await race(p), null, "the dialog has not answered");
  pressEscape(window);
  const settled = await race(p);
  assert.deepEqual(settled, { v: false }, "Escape cancels");
  assert.equal(modal.style.display, "none");
});

test("Escape cancels a callback confirm without running its action", async () => {
  const { window, modal } = setup();
  let confirmed = 0;
  window.__cd("Revert?", "Revert to here?", "Revert", () => { confirmed += 1; });
  clickBackdrop(window, modal);
  assert.notEqual(modal.style.display, "none", "a click outside does not close it");
  pressEscape(window);
  assert.equal(modal.style.display, "none", "Escape closes it");
  assert.equal(confirmed, 0, "the action did not run");
});

test("Escape with no dialog open does nothing", () => {
  const { window, modal } = setup();
  assert.equal(modal.style.display, "none");
  pressEscape(window);
  assert.equal(modal.style.display, "none");
});

test("the confirm button still confirms once", async () => {
  const { window, modal } = setup();
  const p = window.__cda("Delete?", "Delete this chat?", "Delete");
  modal.querySelector(".btn-danger").click();
  assert.deepEqual(await race(p), { v: true });
  pressEscape(window);
  assert.deepEqual(await race(p), { v: true }, "a later Escape changes nothing");
});
