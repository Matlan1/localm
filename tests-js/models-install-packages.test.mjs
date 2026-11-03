// SPDX-License-Identifier: AGPL-3.0-or-later
// The Models page "install packages" control: shown only on a row whose own
// code needs packages localm can add, it lists what would be installed and
// installs nothing unless the user agrees, then installs the plan it showed.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages } from "./harness.mjs";

const settle = (ms = 0) => new Promise((r) => setTimeout(r, ms));

function sse(events) {
  const text = events.map((e) => "data: " + JSON.stringify(e) + "\n\n").join("");
  let sent = false;
  return {
    getReader() {
      return {
        read: async () => {
          if (sent) return { done: true };
          sent = true;
          return { done: false, value: new TextEncoder().encode(text) };
        },
      };
    },
  };
}

const PLAN = { plan_id: "p1", modules: ["timm"], packages: ["timm"],
               adds: [{ name: "timm", version: "1.0.30", source: "pypi.org" },
                      { name: "torchvision", version: "0.24.0", source: "pypi.org" }] };

function makeFetch(models, calls, { end = { type: "end", ok: true, installed: ["timm==1.0.30"],
                                            failed: [], error: "" } } = {}) {
  return async (url, opts = {}) => {
    const u = String(url);
    calls.push({ u, body: opts.body ? JSON.parse(opts.body) : null });
    if (u === "/api/models/packages/plan") {
      return { ok: true, status: 200, json: async () => PLAN };
    }
    if (u === "/api/models/packages/install") {
      return { ok: true, status: 200, json: async () => ({ status: "running", lines: [] }) };
    }
    if (u === "/api/models/packages/install/events") {
      return { ok: true, status: 200,
               body: sse([{ type: "log", line: "Installing: timm 1.0.30" }, end]) };
    }
    if (u === "/api/models" || u.startsWith("/api/models?")) {
      return { ok: true, status: 200, json: async () => ({ models, active: null }) };
    }
    return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
  };
}

const NEEDS = { name: "kimi", active: false, loaded: false, model_type: "llm", size_bytes: null,
                packages: { missing: ["timm"], addable: ["timm"], installable: true } };
const BLOCKED = { name: "blocked", active: false, loaded: false, model_type: "llm",
                  size_bytes: null,
                  packages: { missing: ["ebfm"], addable: [], installable: false } };
const PLAIN = { name: "plain", active: false, loaded: false, model_type: "llm", size_bytes: 1 };

function rowButton(window, model, label) {
  const row = [...window.document.querySelectorAll("#models-table tbody tr")]
    .find((tr) => tr.textContent.includes(model));
  return row && [...row.querySelectorAll("button")].find((b) => b.textContent === label);
}

function modalButton(window, label) {
  return [...window.document.querySelectorAll("#modal-body button")]
    .find((b) => b.textContent === label);
}

async function open(models, opts) {
  const calls = [];
  const toasts = [];
  const { window } = loadAppWithPages({ fetchImpl: makeFetch(models, calls, opts) });
  window.toast = (msg, isError) => toasts.push({ msg: String(msg), isError: !!isError });
  await window.refreshModelsPage();
  await settle();
  return { window, calls, toasts };
}

test("install packages: shown only where localm can add every missing package", async () => {
  const { window } = await open([NEEDS, BLOCKED, PLAIN]);
  assert.ok(rowButton(window, "kimi", "install packages"));
  assert.equal(rowButton(window, "blocked", "install packages"), undefined);
  assert.equal(rowButton(window, "plain", "install packages"), undefined);
});

test("install packages: cancelling the list installs nothing", async () => {
  const { window, calls } = await open([NEEDS]);
  const clicked = rowButton(window, "kimi", "install packages").onclick();
  await settle(10);
  const body = window.document.getElementById("modal-body").textContent;
  assert.ok(body.includes("timm 1.0.30") && body.includes("torchvision 0.24.0"), body);
  assert.ok(body.includes("from pypi.org"), body);
  modalButton(window, "Cancel").onclick();
  await clicked;
  assert.deepEqual(calls.filter((c) => c.u.startsWith("/api/models/packages")).map((c) => c.u),
                   ["/api/models/packages/plan"]);
});

test("install packages: agreeing installs the plan shown and reports success", async () => {
  const { window, calls, toasts } = await open([NEEDS]);
  const clicked = rowButton(window, "kimi", "install packages").onclick();
  await settle(10);
  modalButton(window, "Install").onclick();
  await clicked;
  const install = calls.find((c) => c.u === "/api/models/packages/install");
  assert.deepEqual(install.body, { plan_id: "p1" });
  assert.deepEqual(calls.find((c) => c.u === "/api/models/packages/plan").body, { model: "kimi" });
  assert.ok(toasts.some((t) => !t.isError && t.msg.includes("'kimi' can now be loaded")),
            JSON.stringify(toasts));
});

test("install packages: a failed install is reported as an error with the reason", async () => {
  const { window, toasts } = await open([NEEDS], {
    end: { type: "end", ok: false, installed: [], failed: [], error: "uv could not install" } });
  const clicked = rowButton(window, "kimi", "install packages").onclick();
  await settle(10);
  modalButton(window, "Install").onclick();
  await clicked;
  assert.ok(toasts.some((t) => t.isError && t.msg.includes("uv could not install")),
            JSON.stringify(toasts));
});
