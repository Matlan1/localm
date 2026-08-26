// SPDX-License-Identifier: AGPL-3.0-or-later
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

function ok(j) {
  return { ok: true, status: 200, json: async () => j, text: async () => "" };
}

// A stateful stub for /api/image/workflows so select/delete/upload round-trip.
function makeFetch(calls) {
  let workflows = [{ name: "a.json", is_active: true, size_bytes: 10 }];
  let selected = "a.json";
  return async (url, opts = {}) => {
    const method = opts.method || "GET";
    if (url === "/api/image/workflows" && method === "GET") {
      return ok({ workflows, selected });
    }
    if (url === "/api/image/workflows" && method === "POST") {
      const b = JSON.parse(opts.body);
      calls.push(["upload", b]);
      workflows = [{ name: b.name, is_active: true, size_bytes: 5 }];
      selected = b.name;
      return ok({ name: b.name, workflows, selected });
    }
    if (url === "/api/image/workflows/select" && method === "POST") {
      const b = JSON.parse(opts.body);
      calls.push(["select", b]);
      selected = b.name;
      workflows = workflows.map((w) => ({ ...w, is_active: w.name === selected }));
      return ok({ selected, workflows });
    }
    if (url.startsWith("/api/image/workflows/") && method === "DELETE") {
      const name = decodeURIComponent(url.split("/").pop());
      calls.push(["delete", name]);
      workflows = workflows.filter((w) => w.name !== name);
      return ok({ workflows, selected });
    }
    return ok({ models: [], active: "", conversations: [], plugins: [] });
  };
}

async function drain() {
  for (let i = 0; i < 8; i++) await new Promise((r) => setTimeout(r, 0));
}

test("workflow panel renders default + uploaded, and select round-trips", async () => {
  const calls = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(calls) });
  runScript(win, 'refreshWorkflowPanel("image");');
  await drain();
  const box = win.document.querySelector("#img-workflows");

  const rows = box.querySelectorAll(".workflow-row");
  assert.equal(rows.length, 2, "Built-in default + one uploaded workflow");
  const picks = [...box.querySelectorAll(".workflow-pick")];
  assert.ok(picks.some((p) => /Built-in default/.test(p.textContent)), "default row present");
  const activePick = picks.find((p) => /a\.json/.test(p.textContent));
  assert.ok(activePick, "a.json row present");
  // Selection is marked by the row's accent bar (.workflow-row.active) plus
  // aria-current, not by a radio dot/ring glyph.
  assert.ok(activePick.closest(".workflow-row").classList.contains("active"),
            "a.json's row carries .active (the accent-bar selection marker)");
  assert.equal(activePick.getAttribute("aria-current"), "true",
               "a.json is announced as the current workflow");
  const defaultPick = picks.find((p) => /Built-in default/.test(p.textContent));
  assert.ok(!defaultPick.closest(".workflow-row").classList.contains("active"),
            "the unselected row is not marked active");
  assert.equal(defaultPick.getAttribute("aria-current"), null,
               "only the selected row carries aria-current");
  assert.equal(box.querySelectorAll('[data-icon-name="dot"], [data-icon-name="ring"]').length, 0,
               "no radio dot/ring glyphs remain in the workflow list");

  // Switch to the built-in default (sends name: null).
  [...box.querySelectorAll(".workflow-pick")]
    .find((p) => /Built-in default/.test(p.textContent)).click();
  await drain();
  const sel = calls.find((c) => c[0] === "select");
  assert.deepEqual(sel[1], { name: null }, "selecting default clears with null");

  // a.json is now inactive -> it gets a Delete button; delete it.
  const box2 = win.document.querySelector("#img-workflows");
  const delBtn = box2.querySelector(".workflow-del");
  assert.ok(delBtn, "an inactive workflow can be deleted");
  delBtn.click();
  await drain();
  // Deletion confirms via the in-page confirmDanger modal, not window.confirm:
  // click its danger button.
  const ok = win.document.querySelector("#modal-body .btn-danger");
  assert.ok(ok, "delete-workflow confirm modal shown");
  ok.click();
  await drain();
  assert.ok(calls.find((c) => c[0] === "delete" && c[1] === "a.json"), "DELETE sent");
});

test("upload parses the file and POSTs the workflow JSON", async () => {
  const calls = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(calls) });
  runScript(win, "window._uploadWorkflow = uploadWorkflow;");
  const wf = { "3": { class_type: "KSampler", inputs: {} } };
  const fakeInput = {
    files: [{ name: "my.json", text: async () => JSON.stringify(wf) }],
    value: "Z:/fake/my.json",
  };
  await win._uploadWorkflow("image", fakeInput);
  await drain();
  const up = calls.find((c) => c[0] === "upload");
  assert.ok(up, "an upload was POSTed");
  assert.equal(up[1].name, "my.json");
  assert.deepEqual(up[1].workflow, wf, "the parsed workflow JSON is sent");
  assert.equal(up[1].activate, true, "uploaded workflow is activated");
  assert.equal(fakeInput.value, "", "the file input is cleared after upload");
});
