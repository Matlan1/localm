// SPDX-License-Identifier: AGPL-3.0-or-later
// Settings > Updates > Integrity (pages/models.js): the result of the integrity
// check, its file list, Repair and Keep my changes, and the notice a startup check
// that finds differing files shows once per result when its setting is on.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

const RESULT = {
  available: true, version: "1.0", count: 2, at: 123.5,
  differ: ["localm/a.py"], missing: ["docs/c.md"], unreadable: [], kept: ["localm/b.py"],
  runtime: { present: true, listed: true, differ: [], missing: [] },
};

function app(states, posts = []) {
  let i = 0;
  const fetchImpl = async (url, opts = {}) => {
    const u = String(url);
    if (u.startsWith("/api/integrity") && (opts.method || "GET") === "POST") {
      posts.push({ url: u, body: opts.body ? JSON.parse(opts.body) : null });
      const reply = u.endsWith("/repair")
        ? { repaired: true, files: ["localm/a.py"], failed: [], result: { ...RESULT, count: 0, differ: [], missing: [] } }
        : u.endsWith("/keep")
          ? { kept: ["localm/a.py", "docs/c.md"], result: { ...RESULT, count: 0, differ: [], missing: [] } }
          : RESULT;
      return { ok: true, status: 200, json: async () => reply };
    }
    if (u === "/api/integrity") {
      const s = states.length ? states[Math.min(i++, states.length - 1)]
        : { pending: false, running: false, result: null };
      return { ok: true, status: 200, json: async () => s };
    }
    return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
  };
  return loadAppWithPages({ fetchImpl }).window;
}

const text = (w, id) => w.document.getElementById(id).textContent;
const items = (w) => [...w.document.querySelectorAll("#integrity-list li")].map((li) => li.textContent);

test("a result with differing files lists each one and offers Repair and Keep", () => {
  const w = app([]);
  w.renderIntegrity({ result: RESULT, notice: true });
  assert.equal(text(w, "integrity-status"), "Files that differ from localm 1.0: 2.");
  assert.deepEqual(items(w), ["differs: localm/a.py", "missing: docs/c.md", "kept as yours: localm/b.py"]);
  assert.equal(w.document.getElementById("integrity-repair").hidden, false);
  assert.equal(w.document.getElementById("integrity-keep").hidden, false);
});

test("a matching install hides Repair and Keep, and a runtime without a list says so", () => {
  const w = app([]);
  w.renderIntegrity({ result: { ...RESULT, count: 0, differ: [], missing: [], kept: [],
    runtime: { present: true, listed: false, differ: [], missing: [] } } });
  assert.equal(text(w, "integrity-status"), "Every file matches localm 1.0.");
  assert.match(items(w)[0], /no list of its files yet/);
  assert.equal(w.document.getElementById("integrity-repair").hidden, true);
  assert.equal(w.document.getElementById("integrity-keep").hidden, true);
});

test("files that could not be read are listed, and a runtime list that cannot be read says why", () => {
  const w = app([]);
  w.renderIntegrity({ result: { ...RESULT, count: 0, differ: [], missing: [], kept: [],
    unreadable: ["localm/c.py"],
    runtime: { present: true, listed: false, differ: [], missing: [], unreadable: ["llama.dll"],
               error: "PermissionError: Permission denied" } } });
  assert.deepEqual(items(w), ["could not be read: localm/c.py", "could not be read: llama.dll",
    "The native runtime's list of its files could not be read (PermissionError: Permission denied)."]);
});

test("no list to compare with gives the reason; no result yet says when it runs", () => {
  const w = app([]);
  w.renderIntegrity({ result: { available: false, reason: "this install has no list" } });
  assert.equal(text(w, "integrity-status"), "Not checked: this install has no list");
  w.renderIntegrity({ result: null, pending: true });
  assert.match(text(w, "integrity-status"), /shortly after localm starts/);
});

for (const notice of [true, false]) {
  test(`the startup result shows a notice only with the notice setting ${notice ? "on" : "off"}`, async () => {
    const w = app([{ at_start: true, notice, pending: false, running: false, result: RESULT }]);
    const toast = w.document.getElementById("toast");
    toast.textContent = "";
    await w.integrityCheckState();
    if (notice) assert.match(toast.textContent, /differ from the 1\.0 release: 2/);
    else assert.equal(toast.textContent, "");
  });
}

test("the notice is shown once per result", async () => {
  const w = app([{ at_start: true, notice: true, pending: false, running: false, result: RESULT }]);
  const toast = w.document.getElementById("toast");
  await w.integrityCheckState();
  assert.match(toast.textContent, /differ/);
  toast.textContent = "";
  await w.integrityCheckState();
  assert.equal(toast.textContent, "");
});

test("Repair and Keep post to their routes and show the new result", async () => {
  const posts = [];
  const w = app([], posts);
  w.renderIntegrity({ result: RESULT });
  await w.integrityRepair();
  assert.equal(posts[0].url, "/api/integrity/repair");
  assert.match(text(w, "integrity-action-status"), /Repaired files: 1/);
  assert.equal(text(w, "integrity-status"), "Every file matches localm 1.0; 1 kept as yours.");
  await w.integrityKeep();
  assert.equal(posts[1].url, "/api/integrity/keep");
  assert.match(text(w, "integrity-action-status"), /Kept as yours: 2 file/);
  await w.integrityRun(true);
  assert.deepEqual(posts[2], { url: "/api/integrity/check", body: { full: true } });
});

test("only runtime files differing offers Repair, which installs the runtime again", async () => {
  const rtResult = { ...RESULT, count: 1, differ: [], missing: [], kept: [],
    runtime: { present: true, listed: true, differ: ["llama.dll"], missing: [] } };
  const w = app([]);
  w.renderIntegrity({ result: rtResult });
  assert.equal(w.document.getElementById("integrity-repair").hidden, false);
  assert.equal(w.document.getElementById("integrity-keep").hidden, true);
  assert.ok(items(w).includes("The native runtime is not part of the release: Repair installs it again."));
  const replies = [
    { repaired: true, files: [], failed: [], result: { ...rtResult, count: 0,
      runtime: { present: true, listed: true, differ: [], missing: [] } },
    runtime: { needed: true, repaired: true, backend: "vulkan", build: "b1", files: ["llama.dll"],
      error: null, fix: null, note: null } },
    { repaired: true, files: [], failed: [], result: rtResult,
      runtime: { needed: true, repaired: false, backend: "vulkan", build: "b1", files: ["llama.dll"],
        error: "network access is off, so the runtime cannot be downloaded",
        fix: "turn it on; then run: localm integrity --repair", note: null } },
    { repaired: false, error: "the release 1.0 has no build to repair from",
      runtime: { needed: true, repaired: true, backend: "vulkan", build: "b1", files: ["llama.dll"],
        error: null, fix: null, note: null } },
  ];
  const fetchImpl = async (url, opts = {}) => {
    if (String(url) === "/api/integrity/repair" && opts.method === "POST") {
      const reply = replies.shift();
      return { ok: true, status: 200, json: async () => reply };
    }
    return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
  };
  const w2 = loadAppWithPages({ fetchImpl }).window;
  w2.renderIntegrity({ result: rtResult });
  await w2.integrityRepair();
  assert.equal(text(w2, "integrity-action-status"),
    "Installed the vulkan runtime b1 again: 1 file(s) of it repaired.");
  assert.equal(w2.document.getElementById("integrity-repair").hidden, true);
  await w2.integrityRepair();
  assert.equal(text(w2, "integrity-action-status"),
    "The native runtime was not repaired: network access is off, so the runtime cannot be "
    + "downloaded. To repair it: turn it on; then run: localm integrity --repair");
  await w2.integrityRepair();
  assert.equal(text(w2, "integrity-action-status"),
    "Failed: the release 1.0 has no build to repair from "
    + "Installed the vulkan runtime b1 again: 1 file(s) of it repaired.");
});

// The two settings render into the Integrity block, which shows without an update
// channel, and its own Save sends only them.
test("the integrity settings show and save without an update channel", async () => {
  const schema = { fields: [
    { key: "update_allow_prerelease", widget: "toggle", label: "Offer prerelease updates",
      help: "h", group: "Updates", owner: "core", admin_only: true, default: false },
    { key: "integrity_check_at_start", widget: "toggle", label: "Check integrity at startup",
      help: "h", group: "Updates", owner: "core", default: true },
    { key: "integrity_notice", widget: "toggle", label: "Show a notice when files differ",
      help: "h", group: "Updates", owner: "core", default: true },
  ] };
  const patches = [];
  const fetchImpl = async (url, opts = {}) => {
    const u = String(url);
    const method = opts.method || "GET";
    const reply = (body) => ({ ok: true, status: 200, text: async () => "", json: async () => body });
    if (u.includes("/api/capabilities")) {
      return reply({ scopes: [], open: false, plugins: [], update_available: false,
        core: { chat: true, models: true, plugins: true, settings: true } });
    }
    if (u === "/v1/config/schema") return reply(schema);
    if (u === "/v1/config" && method === "PATCH") { patches.push(JSON.parse(opts.body)); return reply({}); }
    if (u === "/v1/config") return reply({ integrity_check_at_start: true, integrity_notice: true });
    if (u === "/api/integrity") return reply({ pending: false, running: false, result: null });
    return reply({ models: [], active: "", conversations: [], plugins: [] });
  };
  const { window } = loadAppWithPages({ fetchImpl });
  const drain = async () => { for (let i = 0; i < 8; i++) await new Promise((r) => setTimeout(r, 0)); };
  await drain();
  runScript(window, "refreshSettingsPage();");
  await drain();
  const doc = window.document;
  const block = doc.getElementById("integrity-settings-block");
  assert.equal(doc.getElementById("app-update-block").hidden, true);
  assert.equal(block.hidden, false);
  for (const key of ["integrity_check_at_start", "integrity_notice"]) {
    assert.ok(block.contains(doc.querySelector(`[data-field-key="${key}"]`)), key);
  }
  assert.ok(!block.contains(doc.querySelector('[data-field-key="update_allow_prerelease"]')));
  doc.querySelector('[data-field-key="integrity_check_at_start"] input').click();
  doc.getElementById("integrity-settings-save").click();
  await drain();
  assert.deepEqual(patches, [{ integrity_check_at_start: false }]);
});
