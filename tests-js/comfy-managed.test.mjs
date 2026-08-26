// SPDX-License-Identifier: AGPL-3.0-or-later
// The managed-ComfyUI panel in Settings > Media: a compact box at the top of the
// Media section, ahead of the three per-plugin boxes. Not installed shows a Set-up
// button POSTing /api/comfy/setup; installed shows "installed at <path>", the
// comfy_target control with its own Save, and a Remove button POSTing
// /api/comfy/remove.
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

const SCHEMA = { fields: [
  { key: "comfy_workdir", widget: "folder", label: "ComfyUI folder", help: "",
    group: "Media", owner: "image", default: "/shared" },
]};
// SCHEMA plus the comfy_target coexistence field.
const SCHEMA_WITH_TARGET = { fields: [
  ...SCHEMA.fields,
  { key: "comfy_target", widget: "select", label: "ComfyUI to use", help: "",
    group: "Media", owner: "image", options: ["own", "user"], default: "own" },
]};
const MEDIA = { plugins: [
  { plugin: "image", label: "Image", fields: [] },
]};

const INSTALLED = {
  installed: true, state: "installed", path: "/home/user/.localm/comfyui",
  models_dir: "/home/user/.localm/comfyui-models",
  api_url: "http://127.0.0.1:8189", target: "own",
  managed_active: false,
};
const NOT_INSTALLED = {
  installed: false, state: "not_installed", path: null, api_url: "http://127.0.0.1:8189",
  target: "own", managed_active: false,
};
// A checkout abandoned mid-setup: not installed, but the folder exists.
const CORRUPT = {
  installed: false, state: "corrupt", path: "/home/user/.localm/comfyui",
  api_url: "http://127.0.0.1:8189", target: "own", managed_active: false,
};
const INSTALLING = {
  installed: false, state: "installing", path: null,
  api_url: "http://127.0.0.1:8189", target: "own", managed_active: false,
};

function makeFetch(calls, { installed, state, schema = SCHEMA, managedActive, statusExtra }) {
  const STATES = { installed: INSTALLED, corrupt: CORRUPT, installing: INSTALLING };
  return async (url, opts = {}) => {
    const u = String(url);
    const method = opts.method || "GET";
    calls.push({ url: u, method, opts });
    if (u === "/v1/config/schema")
      return { ok: true, status: 200, json: async () => schema, text: async () => "" };
    if (u === "/v1/config" && method === "PATCH")
      return { ok: true, status: 200, json: async () => ({ ok: true }), text: async () => "" };
    if (u === "/v1/media/config" && method === "GET")
      return { ok: true, status: 200, json: async () => MEDIA, text: async () => "" };
    if (u === "/v1/comfy/status")
      return { ok: true, status: 200, json: async () => ({ alive: false, launched_by_localm: false }), text: async () => "" };
    if (u === "/api/comfy/managed-status") {
      const base = state ? STATES[state] : (installed ? INSTALLED : NOT_INSTALLED);
      const body = { ...base, ...(statusExtra || {}) };
      return { ok: true, status: 200, text: async () => "",
               json: async () => (managedActive === undefined ? body
                 : { ...body, managed_active: managedActive }) };
    }
    if (u.startsWith("/api/comfy/update") && method === "POST")
      return { ok: true, status: 200, json: async () => ({ job_id: "job123" }), text: async () => "" };
    if (u === "/api/comfy/setup" && method === "POST")
      return { ok: true, status: 200, json: async () => ({ job_id: "job123" }), text: async () => "" };
    if (u === "/api/comfy/remove" && method === "POST")
      return { ok: true, status: 200, json: async () => ({ status: "removed", removed: [INSTALLED.path] }), text: async () => "" };
    if (u === "/api/comfy/repair" && method === "POST")
      return { ok: true, status: 200, json: async () => ({ job_id: "job123", cleared: [CORRUPT.path] }), text: async () => "" };
    return { ok: true, status: 200, text: async () => "",
             json: async () => ({ models: [], active: "", conversations: [], plugins: [] }) };
  };
}

async function render(win) {
  // streamJob is stubbed so a setup click resolves without a real SSE job stream.
  runScript(win, "streamJob = () => Promise.resolve({ status: 'done' });");
  runScript(win, "refreshSettingsPage();");
  for (let i = 0; i < 16; i++) await new Promise((r) => setTimeout(r, 0));
}

test("not installed -> a Set-up button renders (and status was fetched)", async () => {
  const calls = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(calls, { installed: false }) });
  await render(win);
  const btn = win.document.querySelector(".comfy-managed-setup-btn");
  assert.ok(btn, "Set-up button rendered");
  assert.equal(btn.type, "button", "type=button so it never submits the settings form");
  assert.ok(calls.some((c) => c.url === "/api/comfy/managed-status"), "managed-status fetched");
  assert.ok(!win.document.querySelector(".comfy-managed-remove-btn"), "no Remove button when not installed");
});

test("clicking Set up POSTs /api/comfy/setup", async () => {
  const calls = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(calls, { installed: false }) });
  await render(win);
  win.document.querySelector(".comfy-managed-setup-btn").onclick();
  await new Promise((r) => setTimeout(r, 0));
  await new Promise((r) => setTimeout(r, 0));
  const post = calls.find((c) => c.url === "/api/comfy/setup" && c.method === "POST");
  assert.ok(post, "Set up POSTed /api/comfy/setup");
});

test("installed -> shows 'installed at <path>' + a Remove button, no Set-up button", async () => {
  const calls = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(calls, { installed: true }) });
  await render(win);
  const doc = win.document;
  assert.ok(doc.querySelector(".comfy-managed-remove-btn"), "Remove button rendered");
  assert.ok(!doc.querySelector(".comfy-managed-setup-btn"), "no Set-up button when installed");
  const panel = doc.querySelector(".media-comfy-box");
  assert.ok(panel, "media-comfy-box panel present");
  assert.match(panel.textContent, /installed at/i, "shows 'installed at'");
  assert.match(panel.textContent, /\.localm[/\\]comfyui/, "shows the install path");
});

test("installed but comfy_target is 'user' -> the pill discloses generation is NOT using the managed instance", async () => {
  const calls = [];
  const { window: win } = loadAppWithPages(
    { fetchImpl: makeFetch(calls, { installed: true, managedActive: false }) });
  await render(win);
  const panel = win.document.querySelector(".media-comfy-box");
  const pill = panel.querySelector(".comfy-pill");
  assert.match(pill.textContent, /not in use/i, "pill discloses it is not routing here");
  assert.doesNotMatch(pill.className, /\bok\b/, "not styled as active when not in use");
  assert.match(panel.textContent, /your OWN ComfyUI/i,
    "explains generation is using the user's own ComfyUI instead");
});

test("installed and comfy_target is 'own' -> the pill discloses generation IS using the managed instance", async () => {
  const calls = [];
  const { window: win } = loadAppWithPages(
    { fetchImpl: makeFetch(calls, { installed: true, managedActive: true }) });
  await render(win);
  const panel = win.document.querySelector(".media-comfy-box");
  const pill = panel.querySelector(".comfy-pill");
  assert.match(pill.textContent, /in use/i, "pill discloses it is routing here");
  assert.doesNotMatch(pill.textContent, /not in use/i);
  assert.match(pill.className, /\bok\b/, "styled as active when in use");
});

test("clicking Remove POSTs /api/comfy/remove", async () => {
  const calls = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(calls, { installed: true }) });
  await render(win);
  // Remove goes through confirmDanger; stub it to auto-confirm.
  runScript(win, "confirmDanger = (t, m, l, onConfirm) => onConfirm();");
  win.document.querySelector(".comfy-managed-remove-btn").onclick();
  await new Promise((r) => setTimeout(r, 0));
  await new Promise((r) => setTimeout(r, 0));
  const post = calls.find((c) => c.url === "/api/comfy/remove" && c.method === "POST");
  assert.ok(post, "Remove POSTed /api/comfy/remove");
});

test("installed + target field in schema -> the coexistence control renders inside the top box, ahead of the three-mode grid", async () => {
  const calls = [];
  const { window: win } = loadAppWithPages(
    { fetchImpl: makeFetch(calls, { installed: true, schema: SCHEMA_WITH_TARGET }) });
  await render(win);
  const doc = win.document;
  const box = doc.querySelector(".media-comfy-box");
  assert.ok(box, "media-comfy-box present");

  const targetCtrl = box.querySelector('select[data-key="comfy_target"]');
  assert.ok(targetCtrl, "comfy_target control renders inside the top box");

  // The compact box precedes the three-mode grid in DOM order.
  const grid = doc.querySelector("#settings-sec-media .media-settings-grid");
  assert.ok(grid, "the three-mode grid exists");
  const pos = box.compareDocumentPosition(grid);
  assert.ok(pos & win.Node.DOCUMENT_POSITION_FOLLOWING,
    "the comfy box comes before the media grid in the DOM");

  // The top box has its own Save button, distinct from the per-plugin
  // ".media-save" buttons and the Remove button.
  const saveButtons = [...box.querySelectorAll(".actions button")]
    .filter((b) => b.textContent === "Save");
  assert.equal(saveButtons.length, 1, "the top box has exactly one Save button");
});

test("changing comfy_target and clicking the top box's Save PATCHes /v1/config with just that key", async () => {
  const calls = [];
  const { window: win } = loadAppWithPages(
    { fetchImpl: makeFetch(calls, { installed: true, schema: SCHEMA_WITH_TARGET }) });
  await render(win);
  const doc = win.document;
  const box = doc.querySelector(".media-comfy-box");
  const targetCtrl = box.querySelector('select[data-key="comfy_target"]');
  targetCtrl.value = "user";
  targetCtrl.dispatchEvent(new win.Event("change", { bubbles: true }));

  const save = [...box.querySelectorAll(".actions button")]
    .find((b) => b.textContent === "Save");
  assert.ok(save, "Save button present");
  save.onclick();
  await new Promise((r) => setTimeout(r, 0));
  await new Promise((r) => setTimeout(r, 0));

  const patch = calls.find((c) => c.url === "/v1/config" && c.method === "PATCH");
  assert.ok(patch, "Save PATCHed /v1/config");
  const body = JSON.parse(patch.opts.body);
  assert.equal(body.comfy_target, "user", "the changed select's value is in the PATCH body");
  assert.equal(Object.keys(body).length, 1, "exactly this one key is sent, nothing else");
});

test("corrupt -> a Repair button renders, not Set-up or Remove", async () => {
  const calls = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(calls, { state: "corrupt" }) });
  await render(win);
  const doc = win.document;
  const repair = doc.querySelector(".comfy-managed-repair-btn");
  assert.ok(repair, "Repair button rendered");
  assert.equal(repair.type, "button");
  assert.ok(!doc.querySelector(".comfy-managed-setup-btn"),
    "no Set-up button (it would just 409 'already exists')");
  assert.ok(!doc.querySelector(".comfy-managed-remove-btn"),
    "no Remove button (that only appears once genuinely installed)");
  const panel = doc.querySelector(".media-comfy-box");
  assert.match(panel.textContent, /incomplete/i);
});

test("clicking Repair confirms, then POSTs /api/comfy/repair", async () => {
  const calls = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(calls, { state: "corrupt" }) });
  await render(win);
  // Repair goes through confirmDanger; stub it to auto-confirm.
  runScript(win, "confirmDanger = (t, m, l, onConfirm) => onConfirm();");
  win.document.querySelector(".comfy-managed-repair-btn").onclick();
  await new Promise((r) => setTimeout(r, 0));
  await new Promise((r) => setTimeout(r, 0));
  const post = calls.find((c) => c.url === "/api/comfy/repair" && c.method === "POST");
  assert.ok(post, "Repair POSTed /api/comfy/repair");
});

test("a successful Repair re-renders into the installed view", async () => {
  const calls = [];
  let installedNow = false;
  const fetchImpl = async (url, opts = {}) => {
    if (String(url) === "/api/comfy/managed-status") {
      const body = installedNow ? INSTALLED : CORRUPT;
      return { ok: true, status: 200, text: async () => "", json: async () => body };
    }
    return makeFetch(calls, {})(url, opts);
  };
  const { window: win } = loadAppWithPages({ fetchImpl });
  runScript(win, "streamJob = () => Promise.resolve({ status: 'done' });");
  runScript(win, "confirmDanger = (t, m, l, onConfirm) => onConfirm();");
  await render(win);
  installedNow = true;   // the repaired instance is ready by the time streamJob resolves
  win.document.querySelector(".comfy-managed-repair-btn").onclick();
  for (let i = 0; i < 8; i++) await new Promise((r) => setTimeout(r, 0));
  assert.ok(win.document.querySelector(".comfy-managed-remove-btn"),
    "re-rendered into the normal installed/Remove view");
  assert.ok(!win.document.querySelector(".comfy-managed-repair-btn"));
});

test("installing -> no action button, no dead 409-bound Set-up click available", async () => {
  const calls = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(calls, { state: "installing" }) });
  await render(win);
  const doc = win.document;
  assert.ok(!doc.querySelector(".comfy-managed-setup-btn"), "no Set-up button while installing");
  assert.ok(!doc.querySelector(".comfy-managed-repair-btn"), "no Repair button while installing");
  assert.ok(!doc.querySelector(".comfy-managed-remove-btn"), "no Remove button while installing");
  const panel = doc.querySelector(".media-comfy-box");
  assert.match(panel.textContent, /in progress/i);
  assert.doesNotMatch(panel.textContent, /another tab or session/i,
    "must not assert who started it - this page cannot know, and it is often this "
    + "very tab after a navigate-away-and-back losing its local job id");
});

// The "installing" panel attaches its stream during renderManagedComfyPanel itself,
// so streamJob must already be stubbed on the first render. render() re-stubs
// streamJob and would clobber that, so these tests render through this helper.
async function renderNoStreamStub(win) {
  runScript(win, "refreshSettingsPage();");
  for (let i = 0; i < 16; i++) await new Promise((r) => setTimeout(r, 0));
}

test("installing -> finds the running job via /api/activity and streams its real output", async () => {
  const calls = [];
  const activityOp = { id: "job-reattach-1", kind: "comfy-setup", status: "running",
    label: "ComfyUI setup", created_at: 1000 };
  const fetchImpl = async (url, opts = {}) => {
    const u = String(url);
    if (u === "/api/activity")
      return { ok: true, status: 200, text: async () => "",
               json: async () => ({ now: 1010, operations: [activityOp] }) };
    return makeFetch(calls, { state: "installing" })(url, opts);
  };
  const { window: win } = loadAppWithPages({ fetchImpl });
  runScript(win, `streamJob = (id, onLine) => {
    window.__streamedJobId = id;
    onLine("Cloning ComfyUI ...");
    onLine("Installing requirements ...");
    return new Promise(() => {});   // never resolves - job still running
  };`);
  await renderNoStreamStub(win);
  assert.equal(win.__streamedJobId, "job-reattach-1",
    "attached to the actual running job found via /api/activity, not a guess");
  const panel = win.document.querySelector(".media-comfy-box");
  assert.match(panel.textContent, /Cloning ComfyUI/, "real job output is shown, not a static message");
  assert.match(panel.textContent, /Installing requirements/);
});

test("installing -> when the job later finishes, the panel re-renders into the installed view", async () => {
  const calls = [];
  const activityOp = { id: "job-reattach-2", kind: "comfy-setup", status: "running",
    label: "ComfyUI setup", created_at: 1000 };
  // Flipped from inside the streamJob stub, at the moment it resolves.
  let installedNow = false;
  const fetchImpl = async (url, opts = {}) => {
    const u = String(url);
    if (u === "/api/activity")
      return { ok: true, status: 200, text: async () => "",
               json: async () => ({ now: 1010, operations: [activityOp] }) };
    if (u === "/api/comfy/managed-status")
      return { ok: true, status: 200, text: async () => "",
               json: async () => (installedNow ? INSTALLED : INSTALLING) };
    return makeFetch(calls, {})(url, opts);
  };
  const { window: win } = loadAppWithPages({ fetchImpl });
  win.__finishSetup = () => { installedNow = true; };
  runScript(win, `streamJob = () => {
    window.__finishSetup();
    return Promise.resolve({ status: "done" });
  };`);
  await renderNoStreamStub(win);
  assert.equal(installedNow, true, "the stub actually ran (proves streamJob was really invoked)");
  assert.ok(win.document.querySelector(".comfy-managed-remove-btn"),
    "re-rendered into the normal installed/Remove view once the job it attached to ended");
});

test("installing -> /api/activity finds no matching job (e.g. owned by another key) -> says so, does not loop", async () => {
  const calls = [];
  const fetchImpl = async (url, opts = {}) => {
    const u = String(url);
    if (u === "/api/activity")
      return { ok: true, status: 200, text: async () => "",
               json: async () => ({ now: 1010, operations: [] }) };
    return makeFetch(calls, { state: "installing" })(url, opts);
  };
  const { window: win } = loadAppWithPages({ fetchImpl });
  await render(win);
  const panel = win.document.querySelector(".media-comfy-box");
  assert.match(panel.textContent, /not available/i);
  const statusCalls = calls.filter((c) => c.url === "/api/comfy/managed-status").length;
  assert.ok(statusCalls <= 1, `must not loop re-checking status: saw ${statusCalls} calls`);
});

// --------------------------------------------------------------------------- //
//  Update                                                                      //
// --------------------------------------------------------------------------- //

const UPDATE_DUE = {
  pinned_commit: "fe4195f7", pinned_version: "v0.31.1",
  installed_commit: "8f40b43e", installed_version: "v0.9.2",
  update_available: true, updatable: true, update_blocked_reason: "",
};

test("installed + an update available -> an Update button and the two versions render", async () => {
  const calls = [];
  const { window: win } = loadAppWithPages({
    fetchImpl: makeFetch(calls, { installed: true, statusExtra: UPDATE_DUE }) });
  await render(win);
  const btn = win.document.querySelector(".comfy-managed-update-btn");
  assert.ok(btn, "Update button rendered");
  assert.equal(btn.disabled, false, "updatable install -> Update is clickable");
  const txt = win.document.querySelector(".comfy-managed-version").textContent;
  assert.ok(txt.includes("v0.9.2"), "says which version is installed: " + txt);
  assert.ok(txt.includes("v0.31.1"), "says which version localm ships: " + txt);
});

test("clicking Update POSTs /api/comfy/update", async () => {
  const calls = [];
  const { window: win } = loadAppWithPages({
    fetchImpl: makeFetch(calls, { installed: true, statusExtra: UPDATE_DUE }) });
  await render(win);
  win.document.querySelector(".comfy-managed-update-btn").onclick();
  for (let i = 0; i < 4; i++) await new Promise((r) => setTimeout(r, 0));
  const post = calls.find((c) => c.url.startsWith("/api/comfy/update") && c.method === "POST");
  assert.ok(post, "Update POSTed /api/comfy/update");
  assert.ok(!post.url.includes("reinstall_requirements"),
            "dependencies are NOT reinstalled unless asked: " + post.url);
});

test("ticking the dependencies box forwards reinstall_requirements", async () => {
  const calls = [];
  const { window: win } = loadAppWithPages({
    fetchImpl: makeFetch(calls, { installed: true, statusExtra: UPDATE_DUE }) });
  await render(win);
  win.document.querySelector(".comfy-managed-reinstall-box").checked = true;
  win.document.querySelector(".comfy-managed-update-btn").onclick();
  for (let i = 0; i < 4; i++) await new Promise((r) => setTimeout(r, 0));
  const post = calls.find((c) => c.url.startsWith("/api/comfy/update") && c.method === "POST");
  assert.ok(post.url.includes("reinstall_requirements=true"),
            "the checkbox must reach the route, or a pin that changed deps leaves a "
            + "checkout without them: " + post.url);
});

test("typing a commit forwards it to the update route", async () => {
  const calls = [];
  const { window: win } = loadAppWithPages({
    fetchImpl: makeFetch(calls, { installed: true, statusExtra: UPDATE_DUE }) });
  await render(win);
  win.document.querySelector(".comfy-managed-commit-box").value = "abc123def ";
  win.document.querySelector(".comfy-managed-update-btn").onclick();
  for (let i = 0; i < 4; i++) await new Promise((r) => setTimeout(r, 0));
  const post = calls.find((c) => c.url.startsWith("/api/comfy/update") && c.method === "POST");
  assert.ok(post.url.includes("commit=abc123def"),
            "the field must reach the route, or the advanced override does nothing: "
            + post.url);
});

test("leaving the commit field blank omits it (falls back to the shipped pin)", async () => {
  const calls = [];
  const { window: win } = loadAppWithPages({
    fetchImpl: makeFetch(calls, { installed: true, statusExtra: UPDATE_DUE }) });
  await render(win);
  win.document.querySelector(".comfy-managed-update-btn").onclick();
  for (let i = 0; i < 4; i++) await new Promise((r) => setTimeout(r, 0));
  const post = calls.find((c) => c.url.startsWith("/api/comfy/update") && c.method === "POST");
  assert.ok(!post.url.includes("commit="),
            "no commit entered -> no ?commit= at all, not an empty-string override: "
            + post.url);
});

test("a non-git install -> Update is disabled and the REASON is visible text", async () => {
  const reason = "This managed ComfyUI has no git history (it was installed via the "
    + "non-git copy fallback), so a pinned-version update is not possible. Remove it "
    + "and set it up again to move to the version localm ships.";
  const calls = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(calls, {
    installed: true,
    statusExtra: { ...UPDATE_DUE, updatable: false, update_blocked_reason: reason } }) });
  await render(win);
  const btn = win.document.querySelector(".comfy-managed-update-btn");
  assert.ok(btn, "the button still renders, so the state is explained rather than absent");
  assert.equal(btn.disabled, true, "a non-git install cannot take a pinned update");
  assert.ok(win.document.body.textContent.includes("no git history"),
            "the refusal must be readable ON THE PAGE, not only in a title attribute");
  assert.ok(!win.document.querySelector(".comfy-managed-reinstall-box"),
            "no dependencies checkbox when the update cannot run at all");
});

test("an unreadable marker -> UNKNOWN, not a silent 'up to date'", async () => {
  const calls = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(calls, {
    installed: true,
    statusExtra: { ...UPDATE_DUE, update_available: null, installed_commit: null,
                   installed_version: null } }) });
  await render(win);
  const txt = win.document.querySelector(".comfy-managed-version").textContent;
  assert.ok(txt.includes("unknown") || txt.includes("Could not read"),
            "must not claim up-to-date when it could not look: " + txt);
  assert.equal(win.document.querySelector(".comfy-managed-update-btn").disabled, false,
               "unknown must still allow an update - it rolls back if it fails");
});

test("already at the shipped pin -> says up to date, Update still available", async () => {
  const calls = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(calls, {
    installed: true,
    statusExtra: { ...UPDATE_DUE, update_available: false,
                   installed_commit: "fe4195f7", installed_version: "v0.31.1" } }) });
  await render(win);
  const txt = win.document.querySelector(".comfy-managed-version").textContent;
  assert.ok(txt.includes("Up to date"), txt);
  assert.ok(win.document.querySelector(".comfy-managed-update-btn"),
            "still offered: re-running it re-verifies the localm patch set");
});

test("a failed update shows the job's OWN reason as page text, not just 'update failed'", async () => {
  const calls = [];
  const { window: win } = loadAppWithPages({
    fetchImpl: makeFetch(calls, { installed: true, statusExtra: UPDATE_DUE }) });
  await render(win);
  // A failing job whose output is the wrapped rollback-also-failed message.
  runScript(win, `streamJob = (id, onLine) => {
    onLine("Fetching ComfyUI updates ...");
    onLine("The rollback to 8f40b43e0204 ALSO failed (fatal: bad object); the");
    onLine("managed ComfyUI may be in a mixed state - reinstall it with");
    onLine("'localm comfy remove' then 'localm comfy setup'.");
    return Promise.resolve({ status: "failed" });
  };`);
  win.document.querySelector(".comfy-managed-update-btn").onclick();
  for (let i = 0; i < 6; i++) await new Promise((r) => setTimeout(r, 0));

  const err = win.document.querySelector(".comfy-managed-update-error");
  assert.ok(err, "a failure must render its reason on the page");
  assert.ok(err.textContent.includes("ALSO failed"),
            "the real cause must survive the console's line wrapping: " + err.textContent);
  assert.ok(err.textContent.includes("mixed state"), err.textContent);
});

test("a retried update does not stack the previous failure's reason", async () => {
  const calls = [];
  const { window: win } = loadAppWithPages({
    fetchImpl: makeFetch(calls, { installed: true, statusExtra: UPDATE_DUE }) });
  await render(win);
  runScript(win, `streamJob = (id, onLine) => { onLine("first failure reason"); return Promise.resolve({ status: "failed" }); };`);
  win.document.querySelector(".comfy-managed-update-btn").onclick();
  for (let i = 0; i < 6; i++) await new Promise((r) => setTimeout(r, 0));
  runScript(win, `streamJob = (id, onLine) => { onLine("second failure reason"); return Promise.resolve({ status: "failed" }); };`);
  win.document.querySelector(".comfy-managed-update-btn").onclick();
  for (let i = 0; i < 6; i++) await new Promise((r) => setTimeout(r, 0));

  const errs = win.document.querySelectorAll(".comfy-managed-update-error");
  assert.equal(errs.length, 1, "only the CURRENT failure may be shown");
  assert.ok(errs[0].textContent.includes("second"), errs[0].textContent);
  assert.ok(!errs[0].textContent.includes("first"), "a stale reason must not survive");
});

// --------------------------------------------------------------------------- //
//  "Still working" indicator: spinner + elapsed readout while a job runs      //
// --------------------------------------------------------------------------- //

test("Set up: a spinner and elapsed readout appear on the button while the job runs", async () => {
  const calls = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(calls, { installed: false }) });
  await render(win);
  runScript(win, "streamJob = () => new Promise(() => {});");   // never resolves - job still running
  win.document.querySelector(".comfy-managed-setup-btn").onclick();
  for (let i = 0; i < 4; i++) await new Promise((r) => setTimeout(r, 0));
  const btn = win.document.querySelector(".comfy-managed-setup-btn");
  assert.ok(btn.disabled, "button disabled while the job runs");
  assert.ok(btn.querySelector(".comfy-managed-spinner"), "spinner shown on the button");
  const readout = win.document.querySelector(".comfy-managed-elapsed");
  assert.ok(readout, "elapsed readout rendered");
  assert.match(readout.textContent, /still working/i);
});

test("Set up: a failed job removes the spinner and elapsed readout (no re-render to clear them)", async () => {
  const calls = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(calls, { installed: false }) });
  await render(win);
  runScript(win, `streamJob = (id, onLine) => {
    onLine("some output");
    return Promise.resolve({ status: "failed" });
  };`);
  win.document.querySelector(".comfy-managed-setup-btn").onclick();
  for (let i = 0; i < 6; i++) await new Promise((r) => setTimeout(r, 0));
  const btn = win.document.querySelector(".comfy-managed-setup-btn");
  assert.equal(btn.disabled, false, "button re-enabled after failure");
  assert.ok(!btn.querySelector(".comfy-managed-spinner"), "spinner removed after failure");
  assert.ok(!win.document.querySelector(".comfy-managed-elapsed"), "elapsed readout removed after failure");
  assert.match(win.document.querySelector(".comfy-managed-log").textContent, /some output/,
    "the log itself stays visible - only the indicator is torn down");
});

test("Set up: a successful job leaves no spinner or elapsed readout behind (panel re-renders into the installed view)", async () => {
  const calls = [];
  let installedNow = false;
  const fetchImpl = async (url, opts = {}) => {
    if (String(url) === "/api/comfy/managed-status") {
      const body = installedNow ? { installed: true, state: "installed", path: "/x", target: "own", managed_active: false }
                                 : { installed: false, state: "not_installed", path: null, target: "own", managed_active: false };
      return { ok: true, status: 200, text: async () => "", json: async () => body };
    }
    return makeFetch(calls, {})(url, opts);
  };
  const { window: win } = loadAppWithPages({ fetchImpl });
  win.__markInstalled = () => { installedNow = true; };
  // render() resets streamJob to its default stub, so the custom one is set after it.
  await render(win);
  runScript(win, `streamJob = () => { window.__markInstalled(); return Promise.resolve({ status: "done" }); };`);
  win.document.querySelector(".comfy-managed-setup-btn").onclick();
  for (let i = 0; i < 8; i++) await new Promise((r) => setTimeout(r, 0));
  assert.ok(win.document.querySelector(".comfy-managed-remove-btn"), "re-rendered into the installed view");
  assert.ok(!win.document.querySelector(".comfy-managed-spinner"), "no leftover spinner");
  assert.ok(!win.document.querySelector(".comfy-managed-elapsed"), "no leftover elapsed readout");
});

test("Update: a spinner and elapsed readout appear on the button while the job runs", async () => {
  const calls = [];
  const { window: win } = loadAppWithPages({
    fetchImpl: makeFetch(calls, { installed: true, statusExtra: UPDATE_DUE }) });
  await render(win);
  runScript(win, "streamJob = () => new Promise(() => {});");
  win.document.querySelector(".comfy-managed-update-btn").onclick();
  for (let i = 0; i < 4; i++) await new Promise((r) => setTimeout(r, 0));
  const btn = win.document.querySelector(".comfy-managed-update-btn");
  assert.ok(btn.querySelector(".comfy-managed-spinner"), "spinner shown while updating");
  assert.match(win.document.querySelector(".comfy-managed-elapsed").textContent, /still working/i);
});

test("Update: the spinner and elapsed readout are removed once a failed update resolves", async () => {
  const calls = [];
  const { window: win } = loadAppWithPages({
    fetchImpl: makeFetch(calls, { installed: true, statusExtra: UPDATE_DUE }) });
  await render(win);
  runScript(win, `streamJob = (id, onLine) => { onLine("boom"); return Promise.resolve({ status: "failed" }); };`);
  win.document.querySelector(".comfy-managed-update-btn").onclick();
  for (let i = 0; i < 6; i++) await new Promise((r) => setTimeout(r, 0));
  const btn = win.document.querySelector(".comfy-managed-update-btn");
  assert.ok(!btn.querySelector(".comfy-managed-spinner"), "spinner gone after the update settles");
  assert.ok(!win.document.querySelector(".comfy-managed-elapsed"), "elapsed readout gone after the update settles");
});

test("Repair: a spinner and elapsed readout appear on the button while the job runs", async () => {
  const calls = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(calls, { state: "corrupt" }) });
  await render(win);
  runScript(win, "confirmDanger = (t, m, l, onConfirm) => onConfirm();");
  runScript(win, "streamJob = () => new Promise(() => {});");
  win.document.querySelector(".comfy-managed-repair-btn").onclick();
  for (let i = 0; i < 4; i++) await new Promise((r) => setTimeout(r, 0));
  const btn = win.document.querySelector(".comfy-managed-repair-btn");
  assert.ok(btn.querySelector(".comfy-managed-spinner"), "spinner shown while repairing");
  assert.match(win.document.querySelector(".comfy-managed-elapsed").textContent, /still working/i);
});

test("Repair: the spinner is removed once a failed repair resolves", async () => {
  const calls = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(calls, { state: "corrupt" }) });
  await render(win);
  runScript(win, "confirmDanger = (t, m, l, onConfirm) => onConfirm();");
  runScript(win, `streamJob = (id, onLine) => { onLine("boom"); return Promise.resolve({ status: "failed" }); };`);
  win.document.querySelector(".comfy-managed-repair-btn").onclick();
  for (let i = 0; i < 6; i++) await new Promise((r) => setTimeout(r, 0));
  const btn = win.document.querySelector(".comfy-managed-repair-btn");
  assert.ok(!btn.querySelector(".comfy-managed-spinner"), "spinner gone after the repair settles");
  assert.ok(!win.document.querySelector(".comfy-managed-elapsed"), "elapsed readout gone after the repair settles");
});

test("Reattach to an in-progress setup (installing state): a spinner appears next to the status pill", async () => {
  // There is no button in the "installing" state, so the indicator anchors to the pill.
  const calls = [];
  const activityOp = { id: "job-reattach-spinner", kind: "comfy-setup", status: "running",
    label: "ComfyUI setup", created_at: 1000 };
  const fetchImpl = async (url, opts = {}) => {
    const u = String(url);
    if (u === "/api/activity")
      return { ok: true, status: 200, text: async () => "",
               json: async () => ({ now: 1010, operations: [activityOp] }) };
    return makeFetch(calls, { state: "installing" })(url, opts);
  };
  const { window: win } = loadAppWithPages({ fetchImpl });
  runScript(win, `streamJob = (id, onLine) => {
    onLine("Cloning ComfyUI ...");
    return new Promise(() => {});   // never resolves - job still running
  };`);
  await renderNoStreamStub(win);
  const pill = win.document.querySelector(".comfy-pill");
  assert.ok(pill.querySelector(".comfy-managed-spinner"),
    "spinner shown next to the pill (there is no button in this state)");
  assert.match(win.document.querySelector(".comfy-managed-elapsed").textContent, /still working/i);
});
