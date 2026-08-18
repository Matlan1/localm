// SPDX-License-Identifier: AGPL-3.0-or-later
// S5 GUI-button slice: the managed-ComfyUI panel in Settings > Media - its own
// compact box at the TOP of the Media section, ahead of the three per-plugin
// (image/music/video) boxes. When no managed instance is installed it shows a
// "Set up localm's own ComfyUI" button that POSTs /api/comfy/setup (dispatched
// as a job, streamed). When one IS installed it shows "installed at <path>", the
// S1 coexistence control (comfy_target - a core schema field with group="Media",
// otherwise skipped from the flat form) with its own small Save, and a Remove
// button that POSTs /api/comfy/remove. There used to be a second
// managed_comfy_enabled toggle alongside comfy_target; it was retired (the two
// were ANDed with no reachable state where they meaningfully disagreed) - see
// localm/media/managed_comfy.py's module docstring.
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

const SCHEMA = { fields: [
  { key: "comfy_workdir", widget: "folder", label: "ComfyUI folder", help: "",
    group: "Media", owner: "image", default: "/shared" },
]};
// A schema that also carries the S1 coexistence field, for the tests that
// exercise it specifically (kept separate from SCHEMA above so the minimal
// no-target-field case - Remove must still render even then - stays covered).
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
// A checkout abandoned mid-setup (a crashed process, a closed tab) - the exact
// dead end #653-follow-up fixed: is_managed_comfy_installed() correctly says
// false, but the folder exists, so the OLD status response looked identical
// to NOT_INSTALLED and Set-up would just hit the route's own 409.
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
  // Conserve-mode review finding: managed_active was fetched into `st` but
  // never read/displayed anywhere - a GUI-only user completing setup had no
  // durable way to tell whether generation actually routes to the managed
  // instance (the CLI's `comfy setup`/`comfy status` say so explicitly).
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
  // Remove is destructive: it goes through confirmDanger. Stub it to auto-confirm.
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

  // "its own little thing on the top": the compact box precedes the three-mode
  // grid in DOM order, not after it.
  const grid = doc.querySelector("#settings-sec-media .media-grid");
  assert.ok(grid, "the three-mode grid exists");
  const pos = box.compareDocumentPosition(grid);
  assert.ok(pos & win.Node.DOCUMENT_POSITION_FOLLOWING,
    "the comfy box comes before the media grid in the DOM");

  // The top box also has its own Save button (distinct from the per-plugin
  // ".media-save" buttons and the Remove button).
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

// #653-follow-up: state="corrupt" (an abandoned setup attempt, is_managed_
// comfy_installed()=false but the checkout dir exists) used to be visually
// IDENTICAL to state="not_installed" - a Set-up button that would just 409
// "already exists", with no Remove button either (that only ever appears when
// installed=true) - a genuine dead end reachable only via the CLI.

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
  // Repair clears a folder, so it goes through confirmDanger like Remove does.
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

// #1332 follow-up: the panel used to show a static "please wait" for an
// "installing" state with no way to tell it apart from a genuinely stuck job -
// reported live when switching Settings tabs away and back showed the exact
// same message with no progress, indistinguishable from a hang. ADR-0008's
// /api/activity is the one route that finds a running job without already
// holding its id; the panel should use it to re-attach a live log, exactly
// like the Setup button's own click flow does for a job it started itself.

// NOTE: these two tests need a custom streamJob stub in effect for the very
// FIRST render (the "installing" panel attaches during renderManagedComfyPanel
// itself, not from a later click), so they cannot use the shared render()
// helper above - render() unconditionally re-stubs streamJob to its own
// Promise.resolve({status:"done"}) default immediately before calling
// refreshSettingsPage(), which would silently clobber a stub set beforehand.
// (First version of this test did exactly that: the clobbered default made
// streamJob resolve instantly, and since the mocked managed-status never
// leaves "installing", the fix's own "re-render on success" then recursed
// forever - a real infinite loop caught by CPU time, not a settings.js bug.)
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
  // Flips true from INSIDE the streamJob stub, at the moment it resolves -
  // not before render() starts - so the test actually exercises "state was
  // installing at attach time, then flips once the job ends", rather than
  // starting already-installed and never touching the reattach path at all.
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
  // Must NOT keep re-fetching forever - state stays "installing" every time,
  // so a recursive re-render here would be an unbounded loop.
  const statusCalls = calls.filter((c) => c.url === "/api/comfy/managed-status").length;
  assert.ok(statusCalls <= 1, `must not loop re-checking status: saw ${statusCalls} calls`);
});

// --------------------------------------------------------------------------- //
//  Update: the action the GUI never had                                        //
// --------------------------------------------------------------------------- //
// The panel offered Set up / Repair / Remove and no Update, while the CLI had
// `localm comfy update` all along - so a GUI-only user could install a managed
// ComfyUI and never move it off the pin they installed on.

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

test("a non-git install -> Update is disabled and the REASON is visible text", async () => {
  // The whole point of surfacing this in the GUI: a disabled button with the
  // explanation only in a hover title is still an opaque dead end.
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
  // update_managed_comfy distinguishes states a user must be able to tell apart:
  // rolled back cleanly, rolled back but the patch re-apply failed, and the rollback
  // ITSELF failed (a genuinely mixed install). Rendering all of them as "update
  // failed" swallows exactly the distinction that module goes to trouble to make.
  const calls = [];
  const { window: win } = loadAppWithPages({
    fetchImpl: makeFetch(calls, { installed: true, statusExtra: UPDATE_DUE }) });
  await render(win);
  // A failing job whose output is the long, WRAPPED rollback-also-failed message.
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
  // The TAIL, not just the final line: the last line alone is the fragment
  // "'localm comfy remove' then 'localm comfy setup'." with the actual cause lost.
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
// A long silent stretch in the streamed log (a large git clone, a slow pip
// install, a multi-GB download with no per-line progress) used to look
// identical to a hung job - the disabled button and a static log gave no
// "still alive" signal. Covers both ways a job can be in flight: a fresh
// click, and the reattach-after-reload path exercised above (INSTALLING).

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
  await render(win);   // render() always resets streamJob to its own default stub -
  // set the custom one AFTER, or it gets clobbered right back (see the big
  // comment above renderNoStreamStub explaining exactly this gotcha).
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
  // No button exists in the "installing" reattach state (see the test above
  // asserting exactly that), so the indicator must anchor somewhere else -
  // the status pill - rather than silently not showing at all.
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
