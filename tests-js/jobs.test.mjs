// SPDX-License-Identifier: AGPL-3.0-or-later
// jsdom tests for the jobs plugin client_entry (localm/plugins/builtin/jobs/
// static/jobs.js).
//
// Unlike the GUI's app.js (a top-level script), jobs.js is an ES MODULE that
// exports register(ctx). loadClientPlugins() in app.js does
//   const mod = await import(`${base}/${p.client_entry}`); await mod.register(ctx)
// with ctx = { registerTTS, toast, authHeaders, voicesChanged }. We mirror that
// here: build a jsdom document that has a <main id="main"> (the GUI views
// container), install it as the global document/window, stub fetch with canned
// /api/jobs data, import the module, call register(ctx), trigger the jobs view,
// and assert the list renders + Run-now POSTs.

import { test } from "node:test";
import assert from "node:assert/strict";
import { JSDOM } from "jsdom";
import { readFileSync } from "node:fs";
import { fileURLToPath, pathToFileURL } from "node:url";
import { dirname, join } from "node:path";

const HERE = dirname(fileURLToPath(import.meta.url));
const JOBS_JS = join(HERE, "..", "localm", "plugins", "builtin", "jobs", "static", "jobs.js");

// A canned job (matches the backend Job dict: asdict(Job)).
const JOB = {
  id: "abc123",
  name: "Nightly digest",
  schedule_kind: "interval",
  schedule: 3600,
  task_kind: "chat",
  prompt: "summarise my notes",
  model: null,
  cwd: null,
  scope: null,
  enabled: true,
  created: 1700000000,
  last_run: 1700003600,
  last_status: "ok",
  last_result_id: "2026-06-17T00-00-00",
};

// Installed models as /api/models returns them: {models:[{name,active,...}],active}.
const MODELS = [
  { name: "chat-model", active: true },
  { name: "coder-model", active: false },
];

// Build a fresh jsdom env per test, install it as the module's globals, and
// return { window, calls } where calls records every fetch (method + url).
function makeEnv({ jobs = [JOB], models = MODELS, active = "chat-model" } = {}) {
  const dom = new JSDOM(
    `<!DOCTYPE html><html><body><main id="main"></main></body></html>`,
    { url: "http://localhost:8642/" });
  const win = dom.window;
  const calls = [];

  win.confirm = () => true;           // Delete uses confirm(); auto-accept.
  win.fetch = async (url, opts = {}) => {
    const method = (opts.method || "GET").toUpperCase();
    calls.push({ url: String(url), method, body: opts.body });
    if (method === "GET" && /\/api\/jobs$/.test(url)) {
      return { ok: true, status: 200, json: async () => ({ jobs }) };
    }
    if (method === "GET" && /\/api\/models$/.test(url)) {
      return { ok: true, status: 200, json: async () => ({ models, active }) };
    }
    if (method === "POST" && /\/api\/jobs\/[^/]+\/run$/.test(url)) {
      return { ok: true, status: 200,
               json: async () => ({ result_id: "r1", status: "ok", output: "done" }) };
    }
    if (method === "GET" && /\/api\/jobs\/[^/]+\/results$/.test(url)) {
      return { ok: true, status: 200,
               json: async () => ({ id: JOB.id, results: [
                 { result_id: "r1", status: "ok", output: "the result text",
                   error: null, finished: 1700003600 }] }) };
    }
    if (method === "PUT" || method === "DELETE" || method === "POST") {
      return { ok: true, status: 200, json: async () => ({}) };
    }
    return { ok: true, status: 200, json: async () => ({}) };
  };

  // Expose the jsdom realm as the module's ambient globals. jobs.js only touches
  // document / window / fetch / setTimeout, all sourced from here.
  global.window = win;
  global.document = win.document;
  global.fetch = win.fetch;
  global.confirm = win.confirm;
  return { win, calls };
}

// Import the module fresh (cache-busted) so each test starts with no shared
// module-level state and the just-installed globals.
async function importJobs() {
  const url = pathToFileURL(JOBS_JS).href + `?t=${Date.now()}_${Math.random()}`;
  return import(url);
}

// A small async settle so awaited fetch microtasks in refresh() resolve before
// we assert against the DOM.
const settle = () => new Promise((r) => setTimeout(r, 0));

test("register builds a #view-jobs container in #main", async () => {
  const { win } = makeEnv();
  const mod = await importJobs();
  assert.equal(typeof mod.register, "function", "jobs.js must export register()");
  await mod.register({ toast: () => {}, authHeaders: () => ({}) });
  const view = win.document.getElementById("view-jobs");
  assert.ok(view, "register() must create #view-jobs");
  assert.ok(win.document.getElementById("main").contains(view),
    "#view-jobs lives inside #main");
});

test("the jobs view renders canned job names via textContent", async () => {
  const { win } = makeEnv();
  const mod = await importJobs();
  await mod.register({ toast: () => {}, authHeaders: () => ({}) });

  // Trigger the view exactly as app.js does: the chained window.onViewShown.
  assert.equal(typeof win.onViewShown, "function",
    "register() must install a chained window.onViewShown");
  win.onViewShown("jobs");
  await settle();

  const view = win.document.getElementById("view-jobs");
  assert.match(view.textContent, /Nightly digest/, "job name is rendered");
});

test("Run now issues a POST to /api/jobs/<id>/run", async () => {
  const { win, calls } = makeEnv();
  const mod = await importJobs();
  await mod.register({ toast: () => {}, authHeaders: () => ({}) });
  win.onViewShown("jobs");
  await settle();

  // Find the Run-now button for the rendered job (data-action marks intent).
  const btn = win.document.querySelector(
    `#view-jobs [data-action="run"][data-id="${JOB.id}"]`);
  assert.ok(btn, "a Run-now button is rendered for the job");
  btn.click();
  await settle();

  const ran = calls.find(
    (c) => c.method === "POST" && c.url.endsWith(`/api/jobs/${JOB.id}/run`));
  assert.ok(ran, "clicking Run now POSTs to /api/jobs/<id>/run");
});

test("the Model field is a dropdown populated from /api/models", async () => {
  const { win, calls } = makeEnv();
  const mod = await importJobs();
  await mod.register({ toast: () => {}, authHeaders: () => ({}) });
  await settle();   // let the async populateModels() fetch resolve

  const sel = win.document.getElementById("jobs-model");
  assert.ok(sel, "the Model field exists");
  assert.equal(sel.tagName, "SELECT",
    "the Model field is a <select>, not a free-text input");
  const values = [...sel.options].map((o) => o.value);
  assert.ok(values.includes(""), "keeps a blank = active/default option");
  assert.ok(values.includes("chat-model") && values.includes("coder-model"),
    "installed model names are added as options");
  assert.ok(calls.some((c) => c.method === "GET" && /\/api\/models$/.test(c.url)),
    "the dropdown is sourced from GET /api/models");
});

// --- the rag (knowledge re-sync) task kind ---------------------------------
// A rag job re-syncs a NAMED COLLECTION against the folders it was indexed
// from, so it is fully specified without a prompt - but it is useless without
// the collection. The form has to invert its required-field rule for that kind,
// which is exactly the kind of branch that silently rots.

async function openForm() {
  const env = makeEnv();
  const mod = await importJobs();
  const toasts = [];
  await mod.register({ toast: (m, bad) => toasts.push({ m, bad }),
                       authHeaders: () => ({}) });
  env.win.onViewShown("jobs");
  await settle();
  return { ...env, toasts };
}

function setField(win, id, value) {
  const node = win.document.getElementById(id);
  assert.ok(node, `the form has a #${id} field`);
  node.value = value;
  return node;
}

test("a rag job posts task_kind + collection and needs no prompt", async () => {
  const { win, calls } = await openForm();

  const task = win.document.getElementById("jobs-task");
  assert.ok([...task.options].some((o) => o.value === "rag"),
    "the Task dropdown offers the rag kind");
  setField(win, "jobs-name", "sync manuals");
  task.value = "rag";
  setField(win, "jobs-collection", "manuals");
  // Prompt deliberately left empty.
  win.document.getElementById("jobs-add").click();
  await settle();

  const post = calls.find((c) => c.method === "POST" && /\/api\/jobs$/.test(c.url));
  assert.ok(post, "adding a rag job POSTs to /api/jobs");
  const body = JSON.parse(post.body);
  assert.equal(body.task_kind, "rag");
  assert.equal(body.collection, "manuals");
});

test("a rag job without a collection is refused before any POST", async () => {
  const { win, calls, toasts } = await openForm();

  setField(win, "jobs-name", "sync nothing");
  win.document.getElementById("jobs-task").value = "rag";
  win.document.getElementById("jobs-add").click();
  await settle();

  assert.ok(!calls.some((c) => c.method === "POST" && /\/api\/jobs$/.test(c.url)),
    "no job is created without a collection");
  assert.ok(toasts.some((t) => t.bad && /collection/i.test(t.m)),
    "the user is told a collection is required");
});

test("a chat job still requires a prompt", async () => {
  // Negative control: relaxing the prompt rule for rag must not relax it for
  // every other kind.
  const { win, calls, toasts } = await openForm();

  setField(win, "jobs-name", "no prompt");
  win.document.getElementById("jobs-task").value = "chat";
  win.document.getElementById("jobs-add").click();
  await settle();

  assert.ok(!calls.some((c) => c.method === "POST" && /\/api\/jobs$/.test(c.url)),
    "no chat job is created without a prompt");
  assert.ok(toasts.some((t) => t.bad && /prompt/i.test(t.m)),
    "the user is told a prompt is required");
});

test("a rag job row shows which collection it re-syncs", async () => {
  const { win } = makeEnv({ jobs: [{ ...JOB, task_kind: "rag",
                                     collection: "manuals", prompt: "" }] });
  const mod = await importJobs();
  await mod.register({ toast: () => {}, authHeaders: () => ({}) });
  win.onViewShown("jobs");
  await settle();

  const view = win.document.getElementById("view-jobs");
  assert.match(view.textContent, /collection: manuals/,
    "the job row names the collection");
});

test("authHeaders() is applied to the list fetch", async () => {
  const { win, calls } = makeEnv();
  const mod = await importJobs();
  let used = false;
  await mod.register({ toast: () => {}, authHeaders: () => { used = true; return {}; } });
  win.onViewShown("jobs");
  await settle();
  assert.ok(used, "register/refresh calls ctx.authHeaders()");
  assert.ok(calls.some((c) => c.method === "GET" && /\/api\/jobs$/.test(c.url)),
    "the list is fetched from GET /api/jobs");
});

// --- the .data-table list pattern (JOBS-DATA-TABLE) -------------------------
// Jobs used to hand-roll a flexbox card list (.job-row/.job-head/.job-meta/
// .job-actions), which is why it was the one list page with no row hover: the
// hover, name-cell icon layout and mobile card stacking all hang off
// `.data-table` in style.css and were never restated here. These assert the
// STRUCTURE that earns those rules, not the rules themselves (CSS is not applied
// in jsdom) - the shared stylesheet is what turns the structure into the look.

async function renderedJobs(jobs) {
  const env = makeEnv(jobs ? { jobs } : undefined);
  const mod = await importJobs();
  await mod.register({ toast: () => {}, authHeaders: () => ({}) });
  env.win.onViewShown("jobs");
  await settle();
  return env;
}

test("the jobs list renders as a .data-table, one tbody row per job", async () => {
  const second = { ...JOB, id: "def456", name: "Weekly sync", enabled: false };
  const { win } = await renderedJobs([JOB, second]);

  const table = win.document.querySelector("#view-jobs table.data-table");
  assert.ok(table, "the list is a <table class=data-table>, not a div card list");
  assert.equal(table.querySelectorAll("tbody tr").length, 2,
    "one tbody row per job");
  assert.deepEqual(
    [...table.querySelectorAll("thead th")].map((th) => th.textContent),
    ["Name", "State", "Schedule", "Task", "Last run", ""],
    "the header names every column, with a trailing actions column");

  // The hand-rolled card classes are gone: leaving them behind would mean two
  // list vocabularies on one page, which is the thing being removed.
  assert.equal(win.document.querySelectorAll("#view-jobs .job-row").length, 0,
    "no .job-row cards remain in the list");
  assert.equal(win.document.querySelectorAll("#view-jobs .job-actions").length, 0,
    "no .job-actions button rows remain in the list");
});

test("each row's cells carry the job's own fields", async () => {
  const { win } = await renderedJobs();
  const tds = [...win.document.querySelectorAll("#view-jobs tbody tr td")];
  assert.equal(tds.length, 6, "six cells, matching the six columns");
  assert.match(tds[0].textContent, /Nightly digest/, "name cell");
  assert.match(tds[1].textContent, /enabled/, "state cell says the schedule is armed");
  assert.match(tds[1].textContent, /ok/, "state cell carries the last-run status pill");
  assert.ok(tds[1].querySelector(".job-state.st-ok"),
    "the last-run status is a pill (gui-design.md rule 6), not bare text");
  assert.match(tds[2].textContent, /every 1h/, "schedule cell");
  assert.match(tds[3].textContent, /chat/, "task cell");
  assert.notEqual(tds[4].textContent, "never", "last-run cell is a formatted time");
});

test("the name cell leads with the shared icon helper", async () => {
  // window.iconEl is the GUI's own icon renderer (app/icons.js), reachable from a
  // client plugin via app/main.js's window-export loop. Spying on it is the only
  // way to tell "used the shared icon set" from "drew something icon-shaped".
  const env = makeEnv();
  const seen = [];
  env.win.iconEl = (name, cls) => {
    seen.push({ name, cls });
    const n = env.win.document.createElement("span");
    n.className = cls;
    n.dataset.iconName = name;
    return n;
  };
  const mod = await importJobs();
  await mod.register({ toast: () => {}, authHeaders: () => ({}) });
  env.win.onViewShown("jobs");
  await settle();

  const nameTd = env.win.document.querySelector("#view-jobs tbody tr td.name-cell");
  assert.ok(nameTd, "the first cell is a .name-cell");
  // Filter to the ROW icon's own call rather than asserting the whole history:
  // the card heads legitimately call window.iconEl too, so a whole-history match
  // would key this test on traffic that is not its subject.
  assert.deepEqual(seen.filter((c) => c.cls === "ic ic-job"),
    [{ name: "clock", cls: "ic ic-job" }],
    "the row icon comes from window.iconEl, using the jobs surface's own clock icon,"
    + " exactly once for the one job");
  // The icon/name line sits on an inner .cell-line span, not on the <td>: the flex
  // row has to live on a child, because a display:flex <td> stops being a
  // table-cell and its border-bottom then draws under its own content instead of
  // at the row's foot, breaking the separator partway across the row.
  const line = nameTd.querySelector(":scope > .cell-line");
  assert.ok(line, "the name cell wraps its contents in a .cell-line");
  assert.equal(nameTd.children.length, 1,
    "the .cell-line is the name cell's only child, so the flex row is never on the td");
  assert.equal(line.firstElementChild.dataset.iconName, "clock",
    "the icon LEADS the name cell");
  assert.equal(nameTd.querySelector(".name").textContent, JOB.name,
    "the name follows it in a .name span");
});

test("row action buttons use the in-table tiers, not the page .btn-* classes", async () => {
  // docs/gui-design.md rule 3: inside a dense .data-table the compact
  // `.data-table button` styling wins, so a row action states its tier with
  // .primary / .secondary / .danger. A .btn-secondary here renders as a large
  // page-level button crammed into a table cell.
  const { win } = await renderedJobs();
  const cells = win.document.querySelectorAll("#view-jobs tbody tr td");
  // Named rather than indexed blind: without it, a row that renders no cells at
  // all makes this test die on a TypeError, which reads as a broken test instead
  // of as the missing table it actually is.
  assert.ok(cells.length, "the row renders table cells to look in");
  const actions = cells[cells.length - 1];
  const tiers = [...actions.querySelectorAll("button")].map(
    (b) => [b.dataset.action, b.className]);
  assert.deepEqual(tiers, [
    ["run", "primary"],
    ["toggle", "secondary"],
    ["results", "secondary"],
    ["delete", "danger"],
  ], "Run now is the primary row action, Delete is the danger one");
  assert.equal(win.document.querySelectorAll('#view-jobs tbody [class*="btn-"]').length, 0,
    "no page-level .btn-* button survives inside the table");
});

// --- delete confirmation (JOBS-CONFIRM-DANGER) ------------------------------
// Delete used to call the native confirm(). Some mobile / PWA browsers suppress
// window.confirm() outright, where it returns falsy and the delete silently never
// happens. confirmDanger() is the app's own in-page modal and is what every other
// destructive action in the GUI uses.

// Install a controllable window.confirmDanger and a confirm() that RECORDS being
// called. The native fallback still exists for the no-GUI-shell case, so a test
// that only asserted "the delete was confirmed somehow" would pass either way;
// these assert WHICH one ran.
function withConfirmSpy(win, { autoConfirm = true } = {}) {
  const seen = { danger: [], native: 0 };
  win.confirmDanger = (title, message, label, onConfirm) => {
    seen.danger.push({ title, message, label });
    if (autoConfirm) onConfirm();
  };
  const native = () => { seen.native += 1; return true; };
  win.confirm = native;
  global.confirm = native;
  return seen;
}

test("Delete confirms with the app's confirmDanger modal, never native confirm()", async () => {
  const env = makeEnv();
  const seen = withConfirmSpy(env.win);
  const mod = await importJobs();
  await mod.register({ toast: () => {}, authHeaders: () => ({}) });
  env.win.onViewShown("jobs");
  await settle();

  env.win.document.querySelector(
    `#view-jobs [data-action="delete"][data-id="${JOB.id}"]`).click();
  await settle();

  assert.equal(seen.native, 0, "the native confirm() dialog is never used");
  assert.equal(seen.danger.length, 1, "confirmDanger() is called exactly once");
  assert.match(seen.danger[0].title, /Nightly digest/,
    "the modal names the job being deleted");
  assert.match(seen.danger[0].message, /results/i,
    "and says the results go with it");
  assert.equal(seen.danger[0].label, "Delete", "the confirm button says what it does");
  assert.ok(
    env.calls.some((c) => c.method === "DELETE" && c.url.endsWith(`/api/jobs/${JOB.id}`)),
    "confirming issues the DELETE");
});

test("declining the confirm deletes nothing", async () => {
  // Assert on the DATA (no DELETE was issued, the row is still listed), not on a
  // status code: converting a synchronous `if (!confirm()) return` into a callback
  // is exactly the edit that can make the action fire before the user answers, and
  // a request count is the only thing that reports that.
  const env = makeEnv();
  const seen = withConfirmSpy(env.win, { autoConfirm: false });
  const mod = await importJobs();
  await mod.register({ toast: () => {}, authHeaders: () => ({}) });
  env.win.onViewShown("jobs");
  await settle();

  env.win.document.querySelector(
    `#view-jobs [data-action="delete"][data-id="${JOB.id}"]`).click();
  await settle();

  assert.equal(seen.danger.length, 1, "the confirm was asked for");
  assert.equal(seen.native, 0, "and not via the native dialog");
  assert.ok(!env.calls.some((c) => c.method === "DELETE"),
    "no DELETE is issued while the confirm is unanswered");
  assert.ok(env.win.document.querySelector(
    `#view-jobs [data-action="delete"][data-id="${JOB.id}"]`),
    "the job row is still listed");
});

// --- the schedule builder's dynamic fields (updateSchedDetails) -------------
// The Add-job form swaps its detail fields whenever the Schedule preset changes,
// and each preset builds a DIFFERENT schedule_kind/schedule pair. That behaviour
// was correct and completely untested, which made it the thing most likely to be
// broken silently by an edit to the surrounding markup. These are a
// characterization guard, not a proof of the data-table change: they hold on both
// sides of it by design, and exist so the NEXT edit here cannot pass unnoticed.

async function pickPreset(win, value) {
  const sel = win.document.getElementById("jobs-sched-preset");
  assert.ok(sel, "the form has a Schedule preset dropdown");
  sel.value = value;
  sel.onchange();          // the real handler updateSchedDetails() is bound here
  await settle();
  return sel;
}

test("the schedule preset swaps the detail fields, leaving none of the old ones", async () => {
  const { win } = await openForm();
  const has = (id) => !!win.document.getElementById(id);

  // Default preset is "hours".
  assert.ok(has("jobs-sched-hours"), "hours preset renders the hours input");

  await pickPreset(win, "cron");
  assert.ok(has("jobs-sched-cron"), "cron preset renders the cron expression input");
  assert.ok(!has("jobs-sched-hours"),
    "and the previous preset's input is REMOVED, not left behind hidden");

  await pickPreset(win, "week");
  assert.ok(has("jobs-sched-day") && has("jobs-sched-time"),
    "week preset renders both a weekday and a time field");
  assert.ok(!has("jobs-sched-cron"), "the cron input is gone again");

  await pickPreset(win, "interval");
  assert.ok(has("jobs-sched-interval"), "interval preset renders the seconds input");
  assert.ok(!has("jobs-sched-day") && !has("jobs-sched-time"),
    "the week fields are gone");

  await pickPreset(win, "day");
  assert.ok(has("jobs-sched-time"), "day preset renders just a time field");
  assert.ok(!has("jobs-sched-day"), "with no weekday field");
});

test("each schedule preset posts its own schedule_kind and schedule", async () => {
  // The presets are the whole point of the builder: 'day'/'week' are CRON under
  // the hood while 'hours'/'interval' are second counts. Asserting the POSTed
  // pair is the only way to catch a preset quietly building the wrong one.
  const cases = [
    { preset: "hours", set: ["jobs-sched-hours", "6"], kind: "interval", schedule: 21600 },
    { preset: "interval", set: ["jobs-sched-interval", "90"], kind: "interval", schedule: 90 },
    { preset: "day", set: ["jobs-sched-time", "07:30"], kind: "cron", schedule: "30 7 * * *" },
    { preset: "cron", set: ["jobs-sched-cron", "*/5 * * * *"], kind: "cron", schedule: "*/5 * * * *" },
  ];
  for (const c of cases) {
    const { win, calls } = await openForm();
    setField(win, "jobs-name", "sched " + c.preset);
    setField(win, "jobs-prompt", "do the thing");
    await pickPreset(win, c.preset);
    setField(win, c.set[0], c.set[1]);
    win.document.getElementById("jobs-add").click();
    await settle();

    const post = calls.find((x) => x.method === "POST" && /\/api\/jobs$/.test(x.url));
    assert.ok(post, `the ${c.preset} preset POSTs a job`);
    const body = JSON.parse(post.body);
    assert.equal(body.schedule_kind, c.kind, `${c.preset} -> schedule_kind`);
    assert.equal(body.schedule, c.schedule, `${c.preset} -> schedule`);
  }
});

test("the week preset builds a cron with the chosen weekday", async () => {
  const { win, calls } = await openForm();
  setField(win, "jobs-name", "weekly");
  setField(win, "jobs-prompt", "do the thing");
  await pickPreset(win, "week");
  setField(win, "jobs-sched-day", "5");        // Friday
  setField(win, "jobs-sched-time", "23:05");
  win.document.getElementById("jobs-add").click();
  await settle();

  const post = calls.find((x) => x.method === "POST" && /\/api\/jobs$/.test(x.url));
  const body = JSON.parse(post.body);
  assert.equal(body.schedule_kind, "cron");
  assert.equal(body.schedule, "5 23 * * 5",
    "minute hour * * weekday - the weekday must reach the last cron field");
});

test("a non-numeric custom interval is refused before any POST", async () => {
  const { win, calls, toasts } = await openForm();
  setField(win, "jobs-name", "bad interval");
  setField(win, "jobs-prompt", "do the thing");
  await pickPreset(win, "interval");
  setField(win, "jobs-sched-interval", "every hour please");
  win.document.getElementById("jobs-add").click();
  await settle();

  assert.ok(!calls.some((c) => c.method === "POST" && /\/api\/jobs$/.test(c.url)),
    "no job is created from an unparseable interval");
  assert.ok(toasts.some((t) => t.bad && /whole number/i.test(t.m)),
    "and the user is told what a valid interval looks like");
});
