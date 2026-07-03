// SPDX-License-Identifier: AGPL-3.0-or-later
// Jobs plugin client entry (ES module).
//
// loadClientPlugins() in the GUI (localm/plugins/gui/static/app.js) imports this
// module for the active jobs plugin and calls register(ctx) with
//   ctx = { registerTTS, toast, authHeaders, voicesChanged }.
// We use ctx.toast(msg, isError) for notifications and ctx.authHeaders() (bearer
// + content-type) on every fetch.
//
// The GUI's view sections live inside <main id="main">. The kernel sections are
// hardcoded in index.html; a plugin tab's section is NOT created for us, so this
// module builds its own #view-jobs section into #main (idempotently) and chains
// window.onViewShown so the list refreshes whenever the Jobs tab is shown.
//
// SECURITY: every server/job-originating string reaches the DOM only via
// textContent (or value), never innerHTML. Mirrors app.js's XSS rule.

const API = "/api/jobs";

// --- tiny DOM helpers (self-contained; app.js's `el` is not in this module's
//     scope, and we want this testable in a bare jsdom document) ---
function el(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined && text !== null) n.textContent = String(text);
  return n;
}
function clear(node) {
  node.replaceChildren();
}

// A colored run-status pill (.job-state st-<status>). Unknown statuses fall back to
// the neutral base pill. Real backend statuses are ok / error / skipped; running is
// a transient UI state, pending / paused round out the vocabulary.
const JOB_STATUS_CLASS = {
  ok: "st-ok", done: "st-done", error: "st-error", failed: "st-failed",
  skipped: "st-skipped", running: "st-running", pending: "st-pending", paused: "st-paused",
};
function statusPill(status) {
  const key = String(status || "").toLowerCase();
  const cls = JOB_STATUS_CLASS[key];
  return el("span", "job-state" + (cls ? " " + cls : ""), status);
}

// A designed empty state, reusing the GUI's shared window.emptyState (icon + text +
// hint) when present. The isolated jsdom unit test imports this module without the
// GUI shell, so window.emptyState is absent there; fall back to a text-only
// .empty-state so both paths render something intentional rather than a bare line.
function emptyStateEl(iconName, text, hint) {
  if (typeof window !== "undefined" && typeof window.emptyState === "function") {
    return window.emptyState(iconName, text, hint);
  }
  const box = el("div", "empty-state");
  box.appendChild(el("div", "empty-state-text", text));
  if (hint) box.appendChild(el("div", "empty-state-hint", hint));
  return box;
}

// --- schedule / time formatting -------------------------------------------
function fmtSchedule(job) {
  if (job.schedule_kind === "cron") return `cron: ${job.schedule}`;
  const s = Number(job.schedule) || 0;
  if (s % 86400 === 0 && s >= 86400) return `every ${s / 86400}d`;
  if (s % 3600 === 0 && s >= 3600) return `every ${s / 3600}h`;
  if (s % 60 === 0 && s >= 60) return `every ${s / 60}m`;
  return `every ${s}s`;
}
function fmtTime(epoch) {
  if (!epoch) return "never";
  try {
    return new Date(epoch * 1000).toLocaleString();
  } catch {
    return String(epoch);
  }
}

export function register(ctx) {
  ctx = ctx || {};
  const toast = typeof ctx.toast === "function" ? ctx.toast : () => {};
  const authHeaders =
    typeof ctx.authHeaders === "function" ? ctx.authHeaders : () => ({});

  // Build (or find) the #view-jobs section inside the GUI views container.
  const main = document.getElementById("main") || document.body;
  let view = document.getElementById("view-jobs");
  if (!view) {
    view = el("section", "view");
    view.id = "view-jobs";
    main.appendChild(view);
  }

  const page = el("div", "page");
  view.replaceChildren(page);
  page.appendChild(el("h2", null, "Jobs"));
  page.appendChild(el("div", "sub",
    "Scheduled recurring tasks. Each job runs a chat or coder prompt on an " +
    "interval or cron schedule. Runs happen in the background while the server " +
    "is up; use Run now to trigger one immediately."));

  // Add-job form card.
  page.appendChild(buildForm());
  // Jobs list card.
  const listCard = el("div", "card");
  listCard.appendChild(el("h3", null, "Scheduled jobs"));
  const listEl = el("div", "jobs-list");
  listCard.appendChild(listEl);
  page.appendChild(listCard);

  // Results panel (a simple inline panel, hidden until opened).
  const panel = el("div", "card jobs-results");
  panel.style.display = "none";
  page.appendChild(panel);

  // --- networking --------------------------------------------------------
  async function api(path, opts = {}) {
    const r = await fetch(API + path, { ...opts, headers: authHeaders() });
    let data = null;
    try {
      data = await r.json();
    } catch {
      data = null;
    }
    if (!r.ok) {
      let msg = (data && (data.detail || data.error)) || `HTTP ${r.status}`;
      // FastAPI validation errors give detail as a list of {loc,msg,...}; join
      // them into something readable instead of falling back to "HTTP 422".
      if (Array.isArray(msg)) {
        msg = msg.map((d) => d && d.msg ? d.msg : "").filter(Boolean).join("; ")
              || `HTTP ${r.status}`;
      }
      throw new Error(typeof msg === "string" ? msg : `HTTP ${r.status}`);
    }
    return data;
  }

  async function refresh() {
    try {
      const data = await api("");
      renderList((data && data.jobs) || []);
    } catch (e) {
      clear(listEl);
      listEl.appendChild(el("div", "sub", "Could not load jobs: " + e.message));
    }
  }

  function renderList(jobs) {
    clear(listEl);
    if (!jobs.length) {
      listEl.appendChild(emptyStateEl("clock", "No jobs yet",
        "Add one above to schedule a recurring task."));
      return;
    }
    for (const job of jobs) listEl.appendChild(renderJob(job));
  }

  function renderJob(job) {
    const row = el("div", "job-row");

    const head = el("div", "job-head");
    head.appendChild(el("span", "job-name", job.name));
    head.appendChild(el("span", "job-state " + (job.enabled ? "on" : "off"),
      job.enabled ? "enabled" : "disabled"));
    // The last run's outcome as a colored pill (green ok / red error / amber skipped)
    // instead of a parenthetical in the meta line.
    if (job.last_status) head.appendChild(statusPill(job.last_status));
    row.appendChild(head);

    const meta = el("div", "job-meta sub");
    meta.appendChild(el("span", null, fmtSchedule(job)));
    meta.appendChild(el("span", null, "task: " + job.task_kind));
    meta.appendChild(el("span", null, "last run: " + fmtTime(job.last_run)));
    row.appendChild(meta);

    const actions = el("div", "job-actions");

    const run = el("button", "btn-secondary", "Run now");
    run.dataset.action = "run";
    run.dataset.id = job.id;
    run.onclick = (e) => runNow(job, e.currentTarget);
    actions.appendChild(run);

    const toggle = el("button", "btn-secondary", job.enabled ? "Disable" : "Enable");
    toggle.dataset.action = "toggle";
    toggle.dataset.id = job.id;
    toggle.onclick = () => setEnabled(job, !job.enabled);
    actions.appendChild(toggle);

    const results = el("button", "btn-secondary", "Results");
    results.dataset.action = "results";
    results.dataset.id = job.id;
    results.onclick = () => showResults(job);
    actions.appendChild(results);

    const del = el("button", "btn-danger", "Delete");
    del.dataset.action = "delete";
    del.dataset.id = job.id;
    del.onclick = () => remove(job);
    actions.appendChild(del);

    row.appendChild(actions);
    return row;
  }

  async function runNow(job, btn) {
    // A cold "Run now" can take 10-60s while the server loads the model. Disable
    // the button and show progress so it never looks hung (which invites a
    // second click -> a 409 "already in progress" that reads as a failure).
    const prev = btn ? btn.textContent : "";
    if (btn) { btn.disabled = true; btn.textContent = "Running..."; }
    try {
      const res = await api(`/${encodeURIComponent(job.id)}/run`, { method: "POST" });
      if (res && res.status === "error") {
        toast(`Job "${job.name}" failed: ${res.error || "error"}`, true);
      } else {
        toast(`Job "${job.name}" ran (${(res && res.status) || "ok"})`);
      }
    } catch (e) {
      toast(`Run failed: ${e.message}`, true);
    } finally {
      if (btn) { btn.disabled = false; btn.textContent = prev; }
      refresh();
    }
  }

  async function setEnabled(job, enabled) {
    try {
      await api(`/${encodeURIComponent(job.id)}`, {
        method: "PUT",
        body: JSON.stringify({ enabled }),
      });
      toast(`Job "${job.name}" ${enabled ? "enabled" : "disabled"}`);
    } catch (e) {
      toast(`Update failed: ${e.message}`, true);
    }
    refresh();
  }

  async function remove(job) {
    if (!confirm(`Delete job "${job.name}"? This also removes its results.`)) return;
    try {
      await api(`/${encodeURIComponent(job.id)}`, { method: "DELETE" });
      toast(`Job "${job.name}" deleted`);
    } catch (e) {
      toast(`Delete failed: ${e.message}`, true);
    }
    refresh();
  }

  async function showResults(job) {
    panel.style.display = "block";
    clear(panel);
    const header = el("div", "row");
    header.appendChild(el("h3", null, `Results - ${job.name}`));
    const close = el("button", "btn-secondary", "close");
    close.onclick = () => { panel.style.display = "none"; };
    header.appendChild(close);
    panel.appendChild(header);

    const body = el("div", "jobs-results-body");
    body.appendChild(el("div", "sub", "loading..."));
    panel.appendChild(body);

    try {
      const data = await api(`/${encodeURIComponent(job.id)}/results`);
      const results = (data && data.results) || [];
      clear(body);
      if (!results.length) {
        body.appendChild(emptyStateEl("clock", "No runs yet",
          "This job has not run yet - use Run now to trigger it."));
        return;
      }
      for (const r of results) {
        const item = el("div", "job-result");
        const line = el("div", "job-meta sub");
        line.appendChild(el("span", null, fmtTime(r.finished || r.started)));
        line.appendChild(statusPill(r.status || "?"));
        item.appendChild(line);
        const pre = el("pre", "job-log");
        pre.textContent = r.status === "error"
          ? (r.error || "(error, no detail)")
          : (r.output || "(no output)");
        item.appendChild(pre);
        body.appendChild(item);
      }
    } catch (e) {
      clear(body);
      body.appendChild(el("div", "sub", "Could not load results: " + e.message));
    }
  }

  // --- add-job form ------------------------------------------------------
  function buildForm() {
    const card = el("div", "card");
    card.appendChild(el("h3", null, "Add job"));

    const name = inputRow("Name", "text", "jobs-name", "Nightly digest");
    const taskKind = selectRow("Task", "jobs-task", [
      ["chat", "chat - run a prompt through the model"],
      ["coder", "coder - run a coder agent in a directory"],
    ]);
    const prompt = el("div", "");
    prompt.appendChild(el("label", null, "Prompt"));
    const promptTa = el("textarea");
    promptTa.id = "jobs-prompt";
    promptTa.rows = 3;
    promptTa.placeholder = "what the job should do";
    prompt.appendChild(promptTa);

    const schedKind = selectRow("Schedule kind", "jobs-sched-kind", [
      ["interval", "interval (seconds)"],
      ["cron", "cron (5-field)"],
    ]);
    const sched = inputRow("Schedule", "text", "jobs-sched", "3600");
    const schedHint = el("div", "sub",
      "interval: seconds between runs (e.g. 3600). cron: minute hour dom month dow.");

    // Model is a dropdown of installed models (blank = active/default). It is
    // populated asynchronously from /api/models so the form renders immediately.
    const model = selectRow("Model (optional)", "jobs-model", [
      ["", "active / default model"],
    ]);
    populateModels(model.querySelector("select"));
    const cwd = inputRow("Working dir (coder jobs)", "text", "jobs-cwd",
      "required for coder jobs");
    const scope = inputRow("Scope glob (optional, coder)", "text", "jobs-scope",
      "file-access glob for coder");

    const actions = el("div", "actions");
    const add = el("button", "btn-primary", "Add job");
    add.id = "jobs-add";
    actions.appendChild(add);

    [name, taskKind, prompt, schedKind, sched, schedHint, model, cwd, scope, actions]
      .forEach((n) => card.appendChild(n));

    add.onclick = async () => {
      const kind = schedKind.querySelector("select").value;
      let schedule = sched.querySelector("input").value.trim();
      if (kind === "interval") {
        // The "3600" in the field is a placeholder, not a value: an untouched
        // field is empty -> parseInt("")=NaN -> JSON null -> a cryptic 422.
        // Default the blank case to 3600 and reject a non-numeric interval
        // (e.g. "1 hour") instead of silently coercing it to a 1-second job.
        if (!/^\d+$/.test(schedule)) schedule = schedule === "" ? "3600" : "";
        schedule = parseInt(schedule, 10);
        if (Number.isNaN(schedule) || schedule < 1) {
          toast("Interval must be a whole number of seconds (e.g. 3600).", true);
          return;
        }
      }
      const payload = {
        name: name.querySelector("input").value.trim(),
        task_kind: taskKind.querySelector("select").value,
        prompt: promptTa.value,
        schedule_kind: kind,
        schedule,
        model: model.querySelector("select").value.trim() || null,
        cwd: cwd.querySelector("input").value.trim() || null,
        scope: scope.querySelector("input").value.trim() || null,
      };
      if (!payload.name) { toast("A job name is required", true); return; }
      if (!String(payload.prompt).trim()) { toast("A prompt is required", true); return; }
      try {
        await api("", { method: "POST", body: JSON.stringify(payload) });
        toast(`Job "${payload.name}" added`);
        // Reset the text fields so the form is ready for the next job.
        name.querySelector("input").value = "";
        promptTa.value = "";
        refresh();
      } catch (e) {
        toast(`Could not add job: ${e.message}`, true);
      }
    };

    return card;
  }

  // Fill a <select> with the installed models (from /api/models). Best-effort:
  // on any failure the field keeps just its "active / default" option and stays
  // usable. Uses raw fetch (not api(), which is namespaced to /api/jobs).
  async function populateModels(sel) {
    try {
      const r = await fetch("/api/models", { headers: authHeaders() });
      if (!r.ok) return;
      const data = await r.json();
      for (const m of (data && data.models) || []) {
        const name = m && m.name;
        if (!name) continue;
        const opt = document.createElement("option");
        opt.value = name;
        opt.textContent = m.active ? `${name} (active)` : name;
        sel.appendChild(opt);
      }
    } catch {
      /* leave just the default option */
    }
  }

  function inputRow(label, type, id, placeholder) {
    const wrap = el("div", "");
    wrap.appendChild(el("label", null, label));
    const inp = document.createElement("input");
    inp.type = type;
    inp.id = id;
    if (placeholder) inp.placeholder = placeholder;
    wrap.appendChild(inp);
    return wrap;
  }
  function selectRow(label, id, options) {
    const wrap = el("div", "");
    wrap.appendChild(el("label", null, label));
    const sel = document.createElement("select");
    sel.id = id;
    for (const [value, text] of options) {
      const opt = document.createElement("option");
      opt.value = value;
      opt.textContent = text;
      sel.appendChild(opt);
    }
    wrap.appendChild(sel);
    return wrap;
  }

  // Refresh whenever the Jobs tab is shown. Chain, do not clobber, the existing
  // window.onViewShown (pages.js installs its own; other plugins may too).
  const prev = window.onViewShown;
  window.onViewShown = (name) => {
    if (prev) prev(name);
    if (name === "jobs") refresh();
  };

  // If the Jobs tab is already the active view at load (e.g. restored), populate
  // it now. Otherwise the first showView("jobs") triggers the refresh above.
  if (view.classList.contains("active")) refresh();
}
