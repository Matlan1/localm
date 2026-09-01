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

// A .card-head (category-hued icon + heading), matching every other GUI card
// (docs/gui-design.md rule 4). Reuses the shared window.iconEl when present; the
// isolated jsdom unit test loads this module without the GUI shell (no
// window.iconEl), same blind spot as emptyStateEl above, so the heading still
// renders on its own without an icon there.
function cardHead(iconName, cat, tag, text) {
  const head = el("div", "card-head");
  if (typeof window !== "undefined" && typeof window.iconEl === "function") {
    head.appendChild(window.iconEl(iconName, "ic cat-ic " + cat));
  }
  const txt = el("div", "card-head-text");
  txt.appendChild(el(tag, null, text));
  head.appendChild(txt);
  return head;
}

// --- GUI shell helpers ------------------------------------------------------
// app/main.js defines window.<name> as a live getter for every export of the GUI
// modules, so a client plugin reaches iconEl (app/icons.js) and confirmDanger
// (app/helpers.js) through window rather than importing across the GUI's own file
// layout - helpers.js wires the shared modal at import time and would throw in a
// document that has none. Each bridge below degrades the same way emptyStateEl
// does: the isolated jsdom unit test loads this module with no GUI shell, and the
// fallback keeps the affordance real there instead of silently dropping it.

// A row's leading content-type icon. Without the shell there is no SVG to draw, so
// the placeholder keeps the .name-cell layout identical and the name still renders.
function iconElFor(name, cls) {
  if (typeof window !== "undefined" && typeof window.iconEl === "function") {
    return window.iconEl(name, cls);
  }
  return el("span", cls);
}

// Confirm a destructive action with the app's own themed modal. The native
// confirm() fallback is the pre-existing behaviour, kept only for the no-shell
// case: it still blocks the action, so a delete is never silently unconfirmed.
function confirmDangerous(title, message, confirmLabel, onConfirm) {
  if (typeof window !== "undefined" && typeof window.confirmDanger === "function") {
    window.confirmDanger(title, message, confirmLabel, onConfirm);
    return;
  }
  if (confirm(`${title} ${message}`)) onConfirm();
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
    "Scheduled recurring tasks. Each job runs a chat or coder prompt, or " +
    "re-syncs a knowledge collection, on an interval or cron schedule. Runs " +
    "happen in the background while the server is up; use Run now to trigger " +
    "one immediately."));
  const schedWarnEl = el("div", "key-warn");
  schedWarnEl.hidden = true;
  page.appendChild(schedWarnEl);

  // Add-job form card.
  page.appendChild(buildForm());
  // Jobs list card.
  const listCard = el("div", "card");
  listCard.appendChild(cardHead("clock", "cat-teal", "h3", "Scheduled jobs"));
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
      const warning = data && data.scheduler_warning;
      schedWarnEl.hidden = !warning;
      schedWarnEl.textContent = warning
        ? `Scheduled jobs are not running: ${warning}` : "";
    } catch (e) {
      clear(listEl);
      listEl.appendChild(emptyStateEl("warning", "Could not load jobs", e.message));
      schedWarnEl.hidden = true;
      schedWarnEl.textContent = "";
    }
  }

  // The jobs list is a <table class="data-table">, the same primitive Models,
  // Knowledge and Plugins use, so it inherits that pattern's row hover, name-cell
  // icon layout and mobile card stacking instead of restating them (JOBS-DATA-TABLE).
  const JOB_COLUMNS = ["Name", "State", "Schedule", "Task", "Last run", ""];

  function renderList(jobs) {
    clear(listEl);
    if (!jobs.length) {
      listEl.appendChild(emptyStateEl("clock", "No jobs yet",
        "Add one above to schedule a recurring task."));
      return;
    }
    const table = el("table", "data-table");
    const thead = el("thead");
    const hr = el("tr");
    for (const h of JOB_COLUMNS) hr.appendChild(el("th", "", h));
    thead.appendChild(hr);
    table.appendChild(thead);
    const tbody = el("tbody");
    for (const job of jobs) tbody.appendChild(renderJob(job));
    table.appendChild(tbody);
    listEl.appendChild(table);
  }

  // A row action button. Inside a dense .data-table the compact `.data-table
  // button` styling wins, so the tier is spelled .primary / .secondary / .danger
  // rather than .btn-* (docs/gui-design.md rule 3). data-action / data-id state
  // the row's intent for tests and for anything reading the DOM.
  function rowBtn(action, job, cls, label) {
    const b = el("button", cls, label);
    b.dataset.action = action;
    b.dataset.id = job.id;
    return b;
  }

  function renderJob(job) {
    const tr = el("tr");

    // Name cell: content-type icon then the name (gui-design.md rule 1). `clock`
    // is the jobs surface's own icon (plugin.toml), so a row leads with the same
    // mark as the tab it lives under.
    const nameTd = el("td", "name-cell");
    // The flex line lives on this inner span, never on the td: a display:flex td
    // stops being a table-cell, so its border-bottom draws under its own content
    // instead of at the row's foot and breaks the separator (style.css).
    const nameLine = el("span", "cell-line");
    nameTd.appendChild(nameLine);
    nameLine.appendChild(iconElFor("clock", "ic ic-job"));
    nameLine.appendChild(el("span", "name", job.name));
    tr.appendChild(nameTd);

    // Whether the schedule is armed, then the last run's outcome as a colored
    // pill (green ok / red error / amber skipped) - two independent facts, so
    // two pills rather than one merged word.
    const stateTd = el("td");
    stateTd.appendChild(el("span", "job-state " + (job.enabled ? "on" : "off"),
      job.enabled ? "enabled" : "disabled"));
    if (job.last_status) stateTd.appendChild(statusPill(job.last_status));
    tr.appendChild(stateTd);

    tr.appendChild(el("td", "mono shrink-cell", fmtSchedule(job)));

    // Which collection a rag job re-syncs is the one thing that distinguishes
    // two otherwise identical rows, so show it rather than making the user open
    // the job to find out.
    const taskTd = el("td", "grow-cell", job.task_kind);
    if (job.task_kind === "rag" && job.collection) {
      taskTd.appendChild(el("div", "sub", "collection: " + job.collection));
    }
    tr.appendChild(taskTd);

    tr.appendChild(el("td", "mono shrink-cell", fmtTime(job.last_run)));

    const actionsTd = el("td", "actions-cell");
    const run = rowBtn("run", job, "primary", "Run now");
    run.onclick = (e) => runNow(job, e.currentTarget);
    const toggle = rowBtn("toggle", job, "secondary", job.enabled ? "Disable" : "Enable");
    toggle.onclick = () => setEnabled(job, !job.enabled);
    const results = rowBtn("results", job, "secondary", "Results");
    results.onclick = () => showResults(job);
    const del = rowBtn("delete", job, "danger", "Delete");
    del.onclick = () => remove(job);
    for (const b of [run, toggle, results, del]) actionsTd.appendChild(b);
    tr.appendChild(actionsTd);

    return tr;
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

  // Deleting is confirmed with the app's themed modal, like every other
  // destructive action in the GUI. window.confirm() is suppressed outright by
  // some mobile / PWA browsers, which would have made this delete fire with no
  // prompt at all on exactly those clients.
  function remove(job) {
    confirmDangerous(`Delete job "${job.name}"?`, "This also removes its results.",
      "Delete", async () => {
        try {
          await api(`/${encodeURIComponent(job.id)}`, { method: "DELETE" });
          toast(`Job "${job.name}" deleted`);
        } catch (e) {
          toast(`Delete failed: ${e.message}`, true);
        }
        refresh();
      });
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
      body.appendChild(emptyStateEl("warning", "Could not load results", e.message));
    }
  }

  // --- add-job form ------------------------------------------------------
  function buildForm() {
    const card = el("div", "card");
    card.appendChild(cardHead("plus", "cat-teal", "h3", "Add job"));

    const name = inputRow("Name", "text", "jobs-name", "Nightly digest");
    const taskKind = selectRow("Task", "jobs-task", [
      ["chat", "chat - run a prompt through the model"],
      ["coder", "coder - run a coder agent in a directory"],
      ["rag", "rag - re-sync a knowledge collection with its folders"],
    ]);
    const prompt = el("div", "");
    prompt.appendChild(el("label", null, "Prompt"));
    const promptTa = el("textarea");
    promptTa.id = "jobs-prompt";
    promptTa.rows = 3;
    promptTa.placeholder = "what the job should do";
    prompt.appendChild(promptTa);

    // Human-friendly schedule picker
    const schedContainer = el("div", "jobs-sched-container");
    const schedKind = selectRow("Schedule", "jobs-sched-preset", [
      ["hours", "Every N hours"],
      ["day", "Every day"],
      ["week", "Every week"],
      ["interval", "Custom interval (seconds)"],
      ["cron", "Custom cron (5-field)"],
    ]);
    const schedDetails = el("div", "jobs-sched-details");
    
    function updateSchedDetails() {
      clear(schedDetails);
      const val = schedKind.querySelector("select").value;
      if (val === "hours") {
        const wrap = el("div", "");
        wrap.appendChild(el("span", null, "Every "));
        const inp = document.createElement("input");
        inp.type = "number"; inp.min = "1"; inp.value = "6"; inp.id = "jobs-sched-hours";
        inp.style.width = "4em";
        wrap.appendChild(inp);
        wrap.appendChild(el("span", null, " hours"));
        schedDetails.appendChild(wrap);
      } else if (val === "day") {
        const wrap = el("div", "");
        wrap.appendChild(el("span", null, "At time: "));
        const inp = document.createElement("input");
        inp.type = "time"; inp.value = "08:00"; inp.id = "jobs-sched-time";
        wrap.appendChild(inp);
        schedDetails.appendChild(wrap);
      } else if (val === "week") {
        const wrap = el("div", "");
        wrap.appendChild(el("span", null, "On "));
        const daySel = document.createElement("select");
        daySel.id = "jobs-sched-day";
        ["Sun","Mon","Tue","Wed","Thu","Fri","Sat"].forEach((d, i) => {
          const opt = document.createElement("option"); opt.value = i; opt.textContent = d;
          if(i===1) opt.selected = true;
          daySel.appendChild(opt);
        });
        wrap.appendChild(daySel);
        wrap.appendChild(el("span", null, " at time: "));
        const inp = document.createElement("input");
        inp.type = "time"; inp.value = "08:00"; inp.id = "jobs-sched-time";
        wrap.appendChild(inp);
        schedDetails.appendChild(wrap);
      } else if (val === "interval") {
        schedDetails.appendChild(inputRow("Seconds between runs", "text", "jobs-sched-interval", "3600"));
      } else if (val === "cron") {
        schedDetails.appendChild(inputRow("Cron expression", "text", "jobs-sched-cron", "0 8 * * *"));
        schedDetails.appendChild(el("div", "sub", "minute hour dom month dow."));
      }
    }
    schedKind.querySelector("select").onchange = updateSchedDetails;
    updateSchedDetails();
    schedContainer.appendChild(schedKind);
    schedContainer.appendChild(schedDetails);

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
    const collection = inputRow("Collection (rag jobs)", "text", "jobs-collection",
      "required for rag jobs");

    const actions = el("div", "actions");
    const add = el("button", "btn-primary", "Add job");
    add.id = "jobs-add";
    actions.appendChild(add);

    [name, taskKind, prompt, schedContainer, model, cwd, scope, collection,
      actions].forEach((n) => card.appendChild(n));

    add.onclick = async () => {
      const preset = schedKind.querySelector("select").value;
      let kind = "interval";
      let schedule = "";
      if (preset === "hours") {
        const h = parseInt(document.getElementById("jobs-sched-hours").value, 10);
        if (Number.isNaN(h) || h < 1) { toast("Hours must be a positive number.", true); return; }
        schedule = h * 3600;
      } else if (preset === "day") {
        kind = "cron";
        const t = document.getElementById("jobs-sched-time").value || "08:00";
        const [hh, mm] = t.split(":");
        schedule = `${parseInt(mm, 10)} ${parseInt(hh, 10)} * * *`;
      } else if (preset === "week") {
        kind = "cron";
        const d = document.getElementById("jobs-sched-day").value;
        const t = document.getElementById("jobs-sched-time").value || "08:00";
        const [hh, mm] = t.split(":");
        schedule = `${parseInt(mm, 10)} ${parseInt(hh, 10)} * * ${d}`;
      } else if (preset === "interval") {
        kind = "interval";
        let raw = document.getElementById("jobs-sched-interval").value.trim();
        if (!/^\d+$/.test(raw)) raw = raw === "" ? "3600" : "";
        schedule = parseInt(raw, 10);
        if (Number.isNaN(schedule) || schedule < 1) {
          toast("Interval must be a whole number of seconds (e.g. 3600).", true);
          return;
        }
      } else if (preset === "cron") {
        kind = "cron";
        schedule = document.getElementById("jobs-sched-cron").value.trim();
        if (!schedule) schedule = "0 8 * * *";
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
        collection: collection.querySelector("input").value.trim() || null,
      };
      if (!payload.name) { toast("A job name is required", true); return; }
      // A rag job re-syncs a named collection against the folders it was indexed
      // from, so it is fully specified without a prompt (same as the backend's
      // memory kind). Every other kind still needs one.
      if (payload.task_kind === "rag") {
        if (!payload.collection) {
          toast("A rag job needs a collection to re-sync", true);
          return;
        }
      } else if (!String(payload.prompt).trim()) {
        toast("A prompt is required", true); return;
      }
      try {
        await api("", { method: "POST", body: JSON.stringify(payload) });
        toast(`Job "${payload.name}" added`);
        // Reset the text fields so the form is ready for the next job.
        name.querySelector("input").value = "";
        promptTa.value = "";
        collection.querySelector("input").value = "";
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
