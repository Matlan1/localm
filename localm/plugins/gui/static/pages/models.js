// SPDX-License-Identifier: AGPL-3.0-or-later
/* localm GUI - Models page. */
"use strict";

// --- ES module imports ---
import { pickDirectory } from "../app/picker.js";
import { $, GIB, authHeaders, confirmDanger, downloadRate, el, fmtBytes, fmtDuration, openModal, promptText, renderMarkdown, streamJob, toast } from "../app/helpers.js";
import { t, tn } from "../app/i18n.js";
import { onServerUnreachable } from "../app/init.js";
import { emptyState, iconEl } from "../app/icons.js";
import { modelCache, refreshModels, showKeyGate, switchModel, toastLoadResult } from "../app/models-sidebar.js";
import { refreshPerfEstimate } from "../app/settings-perf.js";
import { showView } from "../app/tabs.js";

/* ================================================================ */
/*  Models page                                                      */
/* ================================================================ */

export function fmtSize(bytes) {
  if (bytes == null) return "";
  return (bytes / GIB).toFixed(2) + " GB";   // binary GiB, labelled GB
}

// An ISO calendar date. mtime is a nullable epoch-seconds float from /api/models.
export function fmtModelDate(mtime) {
  if (mtime == null) return "";
  try { return new Date(mtime * 1000).toISOString().slice(0, 10); }
  catch (_) { return ""; }
}

// The Registered-models table tab (All/LLMs/Embedding/...). Scopes only the
// installed-models TABLE below; the HuggingFace SEARCH has its own Type
// checkboxes (discTypes()).
let currentTypeFilter = "all";

// The Other tab's data-type, not a MODEL_TYPES value. It collects every type
// the strip has no tab for: "unknown", mmproj, and whatever the registry gains
// next.
export const OTHER_TAB = "other";

// Two browser-only display preferences for this list, read from localStorage at
// RENDER time rather than cached in a module variable.
export const SHOW_OTHER_KEY = "localm.showOtherModelsInAll";
export const GROUP_BY_TYPE_KEY = "localm.modelsGroupByType";

// Selectable model types - mirrors localm.model_manager.registry.MODEL_TYPES
// (keep in sync). Powers the per-row one-click set-type control, and gives the
// group-by-type headings a stable order for a type the tab strip does not name.
const MODEL_TYPE_OPTIONS =
  ["llm", "embedding", "mmproj", "diffusion-unet", "text-encoder", "vae", "lora", "unknown"];

// The type reported for a registry entry with no recorded model_type. The route
// still sends `model_type: "llm"` for such an entry, plus
// `model_type_recorded: false`; an absent flag means recorded. Not
// identifier-shaped, so it can never collide with a MODEL_TYPES member and is
// never used as a selector (typeLabel's regex rejects it and returns it as the
// heading).
export const UNTAGGED_TYPE = "(not set)";

// The type a model reads as: UNTAGGED_TYPE when the route reports none is
// recorded, the recorded model_type otherwise, falling back to "llm". Every
// filter, count and grouping decision goes through this one helper.
export function modelTypeOf(m) {
  if (m && m.model_type_recorded === false) return UNTAGGED_TYPE;
  return (m && m.model_type) || "llm";
}

// Which types have a tab of their own, read from the tab strip itself rather
// than from a second list here.
export function typesWithOwnTab() {
  const out = new Set();
  const nav = $("models-tab-nav");
  if (!nav) return out;
  for (const btn of nav.querySelectorAll(".tab-btn")) {
    const t = btn.dataset.type;
    if (t && t !== "all" && t !== OTHER_TAB) out.add(t);
  }
  return out;
}

// A group heading borrows the tab strip's own label. A type with no tab has no
// label to borrow and shows its raw registry value.
export function typeLabel(type) {
  const nav = $("models-tab-nav");
  if (nav && /^[a-z0-9-]+$/i.test(type)) {
    const btn = nav.querySelector(`.tab-btn[data-type="${type}"]`);
    if (btn) {
      const clone = btn.cloneNode(true);
      for (const c of clone.querySelectorAll(".tab-count")) c.remove();
      const text = clone.textContent.trim();
      if (text) return text;
    }
  }
  return type;
}

// Heading order: the tab strip's order first, then any remaining known type,
// then anything the registry grew that neither list names. Only types actually
// PRESENT are ordered.
function _groupOrder(present) {
  const have = new Set(present);
  const seen = new Set();
  const order = [];
  const push = (t) => {
    if (have.has(t) && !seen.has(t)) { seen.add(t); order.push(t); }
  };
  const nav = $("models-tab-nav");
  if (nav) for (const b of nav.querySelectorAll(".tab-btn")) push(b.dataset.type);
  for (const t of MODEL_TYPE_OPTIONS) push(t);
  for (const t of [...present].sort()) {
    if (t !== UNTAGGED_TYPE) push(t);
  }
  // Last: the untagged bucket, after every recorded type.
  push(UNTAGGED_TYPE);
  return order;
}

// Turn an already-sorted model list into a flat render list of heading markers
// and models, all for one table.
export function groupModelsByType(models) {
  const byType = new Map();
  for (const m of models) {
    const t = modelTypeOf(m);
    if (!byType.has(t)) byType.set(t, []);
    byType.get(t).push(m);
  }
  const out = [];
  for (const t of _groupOrder([...byType.keys()])) {
    const list = byType.get(t);
    out.push({ head: true, type: t, label: typeLabel(t), count: list.length });
    for (const m of list) out.push({ model: m });
  }
  return out;
}

// Grouping applies only to a view that can hold more than one type: All and
// Other. The control hides itself on a single-type tab (_syncViewOptControls).
function _groupingActive() {
  if (localStorage.getItem(GROUP_BY_TYPE_KEY) !== "true") return false;
  return currentTypeFilter === "all" || currentTypeFilter === OTHER_TAB;
}

// The note naming how many rows the All tab left out and where to find them.
// Also stands in for the empty state when every model is one of the hidden ones.
export function otherHiddenNote(n) {
  return el("div", "sub models-other-note", tn("models.otherHiddenNote", n));
}

/** A plain, deliberate-click link into the Setup view from the "No models
 *  yet" empty state. Setup never opens on its own (see pages/setup.js) -
 *  this is the ONLY door into it from here, and only a click opens it. */
function setupLink() {
  const p = el("div", "sub");
  const a = document.createElement("a");
  a.href = "#";
  a.textContent = t("models.empty.setupLink");
  a.onclick = (e) => { e.preventDefault(); showView("setup"); };
  p.appendChild(a);
  return p;
}

// Show each display toggle only in the views it applies to.
function _syncViewOptControls() {
  const showOther = $("models-show-other-wrap");
  const group = $("models-group-wrap");
  if (showOther) showOther.hidden = currentTypeFilter !== "all";
  if (group) group.hidden = !(currentTypeFilter === "all" || currentTypeFilter === OTHER_TAB);
}
// Set when a discovery result is chosen, cleared on a successful add or a spec
// edit. {spec, type} - the Add handler attaches model_type only while spec
// still matches exactly what was prefilled.
let pendingPullTypeHint = null;

// Registered-models table columns: label, how to read/compare a row, and
// whether it is sortable (the actions column is not). "kind" picks the
// comparator - "number" (size_bytes/mtime, both nullable) vs "string"
// (case-insensitive). The header row, the click/keydown handlers and the
// comparator all read this one list.
const MODEL_COLUMNS = [
  { key: "name", label: "models.col.name", sortable: true, kind: "string", get: (m) => m.name || "" },
  { key: "model_type", label: "models.col.role", sortable: true, kind: "string", get: (m) => m.model_type || "llm" },
  { key: "source", label: "models.col.source", sortable: true, kind: "string", get: (m) => m.source || "" },
  { key: "size_bytes", label: "models.col.size", sortable: true, kind: "number", get: (m) => m.size_bytes },
  { key: "mtime", label: "models.col.modified", sortable: true, kind: "number", get: (m) => m.mtime },
  { key: "actions", label: "", sortable: false },
];
const _SORTABLE_KEYS = MODEL_COLUMNS.filter((c) => c.sortable).map((c) => c.key);

// Persisted sort choice. Falls back to alphabetical-by-name whenever nothing is
// stored yet, or a stored value no longer names a real column.
let currentSortKey = _SORTABLE_KEYS.includes(localStorage.getItem("localm.modelsSortKey"))
  ? localStorage.getItem("localm.modelsSortKey") : "name";
let currentSortDir = ["asc", "desc"].includes(localStorage.getItem("localm.modelsSortDir"))
  ? localStorage.getItem("localm.modelsSortDir") : "asc";

// Compare two model rows on one column. Numbers (size_bytes, mtime) can be
// null, and a null ALWAYS sorts last, in both directions.
function _compareModelRows(a, b, col, dir) {
  const av = col.get(a);
  const bv = col.get(b);
  if (col.kind === "number") {
    const aNull = av == null;
    const bNull = bv == null;
    if (aNull && bNull) return 0;
    if (aNull) return 1;
    if (bNull) return -1;
    return dir === "asc" ? av - bv : bv - av;
  }
  const as = String(av).toLowerCase();
  const bs = String(bv).toLowerCase();
  if (as < bs) return dir === "asc" ? -1 : 1;
  if (as > bs) return dir === "asc" ? 1 : -1;
  return 0;
}

// Sort a copy of `models` by `sortKey`/`sortDir` ("asc"/"desc"). An unknown or
// non-sortable sortKey returns an unsorted copy rather than throwing.
export function sortModels(models, sortKey, sortDir) {
  const col = MODEL_COLUMNS.find((c) => c.key === sortKey && c.sortable);
  if (!col) return models.slice();
  return models.slice().sort((a, b) => _compareModelRows(a, b, col, sortDir));
}

// Guards refreshModelsPage() against overlapping calls. Every call captures this
// counter BEFORE it awaits anything, then re-checks it immediately before its
// single write into the shared `box`: a call superseded by a newer one discards
// its own render. Each render path writes with ONE replaceChildren().
let _modelsRenderGen = 0;
let _pullShortcutsLoaded = false;   // guards _loadPullShortcuts() below, fetched once

export async function refreshModelsPage() {
  const myGen = ++_modelsRenderGen;
  // Fire-and-forget, once per page lifetime. This authenticated read must run
  // only after the boot auth probe has confirmed the client is authed, which is
  // what onViewShown (dispatch.js) gates refreshModelsPage() on.
  if (!_pullShortcutsLoaded) { _pullShortcutsLoaded = true; _loadPullShortcuts(); }
  await refreshModels();

  const box = $("models-table");
  if (myGen !== _modelsRenderGen) return;
  // `box` is NOT cleared here: every write below REPLACES its content in one
  // call, so the scroll container never sits empty across the fetch.

  // Fetched unfiltered and narrowed to the active tab below, so the per-type tab
  // counts can cover the whole registry.
  let models = [];
  try {
    const r = await fetch("/api/models", { headers: authHeaders() });
    if (r.status === 401) {
      // Expired or absent session. Show the in-page key gate rather than falling
      // through to the models=[] path, which would read as "No models yet". Not
      // gated on myGen.
      showKeyGate(t("models.keyRequired"));
      return;
    }
    if (!r.ok) {
      // A non-401 error (403 = key lacks models.read, 500, 503, ...) also
      // returns a body with no `models` array. Surface the status instead of
      // falling through to the empty-list path.
      if (myGen !== _modelsRenderGen) return;
      box.replaceChildren(el("div", "sub", t("models.loadError", { status: r.status })));
      return;
    }
    const data = await r.json();
    models = (data && Array.isArray(data.models)) ? data.models : [];
  } catch (e) {
    if (myGen !== _modelsRenderGen) return;
    box.replaceChildren(el("div", "sub", t("models.loadException", { message: e.message })));
    return;
  }

  if (myGen !== _modelsRenderGen) return;
  // Counts come from the WHOLE registry, so a tab keeps showing how many it
  // holds while a different one is active.
  syncTabCounts(models);
  _syncViewOptControls();

  // Narrow to the active tab. A named type is the same comparison the
  // /api/models route makes. The two multi-type views differ from it:
  //   Other - every model whose type has no tab of its own.
  //   All   - those same models are left OUT by default and merged back in on
  //           demand; whatever it leaves out is counted on Other and named below.
  const tabbed = typesWithOwnTab();
  const mergeOther = localStorage.getItem(SHOW_OTHER_KEY) === "true";
  let hiddenOther = 0;
  if (currentTypeFilter !== "all" && currentTypeFilter !== OTHER_TAB) {
    models = models.filter((m) => modelTypeOf(m) === currentTypeFilter);
  } else if (tabbed.size === 0) {
    // No strip to read, so "has a tab of its own" has no answer: leave the list
    // whole rather than treating every type as an Other.
  } else if (currentTypeFilter === OTHER_TAB) {
    models = models.filter((m) => !tabbed.has(modelTypeOf(m)));
  } else if (!mergeOther) {
    const before = models.length;
    models = models.filter((m) => tabbed.has(modelTypeOf(m)));
    hiddenOther = before - models.length;
  }

  if (!models.length) {
    // The hidden-rows note when the registry is not empty and All is merely
    // hiding all of it; the empty state otherwise.
    if (hiddenOther) {
      box.replaceChildren(otherHiddenNote(hiddenOther));
    } else {
      box.replaceChildren(emptyState("models", t("models.empty.text"), t("models.empty.hint")));
      box.appendChild(setupLink());
    }
    return;
  }

  models = sortModels(models, currentSortKey, currentSortDir);
  // Sorted FIRST, then partitioned, so each section keeps the active column
  // sort's order.
  const renderList = _groupingActive()
    ? groupModelsByType(models)
    : models.map((m) => ({ model: m }));

  const table = el("table", "data-table");
  const thead = el("thead");
  const hr = el("tr");
  for (const col of MODEL_COLUMNS) {
    const th = el("th", "", col.label ? t(col.label) : "");
    if (col.sortable) {
      th.classList.add("sortable");
      th.tabIndex = 0;
      th.setAttribute("role", "columnheader");
      const active = col.key === currentSortKey;
      th.setAttribute("aria-sort", active ? (currentSortDir === "asc" ? "ascending" : "descending") : "none");
      if (active) {
        th.appendChild(iconEl("caret", "ic sort-arrow" + (currentSortDir === "desc" ? " sort-desc" : "")));
      }
      const activate = () => {
        if (currentSortKey === col.key) {
          currentSortDir = currentSortDir === "asc" ? "desc" : "asc";
        } else {
          currentSortKey = col.key;
          currentSortDir = "asc";
        }
        localStorage.setItem("localm.modelsSortKey", currentSortKey);
        localStorage.setItem("localm.modelsSortDir", currentSortDir);
        refreshModelsPage();
      };
      th.onclick = activate;
      th.onkeydown = (e) => {
        if (e.key === "Enter" || e.key === " ") { e.preventDefault(); activate(); }
      };
    }
    hr.appendChild(th);
  }
  thead.appendChild(hr);
  table.appendChild(thead);
  const tbody = el("tbody");
  for (const entry of renderList) {
    if (entry.head) {
      // A group heading spans the whole row rather than sitting in a column, so
      // the data columns keep the widths the header row set for them.
      const headTr = el("tr", "group-head");
      const headTh = el("th", "group-head-cell");
      headTh.colSpan = MODEL_COLUMNS.length;
      headTh.setAttribute("scope", "colgroup");
      headTh.appendChild(el("span", "group-head-label", entry.label));
      headTh.appendChild(el("span", "group-head-count", String(entry.count)));
      headTr.appendChild(headTh);
      tbody.appendChild(headTr);
      continue;
    }
    const m = entry.model;
    const tr = el("tr");
    const nameTd = el("td", "name-cell");
    // The icon/name/badge flex line lives on this inner span, never on the td: a
    // display:flex td stops being a table-cell, so its border-bottom draws under
    // its own content rather than at the row's foot.
    const nameLine = el("span", "cell-line");
    nameTd.appendChild(nameLine);
    nameLine.appendChild(iconEl("models", "ic ic-model"));
    nameLine.appendChild(el("span", "name", m.name));
    // Architecture/MoE badges from the model's own file header, read at
    // registration/pull time. Falsy-checked (m.architecture,
    // m.expert_count > 0), never defaulted: a model whose header carries
    // neither shows neither badge.
    if (m.architecture) {
      nameLine.appendChild(el("span", "arch-badge", m.architecture));
    }
    if (m.expert_count > 0) {
      nameLine.appendChild(el("span", "moe-badge moe-confirmed", t(MOE_LABEL.confirmed)));
    }
    const visBadge = visionBadge(m.vision);
    if (visBadge) nameLine.appendChild(visBadge);
    if (m.active) nameLine.appendChild(el("span", "active-tag job-state st-ok", t("models.tag.active")));
    // Independent of "active": a model can sit resident in VRAM without being
    // the one currently serving requests. Wears job-state's "on" variant rather
    // than active-tag, so it looks distinct and does not trigger the
    // tr:has(.active-tag) row highlight.
    else if (m.loaded) nameLine.appendChild(el("span", "loaded-tag job-state on", t("models.tag.loaded")));
    // The file behind this entry is gone (moved or deleted). The relocate
    // action in the actions cell below is the fix.
    if (m.missing) nameLine.appendChild(el("span", "missing-tag job-state st-error", t("models.tag.missing")));
    tr.appendChild(nameTd);
    
    // Role column. The set-type control IS this column - one pill that both
    // shows the type and changes it, wearing the .type-<name> colour.
    const roleTd = el("td", "mono shrink-cell");
    // An entry with nothing recorded wears "unset", never an "llm" badge.
    const untagged = modelTypeOf(m) === UNTAGGED_TYPE;
    const roleType = untagged ? "unset" : (m.model_type || "llm");

    // On every row, outside the LLM-only gate below. Changing it POSTs the
    // chosen type, then re-renders (so an unknown->llm switch reveals
    // use/alias).
    const typeSel = el("select", "model-type-select type-badge type-" + roleType);
    typeSel.title = untagged
      ? t("models.typeSelect.titleUnset")
      : t("models.typeSelect.titleChange");
    typeSel.setAttribute("aria-label", t("models.typeSelect.ariaLabel", { name: m.name }));
    if (untagged) {
      // A placeholder, not a value: "not set" is not a MODEL_TYPES member and
      // the set-type route rejects it, so it is disabled and can never be
      // chosen or posted. Rendered only while no type is recorded.
      const none = el("option", "", t("models.typeSelect.notSet"));
      none.value = "";
      none.disabled = true;
      none.selected = true;
      typeSel.appendChild(none);
    }
    for (const t of MODEL_TYPE_OPTIONS) {
      const opt = el("option", "", t);
      opt.value = t;
      if (!untagged && (m.model_type || "llm") === t) opt.selected = true;
      typeSel.appendChild(opt);
    }
    typeSel.onchange = async () => {
      typeSel.disabled = true;
      try {
        const r = await fetch("/api/models/type", {
          method: "POST", headers: authHeaders(),
          body: JSON.stringify({ model: m.name, model_type: typeSel.value }),
        });
        const data = await r.json().catch(() => ({}));
        if (!r.ok) { toast(data.detail || t("models.setType.failed"), true); return; }
        toast(t("models.setType.toast", { name: m.name, type: typeSel.value }));
        refreshModelsPage();
        refreshPerfEstimate();
      } finally { typeSel.disabled = false; }
    };
    roleTd.appendChild(typeSel);
    tr.appendChild(roleTd);

    // The source clips to an ellipsis, with the full value on hover.
    const sourceTd = el("td", "mono clip-cell", m.source || "");
    if (m.source) sourceTd.title = m.source;
    tr.appendChild(sourceTd);
    tr.appendChild(el("td", "mono shrink-cell", fmtSize(m.size_bytes)));
    tr.appendChild(el("td", "mono shrink-cell", fmtModelDate(m.mtime)));

    const actions = el("td", "actions-cell");
    const detail = el("button", "secondary", t("models.action.info"));
    detail.onclick = () => showModelDetail(m.name);
    actions.appendChild(detail);

    // Not gated on isLlm below: any model type can be an externally-referenced
    // file that gets moved. Rendered only when the file is missing.
    if (m.missing) {
      const relocateBtn = el("button", "secondary", t("models.action.relocate"));
      relocateBtn.title = t("models.relocate.title");
      relocateBtn.onclick = async () => {
        const path = await promptText(
          t("models.relocate.prompt", { name: m.name }),
          m.last_path || "");
        if (!path || !path.trim()) return;
        const r = await fetch("/api/models/relocate", {
          method: "POST", headers: authHeaders(),
          body: JSON.stringify({ model: m.name, new_path: path.trim() }),
        });
        const data = await r.json().catch(() => ({}));
        if (r.ok) { toast(t("models.relocate.toast", { name: m.name })); refreshModelsPage(); }
        else toast(data.detail || t("models.relocate.failed"), true);
      };
      actions.appendChild(relocateBtn);
    }

    // Only LLMs support use/alias/remove
    const isLlm = !m.model_type || m.model_type === "llm";
    if (isLlm) {
      if (!m.active) {
        const use = el("button", "primary", t("models.action.use"));
        use.onclick = async () => {
          // An inline busy label on the button itself, alongside the sidebar
          // #status-text pill switchModel() drives.
          const prevLabel = use.textContent;
          use.disabled = true;
          use.textContent = t("models.use.loading");
          try {
            const res = await switchModel(m.name);
            if (!res || (res.status !== "superseded" && res.status !== "cancelled")) {
              toastLoadResult(res, m.name);
              refreshModelsPage();
              // Keep the Settings "Live tuning" VRAM estimate, which defaults
              // to the active model, in sync.
              refreshPerfEstimate();
            }
          } catch (e) {
            toast(t("models.use.loadFailed", { message: e.message }), true);
          } finally { use.disabled = false; use.textContent = prevLabel; }
        };
        actions.appendChild(use);
      }
      const aliasBtn = el("button", "secondary", t("models.action.alias"));
      aliasBtn.onclick = async () => {
        const name = await promptText(t("models.alias.prompt", { name: m.name }));
        if (!name) return;
        const r = await fetch("/api/models/alias", {
          method: "POST", headers: authHeaders(),
          body: JSON.stringify({ model: m.name, alias: name.trim() }),
        });
        const data = await r.json().catch(() => ({}));
        // Report the alias the SERVER stored, not the raw text: aliases are
        // sanitized server-side ("daily driver" -> "daily-driver").
        if (r.ok) {
          toast(t("models.alias.toast", { alias: data.alias || name.trim() }));
          refreshModelsPage();
        } else toast(data.detail || t("models.alias.failed"), true);
      };
      actions.appendChild(aliasBtn);
      const renameBtn = el("button", "secondary", t("models.action.rename"));
      renameBtn.title = t("models.rename.title");
      renameBtn.onclick = async () => {
        const name = await promptText(t("models.rename.prompt", { name: m.name }), m.name);
        if (!name || name.trim() === m.name) return;
        const r = await fetch("/api/models/rename", {
          method: "POST", headers: authHeaders(),
          body: JSON.stringify({ model: m.name, new_name: name.trim() }),
        });
        const data = await r.json().catch(() => ({}));
        // Same as alias above: the server sanitizes the name, so report what it
        // actually stored, not the raw text typed in.
        if (r.ok) {
          let msg = t("models.rename.toast", { name: data.new_name || name.trim() });
          // The server's migration notes - what it updated, and what it could
          // NOT reach (e.g. a per-project .localcoder/config.toml) - are shown
          // to the user, not left in the server log alone.
          if (Array.isArray(data.notes) && data.notes.length) {
            msg += ". " + data.notes.join(" ");
          }
          toast(msg);
          refreshModelsPage();
        } else toast(data.detail || t("models.rename.failed"), true);
      };
      actions.appendChild(renameBtn);
      if (m.loaded) {
        const unload = el("button", "secondary", t("models.action.unload"));
        unload.title = t("models.unload.title");
        unload.onclick = async () => {
          unload.disabled = true;
          try {
            const r = await fetch("/api/models/unload", {
              method: "POST", headers: authHeaders(),
              body: JSON.stringify({ model: m.name }),
            });
            const data = await r.json().catch(() => ({}));
            if (!r.ok) { toast(data.detail || t("models.unload.failed"), true); return; }
            // unload_one_model() (http_server.py) answers HTTP 200 for an
            // in-use engine too - a request is mid-generation against it, so
            // nothing was released. Report that rather than "Unloaded".
            if (data.status === "in_use") {
              toast(t("models.unload.inUse", { name: m.name }), true);
              refreshModelsPage();
              return;
            }
            toast(t("models.unload.toast", { name: m.name }));
            refreshModelsPage();
            refreshPerfEstimate();
          } finally { unload.disabled = false; }
        };
        actions.appendChild(unload);
      }
      if (!m.active) {
        const rm = el("button", "danger", t("models.action.remove"));
        rm.onclick = () => {
          confirmDanger(t("models.remove.confirmTitle", { name: m.name }),
            t("models.remove.confirmBody"), t("models.remove.confirmLabel"), async () => {
              const r = await fetch("/api/models/remove", {
                method: "POST", headers: authHeaders(),
                body: JSON.stringify({ model: m.name }),
              });
              const data = await r.json().catch(() => ({}));
              if (!r.ok) { toast(data.detail || t("models.remove.failed"), true); return; }
              const end = await streamJob(data.job_id, null);
              // "disconnected" (streamJob gave up reconnecting, or the job was
              // already gone) is not the same outcome as the remove having
              // failed. refreshModelsPage() below shows the current state either
              // way.
              if (end.status === "done") {
                toast(t("models.remove.toast", { name: m.name }));
              } else if (end.status === "disconnected") {
                toast(t("models.remove.disconnected"), true);
              } else {
                toast(t("models.remove.failed"), true);
              }
              refreshModelsPage();
            });
        };
        actions.appendChild(rm);
      }
    }
    tr.appendChild(actions);
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
  if (myGen !== _modelsRenderGen) return;
  // One call, so the old table stays on screen until the new one is ready.
  box.replaceChildren(...(hiddenOther ? [otherHiddenNote(hiddenOther), table] : [table]));
}

export async function showModelDetail(name) {
  const r = await fetch(`/v1/models/${encodeURIComponent(name)}`, {
    headers: authHeaders() });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) { toast(data.detail || t("models.detail.lookupFailed"), true); return; }
  openModal(t("models.detail.title", { name }), (body) => {
    // Same three-state read as the list: `model_type` alone would render the
    // route's "llm" default as though it had been chosen.
    const modelUntagged = data.model_type_recorded === false;
    const modelType = modelUntagged ? "unset" : (data.model_type || "llm");
    // rowKind is a stable internal discriminator, never localized: the
    // "Type" row's own display label IS translated, so the special-case
    // branch below cannot key off it directly.
    const rows = [
      ["path", t("models.detail.path"), data.path],
      ["type", t("models.detail.type"), null],
      ["source", t("models.detail.source"), data.source],
      ["size", t("models.detail.size"), fmtSize(data.size_bytes)],
      ["sha256", t("models.detail.sha256"), data.sha256 || t("models.detail.sha256Pending")],
      ["aliases", t("models.detail.aliases"),
        data.aliases.length ? data.aliases.join(", ") : t("chat.none")],
      ["status", t("models.detail.status"),
        data.active ? (data.loaded ? t("models.detail.statusActiveLoaded")
                                    : t("models.detail.statusActiveNotLoaded"))
                    : t("models.detail.statusRegistered")],
    ];
    for (const [rowKind, label, v] of rows) {
      const row = el("div", "log-entry");
      row.appendChild(el("span", "t", label));
      if (rowKind === "type") {
        row.appendChild(el("span", "type-badge type-" + modelType,
                            modelUntagged ? t("models.typeSelect.notSet") : modelType));
        // Rendered beside the type, not as a row of its own, so an unconfirmed
        // capability renders nothing at all.
        const visBadge = visionBadge(data.vision);
        if (visBadge) row.appendChild(visBadge);
      } else row.appendChild(document.createTextNode(String(v)));
      body.appendChild(row);
    }
  });
}

/* ---- model discovery (HuggingFace search + VRAM fit badges) ---- */

export function fmtCount(n) {
  if (n == null) return "0";
  if (n >= 1e6) return (n / 1e6).toFixed(1) + "M";
  if (n >= 1e3) return (n / 1e3).toFixed(1) + "k";
  return String(n);
}

// Parameter COUNT (e.g. gguf.total / safetensors.total from discover.py), not a
// byte size - fmtSize is the formatter for bytes.
export function fmtParamCount(n) {
  if (!n) return "";
  if (n >= 1e9) return (n / 1e9).toFixed(1) + "B params";
  if (n >= 1e6) return (n / 1e6).toFixed(1) + "M params";
  return String(n) + " params";
}

export const FIT_TEXT = { "fits": "fits your VRAM", "tight": "tight fit",
                   "too-big": "needs partial CPU offload" };

// moe: "confirmed" (the model's own architecture header says MoE) or "likely"
// (a name-pattern guess, e.g. "8x7B"/"A3B" in the repo id - see discover.py's
// _moe_signal). The label and tooltip keep the two distinct.
//
// Values are catalog key names; t() resolves them at each render site below.
const MOE_LABEL = { confirmed: "models.moe.confirmed", likely: "models.moe.likely" };

// Model CAPABILITY pills - what a model can DO, separate from the
// .fit.fits/.tight/.too-big scale, which grades file SIZE against VRAM. Vision
// is the only capability with a detector (registry.model_vision_capability, the
// same lookup the load path uses).
//
// Values are catalog key names; t() resolves them at each render site below.
const CAP_LABEL = { vision: "models.cap.vision" };
const CAP_TITLE = { vision: "models.cap.visionTitle" };

/** The vision pill for a `vision` field, or null when there must be no pill.
 *
 *  Shared by the list row and the detail modal. Tests `=== true`, never
 *  truthiness: the server sends true, false, or no key at all, and the absent
 *  case means the model's files could not be inspected. false and absent both
 *  render nothing. */
function visionBadge(v) {
  if (v !== true) return null;
  const badge = el("span", "cap-badge cap-vision", t(CAP_LABEL.vision));
  badge.title = t(CAP_TITLE.vision);
  return badge;
}
// Catalog key name; t() resolves it at each render site below.
const MOE_TITLE = { likely: "models.moe.likelyTitle" };

export const FMT_LABEL = { gguf: "GGUF", hf: "HF" };

/** Fetch the GPU list AND the configured split once per search/files-load, for
 *  the split-fit hint and the VRAM-basis caption below. Empty on any failure
 *  (server unreachable, no scope). */
async function _gpuInfo() {
  try {
    const r = await fetch("/api/gpus", { headers: authHeaders() });
    if (!r.ok) return { gpus: [], gpu_split_indices: [] };
    const data = await r.json();
    return {
      gpus: Array.isArray(data.gpus) ? data.gpus : [],
      gpu_split_indices: Array.isArray(data.gpu_split_indices) ? data.gpu_split_indices : [],
    };
  } catch (e) { return { gpus: [], gpu_split_indices: [] }; }
}

/** The GPU array alone, for the split hint and row rendering. */
async function _splitGpus() { return (await _gpuInfo()).gpus; }

/** Caption naming what the fit-badge VRAM number is. Mirrors
 *  discover.vram_capacity: the number is COMBINED only when 2+ configured split
 *  indices map to detected devices; otherwise it is the single main GPU's.
 *  `gpuInfo` is {gpus, gpu_split_indices} from _gpuInfo(). */
export function vramBasisCaption(totalBytes, gpuInfo) {
  const gib = (totalBytes / GIB).toFixed(0);
  const gpus = Array.isArray(gpuInfo?.gpus) ? gpuInfo.gpus : [];
  const rawSplit = Array.isArray(gpuInfo?.gpu_split_indices) ? gpuInfo.gpu_split_indices : [];
  // Dedup and keep only indices that map to a detected device - the same
  // resolve_gpu_split validation vram_capacity() applies.
  const split = [...new Set(rawSplit.filter((i) => gpus.some((g) => g.index === i)))];
  const tail = " (weights + ~1.5 GB overhead).";
  let basis;
  if (split.length >= 2) {
    basis = `your ${gib} GB VRAM combined across ${split.length} GPUs`;
  } else if (gpus.length > 1) {
    basis = `your main GPU's ${gib} GB (set a split in Settings to use all ${gpus.length})`;
  } else {
    basis = `your ${gib} GB VRAM`;
  }
  return `Badges compare each file against ${basis}${tail}`;
}

/** Non-blocking "might not fit on one GPU, but may fit split across them" hint,
 *  or "" when not applicable: unknown size, fewer than 2 GPUs detected, it
 *  already fits the single largest device, or it would not fit even split
 *  across every device. A rough client-side suggestion; the server's own fit
 *  badges (`m.fit`/`f.fit`) are the authoritative numbers, and a real split load
 *  applies its own per-device VRAM check (gpu_split_shortfall). Uses the same
 *  need-math as the fit badge (weights * 1.10 + ~1.5 GB overhead). Enables
 *  nothing - it only points at the "Split across GPUs" control. */
export function splitFitHint(sizeBytes, gpus) {
  if (!sizeBytes || !Array.isArray(gpus) || gpus.length < 2) return "";
  const frees = gpus.map((g) => (typeof g.free === "number" ? g.free : g.total));
  if (frees.some((f) => typeof f !== "number")) return "";
  const need = sizeBytes * 1.10 + 1.5e9;   // same basis as the fit badge (fit_label)
  const largest = Math.max(...frees);
  const total = frees.reduce((a, b) => a + b, 0);
  if (need <= largest || need > total) return "";
  return `may not fit on one GPU, but may fit split across your ${gpus.length} GPUs - `
       + "see Settings > Split across GPUs";
}

export function discFormats() {
  // The search-page FORMAT toggles -> the formats list. "hf" is the non-gguf /
  // safetensors world (labelled "Safetensors" in the UI). Defaults to gguf when
  // the checkboxes are absent. Empty (both unchecked) is a real state the
  // caller handles.
  const g = $("disc-fmt-gguf"), h = $("disc-fmt-hf");
  if (!g && !h) return ["gguf"];
  const out = [];
  if (!g || g.checked) out.push("gguf");
  if (h && h.checked) out.push("hf");
  return out;
}

const ALL_SEARCH_TYPES =
  ["llm", "embedding", "diffusion-unet", "text-encoder", "vae", "lora", "unknown"];

export function discTypes() {
  // The search-page TYPE checkboxes -> the model_types list, independent of the
  // Registered-models tab. Defaults to all types when the checkboxes are
  // absent. Empty (none ticked) is a real state the caller surfaces, same as no
  // format.
  const boxes = [...document.querySelectorAll(".disc-type")];
  if (!boxes.length) return ALL_SEARCH_TYPES.slice();
  return boxes.filter((b) => b.checked).map((b) => b.value);
}

export function discSource() {
  const s = $("disc-source");
  return s && s.value === "civitai" ? "civitai" : "hf";
}

export function discCivitaiTypes() {
  // Mirrors discTypes() for the CivitAI-only type chips (localm/model_manager/
  // sources.py CIVITAI_TYPE_MAP keys). Defaults to all when absent; empty (none
  // ticked) is a real state the caller surfaces, same as discTypes().
  const boxes = [...document.querySelectorAll(".disc-civitai-type")];
  return boxes.filter((b) => b.checked).map((b) => b.value);
}

export function discCivitaiNsfw() {
  const b = $("disc-civitai-nsfw");
  return !!(b && b.checked);
}

export function discCivitaiLegacyFormats() {
  const b = $("disc-civitai-legacy");
  return !!(b && b.checked);
}

// The model_type hint carried into a pull for a chosen discovery result: the
// badged detected_type, or - when HF gave no confident type ("unknown") and the
// search is narrowed to exactly ONE type - that single type. Null otherwise, so
// the pull auto-detects.
function resolveTypeHint(detectedType) {
  if (detectedType && detectedType !== "unknown") return detectedType;
  const types = discTypes();
  return types.length === 1 ? types[0] : null;
}

function showHfHint() {
  const hint = $("disc-hf-hint");
  if (!hint) return;
  // Non-blocking: HF (transformers) models still download, they just cannot run
  // until torch + transformers are installed.
  hint.textContent = "No transformers runtime detected. HF models will download, "
    + "but need the [gpu] extra (torch + transformers) installed to run.";
  hint.style.display = "block";
}
function hideHfHint() {
  const hint = $("disc-hf-hint");
  if (hint) hint.style.display = "none";
}

// A bare owner/repo (no :file.gguf) tells `localm pull` to fetch the WHOLE
// transformers repo -> the HF backend. Prefills the Add box for the user to
// confirm.
function prefillHfPull(repo, detectedType) {
  $("pull-spec").value = repo;
  $("pull-name").value = repo.split("/").pop();
  const typeHint = resolveTypeHint(detectedType);
  pendingPullTypeHint = typeHint ? { spec: repo, type: typeHint } : null;
  const mmprojSelect = $("pull-mmproj");
  if (mmprojSelect) { mmprojSelect.replaceChildren(); mmprojSelect.style.display = "none"; }
  const nameInput = $("pull-name");
  // scrollIntoView is absent in some environments (e.g. jsdom); guard so the
  // prefill never throws where it is unimplemented.
  if (typeof nameInput.scrollIntoView === "function") {
    nameInput.scrollIntoView({ behavior: "smooth", block: "center" });
  }
  nameInput.focus();
  nameInput.select();
  toast("Review the alias, then click Add to download the full HF model");
}

// One search-result repo: name + type badge + per-format badge(s) + the right
// pull affordance (GGUF -> a per-quant file list; HF -> a whole-repo add). A
// repo tagged with both formats gets both. Each pull affordance resolves its
// model_type hint at click time from the result's detected type and the current
// Type checkboxes (resolveTypeHint).
function discRepoRow(m, gpus) {
  const row = el("div", "disc-repo");
  const head = el("div", "head");
  head.appendChild(iconEl("models", "ic ic-model"));
  head.appendChild(el("span", "name", m.id));
  // Best-effort HF classification, DISPLAY ONLY - never gates whether a result
  // is shown.
  if (m.detected_type) {
    head.appendChild(el("span", "type-badge type-" + m.detected_type, m.detected_type));
  }
  // Architecture family, MoE-ness and param count from discover.py's
  // classified-row fields - display only, never gating which results show, and
  // outside the type-badge colour palette.
  if (m.architecture) {
    head.appendChild(el("span", "arch-badge", m.architecture));
  }
  if (m.moe) {
    const moeBadge = el("span", "moe-badge moe-" + m.moe, MOE_LABEL[m.moe] ? t(MOE_LABEL[m.moe]) : "MoE");
    if (MOE_TITLE[m.moe]) moeBadge.title = t(MOE_TITLE[m.moe]);
    head.appendChild(moeBadge);
  }
  if (m.param_count) {
    head.appendChild(el("span", "param-count", fmtParamCount(m.param_count)));
  }
  const fmts = Array.isArray(m.formats) ? m.formats : ["gguf"];
  for (const f of fmts) head.appendChild(el("span", "fmt-badge fmt-" + f, FMT_LABEL[f] || f));
  // HF repos pull whole, so show total size + a VRAM fit badge inline (from the
  // server's safetensors param estimate), or "size unknown" when there is no
  // estimate. GGUF results are sized per-quant in the files expander instead.
  if (fmts.includes("hf")) {
    if (m.size_bytes) {
      head.appendChild(el("span", "disc-hf-size", fmtSize(m.size_bytes)));
      if (m.fit) head.appendChild(el("span", "fit " + m.fit, FIT_TEXT[m.fit]));
      // "fits"/"tight" already account for a configured split (the server sums
      // capacity across it - see discover.vram_capacity), so no split hint.
      const hint = (m.fit === "fits" || m.fit === "tight")
        ? "" : splitFitHint(m.size_bytes, gpus);
      if (hint) head.appendChild(el("span", "sub split-hint", hint));
    } else {
      head.appendChild(el("span", "disc-hf-size sub", "size unknown"));
    }
  }
  // Downloads + likes as inline SVGs.
  const meta = el("span", "meta disc-stats");
  meta.appendChild(iconEl("download", "meta-ic"));
  meta.appendChild(el("span", "", fmtCount(m.downloads)));
  meta.appendChild(iconEl("heart", "meta-ic"));
  meta.appendChild(el("span", "", fmtCount(m.likes)));
  head.appendChild(meta);
  const filesBox = el("div", "files");
  if (fmts.includes("gguf")) {
    const btn = el("button", "btn-secondary", "files");
    btn.onclick = () => discoverFiles(m.id, filesBox, btn, gpus, m.detected_type);
    head.appendChild(btn);
  }
  if (fmts.includes("hf")) {
    const btn = el("button", "btn-secondary", "add full repo");
    btn.onclick = () => prefillHfPull(m.id, m.detected_type);
    head.appendChild(btn);
  }
  row.appendChild(head);
  row.appendChild(filesBox);
  return row;
}

// CivitAI's own license flag system (allowCommercialUse/allowDerivatives/
// allowNoCredit/allowDifferentLicense), never an SPDX string - CivitAI has no
// SPDX equivalent, so this is displayed as what it is rather than mapped onto
// discRepoRow's HF `license` shape. true/false/absent (three-state, since the
// API may omit a flag) each get their own word - never collapsed to "yes/no".
function civitaiFlag(value) {
  if (value === true) return "yes";
  if (value === false) return "no";
  return "unknown";
}

// allowCommercialUse is an array of the specific permitted uses (e.g.
// ["Image", "Sell"]), not a boolean like the other three flags - shown as the
// actual permitted uses rather than collapsed into a lossy yes/no, and an
// EMPTY array is a real "no commercial use", not "unknown" (only an absent
// field is unknown).
function civitaiCommercialUse(value) {
  if (Array.isArray(value)) return value.length ? value.join(", ") : "no";
  if (value === true) return "yes";
  if (value === false) return "no";
  return "unknown";
}

// One CivitAI search result: name + type badge + license-flag summary + NSFW
// badge (only when flagged; a `minor`-flagged model never reaches this - the
// server hard-excludes it regardless of the NSFW toggle) + downloads + a
// "files" expander for its latest model VERSION. Unlike discRepoRow, CivitAI
// has no per-repo GGUF/HF format split - one row always means one model.
function discCivitaiRow(m) {
  const row = el("div", "disc-repo");
  const head = el("div", "head");
  head.appendChild(iconEl("models", "ic ic-model"));
  head.appendChild(el("span", "name", m.name || "(unnamed)"));
  if (m.type) head.appendChild(el("span", "arch-badge", m.type));
  if (m.nsfw === true) {
    const badge = el("span", "type-badge type-unknown",
      typeof m.nsfwLevel === "number" ? `NSFW (level ${m.nsfwLevel})` : "NSFW");
    badge.title = "CivitAI-flagged NSFW content. Only shown because the NSFW toggle is on.";
    head.appendChild(badge);
  }
  const flags = el("span", "sub civitai-license",
    `commercial use: ${civitaiCommercialUse(m.allowCommercialUse)} · `
    + `derivatives: ${civitaiFlag(m.allowDerivatives)} · `
    + `credit required: ${civitaiFlag(m.allowNoCredit === undefined ? undefined : !m.allowNoCredit)} · `
    + `different license: ${civitaiFlag(m.allowDifferentLicense)}`);
  flags.title = "CivitAI's own permission flags, not an SPDX license - shown as CivitAI reports them.";
  const meta = el("span", "meta disc-stats");
  meta.appendChild(iconEl("download", "meta-ic"));
  meta.appendChild(el("span", "", fmtCount((m.stats || {}).downloadCount)));
  head.appendChild(meta);
  const versions = Array.isArray(m.modelVersions) ? m.modelVersions : [];
  const latestVersion = versions[0] && versions[0].id;
  const filesBox = el("div", "files");
  if (latestVersion != null) {
    const btn = el("button", "btn-secondary", "files");
    btn.onclick = () => discoverCivitaiFiles(latestVersion, filesBox, btn);
    head.appendChild(btn);
  }
  row.appendChild(head);
  row.appendChild(flags);
  row.appendChild(filesBox);
  return row;
}

// A CivitAI safety-scan status pill: CivitAI's own upload-time scan result,
// evidence rather than a guarantee (per ADR-0015) - so it borrows the MoE-pill
// "inferred, not confirmed" visual treatment (dashed border) for anything that
// is not an affirmative clean result, rather than reading as a green light.
function civitaiScanBadge(status) {
  const s = (status || "").trim();
  const clean = /^success$/i.test(s);
  const badge = el("span", "moe-badge " + (clean ? "moe-confirmed" : "moe-likely"),
    "scan: " + (s || "unknown"));
  badge.title = "CivitAI's own upload-time safety scan - evidence, not a guarantee.";
  return badge;
}

export async function discoverCivitaiFiles(versionId, filesBox, btn) {
  if (filesBox.childElementCount) {            // toggle collapse
    filesBox.replaceChildren();
    return;
  }
  if (btn) btn.disabled = true;
  filesBox.replaceChildren(el("div", "sub", "loading file list…"));
  try {
    const legacy = discCivitaiLegacyFormats();
    const r = await fetch("/api/discover/files?repo=" + encodeURIComponent(versionId)
                          + "&source=civitai&legacy_formats=" + (legacy ? "true" : "false"),
                          { headers: authHeaders() });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || r.statusText);
    filesBox.replaceChildren();
    const files = Array.isArray(data.files) ? data.files : [];
    if (!files.length) {
      filesBox.appendChild(el("div", "sub",
        "(no downloadable files - try Show legacy formats if you expect a "
        + "riskier format)"));
    }
    for (const f of files) {
      const row = el("div", "disc-file");
      const fmt = (f.metadata || {}).format || "?";
      row.appendChild(el("span", "quant", fmt));
      const sizeKb = f.sizeKB || 0;
      row.appendChild(el("span", "mono", (sizeKb / (1024 * 1024)).toFixed(2) + " GB"));
      row.appendChild(civitaiScanBadge(f.virusScanResult));
      row.appendChild(el("span", "fname", f.name || String(f.id)));
      const pull = el("button", "btn-secondary", "pull");
      pull.onclick = () => {
        $("pull-spec").value = `civitai:${versionId}:${f.id}`;
        $("pull-name").value = (f.name || `civitai-${versionId}`).replace(/\.[^.]+$/, "");
        const nameInput = $("pull-name");
        if (typeof nameInput.scrollIntoView === "function") {
          nameInput.scrollIntoView({ behavior: "smooth", block: "center" });
        }
        nameInput.focus();
        nameInput.select();
        toast("Review the alias, then click Add to download from CivitAI");
      };
      row.appendChild(pull);
      filesBox.appendChild(row);
    }
  } catch (e) {
    filesBox.replaceChildren(el("div", "sub", "Failed: " + e.message));
  } finally {
    if (btn) btn.disabled = false;
  }
}

export async function discoverSearch() {
  if (discSource() === "civitai") return _discoverSearchCivitai();
  const box = $("disc-results");
  const formats = discFormats();
  const types = discTypes();
  if (!formats.length) {
    box.replaceChildren(el("div", "sub",
      "Select at least one format (GGUF or Safetensors) to search."));
    hideHfHint();
    return;
  }
  if (!types.length) {
    box.replaceChildren(el("div", "sub",
      "Select at least one model type to search for."));
    hideHfHint();
    return;
  }
  box.replaceChildren(el("div", "sub", "searching HuggingFace…"));
  $("disc-search").disabled = true;
  try {
    const q = $("disc-query").value.trim();
    const r = await fetch("/api/discover/search?q=" + encodeURIComponent(q)
                          + "&formats=" + encodeURIComponent(formats.join(","))
                          + "&types=" + encodeURIComponent(types.join(",")),
                          { headers: authHeaders() });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || r.statusText);
    // One GPU probe drives both the basis caption and the per-row split hint.
    const gpuInfo = await _gpuInfo();
    $("disc-vram").textContent = data.vram.total
      ? vramBasisCaption(data.vram.total, gpuInfo)
      : "No GPU VRAM detected - sizes shown without fit badges.";
    // Show the HF-runtime hint only when HF is actually being searched and the
    // runtime is missing (backend flag comes from the server probe).
    if (formats.includes("hf") && data.hf_backend_available === false) showHfHint();
    else hideHfHint();
    box.replaceChildren();
    if (!data.results.length) {
      box.appendChild(el("div", "sub", "(no matching repos found)"));
      return;
    }
    // A persistent legend for the dashed "MoE?" pill, shown once and only when
    // a result on screen carries that inferred-not-confirmed signal.
    if (data.results.some((m) => m.moe === "likely")) {
      box.appendChild(el("div", "sub moe-legend",
        "MoE? = inferred from the model's name, not confirmed by its own header"));
    }
    for (const m of data.results) box.appendChild(discRepoRow(m, gpuInfo.gpus));
  } catch (e) {
    box.replaceChildren(el("div", "sub", "Search failed: " + e.message));
  } finally {
    $("disc-search").disabled = false;
  }
}

async function _discoverSearchCivitai() {
  const box = $("disc-results");
  const types = discCivitaiTypes();
  if (!types.length) {
    box.replaceChildren(el("div", "sub",
      "Select at least one model type to search for."));
    return;
  }
  hideHfHint();
  box.replaceChildren(el("div", "sub", "searching CivitAI…"));
  $("disc-search").disabled = true;
  try {
    const q = $("disc-query").value.trim();
    const nsfw = discCivitaiNsfw();
    const r = await fetch("/api/discover/search?q=" + encodeURIComponent(q)
                          + "&source=civitai"
                          + "&types=" + encodeURIComponent(types.join(","))
                          + "&nsfw=" + (nsfw ? "true" : "false"),
                          { headers: authHeaders() });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || r.statusText);
    $("disc-vram").textContent = "";
    box.replaceChildren();
    const results = Array.isArray(data.results) ? data.results : [];
    if (!results.length) {
      box.appendChild(el("div", "sub", "(no matching CivitAI models found)"));
      return;
    }
    for (const m of results) box.appendChild(discCivitaiRow(m));
  } catch (e) {
    box.replaceChildren(el("div", "sub", "Search failed: " + e.message));
  } finally {
    $("disc-search").disabled = false;
  }
}

export async function discoverFiles(repo, filesBox, btn, gpus, detectedType) {
  if (filesBox.childElementCount) {            // toggle collapse
    filesBox.replaceChildren();
    return;
  }
  btn.disabled = true;
  filesBox.replaceChildren(el("div", "sub", "loading file list…"));
  if (!Array.isArray(gpus)) gpus = await _splitGpus();   // not supplied by the caller
  try {
    const r = await fetch("/api/discover/files?repo=" + encodeURIComponent(repo),
                          { headers: authHeaders() });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || r.statusText);
    filesBox.replaceChildren();
    
    const showMmproj = localStorage.getItem("localm.showMmprojFiles") === "true";
    let filesToShow = data.files;
    if (showMmproj && data.mmprojs && data.mmprojs.length > 0) {
      filesToShow = filesToShow.concat(data.mmprojs);
    }
    
    for (const f of filesToShow) {
      const row = el("div", "disc-file");
      row.appendChild(el("span", "quant", f.quant || "?"));
      const desc = `${(f.size_bytes / GIB).toFixed(1)} GB` +
        (f.n_parts > 1 ? ` (${f.n_parts} parts)` : "");
      row.appendChild(el("span", "mono", desc));
      if (f.fit) row.appendChild(el("span", "fit " + f.fit, FIT_TEXT[f.fit]));
      // "fits"/"tight" already account for a configured split, so no split hint.
      const splitHint = (f.fit === "fits" || f.fit === "tight")
        ? "" : splitFitHint(f.size_bytes, gpus);
      if (splitHint) row.appendChild(el("span", "sub split-hint", splitHint));
      row.appendChild(el("span", "fname", f.file));
      const pull = el("button", "btn-secondary", "pull");
      pull.onclick = () => {
        // Prefill the pull form; the user confirms (and can set an alias)
        // before anything downloads. The suggested alias mirrors the server's
        // default name (file name without .gguf).
        $("pull-spec").value = `${repo}:${f.file}`;
        $("pull-name").value = f.file.replace(/\.gguf$/i, "");
        // Only a REGULAR file's pull carries a type hint, never one drawn from
        // data.mmprojs (the same object reference, merged in by the "show mmproj
        // files" toggle above): the search Type checkboxes have no mmproj entry.
        // The hint resolves at click time from the repo's detected type and
        // those checkboxes.
        const isMmproj = Array.isArray(data.mmprojs) && data.mmprojs.includes(f);
        const typeHint = isMmproj ? null : resolveTypeHint(detectedType);
        pendingPullTypeHint = typeHint
          ? { spec: `${repo}:${f.file}`, type: typeHint } : null;

        // Populate the mmproj dropdown
        const mmprojSelect = $("pull-mmproj");
        mmprojSelect.replaceChildren();
        const noOption = el("option", "", "No vision projector");
        noOption.value = "";
        mmprojSelect.appendChild(noOption);
        
        if (data.mmprojs && data.mmprojs.length > 0) {
          mmprojSelect.style.display = "inline-block";
          
          // Guess the best mmproj by filename-token overlap.
          let bestMatch = "";
          let bestScore = 0;
          
          for (const m of data.mmprojs) {
            const opt = el("option", "", m.file);
            opt.value = `${repo}:${m.file}`;
            mmprojSelect.appendChild(opt);
            
            const fTokens = f.file.toLowerCase().replace(".gguf", "").split("-");
            const mTokens = m.file.toLowerCase().replace(".gguf", "").split("-");
            let score = 0;
            for (const t of fTokens) {
              if (mTokens.includes(t)) score++;
            }
            if (score > bestScore) {
              bestScore = score;
              bestMatch = opt.value;
            }
          }
          if (bestMatch && !f.file.toLowerCase().includes("mmproj")) {
            mmprojSelect.value = bestMatch;
          }
        } else {
          mmprojSelect.style.display = "none";
        }

        const nameInput = $("pull-name");
        // scrollIntoView is absent in some environments (e.g. jsdom); guard so
        // this never throws where it is unimplemented.
        if (typeof nameInput.scrollIntoView === "function") {
          nameInput.scrollIntoView({ behavior: "smooth", block: "center" });
        }
        nameInput.focus();
        nameInput.select();
        toast("Review the alias, then click Pull to start the download");
      };
      row.appendChild(pull);
      filesBox.appendChild(row);
    }
  } catch (e) {
    filesBox.replaceChildren(el("div", "sub", "Failed: " + e.message));
  } finally {
    btn.disabled = false;
  }
}

$("disc-search").onclick = discoverSearch;
$("disc-query").addEventListener("keydown", (e) => {
  if (e.key === "Enter") discoverSearch();
});

const DISC_SOURCE_PLACEHOLDER = {
  hf: "search HuggingFace - empty shows the most downloaded",
  civitai: "search CivitAI - empty shows the most downloaded",
};

function applyDiscSource() {
  const source = discSource();
  const hfBar = $("disc-hf-filters"), civitaiBar = $("disc-civitai-filters");
  if (hfBar) hfBar.style.display = source === "hf" ? "" : "none";
  if (civitaiBar) civitaiBar.style.display = source === "civitai" ? "" : "none";
  const q = $("disc-query");
  if (q) q.placeholder = DISC_SOURCE_PLACEHOLDER[source];
  hideHfHint();
  const box = $("disc-results");
  if (box) box.replaceChildren();
  const vram = $("disc-vram");
  if (vram) vram.textContent = "";
}
if ($("disc-source")) {
  $("disc-source").addEventListener("change", applyDiscSource);
  applyDiscSource();
}

// Mirror a filter checkbox's state onto its chip label as an `.on` class, which
// is what style.css colours.
function _syncChip(box) {
  const chip = box.closest(".disc-chip");
  if (chip) chip.classList.toggle("on", box.checked);
}

// Restore + persist a search filter checkbox (all default on). Keeps its chip in
// sync, and re-runs the search on change when results are already showing.
function _bindDiscToggle(box, key) {
  if (!box) return;
  const saved = localStorage.getItem(key);
  if (saved !== null) box.checked = saved === "true";
  _syncChip(box);
  box.addEventListener("change", () => {
    localStorage.setItem(key, box.checked ? "true" : "false");
    _syncChip(box);
    const results = $("disc-results");
    if (results && results.childElementCount) discoverSearch();
  });
}
_bindDiscToggle($("disc-fmt-gguf"), "localm.discFmtGguf");
_bindDiscToggle($("disc-fmt-hf"), "localm.discFmtHf");
// The explicit model-TYPE checkboxes, persisted per type (keyed by value, e.g.
// localm.discType.vae).
for (const box of document.querySelectorAll(".disc-type")) {
  _bindDiscToggle(box, "localm.discType." + box.value);
}
for (const box of document.querySelectorAll(".disc-civitai-type")) {
  _bindDiscToggle(box, "localm.discCivitaiType." + box.value);
}
// NSFW and legacy-formats are NOT persisted like the type/format chips above -
// both must start off every session (ADR-0015: off by default, an explicit
// per-search choice), never remembered from a prior search.
if ($("disc-civitai-nsfw")) {
  _syncChip($("disc-civitai-nsfw"));
  $("disc-civitai-nsfw").addEventListener("change", () => {
    _syncChip($("disc-civitai-nsfw"));
    const results = $("disc-results");
    if (results && results.childElementCount) discoverSearch();
  });
}
if ($("disc-civitai-legacy")) _syncChip($("disc-civitai-legacy"));

// Curated model shortcuts (`localm pull <alias>`, see MODEL_SHORTCUTS in
// model_manager/registry.py) - a fixed local list, so it works with
// net_mode=off. Picking one prefills the Add box with the resolved repo:file,
// not the bare alias.
//
// Fetched from refreshModelsPage() above, not eagerly at module load: this is an
// authenticated read (MODELS_READ) and must not run before the boot path
// confirms this client is authed. onchange fires no request of its own and is
// wired here unconditionally.
async function _loadPullShortcuts() {
  const sel = $("pull-shortcut");
  if (!sel) return;
  try {
    const r = await fetch("/api/models/shortcuts", { headers: authHeaders() });
    if (!r.ok) return;
    const data = await r.json();
    const shortcuts = Array.isArray(data.shortcuts) ? data.shortcuts : [];
    for (const s of shortcuts) {
      const opt = el("option", null, `${s.alias} (${s.size || "size unknown"})`);
      opt.value = s.spec;
      opt.dataset.alias = s.alias;
      sel.appendChild(opt);
    }
  } catch (e) {
    // Best-effort convenience list - the spec field still works typed by hand.
  }
}

if ($("pull-shortcut")) {
  $("pull-shortcut").onchange = (e) => {
    const opt = e.target.selectedOptions[0];
    if (!opt || !opt.value) return;
    $("pull-spec").value = opt.value;
    $("pull-name").value = opt.dataset.alias || "";
    e.target.selectedIndex = 0;   // revert to the placeholder
  };
}

// Pick a folder on this machine and drop its path into the spec field; the
// /api/models/pull endpoint accepts a local folder/file path.
document.addEventListener("click", async (e) => {
  if (e.target && e.target.id === "pull-browse") {
    const spec = $("pull-spec");
    if (!spec) return;
    const dir = await pickDirectory("Pick a folder that holds the model(s)",
                                    spec.value.trim());
    if (dir) spec.value = dir;
  }
});

// Restart the server in place from Settings: the backend unloads the model, then
// re-execs the same process, so it comes back on the same port.
if ($("server-restart")) {
  $("server-restart").onclick = () => {
    confirmDanger("Restart the server?",
      "This restarts the LocaLM server (the model is unloaded first, then reloaded). " +
      "It will be briefly unavailable, then reconnect automatically.",
      "Restart", async () => {
        const before = window.fetchWhoami ? await window.fetchWhoami() : null;
        try {
          const r = await fetch("/v1/server/restart",
                                { method: "POST", headers: authHeaders() });
          if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
          toast("Server restarting…");
          // The reconnect overlay polls /whoami's instance_id until a NEW
          // process answers, rather than reloading on a bounded count of
          // reachable polls that the still-shutting-down old process can
          // satisfy on its own.
          if (window.onServerUnreachable) {
            setTimeout(() => onServerUnreachable({
              priorInstanceId: before && before.instance_id
            }), 800);
          }
        } catch (e) { toast("Could not restart: " + e.message, true); }
      });
  };
}

// Shut the server down cleanly from Settings; the backend unloads the model
// before exit.
if ($("server-shutdown")) {
  $("server-shutdown").onclick = () => {
    confirmDanger("Shut down the server?",
      "This stops the LocaLM server (the model is unloaded first). You will need to " +
      "start it again from your launcher or terminal.",
      "Shut down", async () => {
        try {
          const r = await fetch("/v1/server/shutdown",
                                { method: "POST", headers: authHeaders() });
          if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
          toast("Server shutting down…");
          // Show the reconnect overlay while the server goes away.
          if (window.onServerUnreachable) setTimeout(() => onServerUnreachable(), 800);
        } catch (e) { toast("Could not shut down: " + e.message, true); }
      });
  };
}

// The GUI form of `localm ps` / `localm stop <id>`. /api/instances returns every
// live instance on this machine, this one included and flagged `self`; the card
// below filters `self` out client-side. A row with `same_install: false` belongs
// to another install and gets a label instead of a Stop button.
//
// Guarded by a generation counter and written with one replaceChildren(), same
// as refreshModelsPage above: a stale in-flight fetch never overwrites a newer
// render, and the box never sits empty across an await.
let _instancesRenderGen = 0;

export async function refreshInstancesCard() {
  const box = $("instances-list");
  if (!box) return;
  const myGen = ++_instancesRenderGen;
  let rows;
  try {
    const r = await fetch("/api/instances", { headers: authHeaders() });
    if (myGen !== _instancesRenderGen) return;
    if (!r.ok) { box.replaceChildren(); return; }   // e.g. a read-only key: hide
    const data = await r.json().catch(() => ({ instances: [] }));
    rows = (data.instances || []).filter((i) => !i.self);
  } catch (e) {
    return;   // transient error - leave the card as it was
  }
  if (myGen !== _instancesRenderGen) return;

  if (!rows.length) {
    box.replaceChildren(emptyState("folder", "No other instances running",
      "Only the server behind this page is running right now."));
    return;
  }

  const table = el("table", "data-table");
  const thead = el("thead");
  const hr = el("tr");
  for (const label of ["Directory", "Surface", "Address", "Status", ""]) {
    hr.appendChild(el("th", "", label));
  }
  thead.appendChild(hr);
  const tbody = el("tbody");
  for (const inst of rows) {
    const tr = el("tr");
    tr.appendChild(el("td", "", inst.root_dir || "(not reported)"));
    tr.appendChild(el("td", "", inst.mode || "?"));
    tr.appendChild(el("td", "", inst.address || ""));
    tr.appendChild(el("td", "", inst.alive ? "live" : "no answer"));
    const actionsTd = el("td");
    if (inst.same_install === false) {
      const note = el("span", "instances-foreign", "other install");
      note.title = "This server belongs to a different LocaLM install, which " +
        "keeps its own data folder. Stop it from that install's own window, or " +
        "from the terminal it was started in.";
      actionsTd.appendChild(note);
      tr.appendChild(actionsTd);
      tbody.appendChild(tr);
      continue;
    }
    const stopBtn = el("button", "btn-secondary btn-danger", "Stop");
    stopBtn.onclick = () => {
      confirmDanger("Stop this instance?",
        `This stops the LocaLM server at ${inst.address} (serving ` +
        `${inst.root_dir || "an unknown directory"}). Its model is unloaded ` +
        "first when possible.",
        "Stop", async () => {
          stopBtn.disabled = true;
          try {
            const resp = await fetch(
              `/api/instances/${encodeURIComponent(inst.instance_id)}/stop`,
              { method: "POST", headers: authHeaders() });
            const d = await resp.json().catch(() => ({}));
            if (!resp.ok) throw new Error(d.detail || resp.statusText);
            toast(`Stopped ${inst.root_dir || inst.instance_id}`);
            refreshInstancesCard();
          } catch (e) {
            toast("Could not stop: " + e.message, true);
            stopBtn.disabled = false;
          }
        });
    };
    actionsTd.appendChild(stopBtn);
    tr.appendChild(actionsTd);
    tbody.appendChild(tr);
  }
  table.appendChild(thead);
  table.appendChild(tbody);
  box.replaceChildren(table);
}
window.refreshInstancesCard = refreshInstancesCard;

// File a bug report from Settings. "Save report" writes an editable markdown
// report to the data folder (safe snapshot, optional log tail - never secrets or
// chat). "Send to maintainer", shown only when capabilities.bugreport_upload
// names an upload endpoint, ALSO files it as a GitHub issue through the proxy. A
// failed upload is reported as a failure; the file is still saved.
// The most recent saved report's markdown + filename, for "Download report".
let _lastBugReport = null;

function _showBugActions({ retry, download }) {
  const r = $("bug-retry"), d = $("bug-download");
  if (r) r.hidden = !retry;
  if (d) d.hidden = !download;
}

// Download the last saved report as a .md.
function downloadBugReport() {
  if (!_lastBugReport || !_lastBugReport.markdown) return;
  const blob = new Blob([_lastBugReport.markdown], { type: "text/markdown" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = _lastBugReport.filename || "bug-report.md";
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export async function submitBugReport(upload, isRetry = false) {
  const desc = ($("bug-desc").value || "").trim();
  const expected = (($("bug-expected") && $("bug-expected").value) || "").trim();
  const happened = (($("bug-happened") && $("bug-happened").value) || "").trim();
  // Either "what were you doing" or "what happened" alone is enough to send,
  // mirroring the server's own "description or what_happened" check.
  if (!desc && !happened) { toast("Describe the problem first", true); return; }
  const includeLog = !!($("bug-include-log") && $("bug-include-log").checked);
  const saveBtn = $("bug-send"), upBtn = $("bug-upload");
  if (saveBtn) saveBtn.disabled = true;
  if (upBtn) upBtn.disabled = true;
  _showBugActions({ retry: false, download: false });   // fresh attempt: reset
  // Browser context; the env snapshot and server state are added server-side.
  // Sanitized + capped on the server; rendered as plain text, never executed.
  const client = {
    userAgent: navigator.userAgent,
    page: location.hash || location.pathname,
    viewport: window.innerWidth + "x" + window.innerHeight,
    console: (window.__localmClientLog || []).slice(-40),
  };
  const out = $("bug-result");
  try {
    const r = await fetch("/api/bug-report", {
      method: "POST",
      headers: { "Content-Type": "application/json", ...authHeaders() },
      body: JSON.stringify({ description: desc, what_i_expected: expected,
        what_happened: happened, include_log: includeLog, client,
        upload: !!upload }),
    });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(data.detail || r.statusText);
    // Rate limited: with auto-retry on (the default), keep the user's text,
    // count down, and retry the send ONCE; with it off, fall through to a plain
    // notice.
    const autoRetry = !$("bug-autoretry") || $("bug-autoretry").checked;
    if (upload && data.rate_limited && !isRetry && autoRetry) {
      const secs = Math.max(1, parseInt(data.retry_after, 10) || 30);
      await countdownRetryBugReport(secs, out);
      return;   // the retry (and the finally below) re-enable the buttons
    }
    const where = data.path || data.filename || "report";
    const sent = upload && data.uploaded;
    const uploadFailed = upload && data.upload_error && !data.rate_limited;
    // Stash the saved report so "Download report" can hand it to the tester.
    _lastBugReport = data.report_markdown
      ? { markdown: data.report_markdown, filename: data.filename || "bug-report.md" }
      : null;
    // On a failed send, offer Retry (re-file the issue) and Download; on
    // success, keep them hidden.
    _showBugActions({ retry: uploadFailed, download: !sent && !!_lastBugReport });
    if (out) {
      out.hidden = false;
      if (sent) {
        out.textContent = "Sent." +
          (data.issue_url ? " Tracking issue: " + data.issue_url : "");
      } else if (data.rate_limited) {
        out.textContent = "Saved: " + where +
          "  -  rate limited; wait a bit and click Send again.";
      } else if (uploadFailed) {
        // Name where it failed (the diagnosed message) and that the report is
        // kept, so it can be retried, downloaded or emailed.
        out.textContent = "Could not send: " +
          (data.upload_message || data.upload_error) +
          "  The report is saved" + (where ? " (" + where + ")" : "") +
          " - retry, download it, or email " + (data.maintainer || "the maintainer") + ".";
      } else {
        out.textContent = "Saved: " + where +
          (data.maintainer ? "  -  send it to " + data.maintainer : "");
      }
    }
    // Keep the description on ANY failed send so Retry re-uses it; clear it once
    // the report is done with (sent, or a plain save with no upload attempt).
    if (sent || (!upload && data.saved)) {
      $("bug-desc").value = "";
      if ($("bug-expected")) $("bug-expected").value = "";
      if ($("bug-happened")) $("bug-happened").value = "";
    }
    if (sent) toast("Bug report sent");
    else if (data.rate_limited) toast("Rate limited; wait a bit and click Send again", true);
    else if (uploadFailed) toast("Could not send: " + (data.upload_message || data.upload_error), true);
    else toast("Bug report saved");
  } catch (e) {
    toast("Could not file report: " + e.message, true);
  } finally {
    if (saveBtn) saveBtn.disabled = false;
    if (upBtn) upBtn.disabled = false;
  }
}

// Show a live countdown in the bug-report result line, then auto-retry the send
// once (isRetry=true, so it never loops). The caller's buttons stay disabled
// throughout.
async function countdownRetryBugReport(secs, out) {
  for (let s = secs; s > 0; s--) {
    if (out) {
      out.hidden = false;
      out.textContent = "Rate limited - retrying in " + s + "s...";
    }
    await new Promise((res) => setTimeout(res, 1000));
  }
  if (out) out.textContent = "Retrying...";
  await submitBugReport(true, true);
}

if ($("bug-send")) $("bug-send").onclick = () => submitBugReport(false);
if ($("bug-upload")) $("bug-upload").onclick = () => submitBugReport(true);
if ($("bug-retry")) $("bug-retry").onclick = () => submitBugReport(true);
if ($("bug-download")) $("bug-download").onclick = () => downloadBugReport();

// Updates: a check-only auto-surface (a throttled startup check in app.js calls
// __localmUpdateCheck) plus an explicit "Update now". The apply runs only on
// that click; the server rolls back and reports on failure.
export async function updateCheck() {
  const out = $("update-status"), applyBtn = $("update-apply");
  try {
    const r = await fetch("/api/update/check", { headers: authHeaders() });
    const d = await r.json().catch(() => ({}));
    if (out) out.hidden = false;
    if (!r.ok) throw new Error(d.detail || r.statusText);
    if (d.error) { if (out) out.textContent = "Could not check: " + d.error; return; }
    if (!d.available) { if (out) out.textContent = "The updater is not configured."; return; }
    if (d.newer && d.asset && d.asset.id) {
      if (out) out.textContent = "Update available: " + d.latest + " (you have " + d.current +
        ")." + (d.notes ? "  " + String(d.notes).slice(0, 200) : "");
      if (applyBtn) applyBtn.hidden = false;
    } else if (d.newer) {
      if (out) out.textContent = "Update " + d.latest + " is available but has no build attached.";
      if (applyBtn) applyBtn.hidden = true;
    } else if (d.comparable === false) {
      // is_newer() returns False both for a genuine tie or older release and for
      // a tag it could not parse as a version at all (nightly, stable,
      // release-5). Keyed like cli/maintenance.py's .get("comparable", True):
      // strict === false, so an absent key means comparable. The wording carries
      // no "up to date" substring.
      if (out) out.textContent = "Could not tell whether " + d.latest +
        " is newer than your version " + d.current +
        " (unrecognized version format) - check the release notes yourself" +
        " before assuming there is nothing newer.";
      if (applyBtn) applyBtn.hidden = true;
    } else {
      if (out) out.textContent = "localm is up to date (running " + d.current + ").";
      if (applyBtn) applyBtn.hidden = true;
    }
  } catch (e) { if (out) { out.hidden = false; out.textContent = "Could not check: " + e.message; } }
}
window.__localmUpdateCheck = updateCheck;

export async function updateApply() {
  const out = $("update-status"), btn = $("update-apply");
  if (btn) btn.disabled = true;
  if (out) { out.hidden = false; out.textContent = "Downloading and applying ..."; }
  try {
    const r = await fetch("/api/update/apply", { method: "POST", headers: authHeaders() });
    const d = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(d.detail || r.statusText);
    if (d.applied) {
      if (out) out.textContent = "Updated to " + (d.version || "") +
        (d.restarting ? ". Restarting ..." :
          (d.klass === "setup" ? ". Re-run setup.bat to finish." : "."));
      if (btn) btn.hidden = true;
    } else if (d.error) {
      if (out) out.textContent = "Update failed (rolled back): " + d.error;
    } else {
      if (out) out.textContent = d.reason || "Nothing to apply.";
    }
  } catch (e) { if (out) out.textContent = "Update failed: " + e.message; }
  finally { if (btn) btn.disabled = false; }
}
if ($("update-check")) $("update-check").onclick = updateCheck;
if ($("update-apply")) $("update-apply").onclick = updateApply;

// Roll back: the GUI form of `localm update --rollback`. Probes first - the GET
// is read-only and never performs the rollback - so the control appears only
// when a backup exists. The server owner-gates the POST and restarts itself
// afterwards (routes/admin.py).
export async function updateRollbackCheck() {
  const block = $("app-rollback-block"), out = $("update-rollback-status");
  if (!block) return;
  try {
    const r = await fetch("/api/update/rollback", { headers: authHeaders() });
    const d = await r.json().catch(() => ({}));
    if (!r.ok || !d.available) { block.hidden = true; return; }
    block.hidden = false;
    if (out) {
      out.hidden = false;
      out.textContent = "This restores " + (d.version || "the previous build") +
        (d.current ? " (you are running " + d.current + ")" : "") + ".";
    }
  } catch (e) {
    // Unconfirmed: stay hidden rather than offer a rollback that may not exist,
    // and log it to the console instead of raising a banner.
    block.hidden = true;
    console.warn("could not check for a rollback backup:", e);
  }
}
window.__localmRollbackCheck = updateRollbackCheck;

export async function updateRollback() {
  const out = $("update-rollback-status"), btn = $("update-rollback");
  confirmDanger("Roll back to the previous build?",
    "This replaces the running LocaLM code with the build from before the last " +
    "update, then restarts the server. Anything the newer build fixed comes back.",
    "Roll back", async () => {
      if (btn) btn.disabled = true;
      if (out) { out.hidden = false; out.textContent = "Rolling back..."; }
      try {
        const r = await fetch("/api/update/rollback",
                              { method: "POST", headers: authHeaders() });
        const d = await r.json().catch(() => ({}));
        if (!r.ok) throw new Error(d.detail || r.statusText);
        if (out) out.textContent = "Rolled back to " + (d.version || "the previous build") +
          ". Restarting...";
        // Not re-enabled: the server is re-execing and the reconnect overlay
        // takes over from here.
        if (btn) btn.hidden = true;
        if (window.onServerUnreachable) setTimeout(() => onServerUnreachable(), 800);
      } catch (e) {
        if (out) out.textContent = "Roll back failed: " + e.message;
        if (btn) btn.disabled = false;
      }
    });
}
if ($("update-rollback")) $("update-rollback").onclick = updateRollback;

// The GUI form of `localm make-launcher --force`. Always passes force=true, so a
// launcher already on disk is rebuilt rather than returning an idempotent no-op.
// A single blocking POST: make_launcher() runs in seconds and reports a
// structured result rather than streaming progress, so there is no job.
export async function rebuildLauncher() {
  const out = $("rebuild-launcher-status"), btn = $("rebuild-launcher");
  if (btn) btn.disabled = true;
  if (out) { out.hidden = false; out.textContent = "Rebuilding..."; }
  try {
    const r = await fetch("/api/app/rebuild-launcher?force=true",
                          { method: "POST", headers: authHeaders() });
    const d = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(d.detail || r.statusText);
    // notes carries a human-readable surface even on a success (e.g. "could not
    // stamp the exe icon"), so it is shown either way.
    const notes = (d.notes || []).join(" ");
    if (out) {
      out.textContent = d.ok
        ? "Launcher rebuilt" + (d.path ? ": " + d.path : "") + (notes ? " (" + notes + ")" : ".")
        : "Could not rebuild the launcher" + (notes ? ": " + notes : ".");
    }
  } catch (e) {
    if (out) out.textContent = "Rebuild failed: " + e.message;
  } finally {
    if (btn) btn.disabled = false;
  }
}
if ($("rebuild-launcher")) $("rebuild-launcher").onclick = rebuildLauncher;

// The inference runtime: the native llama.cpp binaries `localm setup-llama`
// provisions, separate from the "Updates" card above (the Python source tree).
// Check is read-only (GET /api/runtime/check); provisioning streams a job (POST
// /api/runtime/update). This block is the GUI's whole form of
// `localm setup-llama`: install a runtime, switch backend, or choose a build.

// The last SUCCESSFUL /api/runtime/check payload, or null when none has
// succeeded yet. Lets the button be re-labelled the instant the selection
// changes with no round trip, and tells "switch" apart from "install".
let runtimeCheckState = null;

/** The raw GET /api/runtime/check call: no DOM, throws on a non-OK response.
 *  The one place that talks to the endpoint - shared by the Settings runtime
 *  card below and the Setup flow (pages/setup.js), so neither forks it. */
export async function fetchRuntimeCheck() {
  const r = await fetch("/api/runtime/check", { headers: authHeaders() });
  const d = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(d.detail || r.statusText);
  return d;
}

/** The raw POST /api/runtime/update call: no DOM, returns the new job's id.
 *  Shared the same way as fetchRuntimeCheck above. */
export async function postRuntimeUpdate(backend, tag, rollback) {
  const body = {};
  if (backend) body.backend = backend;
  if (rollback) body.rollback = true;
  else if (tag) body.tag = tag;
  const r = await fetch("/api/runtime/update", {
    method: "POST", headers: authHeaders(), body: JSON.stringify(body),
  });
  const d = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(d.detail || r.statusText);
  return d.job_id;
}

/** What pressing the button will do, given the current selection and the last
 *  check. Pure, so the wording is testable without a DOM.
 *
 *  A null *state* means no check has succeeded, so "switch" versus "install" is
 *  not knowable and the label stays neutral. */
export function runtimeApplyLabel(wanted, tag, state) {
  const installed = state && state.installed ? state.backend : null;
  if (wanted === "auto") return installed ? "Re-detect and reinstall" : "Detect and install";
  if (wanted && installed && wanted !== installed) return "Switch to " + wanted;
  if (wanted && installed) return "Reinstall " + wanted;
  if (wanted) return "Install " + wanted;
  if (tag) return "Install build " + tag;
  return installed ? "Update runtime" : "Install runtime";
}

/** Show or hide the action button, and label it for what it would do now.
 *  Called after every check and on every change to the two controls. */
export function syncRuntimeApply() {
  const btn = $("runtime-update-apply");
  if (!btn) return;
  const sel = $("runtime-backend"), tagEl = $("runtime-tag");
  const wanted = sel ? sel.value.trim() : "";
  const tag = tagEl ? tagEl.value.trim() : "";
  btn.textContent = runtimeApplyLabel(wanted, tag, runtimeCheckState);
  // An explicit choice is actionable whatever the check said, including when
  // the check failed outright.
  if (wanted || tag) { btn.hidden = false; return; }
  if (!runtimeCheckState) { btn.hidden = true; return; }
  if (!runtimeCheckState.installed) { btn.hidden = false; return; }
  btn.hidden = !runtimeCheckState.newer;
}

export async function runtimeUpdateCheck() {
  const out = $("runtime-update-status");
  if (out) { out.hidden = false; out.textContent = "Checking..."; }
  try {
    const d = await fetchRuntimeCheck();
    runtimeCheckState = d;
    if (out) {
      if (!d.installed) {
        out.textContent = "No llama.cpp runtime is installed yet - choose a backend below and install one.";
      } else {
        const current = d.current || "an unrecorded build";
        out.textContent = (d.newer
          ? "A different build is available for " + d.backend + ": " + d.target +
            " (you have " + current + ")."
          : "Up to date (" + d.backend + " " + current + ").") +
          (d.pinned ? " Pinned to " + d.pinned + "." : "");
      }
    }
  } catch (e) {
    runtimeCheckState = null;
    if (out) { out.hidden = false; out.textContent = "Could not check: " + e.message; }
  }
  syncRuntimeApply();
  syncRuntimeRollback();
}

/** Show or hide the Roll back button, from the last check's "previous" field. */
export function syncRuntimeRollback() {
  const btn = $("runtime-rollback"), out = $("runtime-rollback-status");
  const prev = runtimeCheckState && runtimeCheckState.installed
    ? runtimeCheckState.previous : null;
  if (btn) btn.hidden = !prev;
  if (out) {
    out.hidden = !prev;
    if (prev) out.textContent = "A previous build is on record: " + prev + ".";
  }
}

/** POST the provision and stream its job. *backend* and *tag* are sent only when
 *  non-empty, so an untouched card re-provisions whatever is installed.
 *  *rollback* sends {rollback: true} instead of a tag, for the Roll back
 *  button. */
export async function runtimeProvision(backend, tag, rollback) {
  const out = $("runtime-update-status"), btn = $("runtime-update-apply");
  const rbBtn = $("runtime-rollback");
  const log = $("runtime-update-log");
  if (btn) btn.disabled = true;
  if (rbBtn) rbBtn.disabled = true;
  if (out) {
    out.hidden = false;
    out.textContent = rollback ? "Rolling back the runtime..." : "Provisioning the runtime...";
  }
  if (log) { log.style.display = ""; log.textContent = ""; }
  let jobId;
  try {
    jobId = await postRuntimeUpdate(backend, tag, rollback);
  } catch (e) {
    if (out) out.textContent = (rollback ? "Roll back failed: " : "Update failed: ") + e.message;
    if (btn) btn.disabled = false;
    if (rbBtn) rbBtn.disabled = false;
    if (log) log.style.display = "none";
    return;
  }
  const tail = [];
  const end = await streamJob(jobId, (line) => {
    if (log) { log.textContent += line + "\n"; log.scrollTop = log.scrollHeight; }
    // A short tail, not just the last line: setup-llama's failure messages wrap
    // across several printed lines.
    if (line && line.trim()) { tail.push(line.trim()); if (tail.length > 6) tail.shift(); }
  });
  const ok = !!(end && end.status === "done");
  if (out) {
    out.textContent = ok ? (rollback ? "Rolled back." : "Runtime provisioned.") :
      (tail.join(" ").trim() || "The update did not finish. See the log below.");
  }
  if (btn) btn.disabled = false;
  if (rbBtn) rbBtn.disabled = false;
  if (ok) {
    // The request has been carried out, so clear the selection.
    if ($("runtime-backend")) $("runtime-backend").value = "";
    if ($("runtime-tag")) $("runtime-tag").value = "";
    runtimeUpdateCheck();   // re-checks; settles both the update and rollback buttons
  } else {
    syncRuntimeApply();     // failed: keep the retry affordance, correctly labelled
    syncRuntimeRollback();
  }
}

export function runtimeUpdateApply() {
  const sel = $("runtime-backend"), tagEl = $("runtime-tag");
  const wanted = sel ? sel.value.trim() : "";
  const tag = tagEl ? tagEl.value.trim() : "";
  const installed = runtimeCheckState && runtimeCheckState.installed
    ? runtimeCheckState.backend : null;
  // Only replacing a working runtime with a DIFFERENT backend confirms first; a
  // first install and a same-backend re-provision go straight through. "auto"
  // counts as different: it resolves from hardware detection and can land on
  // another backend.
  if (installed && wanted && wanted !== installed) {
    confirmDanger(
      "Switch the inference runtime",
      "This replaces the installed " + installed + " runtime with " +
      (wanted === "auto" ? "whichever backend localm detects for this machine"
                         : "a " + wanted + " build") +
      ". It cannot run while a model is loaded, and if the new build does not " +
      "work here localm says so rather than leaving it installed.",
      "Switch", () => runtimeProvision(wanted, tag));
    return;
  }
  runtimeProvision(wanted, tag);
}
/** Roll back the INSTALLED backend to its previous recorded build. Always
 *  confirms first. */
export function runtimeRollbackApply() {
  const state = runtimeCheckState;
  if (!state || !state.installed || !state.previous) return;
  confirmDanger(
    "Roll back the inference runtime",
    "This replaces the installed " + state.backend + " runtime with the " +
    "previous build (" + state.previous + ") and restarts loading with it.",
    "Roll back", () => runtimeProvision("", "", true));
}
if ($("runtime-update-check")) $("runtime-update-check").onclick = runtimeUpdateCheck;
if ($("runtime-update-apply")) $("runtime-update-apply").onclick = runtimeUpdateApply;
if ($("runtime-rollback")) $("runtime-rollback").onclick = runtimeRollbackApply;
// Re-label the moment the selection changes.
if ($("runtime-backend")) $("runtime-backend").onchange = syncRuntimeApply;
if ($("runtime-tag")) $("runtime-tag").oninput = syncRuntimeApply;

// Changelog: the RELEASED history, newest first, in the shared modal. The
// endpoint strips the in-progress [Unreleased] section before serving. Fetched
// from /api/changelog and rendered via renderMarkdown - the same
// DOMPurify(marked) path chat uses. A missing or failed fetch is shown in the
// modal, never left blank.
export async function showChangelog() {
  openModal("Changelog", (body) => {
    const md = el("div", "changelog-md");
    md.textContent = "Loading the changelog...";
    body.appendChild(md);
    fetch("/api/changelog", { headers: authHeaders() })
      .then(async (r) => {
        const d = await r.json().catch(() => ({}));
        if (!r.ok) throw new Error(d.detail || r.statusText);
        if (!d.available) { md.textContent = "No changelog is available in this build."; return; }
        md.textContent = "";
        renderMarkdown(md, d.markdown || "");
      })
      .catch((e) => { md.textContent = "Could not load the changelog: " + e.message; });
  });
}
if ($("changelog-show")) $("changelog-show").onclick = showChangelog;

// Issues: read-only list (textContent only - never raw innerHTML for proxy data).
export async function issuesRefresh() {
  const out = $("issues-list");
  if (out) out.textContent = "Loading ...";
  try {
    const r = await fetch("/api/issues", { headers: authHeaders() });
    const d = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(d.detail || r.statusText);
    if (!out) return;
    if (d.error) { out.textContent = "Could not load: " + d.error; return; }
    const list = d.issues || [];
    // Reached only on a SUCCESSFUL fetch of an empty list; the error paths above
    // and below say "could not load" instead.
    if (!list.length) {
      out.replaceChildren(emptyState("book", "No open issues",
        "Reported bugs that are still open show up here."));
      return;
    }
    out.textContent = "";
    for (const it of list) {
      const row = document.createElement("div");
      row.textContent = "#" + it.number + " " + (it.state || "?") + "  " + (it.title || "");
      if (it.html_url) {
        const a = document.createElement("a");
        a.href = it.html_url; a.target = "_blank"; a.rel = "noopener";
        a.textContent = " (open)";
        row.appendChild(a);
      }
      out.appendChild(row);
    }
  } catch (e) { if (out) out.textContent = "Could not load issues: " + e.message; }
}
if ($("issues-refresh")) $("issues-refresh").onclick = issuesRefresh;

// Export all logs of this instance to a folder the user picks with the shared
// directory-picker modal, then POST the chosen path to /api/logs/export.
if ($("logs-export")) {
  $("logs-export").onclick = async () => {
    const dest = await pickDirectory("Choose a folder for the exported logs");
    if (!dest) return;                       // dismissed
    const btn = $("logs-export");
    btn.disabled = true;
    try {
      const r = await fetch("/api/logs/export", {
        method: "POST",
        headers: { "Content-Type": "application/json", ...authHeaders() },
        body: JSON.stringify({ dest }),
      });
      const data = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(data.detail || r.statusText);
      const out = $("logs-result");
      if (out) {
        out.hidden = false;
        out.textContent = data.copied
          ? `Exported ${data.copied} log file(s) to: ${data.dest}`
          : (data.message || "No logs found to export.");
        // A partial export names which files failed.
        if (data.warning) out.textContent += `  (${data.warning})`;
      }
      toast(data.copied ? `Exported ${data.copied} log file(s)` : "No logs to export",
        !data.copied || !!data.warning);
    } catch (e) {
      toast("Could not export logs: " + e.message, true);
    } finally {
      btn.disabled = false;
    }
  };
}

// Upload files from this device or a phone into <home>/uploads/, where models
// and tools can read them. The POST is multipart: the JSON Content-Type is
// stripped so the browser sets multipart/form-data with its own boundary, while
// the auth + CSRF headers stay. CONFIG_WRITE-gated server-side. The list is
// built with textContent, never innerHTML.
export async function refreshUploadsList() {
  const list = $("upload-list");
  if (!list) return;
  try {
    const r = await fetch("/api/uploads", { headers: authHeaders() });
    if (!r.ok) { list.replaceChildren(); return; }   // e.g. a read-only key: hide
    const data = await r.json().catch(() => ({ items: [] }));
    list.replaceChildren();
    // The empty state for a readable but empty list, distinct from the
    // read-only-key early return above, which leaves the list blank.
    if (!(data.items || []).length) {
      list.appendChild(emptyState("attach", "No uploaded files",
        "Files you send here are readable by models and tools."));
      return;
    }
    for (const it of (data.items || [])) {
      const li = document.createElement("li");
      const span = document.createElement("span");
      span.className = "upload-name";
      span.textContent = `${it.name}  ·  ${fmtBytes(it.bytes)}`;
      const del = document.createElement("button");
      del.className = "btn-secondary upload-del";
      del.textContent = "Remove";
      del.onclick = () => deleteUpload(it.name);
      li.appendChild(span);
      li.appendChild(del);
      list.appendChild(li);
    }
  } catch (e) { /* leave the list as-is on a transient error */ }
}
window.refreshUploadsList = refreshUploadsList;

export async function deleteUpload(name) {
  try {
    const r = await fetch("/api/uploads/" + encodeURIComponent(name),
      { method: "DELETE", headers: authHeaders() });
    if (!r.ok) {
      const d = await r.json().catch(() => ({}));
      throw new Error(d.detail || r.statusText);
    }
    toast("Removed " + name);
    refreshUploadsList();
  } catch (e) {
    toast("Could not remove: " + e.message, true);
  }
}
window.deleteUpload = deleteUpload;

if ($("upload-choose")) {
  $("upload-choose").onclick = () => $("upload-input").click();
}
if ($("upload-input")) {
  $("upload-input").onchange = () => {
    const files = Array.from($("upload-input").files || []);
    const label = $("upload-selected");
    if (label) {
      label.textContent = files.length
        ? `${files.length} file(s) selected: ${files.map((f) => f.name).join(", ")}`
        : "";
    }
  };
}

if ($("upload-send")) {
  $("upload-send").onclick = async () => {
    const input = $("upload-input");
    const files = input && input.files ? Array.from(input.files) : [];
    if (!files.length) { toast("Choose a file to upload first", true); return; }
    const fd = new FormData();
    for (const f of files) fd.append("file", f, f.name);
    const headers = authHeaders();
    delete headers["Content-Type"];   // let the browser set the multipart boundary
    const btn = $("upload-send");
    btn.disabled = true;
    try {
      const r = await fetch("/api/upload", { method: "POST", headers, body: fd });
      const data = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(data.detail || r.statusText);
      const n = (data.uploaded || []).length;
      const out = $("upload-result");
      if (out) {
        out.hidden = false;
        out.textContent = `Uploaded ${n} file(s) to: ${data.dir || "uploads"}`;
      }
      input.value = "";
      const label = $("upload-selected");
      if (label) label.textContent = "";
      toast(`Uploaded ${n} file(s)`);
      refreshUploadsList();
    } catch (e) {
      toast("Upload failed: " + e.message, true);
    } finally {
      btn.disabled = false;
    }
  };
}

$("pull-start").onclick = async () => {
  const spec = $("pull-spec").value.trim();
  if (!spec) { toast("Enter a model spec", true); return; }
  if (spec.startsWith("-")) {
    toast("A model spec can't start with '-'. Use owner/repo, " +
          "owner/repo:file.gguf, or an https URL.", true);
    return;
  }
  const name = $("pull-name").value.trim();
  const mmprojInput = $("pull-mmproj");
  const mmproj = mmprojInput && mmprojInput.style.display !== "none" ? mmprojInput.value : null;
  const sha256Input = $("pull-sha256");
  const sha256 = sha256Input ? sha256Input.value.trim() : "";
  const storeInput = $("pull-store");
  const store = storeInput && storeInput.value ? storeInput.value : null;

  $("pull-start").disabled = true;
  const log = $("pull-log");
  log.style.display = "block";
  log.textContent = "";
  const prog = $("pull-progress");
  const bar = $("pull-bar");
  const pct = $("pull-pct");
  // A live (indeterminate) bar from the start, so a job that fails before
  // emitting any progress still shows as running.
  prog.style.display = "block";
  bar.classList.remove("failed");
  bar.classList.add("indeterminate");
  bar.style.width = "35%";
  pct.textContent = "starting…";
  if ($("pull-file")) { $("pull-file").hidden = true; $("pull-file").textContent = ""; }
  const samples = [];   // rolling {t, downloaded} window for speed/ETA
  try {
    const payload = { spec, name: name || null };
    if (mmproj) payload.mmproj = mmproj;
    if (sha256) payload.sha256 = sha256;
    if (store) payload.store = store;
    // Attach the type hint only while the spec still matches exactly what it was
    // prefilled for; a hand-edited spec falls back to auto-detect.
    if (pendingPullTypeHint && pendingPullTypeHint.spec === spec) {
      payload.model_type = pendingPullTypeHint.type;
    }

    const r = await fetch("/api/models/pull", {
      method: "POST", headers: authHeaders(),
      body: JSON.stringify(payload),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || r.statusText);
    const end = await streamJob(data.job_id, (line) => {
      log.textContent += line + "\n";
      log.scrollTop = log.scrollHeight;
    }, (ev) => {
      // For a multi-file (split GGUF) download, show which file is in flight.
      const fileLine = $("pull-file");
      if (fileLine) {
        if (ev.count > 1) {
          fileLine.hidden = false;
          fileLine.textContent = ev.name
            ? `file ${ev.index} of ${ev.count}: ${ev.name}`
            : `file ${ev.index} of ${ev.count}`;
        } else {
          fileLine.hidden = true;
        }
      }
      if (ev.pct != null && ev.total) {
        bar.classList.remove("indeterminate");
        bar.style.width = ev.pct + "%";
        // Smoothed speed + ETA from a rolling ~10-sample window.
        samples.push({ t: Date.now(), downloaded: ev.downloaded });
        if (samples.length > 10) samples.shift();
        const { bytesPerSec, etaSec } = downloadRate(samples, ev.total);
        let extra = "";
        if (bytesPerSec) extra += `  ·  ${fmtBytes(bytesPerSec)}/s`;
        if (etaSec != null) extra += `  ·  ETA ${fmtDuration(etaSec)}`;
        pct.textContent =
          `${ev.pct.toFixed(0)}%  ·  ${fmtBytes(ev.downloaded)} / ${fmtBytes(ev.total)}${extra}`;
      } else {
        // Unknown total - busy bar with a running byte count
        bar.classList.add("indeterminate");
        bar.style.width = "100%";
        pct.textContent = "downloading…  " + fmtBytes(ev.downloaded);
      }
    });
    if (end.status === "done") {
      bar.classList.remove("indeterminate");
      bar.style.width = "100%";
      pct.textContent = "done";
      toast("Pull finished");
      $("pull-spec").value = "";
      $("pull-name").value = "";
      if (sha256Input) sha256Input.value = "";
      if (storeInput) storeInput.value = "";
      pendingPullTypeHint = null;
      refreshModelsPage();
    } else if (end.status === "disconnected") {
      // streamJob gave up reconnecting, or the job was already gone. Not a job
      // failure: a dropped SSE connection does not stop the pull, so the bar is
      // left as-is rather than painted red or completed.
      pct.textContent = "connection lost - it may still be running";
      toast("Lost connection to the pull - it may still be running in the "
            + "background. Check the Models list in a moment.", true);
    } else {
      // Surface the failure: red bar, exit code, and the inputs kept for a
      // retry.
      bar.classList.remove("indeterminate");
      bar.classList.add("failed");
      const code = end.returncode != null ? `, exit ${end.returncode}` : "";
      pct.textContent = `failed${code} - see log`;
      toast(`Pull failed (${end.status}${code}) - see log`, true);
      // The job's own log is the CLI's console output verbatim (it streams a
      // spawned `localm pull`), so a net_mode=off refusal in there names a
      // terminal command a browser cannot run. Never rewritten in place
      // (the log is a faithful transcript) - append a GUI-native pointer
      // instead, whenever that state is actually true right now. Silently
      // skipped without config:read scope; this is a hint, not the failure.
      try {
        const cr = await fetch("/v1/config", { headers: authHeaders() });
        if (cr.ok) {
          const c = await cr.json();
          if (c.net_mode === "off" && !c.net_allow_model_downloads) {
            log.textContent += "(if this failed because network access is " +
              "off: turn it on, or allow downloads only, in Settings → " +
              "Network)\n";
            log.scrollTop = log.scrollHeight;
          }
        }
      } catch { /* best-effort hint only */ }
    }
  } catch (e) {
    bar.classList.remove("indeterminate");
    bar.classList.add("failed");
    pct.textContent = "failed - see log";
    toast("Pull failed: " + e.message, true);
  } finally {
    $("pull-start").disabled = false;
  }
};


// Per-tab counts. The tab's own label is left alone and the number rides in a
// trailing span, so the text read by dataset.type is untouched. A type with
// nothing registered shows no number rather than a "0"; "All" carries the total.
//
// Other counts every model whose type has no tab of its own, and All counts what
// All actually SHOWS, not the registry total. With the merge toggle off the
// numbers partition the registry exactly (every model is on All or on Other);
// with it on, Other is a subset of All.
export function syncTabCounts(models) {
  const nav = $("models-tab-nav");
  if (!nav) return;
  const tabbed = typesWithOwnTab();
  const counts = new Map();
  let otherCount = 0;
  for (const m of models || []) {
    const t = modelTypeOf(m);
    counts.set(t, (counts.get(t) || 0) + 1);
    if (!tabbed.has(t)) otherCount++;
  }
  const total = (models || []).length;
  const mergeOther = localStorage.getItem(SHOW_OTHER_KEY) === "true";
  for (const btn of nav.querySelectorAll(".tab-btn")) {
    const n = btn.dataset.type === "all"
      ? (mergeOther ? total : total - otherCount)
      : (btn.dataset.type === OTHER_TAB
         ? otherCount
         : (counts.get(btn.dataset.type) || 0));
    let slot = btn.querySelector(".tab-count");
    if (!slot) {
      slot = el("span", "tab-count");
      btn.appendChild(slot);
    }
    slot.textContent = n ? String(n) : "";
  }
}

// Bind tab click handlers
const tabNav = $("models-tab-nav");
if (tabNav) {
  for (const btn of tabNav.querySelectorAll(".tab-btn")) {
    btn.onclick = () => {
      for (const b of tabNav.querySelectorAll(".tab-btn")) {
        b.classList.remove("active");
      }
      btn.classList.add("active");
      currentTypeFilter = btn.dataset.type;
      refreshModelsPage();
    };
  }
}

// Both display toggles: reflect the stored value on load, write it back on
// change, re-render. The render reads localStorage directly, never this
// element's .checked.
function _bindModelsViewToggle(id, key) {
  const box = $(id);
  if (!box) return;
  box.checked = localStorage.getItem(key) === "true";
  box.addEventListener("change", (e) => {
    localStorage.setItem(key, e.target.checked ? "true" : "false");
    refreshModelsPage();
  });
}
_bindModelsViewToggle("models-show-other", SHOW_OTHER_KEY);
_bindModelsViewToggle("models-group-by-type", GROUP_BY_TYPE_KEY);

// Turn a ComfyUI ScanResult (added / skipped / method) into a human toast. The
// scanner's `method` field carries the reason an empty scan found nothing
// ("none (comfy_workdir not configured)", "none (models folder not found under
// {path})"), which is decoded here instead of a bare "Added 0". The internal
// "folder-walk" / "hybrid" method jargon stays out of the visible text.
export function scanResultMessage(data) {
  const added = data.added || 0;
  const skipped = data.skipped || 0;
  const method = String(data.method || "");
  if (method.startsWith("none")) {
    if (method.includes("comfy_workdir not configured")) {
      return "No ComfyUI workdir configured - set it in Settings.";
    }
    const under = "models folder not found under ";
    const i = method.indexOf(under);
    if (i !== -1) {
      const path = method.slice(i + under.length).replace(/\)\s*$/, "");
      return `ComfyUI models folder not found at ${path}.`;
    }
    // Any other "none (...)" reason.
    return "Scan found no ComfyUI models. Check the ComfyUI setup in Settings.";
  }
  if (added === 0 && skipped === 0) {
    // A real scan (folder-walk / hybrid) that turned up nothing: the folders
    // exist but held no models.
    return "Scan complete. No new ComfyUI models found.";
  }
  return `Scan complete. Added ${added} models, skipped ${skipped} existing.`;
}

// Bind Scan button click handler
const scanBtn = $("models-scan-btn");
if (scanBtn) {
  scanBtn.onclick = async () => {
    scanBtn.disabled = true;
    const prog = $("scan-progress");
    const bar = $("scan-bar");
    const pct = $("scan-pct");
    // Live (indeterminate) bar from the start: registering ComfyUI's
    // found_files.items() has no total until the directory walk finishes, then
    // it flips to a real "N of M" once registration starts.
    if (prog) {
      prog.style.display = "block";
      bar.classList.remove("failed");
      bar.classList.add("indeterminate");
      bar.style.width = "35%";
    }
    if (pct) pct.textContent = "Scanning ComfyUI model folders…";
    let lastLine = null;
    let finalResult = null;
    try {
      const r = await fetch("/api/models/scan", {
        method: "POST",
        headers: authHeaders(),
      });
      const data = await r.json().catch(() => ({}));
      if (!r.ok) {
        if (bar) { bar.classList.remove("indeterminate"); bar.classList.add("failed"); }
        if (pct) pct.textContent = "failed - see toast";
        toast(data.detail || "Scan failed", true);
        return;
      }
      const end = await streamJob(data.job_id, (line) => { lastLine = line; }, (ev) => {
        if (ev.phase === "done") {
          finalResult = { added: ev.added || 0, skipped: ev.skipped || 0, method: ev.method || "" };
          return;
        }
        if (ev.total && bar && pct) {
          bar.classList.remove("indeterminate");
          bar.style.width = (ev.done / ev.total * 100) + "%";
          pct.textContent = `Registering model ${ev.done} of ${ev.total}`
            + (ev.name ? `: ${ev.name}` : "");
        }
      });
      if (end.status === "done") {
        if (bar) { bar.classList.remove("indeterminate"); bar.style.width = "100%"; }
        if (pct) pct.textContent = "done";
        toast(finalResult ? scanResultMessage(finalResult) : "Scan complete.");
        refreshModelsPage();
      } else if (end.status === "disconnected") {
        // A lost SSE connection is not a job failure, so the bar is left as-is
        // rather than painted red.
        if (pct) pct.textContent = "connection lost - it may still be running";
        toast("Lost connection to the scan - it may still be running in the "
              + "background. Check the Models list in a moment.", true);
      } else {
        if (bar) { bar.classList.remove("indeterminate"); bar.classList.add("failed"); }
        if (pct) pct.textContent = "failed - see toast";
        toast(lastLine || "Scan failed", true);
      }
    } catch (e) {
      if (bar) { bar.classList.remove("indeterminate"); bar.classList.add("failed"); }
      if (pct) pct.textContent = "failed - see toast";
      toast("Scan failed: " + e.message, true);
    } finally {
      scanBtn.disabled = false;
    }
  };
}

// The guided "Import from ComfyUI..." flow (Add-a-model card). Unlike the Scan
// button above, which always scans whatever comfy_workdir Settings holds, this
// imports from ANY ComfyUI folder as a one-off - previewing counts per category
// before anything registers - without touching the comfy_workdir setting.
// Friendly labels for the dry-run preview breakdown; mirrors
// localm.model_manager.registry.MODEL_TYPES (keep in sync), minus "llm"/"mmproj"
// which a ComfyUI scan never produces.
export const IMPORT_TYPE_LABELS = {
  "diffusion-unet": "Diffusion / checkpoints",
  "text-encoder": "Text encoders",
  vae: "VAEs",
  lora: "LoRAs",
  unknown: "Other",
};

// localm's own managed ComfyUI install path, or null if there isn't one (not
// installed, or this key lacks config:read). A failed or 403 fetch just means
// the quick-fill option does not appear; the Browse flow is unaffected.
async function fetchManagedComfyPath() {
  try {
    const r = await fetch("/api/comfy/managed-status", { headers: authHeaders() });
    if (!r.ok) return null;
    const d = await r.json().catch(() => ({}));
    return d && d.installed && d.path ? d.path : null;
  } catch (e) {
    return null;
  }
}

// A dry-run preview response with a "none (...)" method (no folder configured or
// found) reuses the humanized reasons scanResultMessage decodes for the
// real-scan case. Returns null when the preview is a normal (possibly empty)
// result the caller should render counts for.
export function importComfyPreviewMessage(data) {
  const method = String((data && data.method) || "");
  if (method.startsWith("none")) {
    return scanResultMessage({ added: 0, skipped: 0, method });
  }
  return null;
}

export function openImportComfyModal(initialPath = "") {
  openModal("Import from ComfyUI", (body) => {
    const wrap = el("div", "import-comfy");

    const pathRow = el("div", "row");
    const pathInput = document.createElement("input");
    pathInput.type = "text";
    pathInput.placeholder = "ComfyUI install folder";
    pathInput.value = initialPath;
    const browseBtn = el("button", "btn-secondary", "Browse…");
    browseBtn.type = "button";
    // pickDirectory() opens its OWN modal, and this app has a single shared
    // #modal/#modal-body (openModal() replaces its content, it does not stack),
    // so the picker destroys this modal's DOM, pathInput included. Re-open this
    // modal fresh with the picked path rather than writing into a detached node.
    browseBtn.onclick = async () => {
      const dir = await pickDirectory("Pick a ComfyUI folder", pathInput.value.trim());
      if (dir) openImportComfyModal(dir);
    };
    pathRow.append(pathInput, browseBtn);
    wrap.appendChild(pathRow);

    // Quick-fill for localm's own managed ComfyUI - hidden until the async
    // status check resolves, and stays hidden if there is none.
    const managedRow = el("div", "row");
    managedRow.style.display = "none";
    const managedBtn = el("button", "btn-secondary", "Use localm's own ComfyUI");
    managedBtn.type = "button";
    managedRow.appendChild(managedBtn);
    wrap.appendChild(managedRow);
    fetchManagedComfyPath().then((path) => {
      if (!path) return;
      managedBtn.onclick = () => { pathInput.value = path; };
      managedRow.style.display = "flex";
    });

    const previewBox = el("div", "sub import-comfy-preview");
    wrap.appendChild(previewBox);

    // Live "registering model N of M" text while Import's background job runs -
    // hidden the rest of the time.
    const progressBox = el("div", "sub import-comfy-progress");
    progressBox.style.display = "none";
    wrap.appendChild(progressBox);

    const actions = el("div", "actions");
    const previewBtn = el("button", "btn-secondary", "Preview");
    previewBtn.type = "button";
    const importBtn = el("button", "btn-primary", "Import");
    importBtn.type = "button";
    importBtn.disabled = true;
    actions.append(previewBtn, importBtn);
    wrap.appendChild(actions);
    body.appendChild(wrap);

    // The folder the last successful, non-empty preview covered. Import reuses
    // it rather than re-reading the possibly since-edited input.
    let previewedWorkdir = null;

    previewBtn.onclick = async () => {
      const workdir = pathInput.value.trim();
      if (!workdir) { toast("Choose a ComfyUI folder first", true); return; }
      previewedWorkdir = null;
      importBtn.disabled = true;
      previewBtn.disabled = true;
      previewBox.replaceChildren(el("span", "", "Scanning…"));
      try {
        const r = await fetch("/api/models/scan", {
          method: "POST", headers: authHeaders(),
          body: JSON.stringify({ workdir, dry_run: true }),
        });
        const data = await r.json().catch(() => ({}));
        if (!r.ok) {
          previewBox.replaceChildren(el("span", "", data.detail || "Preview failed"));
          return;
        }
        const errMsg = importComfyPreviewMessage(data);
        if (errMsg) {
          previewBox.replaceChildren(el("span", "", errMsg));
          return;
        }
        previewBox.replaceChildren();
        const counts = data.counts || {};
        const total = data.total_new || 0;
        if (!total) {
          previewBox.appendChild(el("span", "", data.already_registered
            ? `All ${data.already_registered} found model(s) are already registered.`
            : "No models found in that folder."));
          return;
        }
        previewBox.appendChild(el("div", "", `Found ${total} new model(s):`));
        const list = el("ul", "import-comfy-counts");
        for (const [type, n] of Object.entries(counts)) {
          list.appendChild(el("li", "", `${IMPORT_TYPE_LABELS[type] || type}: ${n}`));
        }
        previewBox.appendChild(list);
        if (data.already_registered) {
          previewBox.appendChild(el("div", "sub",
            `${data.already_registered} already registered (skipped).`));
        }
        previewedWorkdir = workdir;
        importBtn.disabled = false;
      } catch (e) {
        previewBox.replaceChildren(el("span", "", "Preview failed: " + e.message));
      } finally {
        previewBtn.disabled = false;
      }
    };

    importBtn.onclick = async () => {
      if (!previewedWorkdir) return;
      importBtn.disabled = true;
      progressBox.style.display = "block";
      progressBox.textContent = "Starting import…";
      let lastLine = null;
      let finalResult = null;
      try {
        const r = await fetch("/api/models/scan", {
          method: "POST", headers: authHeaders(),
          body: JSON.stringify({ workdir: previewedWorkdir, dry_run: false }),
        });
        const data = await r.json().catch(() => ({}));
        if (!r.ok) {
          toast(data.detail || "Import failed", true);
          importBtn.disabled = false;
          progressBox.style.display = "none";
          return;
        }
        const end = await streamJob(data.job_id, (line) => { lastLine = line; }, (ev) => {
          // The final progress event (phase "done") carries the added / skipped
          // / method fields scanResultMessage() renders.
          if (ev.phase === "done") {
            finalResult = { added: ev.added || 0, skipped: ev.skipped || 0, method: ev.method || "" };
            return;
          }
          if (ev.total) {
            progressBox.textContent = `Registering model ${ev.done} of ${ev.total}`
              + (ev.name ? `: ${ev.name}` : "");
          }
        });
        if (end.status === "done") {
          toast(finalResult ? scanResultMessage(finalResult) : "Import finished");
          $("modal").style.display = "none";
          refreshModelsPage();
        } else if (end.status === "disconnected") {
          // A lost stream is not necessarily a lost job.
          progressBox.textContent = "Connection lost - it may still be running";
          toast("Lost connection to the import - it may still be running in "
                + "the background. Check the Models list in a moment.", true);
          importBtn.disabled = false;
        } else {
          progressBox.style.display = "none";
          toast(lastLine || "Import failed", true);
          importBtn.disabled = false;
        }
      } catch (e) {
        toast("Import failed: " + e.message, true);
        importBtn.disabled = false;
        progressBox.style.display = "none";
      }
    };
  });
}

if ($("models-import-comfy-btn")) {
  // Not a bare `= openImportComfyModal` assignment: a DOM onclick handler is
  // called with the click's MouseEvent as its first argument, which would land
  // in openImportComfyModal's initialPath parameter instead of its "" default.
  $("models-import-comfy-btn").onclick = () => openImportComfyModal();
}

// Bind Unload-all button click handler
const unloadAllBtn = $("models-unload-all-btn");
if (unloadAllBtn) {
  unloadAllBtn.onclick = async () => {
    unloadAllBtn.disabled = true;
    try {
      const r = await fetch("/api/models/unload", {
        method: "POST", headers: authHeaders(), body: JSON.stringify({}),
      });
      const data = await r.json().catch(() => ({}));
      if (!r.ok) { toast(data.detail || t("models.unload.failed"), true); return; }
      // unloaded_models lists only chat engines; the shared embedding model has
      // its own lifecycle (localm.inference.embedder) and is reported via
      // embedder_unloaded, so it is counted too.
      const n = (data.unloaded_models || []).length + (data.embedder_unloaded ? 1 : 0);
      // unload_all_models() (http_server.py) reports any pinned (mid-generation)
      // engine in skipped_in_use rather than unloading it - loaded, but not
      // released - so the message names it separately.
      const skipped = Array.isArray(data.skipped_in_use) ? data.skipped_in_use : [];
      let msg;
      if (n && skipped.length) {
        msg = t("models.unloadAll.someStillGenerating", { n, skipped: skipped.length });
      } else if (n) {
        msg = t("models.unloadAll.allUnloaded", { n });
      } else if (skipped.length) {
        msg = t("models.unloadAll.stillGenerating", { skipped: skipped.length });
      } else {
        msg = t("models.unloadAll.nothingLoaded");
      }
      toast(msg, skipped.length > 0);
      refreshModelsPage();
      refreshPerfEstimate();
    } catch (e) {
      toast(t("models.unloadAll.failedException", { message: e.message }), true);
    } finally {
      unloadAllBtn.disabled = false;
    }
  };
}

// The table is painted from a fetched /api/models response, not marked up in
// index.html, so it is redrawn when the interface language changes.
document.addEventListener("localm:language", () => {
  refreshModelsPage();
});

