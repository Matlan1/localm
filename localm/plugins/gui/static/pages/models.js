// SPDX-License-Identifier: AGPL-3.0-or-later
/* localm GUI - Models page (split from pages.js). Classic script: it
   shares the one global lexical environment with app.js and the other
   page scripts, so the helpers it uses ($, el, authHeaders, toast, ...)
   resolve by bare name exactly as before. */
"use strict";

// --- ES module imports (auto-generated boundary; bodies unchanged) ---
import { pickDirectory } from "../app/picker.js";
import { $, GIB, authHeaders, confirmDanger, downloadRate, el, fmtBytes, fmtDuration, openModal, renderMarkdown, streamJob, toast } from "../app/helpers.js";
import { onServerUnreachable } from "../app/init.js";
import { emptyState, iconEl } from "../app/icons.js";
import { modelCache, refreshModels, showKeyGate, switchModel, toastLoadResult } from "../app/models-sidebar.js";
import { refreshPerfEstimate } from "../app/settings-perf.js";

/* ================================================================ */
/*  Models page                                                      */
/* ================================================================ */

export function fmtSize(bytes) {
  if (bytes == null) return "";
  return (bytes / GIB).toFixed(2) + " GB";   // binary GiB, labelled GB (see app.js)
}

// Same shape as picker.js's local fmtDate: an ISO calendar date, compact enough
// for a dense table cell. mtime is a null-able epoch-seconds float from /api/models.
export function fmtModelDate(mtime) {
  if (mtime == null) return "";
  try { return new Date(mtime * 1000).toISOString().slice(0, 10); }
  catch (_) { return ""; }
}

// The Registered-models table tab (All/LLMs/Embedding/...). Scopes only the
// installed-models TABLE below - the HuggingFace SEARCH has its own explicit,
// always-visible Type checkboxes (discTypes()), so what the search covers is
// never inferred silently from this tab.
let currentTypeFilter = "all";
// Set when a discovery result is chosen, cleared on a successful add or a spec
// edit. {spec, type} - the Add handler only attaches model_type when spec still
// matches exactly what was prefilled, so a hand-edited spec silently falls back
// to auto-detect (never sends a stale/wrong type hint).
let pendingPullTypeHint = null;

// Registered-models table columns: label + how to read/compare a row + whether
// it is sortable at all (the actions column never is). "kind" picks the
// comparator - "number" (size_bytes/mtime, both nullable: an unreadable file, a
// UNC timeout, a missing path) vs "string" (case-insensitive). Kept as a single
// descriptor table so the header row, the click/keydown handlers and the
// comparator all read the same column list instead of drifting apart.
const MODEL_COLUMNS = [
  { key: "name", label: "Name", sortable: true, kind: "string", get: (m) => m.name || "" },
  { key: "model_type", label: "Role", sortable: true, kind: "string", get: (m) => m.model_type || "llm" },
  { key: "source", label: "Source", sortable: true, kind: "string", get: (m) => m.source || "" },
  { key: "size_bytes", label: "Size", sortable: true, kind: "number", get: (m) => m.size_bytes },
  { key: "mtime", label: "Modified", sortable: true, kind: "number", get: (m) => m.mtime },
  { key: "actions", label: "", sortable: false },
];
const _SORTABLE_KEYS = MODEL_COLUMNS.filter((c) => c.sortable).map((c) => c.key);

// Persisted sort choice (mirrors the _bindDiscToggle read-on-init pattern this
// file already uses for search filters). Falls back to alphabetical-by-name -
// today's only ordering (the server's sorted(registry.items())) - whenever
// nothing is stored yet, or a stored value no longer names a real column.
let currentSortKey = _SORTABLE_KEYS.includes(localStorage.getItem("localm.modelsSortKey"))
  ? localStorage.getItem("localm.modelsSortKey") : "name";
let currentSortDir = ["asc", "desc"].includes(localStorage.getItem("localm.modelsSortDir"))
  ? localStorage.getItem("localm.modelsSortDir") : "asc";

// Compare two model rows on one column. Numbers (size_bytes, mtime) can be
// null - an unreadable file, a UNC stat timeout, a missing path - and a null
// ALWAYS sorts last, in both directions: coercing it to 0 would rank a broken
// row as the smallest/oldest file rather than as genuinely unknown.
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
// non-sortable sortKey returns an unsorted copy rather than throwing, so a
// stale localStorage value degrades to "no sort" instead of breaking the page.
export function sortModels(models, sortKey, sortDir) {
  const col = MODEL_COLUMNS.find((c) => c.key === sortKey && c.sortable);
  if (!col) return models.slice();
  return models.slice().sort((a, b) => _compareModelRows(a, b, col, sortDir));
}

// Guards refreshModelsPage() against overlapping calls (a rapid double-click
// on a sortable header, a set-type change or use/alias/rename/remove action
// firing while a prior refresh's fetch is still in flight, ...):
// box.replaceChildren() only clears whatever is already in the DOM at the
// moment IT runs, so two calls racing past their own awaited fetch each clear
// once and then each append their own table - both survive, rendering the
// model list duplicated (reproduced live: a rapid double-click on the Name
// header left two <table class="data-table"> in #models-table). Every call
// captures this counter BEFORE it awaits anything, then re-checks it
// immediately before every write into the shared `box` (the clear itself, and
// every render path - error, empty, AND success): a call superseded by a
// newer one discards its own render instead of writing stale/duplicate
// content over (or alongside) the newer call's.
let _modelsRenderGen = 0;

export async function refreshModelsPage() {
  const myGen = ++_modelsRenderGen;
  await refreshModels();

  const box = $("models-table");
  if (myGen !== _modelsRenderGen) return;
  box.replaceChildren();

  // Fetched unfiltered and narrowed to the active tab below. The tabs carry a
  // per-type count, and a count of a type you are NOT looking at cannot come
  // from a ?type= response. Same cost as the default All tab, which already
  // asks for the whole registry; what each tab SHOWS is unchanged, because the
  // filter below is the same comparison the route makes.
  let models = [];
  try {
    const r = await fetch("/api/models", { headers: authHeaders() });
    if (r.status === 401) {
      // Expired/absent session (e.g. a network bind whose loopback key was never
      // seeded): the JSON error body parses fine, so the models=[] fallback below
      // would masquerade as "No models yet". Show the in-page key gate instead,
      // mirroring the sidebar's refreshModels() (models-sidebar.js). Not
      // gated on myGen: an expired session is a real, current fact regardless
      // of which overlapping call happens to observe it first.
      showKeyGate("This LocaLM server requires an API key.");
      return;
    }
    if (!r.ok) {
      // A non-401 error (403 = key lacks models.read, 500, 503, ...) also returns
      // a body with no `models` array; the empty-list fallback would hide the real
      // failure behind "No models yet". Surface the status instead.
      if (myGen !== _modelsRenderGen) return;
      box.appendChild(el("div", "sub", `Could not load models (HTTP ${r.status})`));
      return;
    }
    const data = await r.json();
    models = (data && Array.isArray(data.models)) ? data.models : [];
  } catch (e) {
    if (myGen !== _modelsRenderGen) return;
    box.appendChild(el("div", "sub", "Error loading models: " + e.message));
    return;
  }

  if (myGen !== _modelsRenderGen) return;
  // Counts come from the WHOLE registry, so a tab keeps showing how many it
  // holds while you are looking at a different one.
  syncTabCounts(models);
  // Narrow to the active tab. Deliberately the SAME comparison the /api/models
  // route makes (its own "llm" default for an entry with no recorded type, and
  // an exact match otherwise), so every tab shows exactly the rows it showed
  // when the route did the filtering.
  if (currentTypeFilter !== "all") {
    models = models.filter((m) => (m.model_type || "llm") === currentTypeFilter);
  }

  if (!models.length) {
    box.appendChild(emptyState("models", "No models yet",
      "Pull a model above, or search HuggingFace to add your first one."));
    return;
  }

  models = sortModels(models, currentSortKey, currentSortDir);

  const table = el("table", "data-table");
  const thead = el("thead");
  const hr = el("tr");
  for (const col of MODEL_COLUMNS) {
    const th = el("th", "", col.label);
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
  // Selectable model types - mirrors localm.model_manager.registry.MODEL_TYPES
  // (keep in sync). Powers the per-row one-click set-type control.
  const MODEL_TYPE_OPTIONS =
    ["llm", "embedding", "mmproj", "diffusion-unet", "text-encoder", "vae", "lora", "unknown"];
  for (const m of models) {
    const tr = el("tr");
    const nameTd = el("td", "name-cell");
    // The icon/name/badge flex line lives on this inner span, never on the td: a
    // display:flex td stops being a table-cell, so its border-bottom draws under its
    // own content rather than at the row's foot, which left the separator visibly
    // broken partway across every row (measured 82px out on a tall one).
    const nameLine = el("span", "cell-line");
    nameTd.appendChild(nameLine);
    nameLine.appendChild(iconEl("models", "ic ic-model"));
    nameLine.appendChild(el("span", "name", m.name));
    // F8-PERSIST-ARCH-AND-EXPERT-COUNT: the same real, from-the-file-header
    // architecture/MoE badges the HuggingFace search page shows (discRepoRow
    // below), now available for an already-registered model too - hard
    // metadata read once at registration/pull time, not a name guess.
    // Falsy-checked (m.architecture, m.expert_count > 0), never a fallback
    // default: a model registered before this existed, or whose header could
    // not be read, has these as null/undefined and correctly shows NEITHER
    // badge - never a false "not MoE" claim about a model nobody has checked.
    if (m.architecture) {
      nameLine.appendChild(el("span", "arch-badge", m.architecture));
    }
    if (m.expert_count > 0) {
      nameLine.appendChild(el("span", "moe-badge moe-confirmed", MOE_LABEL.confirmed));
    }
    const visBadge = visionBadge(m.vision);
    if (visBadge) nameLine.appendChild(visBadge);
    if (m.active) nameLine.appendChild(el("span", "active-tag job-state st-ok", "active"));
    // Independent of "active": a model can sit resident in VRAM without being
    // the one currently serving requests - surfaced so a background-loaded
    // model is never invisible/indistinguishable from one that was never
    // loaded at all. Deliberately NOT the active-tag class here (it used to be,
    // which made "loaded" render identically to "active" - .loaded-tag had no
    // CSS of its own - and incorrectly triggered the tr:has(.active-tag) row
    // highlight for a merely-resident model); job-state's "on" variant gives it
    // its own distinct look instead.
    else if (m.loaded) nameLine.appendChild(el("span", "loaded-tag job-state on", "loaded"));
    tr.appendChild(nameTd);
    
    // Role column. The set-type control IS this column - one pill that both shows
    // the type and changes it, wearing the same .type-<name> colour the read-only
    // badge used to, so the column still scans by colour. It was previously a badge
    // here plus a separate <select> over in the actions cell: the same fact twice,
    // and the select was the widest thing in the row.
    const roleTd = el("td", "mono shrink-cell");
    const roleType = m.model_type || "llm";

    // On every row (outside the LLM-only gate below): an 'unknown'/media model
    // otherwise has no controls, and this is how a mis-detected model is
    // reclassified without the CLI. Changing it POSTs the chosen type, then
    // re-renders (so an unknown->llm switch reveals use/alias).
    const typeSel = el("select", "model-type-select type-badge type-" + roleType);
    typeSel.title = "Change this model's type";
    typeSel.setAttribute("aria-label", `Model type for ${m.name}`);
    for (const t of MODEL_TYPE_OPTIONS) {
      const opt = el("option", "", t);
      opt.value = t;
      if ((m.model_type || "llm") === t) opt.selected = true;
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
        if (!r.ok) { toast(data.detail || "Set type failed", true); return; }
        toast(`Set '${m.name}' type to ${typeSel.value}`);
        refreshModelsPage();
        refreshPerfEstimate();
      } finally { typeSel.disabled = false; }
    };
    roleTd.appendChild(typeSel);
    tr.appendChild(roleTd);

    // The source is reference text, not the row's identity, so it clips to an
    // ellipsis with the full value on hover rather than wrapping the row.
    const sourceTd = el("td", "mono clip-cell", m.source || "");
    if (m.source) sourceTd.title = m.source;
    tr.appendChild(sourceTd);
    tr.appendChild(el("td", "mono shrink-cell", fmtSize(m.size_bytes)));
    tr.appendChild(el("td", "mono shrink-cell", fmtModelDate(m.mtime)));

    const actions = el("td", "actions-cell");
    const detail = el("button", "secondary", "info");
    detail.onclick = () => showModelDetail(m.name);
    actions.appendChild(detail);

    // Only LLMs support use/alias/remove in Phase 1
    const isLlm = !m.model_type || m.model_type === "llm";
    if (isLlm) {
      if (!m.active) {
        const use = el("button", "primary", "use");
        use.onclick = async () => {
          // switchModel() already drives the sidebar's #status-text pill
          // for the real load duration (a genuine blocking VRAM-check/evict/
          // load, not fire-and-forget) - but that status line lives inside the
          // off-canvas #sidebar, invisible on mobile until the drawer opens, and
          // easy to miss even on desktop since it is not on this button. Give
          // the button its OWN inline cue too, mirroring settings.js's
          // comfyAction prev-label/busy-label swap (the established pattern for
          // an async button in this codebase) rather than the bigger
          // .reconnect-spinner, which is sized for a full-panel state, not an
          // inline table button.
          const prevLabel = use.textContent;
          use.disabled = true;
          use.textContent = "loading…";
          try {
            const res = await switchModel(m.name);
            if (!res || res.status !== "superseded") {
              toastLoadResult(res, m.name);
              refreshModelsPage();
              // Keep the Settings "Live tuning" VRAM estimate (which defaults to
              // the active model) in sync with a switch made from this page too.
              refreshPerfEstimate();
            }
          } catch (e) {
            toast("Load failed: " + e.message, true);
          } finally { use.disabled = false; use.textContent = prevLabel; }
        };
        actions.appendChild(use);
      }
      const aliasBtn = el("button", "secondary", "alias");
      aliasBtn.onclick = async () => {
        const name = prompt(`New alias for '${m.name}':`);
        if (!name) return;
        const r = await fetch("/api/models/alias", {
          method: "POST", headers: authHeaders(),
          body: JSON.stringify({ model: m.name, alias: name.trim() }),
        });
        const data = await r.json().catch(() => ({}));
        // Report the alias the SERVER stored, not the raw text: aliases are
        // sanitized server-side ("daily driver" -> "daily-driver"), so echoing the
        // input would name a key that does not exist in the registry (REG-562).
        if (r.ok) { toast(`Aliased as '${data.alias || name.trim()}'`); refreshModelsPage(); }
        else toast(data.detail || "Alias failed", true);
      };
      actions.appendChild(aliasBtn);
      const renameBtn = el("button", "secondary", "rename");
      renameBtn.title = "Rename this model (unlike alias, the old name stops working; "
        + "any OTHER name already pointing at the same file is unaffected)";
      renameBtn.onclick = async () => {
        const name = prompt(`Rename '${m.name}' to:`, m.name);
        if (!name || name.trim() === m.name) return;
        const r = await fetch("/api/models/rename", {
          method: "POST", headers: authHeaders(),
          body: JSON.stringify({ model: m.name, new_name: name.trim() }),
        });
        const data = await r.json().catch(() => ({}));
        // Same discipline as alias above: the server sanitizes the name, so
        // report what it actually stored, not the raw text typed in.
        if (r.ok) {
          let msg = `Renamed to '${data.new_name || name.trim()}'`;
          // The server's migration notes - what it updated, and what it could
          // NOT reach (e.g. a per-project .localcoder/config.toml) - must
          // reach the user here, not sit only in the server log (AGENTS.md
          // rule 5: a partial outcome reported and then discarded is the same
          // as hiding it).
          if (Array.isArray(data.notes) && data.notes.length) {
            msg += ". " + data.notes.join(" ");
          }
          toast(msg);
          refreshModelsPage();
        } else toast(data.detail || "Rename failed", true);
      };
      actions.appendChild(renameBtn);
      if (m.loaded) {
        const unload = el("button", "secondary", "unload");
        unload.title = "Release this model from GPU/CPU memory (it reloads "
          + "automatically on the next chat request)";
        unload.onclick = async () => {
          unload.disabled = true;
          try {
            const r = await fetch("/api/models/unload", {
              method: "POST", headers: authHeaders(),
              body: JSON.stringify({ model: m.name }),
            });
            const data = await r.json().catch(() => ({}));
            if (!r.ok) { toast(data.detail || "Unload failed", true); return; }
            // unload_one_model() (http_server.py) answers HTTP 200 for an in-use
            // engine too - a legitimate "not done yet, not an error" outcome (a
            // request is mid-generation against it right now), not the same
            // thing as a real unload. Reporting "Unloaded" here regardless of
            // `status` would claim a VRAM release that did not happen (AGENTS.md
            // rule 5). Mirrors the sidebar quick-unload handler's fix
            // (models-sidebar.js sidebarUnloadBtn.onclick).
            if (data.status === "in_use") {
              toast(`'${m.name}' is still generating - try again once it finishes`, true);
              refreshModelsPage();
              return;
            }
            toast(`Unloaded '${m.name}'`);
            refreshModelsPage();
            refreshPerfEstimate();
          } finally { unload.disabled = false; }
        };
        actions.appendChild(unload);
      }
      if (!m.active) {
        const rm = el("button", "danger", "remove");
        rm.onclick = () => {
          confirmDanger(`Remove '${m.name}'?`,
            "The file is deleted only when this is its last name and it lives " +
            "in the data directory.", "Remove", async () => {
              const r = await fetch("/api/models/remove", {
                method: "POST", headers: authHeaders(),
                body: JSON.stringify({ model: m.name }),
              });
              const data = await r.json().catch(() => ({}));
              if (!r.ok) { toast(data.detail || "Remove failed", true); return; }
              const end = await streamJob(data.job_id, null);
              // "disconnected" (streamJob gave up reconnecting, or the job was
              // already gone) is NOT the same fact as the remove having
              // failed - do not claim it did. refreshModelsPage() below shows
              // the real current state regardless.
              if (end.status === "done") {
                toast(`Removed '${m.name}'`);
              } else if (end.status === "disconnected") {
                toast("Lost connection - check the list below for the current state", true);
              } else {
                toast("Remove failed", true);
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
  box.appendChild(table);
}

export async function showModelDetail(name) {
  const r = await fetch(`/v1/models/${encodeURIComponent(name)}`, {
    headers: authHeaders() });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) { toast(data.detail || "Lookup failed", true); return; }
  openModal("Model - " + name, (body) => {
    const modelType = data.model_type || "llm";
    const rows = [
      ["Path", data.path],
      ["Type", null],
      ["Source", data.source],
      ["Size", fmtSize(data.size_bytes)],
      ["SHA256", data.sha256 || "(not computed yet - hashes lazily on use)"],
      ["Aliases", data.aliases.length ? data.aliases.join(", ") : "(none)"],
      ["Status", data.active ? (data.loaded ? "active, loaded" : "active, not loaded") : "registered"],
    ];
    for (const [k, v] of rows) {
      const row = el("div", "log-entry");
      row.appendChild(el("span", "t", k));
      if (k === "Type") {
        row.appendChild(el("span", "type-badge type-" + modelType, modelType));
        // Beside the type, not as a row of its own: a "Capabilities" row would
        // have to render SOMETHING when there is no confirmed capability, and
        // the only honest something for an unknown is nothing at all. An
        // absent pill says nothing either way, which is the whole contract.
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
// byte size - fmtSize is for bytes. A separate formatter so the two are never
// mixed up at a call site.
export function fmtParamCount(n) {
  if (!n) return "";
  if (n >= 1e9) return (n / 1e9).toFixed(1) + "B params";
  if (n >= 1e6) return (n / 1e6).toFixed(1) + "M params";
  return String(n) + " params";
}

export const FIT_TEXT = { "fits": "fits your VRAM", "tight": "tight fit",
                   "too-big": "needs partial CPU offload" };

// moe: "confirmed" (the model's own architecture header says MoE - reliable)
// or "likely" (a name-pattern guess only, e.g. "8x7B"/"A3B" in the repo id -
// see discover.py's _moe_signal). The label and tooltip make that distinction
// visible; never presented as equally certain.
const MOE_LABEL = { confirmed: "MoE", likely: "MoE?" };

// Model CAPABILITY pills - what a model can DO, deliberately separate from the
// .fit.fits/.tight/.too-big scale, which grades file SIZE against VRAM and
// means something entirely different. Vision is the only capability with a
// real detector today (registry.model_vision_capability, the same lookup the
// load path uses); tool-use and reasoning are a separate, larger design item
// and are NOT inferred here from a filename.
const CAP_LABEL = { vision: "vision" };
const CAP_TITLE = { vision: "Accepts image input on this install - a vision projector (mmproj) or HF vision metadata was found beside the model" };

/** The vision pill for a `vision` field, or null when there must be no pill.
 *
 *  ONE function for the list row AND the detail modal on purpose: two separate
 *  reads of a tri-state is two chances to write `if (v)` and collapse "checked,
 *  text-only" together with "COULD NOT CHECK".
 *
 *  STRICT `=== true`, never truthiness: the server sends true / false / NO KEY
 *  AT ALL, and the absent case means the model's files could not be inspected
 *  (an unmounted drive, a dead UNC share), NOT that it is text-only. false and
 *  absent both render nothing, and nothing is all they may render - a "no
 *  vision" label on a model nobody could check is exactly the false claim the
 *  F8 architecture/MoE badges are written to avoid. */
function visionBadge(v) {
  if (v !== true) return null;
  const badge = el("span", "cap-badge cap-vision", CAP_LABEL.vision);
  badge.title = CAP_TITLE.vision;
  return badge;
}
const MOE_TITLE = { likely: "Inferred from the repo name - not confirmed by the model's own header" };

export const FMT_LABEL = { gguf: "GGUF", hf: "HF" };

/** Fetch the GPU list AND the configured split once per search/files-load (not
 *  once per row) - used for the split-fit hint and the VRAM-basis caption below.
 *  Empty on any failure (server unreachable, no scope) so both simply degrade,
 *  never a broken search. */
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

/** The GPU array alone (for the split hint / row rendering). */
async function _splitGpus() { return (await _gpuInfo()).gpus; }

/** Caption naming what the fit-badge VRAM number actually is, so it never
 *  mislabels a single main-GPU ceiling as the machine "total" (a 2x16 GB box
 *  with no split has a 16 GB main-GPU ceiling, not 32) and always agrees with
 *  the split hint. Mirrors discover.vram_capacity: the number is COMBINED only
 *  when 2+ configured split indices map to detected devices; otherwise it is the
 *  single main GPU's. `gpuInfo` is {gpus, gpu_split_indices} from _gpuInfo(). */
export function vramBasisCaption(totalBytes, gpuInfo) {
  const gib = (totalBytes / GIB).toFixed(0);
  const gpus = Array.isArray(gpuInfo?.gpus) ? gpuInfo.gpus : [];
  const rawSplit = Array.isArray(gpuInfo?.gpu_split_indices) ? gpuInfo.gpu_split_indices : [];
  // Dedup and keep only indices that map to a detected device - the same
  // resolve_gpu_split validation vram_capacity() applies before it combines.
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
 *  across every device. A ROUGH client-side suggestion only - the server's own
 *  fit badges (`m.fit`/`f.fit`) remain the authoritative numbers, and the real
 *  split load still applies a per-device VRAM check (gpu_split_shortfall) that
 *  can refuse it, which is why this is hedged ("may fit") not a promise. It uses
 *  the same need-math as the fit badge (weights * 1.10 + ~1.5 GB overhead) rather
 *  than a raw byte sum so it does not over-promise. Never auto-enables anything
 *  (the "never silently override a user's explicit choice" rule) - it only points
 *  at the "Split across GPUs" control that actually enables a split. */
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
  // the checkboxes are absent (older DOM) so search never silently returns
  // nothing. Empty (both unchecked) is a real state the caller handles.
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
  // The search-page TYPE checkboxes -> the model_types list. Explicit and
  // always visible, independent of the Registered-models tab. Defaults to all
  // types when the checkboxes are absent (older DOM). Empty (none ticked) is a
  // real state the caller surfaces, same as no format.
  const boxes = [...document.querySelectorAll(".disc-type")];
  if (!boxes.length) return ALL_SEARCH_TYPES.slice();
  return boxes.filter((b) => b.checked).map((b) => b.value);
}

// The model_type hint carried into a pull for a chosen discovery result. What
// you SEE badged is what it registers as (detected_type), so the two never
// disagree. When HF gave no confident type ("unknown" - the common case for a
// standalone VAE / text-encoder, whose repos carry no type metadata) but the
// user narrowed the search to exactly ONE type, that single explicit choice is
// the hint. Otherwise let the pull auto-detect (never force "unknown").
function resolveTypeHint(detectedType) {
  if (detectedType && detectedType !== "unknown") return detectedType;
  const types = discTypes();
  return types.length === 1 ? types[0] : null;
}

function showHfHint() {
  const hint = $("disc-hf-hint");
  if (!hint) return;
  // Non-blocking: HF (transformers) models still DOWNLOAD, they just cannot RUN
  // until torch + transformers are installed. State that, do not block the pull.
  hint.textContent = "No transformers runtime detected. HF models will download, "
    + "but need the [gpu] extra (torch + transformers) installed to run.";
  hint.style.display = "block";
}
function hideHfHint() {
  const hint = $("disc-hf-hint");
  if (hint) hint.style.display = "none";
}

// A bare owner/repo (no :file.gguf) tells `localm pull` to fetch the WHOLE
// transformers repo -> the HF backend. Prefill the Add box and let the user
// confirm (they can still just download the files, backend or not).
function prefillHfPull(repo, detectedType) {
  $("pull-spec").value = repo;
  $("pull-name").value = repo.split("/").pop();
  const typeHint = resolveTypeHint(detectedType);
  pendingPullTypeHint = typeHint ? { spec: repo, type: typeHint } : null;
  const mmprojSelect = $("pull-mmproj");
  if (mmprojSelect) { mmprojSelect.replaceChildren(); mmprojSelect.style.display = "none"; }
  const nameInput = $("pull-name");
  // scrollIntoView is absent in some environments (e.g. jsdom); guard so the
  // prefill never throws where it is unimplemented. Browsers have it.
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
// Type checkboxes (resolveTypeHint), so a vae/text-encoder whose HF metadata
// gave no type still registers under the type the user searched for.
function discRepoRow(m, gpus) {
  const row = el("div", "disc-repo");
  const head = el("div", "head");
  head.appendChild(iconEl("models", "ic ic-model"));
  head.appendChild(el("span", "name", m.id));
  // Best-effort HF classification, DISPLAY ONLY - never gates whether a result
  // is shown (HF has no reliable type signal for standalone vae/text-encoder).
  if (m.detected_type) {
    head.appendChild(el("span", "type-badge type-" + m.detected_type, m.detected_type));
  }
  // What the model actually IS: architecture family, MoE-ness, param
  // count - all DISPLAY ONLY, from discover.py's classified-row fields, never
  // gating which results show. Not colored into the type-badge palette (all 7
  // --cat-* hues are already spoken for by MODEL_TYPES) - a shared hue here
  // would misread as "this is the model's type".
  if (m.architecture) {
    head.appendChild(el("span", "arch-badge", m.architecture));
  }
  if (m.moe) {
    const moeBadge = el("span", "moe-badge moe-" + m.moe, MOE_LABEL[m.moe] || "MoE");
    if (MOE_TITLE[m.moe]) moeBadge.title = MOE_TITLE[m.moe];
    head.appendChild(moeBadge);
  }
  if (m.param_count) {
    head.appendChild(el("span", "param-count", fmtParamCount(m.param_count)));
  }
  const fmts = Array.isArray(m.formats) ? m.formats : ["gguf"];
  for (const f of fmts) head.appendChild(el("span", "fmt-badge fmt-" + f, FMT_LABEL[f] || f));
  // HF repos pull whole, so show total size + a VRAM fit badge inline (from the
  // server's safetensors param estimate). When the estimate is unknown (no
  // safetensors metadata) say so - never guess a fit. GGUF results are sized
  // per-quant in the files expander instead.
  if (fmts.includes("hf")) {
    if (m.size_bytes) {
      head.appendChild(el("span", "disc-hf-size", fmtSize(m.size_bytes)));
      if (m.fit) head.appendChild(el("span", "fit " + m.fit, FIT_TEXT[m.fit]));
      // "fits"/"tight" are already combined-aware (server sums capacity
      // across a configured split - see discover.vram_capacity), so a "may
      // not fit on one GPU, go configure a split" suggestion would be
      // stale/contradictory right next to a badge that already accounts for
      // the split.
      const hint = (m.fit === "fits" || m.fit === "tight")
        ? "" : splitFitHint(m.size_bytes, gpus);
      if (hint) head.appendChild(el("span", "sub split-hint", hint));
    } else {
      head.appendChild(el("span", "disc-hf-size sub", "size unknown"));
    }
  }
  // Downloads + likes as inline SVGs (no emoji glyphs on the shipping surface).
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

export async function discoverSearch() {
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
    // A dashed "MoE?" pill's meaning must not depend on a hover-only tooltip
    // (touch devices have no hover at all) - shown once, persistently, only
    // when a result on screen actually carries that inferred-not-confirmed
    // signal, so it never clutters a search with no MoE-named results.
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

export async function discoverFiles(repo, filesBox, btn, gpus, detectedType) {
  if (filesBox.childElementCount) {            // toggle collapse
    filesBox.replaceChildren();
    return;
  }
  btn.disabled = true;
  filesBox.replaceChildren(el("div", "sub", "loading file list…"));
  if (!Array.isArray(gpus)) gpus = await _splitGpus();   // direct call, e.g. from a test
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
      // See the discRepoRow comment above: "fits"/"tight" already account
      // for a configured split, so skip the redundant split-suggestion hint.
      const splitHint = (f.fit === "fits" || f.fit === "tight")
        ? "" : splitFitHint(f.size_bytes, gpus);
      if (splitHint) row.appendChild(el("span", "sub split-hint", splitHint));
      row.appendChild(el("span", "fname", f.file));
      const pull = el("button", "btn-secondary", "pull");
      pull.onclick = () => {
        // Prefill the pull form - the user confirms (and can set an alias)
        // before anything downloads. The suggested alias mirrors the
        // server's default name (file name without .gguf).
        $("pull-spec").value = `${repo}:${f.file}`;
        $("pull-name").value = f.file.replace(/\.gguf$/i, "");
        // A vision-projector companion file is never the searched-for type
        // (there is no mmproj tab/checkbox) - only hint a REGULAR file's own
        // pull, not one drawn from data.mmprojs (same object reference, "show
        // mmproj files" merge above). The hint resolves at click time from the
        // repo's detected type and the current Type checkboxes.
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
        // this per-file pull button matches prefillHfPull's existing guard above.
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

// Mirror a filter checkbox's state onto its chip label as an `.on` class, which
// is what style.css colours (a CSS :checked-combinator does not repaint on a
// programmatic toggle in every engine - see the .disc-chip comment there).
function _syncChip(box) {
  const chip = box.closest(".disc-chip");
  if (chip) chip.classList.toggle("on", box.checked);
}

// Restore + persist a search filter checkbox (all default on). Keep its chip in
// sync, and re-run the search on change when results are already showing so a
// toggle reflects live.
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
// The explicit model-TYPE checkboxes, persisted per type so a chosen scope
// sticks across reloads (keyed by value, e.g. localm.discType.vae).
for (const box of document.querySelectorAll(".disc-type")) {
  _bindDiscToggle(box, "localm.discType." + box.value);
}

// add-models-disk: make adding a model already on disk discoverable - pick a
// folder on this machine and drop its path into the spec field (the /api/models/
// pull endpoint already accepts a local folder/file path). The user no longer has
// to guess that pasting a path works.
document.addEventListener("click", async (e) => {
  if (e.target && e.target.id === "pull-browse") {
    const spec = $("pull-spec");
    if (!spec) return;
    const dir = await pickDirectory("Pick a folder that holds the model(s)",
                                    spec.value.trim());
    if (dir) spec.value = dir;
  }
});

// R18: restart the server in place from Settings - the backend unloads the model,
// then re-execs the same process, so it comes back on the same port. The reconnect
// overlay polls and auto-reconnects once the fresh process is up.
if ($("server-restart")) {
  $("server-restart").onclick = () => {
    confirmDanger("Restart the server?",
      "This restarts the LocaLM server (the model is unloaded first, then reloaded). " +
      "It will be briefly unavailable, then reconnect automatically.",
      "Restart", async () => {
        try {
          const r = await fetch("/v1/server/restart",
                                { method: "POST", headers: authHeaders() });
          if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
          toast("Server restarting…");
          // The server briefly goes away and comes back; the reconnect overlay
          // polls and reconnects automatically once the new process is up.
          if (window.onServerUnreachable) setTimeout(() => onServerUnreachable(), 800);
        } catch (e) { toast("Could not restart: " + e.message, true); }
      });
  };
}

// R18: shut the server down cleanly from Settings (the backend unloads the model
// before exit) instead of force-closing the window. Start it again from the launcher.
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
          // The server is going away; show the reconnect overlay rather than a dead app.
          if (window.onServerUnreachable) setTimeout(() => onServerUnreachable(), 800);
        } catch (e) { toast("Could not shut down: " + e.message, true); }
      });
  };
}

// R47: file a bug report from Settings. "Save report" writes an editable markdown
// report to the data folder (safe snapshot, optional log tail - never secrets or
// chat). "Send to maintainer" (shown only when an upload endpoint is configured,
// via capabilities.bugreport_upload) ALSO files it as a GitHub issue through the
// proxy, so a tester needs no GitHub account. A failed upload is reported honestly
// (the file is still saved), never as success.
// The most recent saved report's markdown + filename, so "Download report" can
// hand the tester the file to send manually when an upload fails.
let _lastBugReport = null;

function _showBugActions({ retry, download }) {
  const r = $("bug-retry"), d = $("bug-download");
  if (r) r.hidden = !retry;
  if (d) d.hidden = !download;
}

// Download the last saved report as a .md so the tester can email/Discord it -
// works from a phone or another LAN device where a server-side path is useless.
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
  // "What were you doing" and "what happened" are the two fields most likely
  // to carry the actual report; either alone is enough to send (mirrors the
  // server's own "description or what_happened" check).
  if (!desc && !happened) { toast("Describe the problem first", true); return; }
  const includeLog = !!($("bug-include-log") && $("bug-include-log").checked);
  const saveBtn = $("bug-send"), upBtn = $("bug-upload");
  if (saveBtn) saveBtn.disabled = true;
  if (upBtn) upBtn.disabled = true;
  _showBugActions({ retry: false, download: false });   // fresh attempt: reset
  // Browser context so the report carries what actually broke in the page (env
  // snapshot + server state are added server-side). Sanitized + capped on the
  // server; rendered as plain text, never executed.
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
    // Rate limited: if auto-retry is on (default), keep the user's text, count down,
    // and retry the send ONCE; if the toggle is off, fall through to a plain notice.
    const autoRetry = !$("bug-autoretry") || $("bug-autoretry").checked;
    if (upload && data.rate_limited && !isRetry && autoRetry) {
      const secs = Math.max(1, parseInt(data.retry_after, 10) || 30);
      await countdownRetryBugReport(secs, out);
      return;   // the retry (and the finally below) re-enable the buttons
    }
    const where = data.path || data.filename || "report";
    const sent = upload && data.uploaded;
    const uploadFailed = upload && data.upload_error && !data.rate_limited;
    // Stash the saved report so "Download report" can hand it to the tester for
    // manual sending (a server-side path is useless from a phone / another device).
    _lastBugReport = data.report_markdown
      ? { markdown: data.report_markdown, filename: data.filename || "bug-report.md" }
      : null;
    // On a failed send, offer Retry (re-file the issue) and Download (send it
    // yourself); on success, keep them hidden.
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
        // Tell the user WHERE it failed (the diagnosed message), and that the
        // report is kept - they can retry, download it, or email it.
        out.textContent = "Could not send: " +
          (data.upload_message || data.upload_error) +
          "  The report is saved" + (where ? " (" + where + ")" : "") +
          " - retry, download it, or email " + (data.maintainer || "the maintainer") + ".";
      } else {
        out.textContent = "Saved: " + where +
          (data.maintainer ? "  -  send it to " + data.maintainer : "");
      }
    }
    // Keep the description on ANY failed send so Retry re-uses it; clear it once the
    // report is genuinely done with (sent, or a plain save with no upload attempt).
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

// Show a live countdown in the bug-report result line, then auto-retry the send once
// (isRetry=true, so it never loops). The caller's buttons stay disabled throughout,
// so this cannot be double-fired.
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

// Updates: check-only auto-surface (a throttled startup check in app.js calls
// __localmUpdateCheck) + an explicit "Update now". localm never self-updates; the
// apply runs only on this click and the server rolls back + reports honestly on
// failure.
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
      // is_newer() returns False BOTH for a genuine tie/older release AND for a
      // tag it could not parse as a version at all (nightly, stable, release-5
      // all measured False). Without this branch the GUI printed "up to date"
      // for the second case - a false reassurance, and pushed UNPROMPTED, since
      // app/settings-perf.js calls this on a throttled startup check as well as
      // on the button. Keyed like cli/maintenance.py's .get("comparable", True):
      // strict === false, so an absent key (an older server) still means
      // comparable and the pre-existing wording is unchanged.
      // Wording deliberately carries NO "up to date" substring, unlike the CLI's
      // otherwise-identical sentence ("...before assuming you are up to date").
      // The CLI ships that inside a bold yellow warning; this is plain text in
      // one status line, where a skimmed tail would read as the very
      // reassurance the branch exists to withhold. It also lets the regression
      // test assert the phrase is ABSENT outright rather than absent-except-here.
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

// Runtime update: the native llama.cpp binaries `localm setup-llama`
// provisions, separate from the "Updates" card above (the Python source
// tree). Check is read-only (GET /api/runtime/check); apply streams a job
// (POST /api/runtime/update), since a real download + native load-test can
// take a while - same shape as the managed-ComfyUI update panel's job log,
// unlike the plain source-tree update above which is a single blocking call.
export async function runtimeUpdateCheck() {
  const out = $("runtime-update-status"), applyBtn = $("runtime-update-apply");
  if (out) { out.hidden = false; out.textContent = "Checking..."; }
  try {
    const r = await fetch("/api/runtime/check", { headers: authHeaders() });
    const d = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(d.detail || r.statusText);
    if (!out) return;
    if (!d.installed) {
      out.textContent = "No llama.cpp runtime is installed yet - run setup first.";
      if (applyBtn) applyBtn.hidden = true;
      return;
    }
    const current = d.current || "an unrecorded build";
    if (d.newer) {
      out.textContent = "A different build is available for " + d.backend + ": " +
        d.target + " (you have " + current + ")." +
        (d.pinned ? " Pinned to " + d.pinned + "." : "");
      if (applyBtn) applyBtn.hidden = false;
    } else {
      out.textContent = "Up to date (" + d.backend + " " + current + ")." +
        (d.pinned ? " Pinned to " + d.pinned + "." : "");
      if (applyBtn) applyBtn.hidden = true;
    }
  } catch (e) { if (out) { out.hidden = false; out.textContent = "Could not check: " + e.message; } }
}

export async function runtimeUpdateApply() {
  const out = $("runtime-update-status"), btn = $("runtime-update-apply");
  const log = $("runtime-update-log");
  if (btn) btn.disabled = true;
  if (out) { out.hidden = false; out.textContent = "Updating the runtime..."; }
  if (log) { log.style.display = ""; log.textContent = ""; }
  let jobId;
  try {
    const r = await fetch("/api/runtime/update", { method: "POST", headers: authHeaders() });
    const d = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(d.detail || r.statusText);
    jobId = d.job_id;
  } catch (e) {
    if (out) out.textContent = "Update failed: " + e.message;
    if (btn) btn.disabled = false;
    if (log) log.style.display = "none";
    return;
  }
  const tail = [];
  const end = await streamJob(jobId, (line) => {
    if (log) { log.textContent += line + "\n"; log.scrollTop = log.scrollHeight; }
    // A short tail, not just the last line: setup-llama's failure messages
    // wrap across several printed lines, so the final line alone is often a
    // fragment with the actual reason lost.
    if (line && line.trim()) { tail.push(line.trim()); if (tail.length > 6) tail.shift(); }
  });
  const ok = !!(end && end.status === "done");
  if (out) {
    out.textContent = ok ? "Runtime updated." :
      (tail.join(" ").trim() || "The update did not finish. See the log below.");
  }
  if (btn) { btn.disabled = false; if (ok) btn.hidden = true; }
  if (ok) runtimeUpdateCheck();
}
if ($("runtime-update-check")) $("runtime-update-check").onclick = runtimeUpdateCheck;
if ($("runtime-update-apply")) $("runtime-update-apply").onclick = runtimeUpdateApply;

// Changelog: show the RELEASED history (newest first) in the shared modal. The
// endpoint strips the in-progress [Unreleased] section before serving, so this
// never shows changes that are not in the running build.
// Fetched from /api/changelog and rendered via renderMarkdown - the same
// DOMPurify(marked) path chat uses, so it is XSS-safe. Always available (it ships
// in every build); a missing/failed fetch is shown in the modal, never left blank.
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
    // gui-design.md rule 7: a designed empty state, not a lone .sub line. Note
    // this is reached only on a SUCCESSFUL fetch of an empty list - the error
    // paths above and below keep saying "could not load", because "nothing is
    // open" and "I could not ask" are different answers.
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

// R30: export all logs of this instance to a folder the user picks (reuses the
// shared directory-picker modal), then POSTs the chosen path to /api/logs/export.
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
        // A partial export reports which files failed - surface it, do not hide it.
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

// R37: upload files from this device or a phone into <home>/uploads/ so models and
// tools can read them (beyond transient chat attachments). The POST is multipart;
// we strip the JSON Content-Type so the browser sets multipart/form-data with its
// own boundary, but keep the auth + CSRF headers. CONFIG_WRITE-gated server-side.
// The list is built with safe DOM nodes (textContent), never innerHTML, so a
// crafted file name cannot inject markup.
export async function refreshUploadsList() {
  const list = $("upload-list");
  if (!list) return;
  try {
    const r = await fetch("/api/uploads", { headers: authHeaders() });
    if (!r.ok) { list.replaceChildren(); return; }   // e.g. a read-only key: hide
    const data = await r.json().catch(() => ({ items: [] }));
    list.replaceChildren();
    // gui-design.md rule 7: an empty <ul> is the blank scroll area the rule
    // names. Distinct from the read-only-key early return above, which leaves
    // the list genuinely blank on purpose - "you may not see this" is not the
    // same statement as "there is nothing here".
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
  const storeInput = $("pull-store");
  const store = storeInput && storeInput.value ? storeInput.value : null;

  $("pull-start").disabled = true;
  const log = $("pull-log");
  log.style.display = "block";
  log.textContent = "";
  const prog = $("pull-progress");
  const bar = $("pull-bar");
  const pct = $("pull-pct");
  // Show a live (indeterminate) bar from the start so a job that fails before
  // emitting any progress is visibly running rather than a blank panel.
  prog.style.display = "block";
  bar.classList.remove("failed");
  bar.classList.add("indeterminate");
  bar.style.width = "35%";
  pct.textContent = "starting…";
  if ($("pull-file")) { $("pull-file").hidden = true; $("pull-file").textContent = ""; }
  const samples = [];   // rolling {t, downloaded} window for speed/ETA (U4)
  try {
    const payload = { spec, name: name || null };
    if (mmproj) payload.mmproj = mmproj;
    if (store) payload.store = store;
    // Only attach the type hint if the spec still matches exactly what it was
    // prefilled for - a hand-edited spec after picking a discovery result
    // silently falls back to today's auto-detect rather than sending a hint
    // for a now-different model.
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
      // R06: for a multi-file (split GGUF) download, show which file is in flight.
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
        // Smoothed speed + ETA from a rolling ~10-sample window (U4).
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
      if (storeInput) storeInput.value = "";
      pendingPullTypeHint = null;
      refreshModelsPage();
    } else if (end.status === "disconnected") {
      // streamJob gave up reconnecting (or the job was already gone) - this is
      // NOT a job failure, just a lost transport: the pull may well still be
      // running or have already finished server-side (verified live: a
      // dropped SSE connection does not stop the job). Never claim "failed"
      // for a fact we do not have; keep the bar as-is rather than painting it
      // red, so it does not look like a completed-and-wrong outcome either.
      pct.textContent = "connection lost - it may still be running";
      toast("Lost connection to the pull - it may still be running in the "
            + "background. Check the Models list in a moment.", true);
    } else {
      // Surface the failure: red bar, exit code, and keep the inputs so the
      // user can see what they typed and retry.
      bar.classList.remove("indeterminate");
      bar.classList.add("failed");
      const code = end.returncode != null ? `, exit ${end.returncode}` : "";
      pct.textContent = `failed${code} - see log`;
      toast(`Pull failed (${end.status}${code}) - see log`, true);
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
// trailing span, so the text a click handler or a test reads by dataset.type is
// untouched. A type with nothing registered shows no number rather than a "0",
// which would add noise to six tabs on a fresh install; "All" carries the total.
//
// The per-type numbers can legitimately sum to LESS than All: a registry entry
// whose model_type has no tab of its own (mmproj) is reachable from All only.
// That is the same set of rows each tab has always shown, now visible.
export function syncTabCounts(models) {
  const nav = $("models-tab-nav");
  if (!nav) return;
  const counts = new Map();
  for (const m of models || []) {
    const t = m.model_type || "llm";
    counts.set(t, (counts.get(t) || 0) + 1);
  }
  for (const btn of nav.querySelectorAll(".tab-btn")) {
    const n = btn.dataset.type === "all"
      ? (models || []).length
      : (counts.get(btn.dataset.type) || 0);
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

// Turn a ComfyUI ScanResult (added / skipped / method) into a human toast. The
// scanner's `method` field carries WHY an empty scan found nothing ("none
// (comfy_workdir not configured)", "none (models folder not found under
// {path})"); surfacing that reason instead of a bare "Added 0" keeps a
// misconfigured scan from failing silently (AGENTS.md rule 5). The internal
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
    // Any other "none (...)" reason: still say something specific, never hide it.
    return "Scan found no ComfyUI models. Check the ComfyUI setup in Settings.";
  }
  if (added === 0 && skipped === 0) {
    // A real scan (folder-walk / hybrid) that turned up nothing: the folders
    // exist but held no models. Not a misconfig, but clearer than "Added 0".
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
    // Live (indeterminate) bar from the start, same shape as the pull flow's
    // #pull-progress - registering ComfyUI's own found_files.items() has no
    // total until the directory walk finishes, so the FIRST real feedback is
    // this busy bar, then it flips to a real "N of M" once registration starts.
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
        // Same distinction streamJob's doc makes for the pull flow: a lost
        // SSE connection is not a job failure, so the bar is left as-is
        // rather than painted red for a fact we do not have.
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

// import-comfy: the guided "Import from ComfyUI..." flow (Add-a-model card).
// Unlike the Scan button above (which always scans whatever comfy_workdir is
// configured in Settings), this lets the user point at ANY ComfyUI folder for
// a one-off import - previewing counts per category before anything registers -
// without touching the persistent comfy_workdir setting. Friendly labels for the
// dry-run preview breakdown; mirrors localm.model_manager.registry.MODEL_TYPES
// (keep in sync), minus "llm"/"mmproj" which a ComfyUI scan never produces.
export const IMPORT_TYPE_LABELS = {
  "diffusion-unet": "Diffusion / checkpoints",
  "text-encoder": "Text encoders",
  vae: "VAEs",
  lora: "LoRAs",
  unknown: "Other",
};

// Best-effort: localm's own managed ComfyUI install path, or null if there
// isn't one (not installed, or this key lacks config:read). Never blocks the
// primary Browse flow - a failed/403 fetch just means the quick-fill option
// doesn't appear.
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

// A dry-run preview response with a "none (...)" method (no folder configured /
// found) reuses the same humanized reasons scanResultMessage already decodes for
// the real-scan case, so the wording is never duplicated. Returns null when the
// preview is a normal (possibly empty) result the caller should render counts for.
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
    // #modal/#modal-body (openModal() replaces its content, it does not
    // stack) - so calling it from inside this already-open modal destroys
    // THIS modal's own DOM (pathInput included) the moment the folder picker
    // renders. Writing the result into `pathInput.value` afterward would land
    // on an already-detached node - no visible effect, the picker just closes
    // with nothing appearing to happen. Re-open this same modal fresh with
    // the picked path instead of trying to keep the old DOM alive across it.
    browseBtn.onclick = async () => {
      const dir = await pickDirectory("Pick a ComfyUI folder", pathInput.value.trim());
      if (dir) openImportComfyModal(dir);
    };
    pathRow.append(pathInput, browseBtn);
    wrap.appendChild(pathRow);

    // Best-effort quick-fill for localm's OWN managed ComfyUI - hidden until the
    // async status check resolves (or stays hidden forever if there is none).
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

    // Live "registering model N of M" text while Import's background job runs
    // (see importBtn.onclick below) - hidden the rest of the time.
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

    // The folder the last successful, non-empty preview covered - Import reuses
    // it rather than re-reading the (possibly since-edited) input, so what gets
    // imported always matches what was previewed.
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
          // The final progress event (phase "done") carries the same
          // added/skipped/method fields the old synchronous response body
          // used to - scanResultMessage() renders it identically either way.
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
          // Same "lost the stream, not necessarily the job" distinction the
          // pull flow makes (streamJob never reports this as a failure).
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
  // called with the click's MouseEvent as its first argument, which would
  // otherwise land in openImportComfyModal's initialPath parameter instead of
  // its "" default (a real bug this introduced once - the button's own click
  // event stringified into the path field on first open).
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
      if (!r.ok) { toast(data.detail || "Unload failed", true); return; }
      // unloaded_models only lists chat engines; the shared embedding model
      // (a separate lifecycle - see localm.inference.embedder) is reported
      // via embedder_unloaded, so count it too or a resident embedder alone
      // reports "Nothing was loaded" despite actually freeing VRAM.
      const n = (data.unloaded_models || []).length + (data.embedder_unloaded ? 1 : 0);
      // unload_all_models() (http_server.py) reports any pinned (mid-generation)
      // engine in skipped_in_use rather than unloading it - it was loaded, just
      // not released. Folding that into the same "Nothing was loaded" (nothing
      // WAS loaded) or a bare "Unloaded n model(s)" (silently drops the skip)
      // would misstate what happened either way (AGENTS.md rule 5), so name it.
      const skipped = Array.isArray(data.skipped_in_use) ? data.skipped_in_use : [];
      let msg;
      if (n && skipped.length) {
        msg = `Unloaded ${n} model(s); ${skipped.length} still generating - try again once finished`;
      } else if (n) {
        msg = `Unloaded ${n} model(s)`;
      } else if (skipped.length) {
        msg = `${skipped.length} model(s) still generating - try again once finished`;
      } else {
        msg = "Nothing was loaded";
      }
      toast(msg, skipped.length > 0);
      refreshModelsPage();
      refreshPerfEstimate();
    } catch (e) {
      toast("Unload failed: " + e.message, true);
    } finally {
      unloadAllBtn.disabled = false;
    }
  };
}

