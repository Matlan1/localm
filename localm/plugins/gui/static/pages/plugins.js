// SPDX-License-Identifier: AGPL-3.0-or-later
/* localm GUI - Plugins page. */
"use strict";

// --- ES module imports ---
import { $, authHeaders, confirmDanger, el, readSSE, toast } from "../app/helpers.js";
import { t, tn } from "../app/i18n.js";
import { emptyState, iconEl } from "../app/icons.js";
import { refreshPluginCommands } from "../app/settings-perf.js";

/* ================================================================ */
/*  Plugins page                                                     */
/* ================================================================ */

/* First-party catalog plugins, driven by the engine (/api/plugins). Each row
   shows install/enable/disable/uninstall depending on its state; chat is the
   protected #0 and has no actions. After any change the catalog table and the
   slash-command hint cache re-derive from one fetch. */
// Each render call supersedes the previous one; the staggered populate below
// checks the token and stops a stale render. The per-row delay is a module var.
export let _catalogRenderToken = 0;
export let _catalogStaggerMs = 24;
// Mirrors the auto_install_plugin_deps setting from the last /api/plugins fetch.
export let _autoInstallDeps = true;

/** A "sub" line reporting a plugin load error, prefixed with the warning icon. */
function pluginErrorLine(name, err) {
  const line = el("div", "sub");
  line.appendChild(iconEl("warning", "btn-ic"));
  line.appendChild(document.createTextNode(`${name}: ${err}`));
  return line;
}

export function _catalogRow(p) {
  const tr = el("tr");
  const nameTd = el("td", "name-cell shrink-cell");
  // The icon/name/badge flex line goes on this inner span, never on the td.
  const nameLine = el("span", "cell-line");
  nameTd.appendChild(nameLine);
  nameLine.appendChild(iconEl("plugins", "ic ic-plugin"));
  nameLine.appendChild(el("span", "name", p.label || p.name));
  tr.appendChild(nameTd);
  const status = p.protected ? t("plugins.pill.protected")
    : p.active ? t("plugins.pill.active")
    : p.installed ? t("plugins.pill.installedOff")
    : t("plugins.pill.available");
  // State renders as a .job-state pill: protected -> on, active -> st-ok,
  // installed-but-off -> st-pending, available -> the base pill.
  const statusCls = p.protected ? "on" : p.active ? "st-ok" : p.installed ? "st-pending" : "";
  const statusTd = el("td", "shrink-cell");
  statusTd.appendChild(el("span", ("job-state " + statusCls).trim(), status));
  tr.appendChild(statusTd);
  const descTd = el("td", "grow-cell", p.description);
  // Warn when a plugin needs other plugins that are not installed.
  const missing = Array.isArray(p.missing_requires) ? p.missing_requires : [];
  if (missing.length) {
    descTd.appendChild(el("div", "sub plugin-missing-req",
      t("plugins.missingRequires",
        { requires: (p.requires || []).join(", "), missing: missing.join(", ") })));
  }
  // Warn when an installed plugin is missing its pip extras (Python packages).
  const missingDeps = Array.isArray(p.missing_deps) ? p.missing_deps : [];
  if (p.installed && missingDeps.length) {
    descTd.appendChild(el("div", "sub plugin-missing-dep",
      t("plugins.missingDeps", { missing: missingDeps.join(", ") })));
  }
  tr.appendChild(descTd);
  const actions = el("td", "actions-cell");
  if (!p.protected) {
    if (!p.installed) {
      actions.appendChild(_catalogBtn("install", p.name, "primary", t("plugins.install")));
    } else {
      actions.appendChild(p.enabled
        ? _catalogBtn("disable", p.name, "", t("plugins.disable"))
        : _catalogBtn("enable", p.name, "primary", t("plugins.enable")));
      // Re-copy this builtin from the bundled store. Only builtins refresh
      // from the store; a third-party install is never a target.
      if (p.builtin) actions.appendChild(_catalogBtn("refresh", p.name, "", t("plugins.refresh")));
      actions.appendChild(_catalogBtn("uninstall", p.name, "danger", t("plugins.uninstall")));
    }
  }
  // One-click install of any missing requirements.
  if (missing.length) {
    const req = el("button", "btn-primary", t("plugins.installRequirements"));
    req.style.marginLeft = "6px";
    req.dataset.reqfor = p.name;
    req.onclick = () => installRequirements(p.name, missing);
    actions.appendChild(req);
  }
  // One-click install of missing pip extras (host-side; a remote client is told
  // to install on the host).
  if (p.installed && missingDeps.length) {
    const dep = el("button", "btn-primary", t("plugins.installDependencies"));
    dep.style.marginLeft = "6px";
    dep.dataset.depsfor = p.name;
    dep.onclick = () => installPluginDeps(p.name);
    actions.appendChild(dep);
  }
  tr.appendChild(actions);
  return tr;
}

export async function renderCatalogPlugins() {
  const box = $("catalog-table");
  if (!box) return;
  const myToken = ++_catalogRenderToken;
  box.replaceChildren();
  let data;
  try {
    const r = await fetch("/api/plugins", { headers: authHeaders() });
    if (!r.ok) throw new Error(r.statusText);
    data = await r.json();
  } catch (e) {
    box.appendChild(emptyState("warning", t("plugins.loadFailed"), e.message));
    return;
  }
  if (myToken !== _catalogRenderToken) return;   // superseded while fetching
  _autoInstallDeps = data.auto_install_plugin_deps !== false;
  const plugins = (data.plugins || []).filter((p) => p.builtin);

  // A status line naming what is loading and what loaded.
  const status = el("div", "sub catalog-status");
  box.appendChild(status);

  const table = el("table", "data-table");
  const thead = el("thead");
  const hr = el("tr");
  for (const h of [t("plugins.col.name"), t("plugins.col.status"), t("plugins.col.description"), ""])
    hr.appendChild(el("th", "", h));
  thead.appendChild(hr);
  table.appendChild(thead);
  const tbody = el("tbody");
  table.appendChild(tbody);
  box.appendChild(table);

  // Populate the rows one after another. The token guard cancels a stale
  // populate if a newer refresh started or the user left the page mid-fill.
  for (let i = 0; i < plugins.length; i++) {
    if (myToken !== _catalogRenderToken) return;
    tbody.appendChild(_catalogRow(plugins[i]));
    status.textContent = t("plugins.loadingProgress", { current: i + 1, total: plugins.length });
    await new Promise((r) => setTimeout(r, _catalogStaggerMs));
  }
  if (myToken !== _catalogRenderToken) return;

  const active = plugins.filter((p) => p.active).length;
  const failed = plugins.filter((p) => p.error).length;
  status.textContent =
    t("plugins.summary.active", { active, total: plugins.length })
    + (failed ? " · " + tn("plugins.summary.failed", failed) : "");

  if (data.errors) {
    // First-party errors only; the External plugins card renders the rest.
    const names = new Set(plugins.map((p) => p.name));
    for (const [name, err] of Object.entries(data.errors)) {
      if (!names.has(name)) continue;
      box.appendChild(pluginErrorLine(name, err));
    }
  }
}

export function _catalogBtn(action, name, cls, label) {
  const b = el("button", cls, label);
  b.style.marginLeft = "6px";
  b.onclick = () => pluginCatalogAction(action, name);
  return b;
}

// Install every plugin a given plugin requires but that is not yet installed.
// Sequential; each result is toasted, then the catalog re-renders.
export async function installRequirements(name, missing) {
  for (const dep of missing) {
    try {
      const r = await fetch(`/api/plugins/${encodeURIComponent(dep)}/install`,
                            { method: "POST", headers: authHeaders() });
      const d = await r.json().catch(() => ({}));
      if (!r.ok) {
        toast(`install ${dep}: ${d.detail || "failed"}`, true);
      } else {
        toast(`${dep}: ${d.status || "installed"}`);
      }
    } catch (e) {
      toast(`install ${dep} failed: ${e.message}`, true);
    }
  }
  renderCatalogPlugins();
  refreshPluginCommands();
}

// Install a plugin's missing pip extras on the HOST, streaming progress. The
// server refuses (403) a non-local caller, which is reported as a note to
// install on the host; `opts.silent` suppresses that note.
export async function installPluginDeps(name, opts = {}) {
  const box = $("catalog-table");
  let r;
  try {
    r = await fetch(`/api/plugins/${encodeURIComponent(name)}/install-deps`,
                    { method: "POST", headers: authHeaders() });
  } catch (e) {
    if (!opts.silent) toast(`install deps failed: ${e.message}`, true);
    return;
  }
  if (r.status === 403) {
    if (!opts.silent) {
      toast(`Install ${name}'s dependencies on the host, e.g.:  `
            + `localm plugin install-deps ${name}`, true);
    }
    return;
  }
  const started = await r.json().catch(() => ({}));
  if (!r.ok) {
    if (!opts.silent) toast(`install deps: ${started.detail || "failed"}`, true);
    return;
  }
  // A small progress panel that updates with the latest installer line.
  const panel = el("div", "dep-install-progress");
  panel.dataset.plugin = name;
  panel.appendChild(el("div", "sub", `Installing ${name} dependencies...`));
  const lineEl = el("div", "mono dep-install-line", "");
  panel.appendChild(lineEl);
  if (box) box.prepend(panel);

  let end = null;
  try {
    const stream = await fetch(
      `/api/plugins/${encodeURIComponent(name)}/install-deps/events`,
      { headers: authHeaders() });
    if (stream.ok) {
      await readSSE(stream, (payload) => {
        let ev;
        try { ev = JSON.parse(payload); } catch { return; }
        if (ev.type === "log") lineEl.textContent = ev.line;
        if (ev.type === "end") end = ev;
      });
    }
  } catch (e) { /* stream ended or dropped; fall through to a re-render */ }
  panel.remove();
  if (end && end.ok) {
    toast(`${name}: dependencies installed`);
  } else {
    toast(`${name}: dependency install `
          + (end ? "failed: " + (end.error || "unknown") : "did not complete"),
          true);
  }
  renderCatalogPlugins();
  refreshPluginCommands();
}

// After an install/enable, kick the host-side dependency install when the
// setting is on and the plugin is still missing packages. Silent.
export async function _maybeAutoInstallDeps(name) {
  if (!_autoInstallDeps) return;
  try {
    const r = await fetch("/api/plugins", { headers: authHeaders() });
    if (!r.ok) return;
    const data = await r.json();
    if (data.auto_install_plugin_deps === false) return;
    const p = (data.plugins || []).find((x) => x.name === name);
    if (p && Array.isArray(p.missing_deps) && p.missing_deps.length) {
      await installPluginDeps(name, { silent: true });
    }
  } catch (e) { /* best-effort */ }
}

export function pluginCatalogAction(action, name) {
  const run = async () => {
    try {
      const r = await fetch(`/api/plugins/${encodeURIComponent(name)}/${action}`,
                            { method: "POST", headers: authHeaders() });
      const d = await r.json().catch(() => ({}));
      if (!r.ok) {
        const msg = d.detail || (r.status === 404 ? t("plugins.error.noSuchPlugin")
                   : r.status === 409 ? t("plugins.error.notAllowed")
                   : t("plugins.error.failed"));
        toast(`${action} ${name}: ${msg}`, true);
        return;
      }
      toast(`${name}: ${d.status || action}`);
    } catch (e) {
      toast(`${action} failed: ${e.message}`, true);
      return;
    }
    // Re-derive from one place: the catalog table and the slash-hint cache.
    renderCatalogPlugins();
    refreshPluginCommands();
    // Tell other open tabs to re-sync, so one parked on this plugin's page
    // leaves a now-disabled view instead of calling its dead routes.
    if (window.bumpPluginsRev) window.bumpPluginsRev();
    // Auto-install the plugin's pip extras when the setting is on (host-only).
    if (action === "install" || action === "enable") _maybeAutoInstallDeps(name);
  };
  if (action === "uninstall") {
    confirmDanger(t("plugins.uninstallConfirm.title", { name }),
      t("plugins.uninstallConfirm.body"), t("plugins.uninstall"), run);
  } else {
    run();
  }
}

/* External (third-party) plugins, from the same engine API as the catalog
   above: /api/plugins lists every INSTALLED plugin, and an external one is an
   installed plugin that is not first-party (builtin = in the bundled store or
   the static catalog). */
export async function refreshPluginsPage() {
  const box = $("plugins-table");
  box.replaceChildren();
  try {
    const r = await fetch("/api/plugins", { headers: authHeaders() });
    if (!r.ok) throw new Error(r.statusText);
    const all = await r.json();
    const builtins = new Set((all.plugins || []).filter((p) => p.builtin)
      .map((p) => p.name));
    const data = {
      plugins: (all.plugins || []).filter((p) => p.installed && !p.builtin),
      // errors is keyed by plugin name, or by directory name for a broken
      // external plugin folder; anything not first-party belongs to this card.
      errors: Object.entries(all.errors || {})
        .filter(([name]) => !builtins.has(name)),
    };
    if (!data.plugins.length) {
      box.appendChild(emptyState("plugins", t("plugins.external.empty.text"),
        t("plugins.external.empty.hint")));
    } else {
      const table = el("table", "data-table");
      const thead = el("thead");
      const hr = el("tr");
      for (const h of [t("plugins.col.name"), t("plugins.col.version"),
                       t("plugins.col.description"), t("plugins.col.tools"), ""])
        hr.appendChild(el("th", "", h));
      thead.appendChild(hr);
      table.appendChild(thead);
      const tbody = el("tbody");
      for (const p of data.plugins) {
        const tr = el("tr");
        const nameTd = el("td", "name-cell shrink-cell");
        const nameLine = el("span", "cell-line");
        nameTd.appendChild(nameLine);
        nameLine.appendChild(iconEl("plugins", "ic ic-plugin"));
        nameLine.appendChild(el("span", "name", p.name));
        tr.appendChild(nameTd);
        tr.appendChild(el("td", "mono shrink-cell", p.version));
        tr.appendChild(el("td", "grow-cell", p.description));
        tr.appendChild(el("td", "mono",
          (p.tool_exports || []).length ? p.tool_exports.join(", ") : ""));
        const actions = el("td", "actions-cell");
        const rm = el("button", "danger", t("plugins.removeButton"));
        rm.onclick = () => {
          confirmDanger(t("plugins.removeConfirm.title", { name: p.name }),
            t("plugins.removeConfirm.body"),
            t("plugins.remove"), async () => {
              const rr = await fetch(
                `/api/plugins/${encodeURIComponent(p.name)}/uninstall`,
                { method: "POST", headers: authHeaders() });
              const dd = await rr.json().catch(() => ({}));
              if (rr.ok) { toast(t("plugins.removed", { name: p.name })); refreshPluginsPage(); }
              else toast(dd.detail || t("plugins.removeFailed"), true);
            });
        };
        actions.appendChild(rm);
        tr.appendChild(actions);
        tbody.appendChild(tr);
      }
      table.appendChild(tbody);
      box.appendChild(table);
    }
    for (const [name, err] of data.errors) {
      box.appendChild(pluginErrorLine(name, err));
    }
  } catch (e) {
    box.appendChild(emptyState("warning", t("plugins.loadFailed"), e.message));
  }
}

$("plugin-install").onclick = async () => {
  const source = $("plugin-source").value.trim();
  if (!source) { toast(t("plugins.enterSourcePath"), true); return; }
  const r = await fetch("/api/plugins/install-external", {
    method: "POST", headers: authHeaders(),
    body: JSON.stringify({ source }),
  });
  const data = await r.json().catch(() => ({}));
  if (r.ok) {
    toast(t("plugins.installedRestartHint", { name: data.name, version: data.version }));
    $("plugin-source").value = "";
    refreshPluginsPage();
  } else {
    toast(data.detail || t("plugins.installFailed"), true);
  }
};

// The catalog and external tables are painted from fetched data, not marked
// up in index.html, so both are re-fetched and redrawn when the interface
// language changes.
document.addEventListener("localm:language", () => {
  renderCatalogPlugins();
  refreshPluginsPage();
});

