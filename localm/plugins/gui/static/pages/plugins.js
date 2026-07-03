// SPDX-License-Identifier: AGPL-3.0-or-later
/* localm GUI - Plugins page (split from pages.js). Classic script: it
   shares the one global lexical environment with app.js and the other
   page scripts, so the helpers it uses ($, el, authHeaders, toast, ...)
   resolve by bare name exactly as before. */
"use strict";

// --- ES module imports (auto-generated boundary; bodies unchanged) ---
import { $, authHeaders, el, readSSE, toast } from "../app/helpers.js";
import { emptyState } from "../app/icons.js";
import { refreshPluginCommands } from "../app/settings-perf.js";

/* ================================================================ */
/*  Plugins page                                                     */
/* ================================================================ */

/* First-party catalog plugins, driven by the engine (/api/plugins). Each row
   shows install/enable/disable/uninstall depending on its state; chat is the
   protected #0 and has no actions. After any change the catalog table AND the
   slash-command hint cache re-derive from one fetch (nav follows in Phase 4). */
// Each call supersedes the previous one (a newer refresh or leaving the page);
// the staggered populate below checks the token so a stale render stops. The
// per-row delay is a module var so tests can drop it to 0.
export let _catalogRenderToken = 0;
export let _catalogStaggerMs = 24;
// Mirrors the auto_install_plugin_deps setting from the last /api/plugins fetch,
// so an install/enable can auto-kick the host-side dependency install.
export let _autoInstallDeps = true;

export function _catalogRow(p) {
  const tr = el("tr");
  const nameTd = el("td", "name-cell");
  nameTd.appendChild(iconEl("plugins", "ic ic-plugin"));
  nameTd.appendChild(el("span", "name", p.label || p.name));
  tr.appendChild(nameTd);
  const status = p.protected ? "protected"
    : p.active ? "active"
    : p.installed ? "installed (off)"
    : "available";
  tr.appendChild(el("td", "mono", status));
  const descTd = el("td", "", p.description);
  // Warn when a plugin needs other plugins that are not installed (B15).
  const missing = Array.isArray(p.missing_requires) ? p.missing_requires : [];
  if (missing.length) {
    descTd.appendChild(el("div", "sub plugin-missing-req",
      `requires ${(p.requires || []).join(", ")} (missing: ${missing.join(", ")})`));
  }
  // Warn when an installed plugin is missing its pip extras (Python packages).
  const missingDeps = Array.isArray(p.missing_deps) ? p.missing_deps : [];
  if (p.installed && missingDeps.length) {
    descTd.appendChild(el("div", "sub plugin-missing-dep",
      `needs Python packages: ${missingDeps.join(", ")}`));
  }
  tr.appendChild(descTd);
  const actions = el("td");
  actions.style.textAlign = "right";
  if (!p.protected) {
    if (!p.installed) {
      actions.appendChild(_catalogBtn("install", p.name, "primary", "Install"));
    } else {
      actions.appendChild(p.enabled
        ? _catalogBtn("disable", p.name, "", "Disable")
        : _catalogBtn("enable", p.name, "primary", "Enable"));
      // Re-copy this builtin from the bundled store if a localm upgrade shipped
      // newer code (the installed copy would otherwise keep shadowing it). Only
      // builtins refresh from the store; a third-party install is never a target.
      if (p.builtin) actions.appendChild(_catalogBtn("refresh", p.name, "", "Refresh"));
      actions.appendChild(_catalogBtn("uninstall", p.name, "danger", "Uninstall"));
    }
  }
  // One-click install of any missing requirements (B15).
  if (missing.length) {
    const req = el("button", "btn-primary", "Install requirements");
    req.style.marginLeft = "6px";
    req.dataset.reqfor = p.name;
    req.onclick = () => installRequirements(p.name, missing);
    actions.appendChild(req);
  }
  // One-click install of missing pip extras (host-side; a remote client is told
  // to install on the host).
  if (p.installed && missingDeps.length) {
    const dep = el("button", "btn-primary", "Install dependencies");
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
    box.appendChild(el("div", "sub", "Could not load plugins: " + e.message));
    return;
  }
  if (myToken !== _catalogRenderToken) return;   // superseded while fetching
  _autoInstallDeps = data.auto_install_plugin_deps !== false;
  const plugins = (data.plugins || []).filter((p) => p.builtin);

  // A status line that tells the user what is loading / what loaded (U2).
  const status = el("div", "sub catalog-status");
  box.appendChild(status);

  const table = el("table", "data-table");
  const thead = el("thead");
  const hr = el("tr");
  for (const h of ["Name", "Status", "Description", ""]) hr.appendChild(el("th", "", h));
  thead.appendChild(hr);
  table.appendChild(thead);
  const tbody = el("tbody");
  table.appendChild(tbody);
  box.appendChild(table);

  // Populate the rows one after another so the catalog visibly fills in instead
  // of flashing all at once (U2). The token guard cancels a stale populate if a
  // newer refresh started or the user left the page mid-fill.
  for (let i = 0; i < plugins.length; i++) {
    if (myToken !== _catalogRenderToken) return;
    tbody.appendChild(_catalogRow(plugins[i]));
    status.textContent = `Loading plugins… (${i + 1}/${plugins.length})`;
    await new Promise((r) => setTimeout(r, _catalogStaggerMs));
  }
  if (myToken !== _catalogRenderToken) return;

  const active = plugins.filter((p) => p.active).length;
  const failed = plugins.filter((p) => p.error).length;
  status.textContent =
    `${active}/${plugins.length} plugins active` + (failed ? ` · ${failed} failed` : "");

  if (data.errors) {
    for (const [name, err] of Object.entries(data.errors)) {
      box.appendChild(el("div", "sub", `⚠ ${name}: ${err}`));
    }
  }
}

export function _catalogBtn(action, name, cls, label) {
  const b = el("button", cls, label);
  b.style.marginLeft = "6px";
  b.onclick = () => pluginCatalogAction(action, name);
  return b;
}

// Install every plugin a given plugin requires but that is not yet installed
// (B15). Best-effort and sequential; each result is toasted, then the catalog
// re-renders so the warnings clear.
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
// server refuses (403) a non-local caller, in which case we tell the user to
// install on the host instead. `opts.silent` suppresses that note for the
// auto-triggered path (a remote GUI just leaves the manual button in place).
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
// setting is on and the plugin is still missing packages. Best-effort and
// silent (a remote GUI leaves the manual "Install dependencies" button).
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

export async function pluginCatalogAction(action, name) {
  if (action === "uninstall" &&
      !confirm(`Uninstall '${name}'? Its plugin files are removed (your data is kept).`)) return;
  try {
    const r = await fetch(`/api/plugins/${encodeURIComponent(name)}/${action}`,
                          { method: "POST", headers: authHeaders() });
    const d = await r.json().catch(() => ({}));
    if (!r.ok) {
      const msg = d.detail || (r.status === 404 ? "no such plugin"
                 : r.status === 409 ? "not allowed in the current state"
                 : "failed");
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
  // R50: tell other open tabs (which may be parked on this plugin's page) so they
  // re-sync and leave a now-disabled view instead of erroring on its dead routes.
  if (window.bumpPluginsRev) window.bumpPluginsRev();
  // Auto-install the plugin's pip extras when the setting is on (host-only; a
  // remote GUI silently leaves the manual "Install dependencies" button).
  if (action === "install" || action === "enable") _maybeAutoInstallDeps(name);
}

export async function refreshPluginsPage() {
  const box = $("plugins-table");
  box.replaceChildren();
  try {
    const r = await fetch("/v1/plugins", { headers: authHeaders() });
    if (!r.ok) throw new Error(r.statusText);
    const data = await r.json();
    if (!data.plugins.length) {
      box.appendChild(emptyState("plugins", "No external plugins installed",
        "Install one from the catalog above to add capabilities."));
    } else {
      const table = el("table", "data-table");
      const thead = el("thead");
      const hr = el("tr");
      for (const h of ["Name", "Version", "Description", "Tools", ""])
        hr.appendChild(el("th", "", h));
      thead.appendChild(hr);
      table.appendChild(thead);
      const tbody = el("tbody");
      for (const p of data.plugins) {
        const tr = el("tr");
        const nameTd = el("td", "name-cell");
        nameTd.appendChild(iconEl("plugins", "ic ic-plugin"));
        nameTd.appendChild(el("span", "name", p.name));
        tr.appendChild(nameTd);
        tr.appendChild(el("td", "mono", p.version));
        tr.appendChild(el("td", "", p.description));
        tr.appendChild(el("td", "mono",
          p.tool_exports.length ? p.tool_exports.join(", ") : ""));
        const actions = el("td");
        actions.style.textAlign = "right";
        const rm = el("button", "danger", "remove");
        rm.onclick = async () => {
          if (!confirm(`Remove plugin '${p.name}'? Its folder in the data directory's plugins/ is deleted.`)) return;
          const rr = await fetch(`/v1/plugins/${encodeURIComponent(p.name)}`, {
            method: "DELETE", headers: authHeaders() });
          const dd = await rr.json();
          if (rr.ok) { toast(`Removed '${p.name}'`); refreshPluginsPage(); }
          else toast(dd.detail || "Remove failed", true);
        };
        actions.appendChild(rm);
        tr.appendChild(actions);
        tbody.appendChild(tr);
      }
      table.appendChild(tbody);
      box.appendChild(table);
    }
    for (const err of data.errors) {
      box.appendChild(el("div", "sub", "⚠ " + err));
    }
  } catch (e) {
    box.appendChild(el("div", "sub", "Could not load plugins: " + e.message));
  }
}

$("plugin-install").onclick = async () => {
  const source = $("plugin-source").value.trim();
  if (!source) { toast("Enter the plugin folder path", true); return; }
  const r = await fetch("/v1/plugins/install", {
    method: "POST", headers: authHeaders(),
    body: JSON.stringify({ source }),
  });
  const data = await r.json();
  if (r.ok) {
    toast(`Installed '${data.name}' ${data.version} - restart localm gui to load its command`);
    $("plugin-source").value = "";
    refreshPluginsPage();
  } else {
    toast(data.detail || "Install failed", true);
  }
};

