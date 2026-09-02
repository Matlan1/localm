// SPDX-License-Identifier: AGPL-3.0-or-later
/* localm GUI - command palette (Ctrl/Cmd+K). */
"use strict";

// --- ES module imports ---
import { newConversation } from "./chat.js";
import { $, el } from "./helpers.js";
import { t } from "./i18n.js";
import { exportConversation } from "./settings-perf.js";
import { VIEWS, isSettingsView, showView } from "./tabs.js";
import { gotoSettingsSection } from "../pages/settings.js";

/* ================================================================ */
/*  Command palette (Ctrl/Cmd+K)                                     */
/* ================================================================ */

// Built fresh on each open, including runtime-added plugin views. View labels
// are read from the live nav buttons.
export function cmdkCommands() {
  const cmds = [];
  for (const v of VIEWS) {
    if (!$("view-" + v)) continue;
    const nav = $("nav-" + v);
    const label = ((nav ? nav.textContent : v) || v).trim() || v;
    cmds.push({ label: t("cmdk.goTo", { view: label }), run: () => showView(v) });
  }
  cmds.push({ label: t("topbar.newChat"), run: () => { newConversation(); showView("chat"); } });
  cmds.push({ label: t("cmdk.toggleTheme"), run: () => $("theme-toggle").click() });
  cmds.push({ label: t("cmdk.exportConversation"), run: () => exportConversation() });
  // Jump to the Keys & devices manager. Offered only while its panel is not
  // gated-hidden for this key.
  const keysCard = $("keys-card");
  if (keysCard && !keysCard.classList.contains("sec-hidden")) {
    cmds.push({ label: t("cmdk.manageKeys"), run: () => {
      showView("settings");
      if (typeof gotoSettingsSection === "function") gotoSettingsSection("keys-card");
    } });
  }
  return cmds;
}

export let _cmdkAll = [], _cmdkShown = [], _cmdkSel = 0;

export function cmdkFilter(query) {
  const q = (query || "").trim().toLowerCase();
  return q ? _cmdkAll.filter((c) => c.label.toLowerCase().includes(q)) : _cmdkAll.slice();
}

export function renderCmdk(query) {
  _cmdkShown = cmdkFilter(query);
  if (_cmdkSel >= _cmdkShown.length) _cmdkSel = Math.max(0, _cmdkShown.length - 1);
  const list = $("cmdk-list");
  list.replaceChildren();
  _cmdkShown.forEach((c, i) => {
    const item = el("div", "cmdk-item" + (i === _cmdkSel ? " sel" : ""), c.label);
    item.onclick = () => runCmdk(i);
    list.appendChild(item);
  });
}

export function cmdkIsOpen() {
  const m = $("cmdk");
  return !!m && m.style.display !== "none";
}

export function openCommandPalette() {
  _cmdkAll = cmdkCommands();
  _cmdkSel = 0;
  $("cmdk-input").value = "";
  renderCmdk("");
  $("cmdk").style.display = "flex";
  $("cmdk-input").focus();
}

export function closeCommandPalette() {
  $("cmdk").style.display = "none";
}

export function runCmdk(index) {
  const cmd = _cmdkShown[index];
  closeCommandPalette();
  if (cmd) cmd.run();
}

$("cmdk-input").addEventListener("input", (e) => { _cmdkSel = 0; renderCmdk(e.target.value); });
$("cmdk").addEventListener("click", (e) => { if (e.target === $("cmdk")) closeCommandPalette(); });
document.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && (e.key === "k" || e.key === "K")) {
    e.preventDefault();
    cmdkIsOpen() ? closeCommandPalette() : openCommandPalette();
    return;
  }
  // Ctrl/Cmd+S saves the active Settings section, on the Settings page only.
  if ((e.ctrlKey || e.metaKey) && (e.key === "s" || e.key === "S")) {
    if (isSettingsView()) {
      e.preventDefault();
      if (window.saveActiveSettingsSection) window.saveActiveSettingsSection();
    }
    return;
  }
  if (!cmdkIsOpen()) return;
  if (e.key === "Escape") { e.preventDefault(); closeCommandPalette(); }
  else if (e.key === "ArrowDown") {
    e.preventDefault(); _cmdkSel = Math.min(_cmdkSel + 1, _cmdkShown.length - 1);
    renderCmdk($("cmdk-input").value);
  } else if (e.key === "ArrowUp") {
    e.preventDefault(); _cmdkSel = Math.max(_cmdkSel - 1, 0);
    renderCmdk($("cmdk-input").value);
  } else if (e.key === "Enter") {
    e.preventDefault(); runCmdk(_cmdkSel);
  }
});

// Trigger the browser's native unsaved-changes prompt on tab close or reload
// while Settings has unsaved edits. An empty returnValue triggers it.
window.addEventListener("beforeunload", (e) => {
  if (window.settingsDirty && window.settingsDirty()) {
    e.preventDefault();
    e.returnValue = "";
  }
});

