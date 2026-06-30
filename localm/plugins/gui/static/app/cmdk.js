// SPDX-License-Identifier: AGPL-3.0-or-later
/* localm GUI - command palette (Ctrl/Cmd+K) (split from app.js). Classic script: it
   shares the one global lexical environment with the other app/* and
   pages/* scripts, so every cross-section reference resolves by bare
   name exactly as before. */
"use strict";

/* ================================================================ */
/*  Command palette (Ctrl/Cmd+K)                                     */
/* ================================================================ */

// Built fresh on open so runtime-added plugin views are included. The view
// labels are taken from the live nav buttons so they match exactly.
function cmdkCommands() {
  const cmds = [];
  for (const v of VIEWS) {
    if (!$("view-" + v)) continue;
    const nav = $("nav-" + v);
    const label = ((nav ? nav.textContent : v) || v).trim() || v;
    cmds.push({ label: "Go to " + label, run: () => showView(v) });
  }
  cmds.push({ label: "New chat", run: () => { newConversation(); showView("chat"); } });
  cmds.push({ label: "Toggle light/dark theme", run: () => $("theme-toggle").click() });
  cmds.push({ label: "Export conversation", run: () => exportConversation() });
  // Direct jump to the owner-only Keys & devices manager (it lives in a Settings
  // sub-section that is otherwise easy to miss). Offered only when this key may
  // actually manage keys - i.e. the panel is not gated-hidden for it.
  const keysCard = $("keys-card");
  if (keysCard && !keysCard.classList.contains("sec-hidden")) {
    cmds.push({ label: "Manage keys & devices", run: () => {
      showView("settings");
      if (typeof gotoSettingsSection === "function") gotoSettingsSection("keys-card");
    } });
  }
  return cmds;
}

let _cmdkAll = [], _cmdkShown = [], _cmdkSel = 0;

function cmdkFilter(query) {
  const q = (query || "").trim().toLowerCase();
  return q ? _cmdkAll.filter((c) => c.label.toLowerCase().includes(q)) : _cmdkAll.slice();
}

function renderCmdk(query) {
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

function cmdkIsOpen() {
  const m = $("cmdk");
  return !!m && m.style.display !== "none";
}

function openCommandPalette() {
  _cmdkAll = cmdkCommands();
  _cmdkSel = 0;
  $("cmdk-input").value = "";
  renderCmdk("");
  $("cmdk").style.display = "flex";
  $("cmdk-input").focus();
}

function closeCommandPalette() {
  $("cmdk").style.display = "none";
}

function runCmdk(index) {
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
  // R09: Ctrl/Cmd+S saves the active Settings section. Only on the Settings page,
  // so every other view keeps the browser's native "Save page".
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

// R10: the browser's native unsaved-changes prompt on tab close / reload while
// Settings has unsaved edits. An empty returnValue is what triggers it; browsers
// ignore any custom message, so we set none.
window.addEventListener("beforeunload", (e) => {
  if (window.settingsDirty && window.settingsDirty()) {
    e.preventDefault();
    e.returnValue = "";
  }
});

