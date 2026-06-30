// SPDX-License-Identifier: AGPL-3.0-or-later
/* localm GUI - per-plugin workflow management (Image/Music/Video pages) (split from pages.js). Classic script: it
   shares the one global lexical environment with app.js and the other
   page scripts, so the helpers it uses ($, el, authHeaders, toast, ...)
   resolve by bare name exactly as before. */
"use strict";

// --- ES module imports (auto-generated boundary; bodies unchanged) ---
import { $, authHeaders, el, toast } from "../app/helpers.js";
import { loginWithKey } from "../app/models-sidebar.js";
import { MEDIA_PLUGIN_ORDER } from "./settings.js";

/* ================================================================ */
/*  Per-plugin workflow management (on the Image/Music/Video pages)   */
/* ================================================================ */

/** Render the workflow panel for a media plugin: the built-in default plus each
 *  uploaded workflow, with select + delete, and an upload control. */
export async function refreshWorkflowPanel(media) {
  // Allowlist the media type before it ever reaches a selector/URL (defensive:
  // today it is only ever called with a hardcoded type).
  if (!MEDIA_PLUGIN_ORDER.includes(media)) return;
  // Query by data-media (not id): the image page uses the "img-" id prefix while
  // the media type is "image", so the data attribute is the stable handle.
  const box = document.querySelector(`[data-media="${media}"]`);
  if (!box) return;
  let data;
  try {
    const r = await fetch(`/api/${media}/workflows`, { headers: authHeaders() });
    if (!r.ok) throw new Error(r.statusText);
    data = await r.json();
  } catch (e) {
    box.replaceChildren(el("div", "sub", "Could not load workflows: " + e.message));
    return;
  }
  box.replaceChildren();
  const list = el("div", "workflow-list");
  // "Built-in default" = no selection (falls back to the committed/legacy template).
  list.appendChild(workflowRow(media, null, "Built-in default",
                               data.selected == null, false));
  for (const w of (data.workflows || [])) {
    list.appendChild(workflowRow(media, w.name, w.name, !!w.is_active, true));
  }
  box.appendChild(list);

  const up = el("div", "workflow-upload");
  const file = document.createElement("input");
  file.type = "file";
  file.accept = ".json,application/json";
  const btn = el("button", "btn-secondary", "Upload + use");
  btn.type = "button";
  btn.onclick = () => uploadWorkflow(media, file);
  up.append(file, btn);
  box.appendChild(up);
}

export function workflowRow(media, name, label, active, deletable) {
  const row = el("div", "workflow-row" + (active ? " active" : ""));
  const pick = el("button", "workflow-pick", (active ? "● " : "○ ") + label);
  pick.type = "button";
  pick.title = active ? "In use" : "Use this workflow";
  pick.onclick = () => selectWorkflow(media, name);
  row.appendChild(pick);
  if (deletable) {
    const del = el("button", "workflow-del", "Delete");
    del.type = "button";
    del.title = "Delete this workflow file";
    del.onclick = () => deleteWorkflow(media, name);
    row.appendChild(del);
  }
  return row;
}

export async function selectWorkflow(media, name) {
  const r = await fetch(`/api/${media}/workflows/select`, {
    method: "POST", headers: authHeaders(), body: JSON.stringify({ name }),
  });
  if (r.ok) { toast("Workflow selected"); refreshWorkflowPanel(media); }
  else toast((await r.json().catch(() => ({}))).detail || "Failed", true);
}

export async function deleteWorkflow(media, name) {
  if (!confirm(`Delete workflow "${name}"?`)) return;
  const r = await fetch(`/api/${media}/workflows/${encodeURIComponent(name)}`, {
    method: "DELETE", headers: authHeaders(),
  });
  if (r.ok) { toast("Deleted"); refreshWorkflowPanel(media); }
  else toast((await r.json().catch(() => ({}))).detail || "Failed", true);
}

export async function uploadWorkflow(media, fileInput) {
  const f = fileInput.files && fileInput.files[0];
  if (!f) { toast("Choose a .json file first", true); return; }
  let wf;
  try {
    wf = JSON.parse(await f.text());
  } catch (e) {
    toast("That file is not valid JSON", true);
    return;
  }
  const r = await fetch(`/api/${media}/workflows`, {
    method: "POST", headers: authHeaders(),
    body: JSON.stringify({ name: f.name, workflow: wf, activate: true }),
  });
  const d = await r.json().catch(() => ({}));
  if (r.ok) {
    toast("Uploaded and selected");
    fileInput.value = "";
    refreshWorkflowPanel(media);
  } else {
    toast(d.detail || "Upload failed", true);
  }
}

export const showMmprojCheckbox = $("show-mmproj-files");
if (showMmprojCheckbox) {
  showMmprojCheckbox.checked = localStorage.getItem("localm.showMmprojFiles") === "true";
  showMmprojCheckbox.addEventListener("change", (e) => {
    localStorage.setItem("localm.showMmprojFiles", e.target.checked ? "true" : "false");
  });
}

$("gui-key-save").onclick = async () => {
  const key = $("gui-api-key").value.trim();
  if (key) {
    const ok = await loginWithKey(key);   // POST /api/session -> server sets the HttpOnly cookie
    // Mark a successful login so a still-401 boot after the reload self-heals a
    // stale shell instead of looping (AUTH-1b).
    if (ok) { try { sessionStorage.setItem("localm.loginOk", "1"); } catch (e) { /* private mode */ } }
  } else {
    // Empty -> sign out (clear the session cookie).
    try {
      await fetch("/api/session/logout", { method: "POST", headers: authHeaders() });
    } catch (e) { /* offline / already cleared */ }
  }
  toast("Key saved - reloading");
  setTimeout(() => location.reload(), 600);
};

