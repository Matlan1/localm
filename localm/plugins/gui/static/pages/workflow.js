// SPDX-License-Identifier: AGPL-3.0-or-later
/* localm GUI - per-plugin workflow management (Image/Music/Video pages) (split from pages.js). Classic script: it
   shares the one global lexical environment with app.js and the other
   page scripts, so the helpers it uses ($, el, authHeaders, toast, ...)
   resolve by bare name exactly as before. */
"use strict";

// --- ES module imports (auto-generated boundary; bodies unchanged) ---
import { $, authHeaders, confirmDanger, el, toast } from "../app/helpers.js";
import { iconEl } from "../app/icons.js";
import { loginWithKey } from "../app/models-sidebar.js";
import { MEDIA_PLUGIN_ORDER } from "./settings.js";

/* ================================================================ */
/*  Per-plugin workflow management (on the Image/Music/Video pages)   */
/* ================================================================ */

// The image plugin splits its routes across two prefixes (workflow management
// under /api/image/..., generation + ComfyUI plumbing under /api/imagine/...) -
// music/video use one prefix for both. comfy-models/comfy-launch live under
// the generation prefix in all three plugins.
const GENERATE_PREFIX = { image: "imagine", music: "music", video: "video" };

// Per-media selected model overrides ({node_id: {input_name: value}}), set by
// the dropdowns below and read by each page's Generate handler. Reset only when
// the ACTIVE workflow changes (selectWorkflow/uploadWorkflow) - a plain panel
// refresh (e.g. re-entering the tab) must not silently drop the user's picks.
export const modelOverrides = { image: {}, music: {}, video: {} };

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

  box.appendChild(comfyLaunchRow(media));
  box.appendChild(await comfyModelPicker(media));
}

/** "Launch ComfyUI" button: starts (or confirms) the configured ComfyUI for
 *  this media plugin without running a generation, then opens it in a new
 *  tab on success and re-probes the model picker (it may now be reachable). */
function comfyLaunchRow(media) {
  const row = el("div", "comfy-launch-row");
  const launchBtn = el("button", "btn-secondary", "Launch ComfyUI");
  launchBtn.type = "button";
  launchBtn.onclick = async () => {
    launchBtn.disabled = true;
    launchBtn.textContent = "Launching…";
    try {
      const r = await fetch(`/api/${GENERATE_PREFIX[media]}/comfy-launch`,
        { method: "POST", headers: authHeaders() });
      const data = await r.json().catch(() => ({}));
      if (data.ok) {
        toast("ComfyUI is running");
        if (data.api_url) window.open(data.api_url, "_blank", "noopener");
        refreshWorkflowPanel(media);
      } else {
        toast(data.message || "Could not start ComfyUI", true);
      }
    } catch (e) {
      toast("Launch failed: " + e.message, true);
    } finally {
      launchBtn.disabled = false;
      launchBtn.textContent = "Launch ComfyUI";
    }
  };
  row.appendChild(launchBtn);
  return row;
}

/** One dropdown per model-file slot the active workflow exposes (resolved
 *  against the live ComfyUI /object_info) - honest about unreachability
 *  (rule 5) rather than a silently-empty picker. Selections are stored in
 *  modelOverrides[media] for the page's Generate handler to send along. */
async function comfyModelPicker(media) {
  const wrap = el("div", "comfy-model-picker");
  let data;
  try {
    const r = await fetch(`/api/${GENERATE_PREFIX[media]}/comfy-models`,
      { headers: authHeaders() });
    data = await r.json();
  } catch (e) {
    wrap.appendChild(el("div", "sub", "Could not check ComfyUI models: " + e.message));
    return wrap;
  }
  if (!data.reachable) {
    wrap.appendChild(el("div", "sub", data.message
      || "ComfyUI is not running - launch it to pick models."));
    return wrap;
  }
  if (!data.slots || !data.slots.length) {
    wrap.appendChild(el("div", "sub", "This workflow has no selectable model files."));
    return wrap;
  }
  wrap.appendChild(el("h5", "comfy-model-picker-head", "Models"));
  // A slot with zero live options means ComfyUI has NONE of that file type
  // installed - distinct from the workflow having nothing to configure at all,
  // and worth calling out since it is exactly the situation a user needs to
  // act on (install the missing file(s)).
  const missing = data.slots.filter((s) => !s.options || !s.options.length);
  if (missing.length) {
    wrap.appendChild(el("div", "sub comfy-model-missing",
      `${missing.length} required model file${missing.length === 1 ? "" : "s"} `
      + "not found in ComfyUI - see below."));
  }
  const overrides = (modelOverrides[media] ??= {});
  for (const slot of data.slots) {
    const row = el("div", "comfy-model-row");
    row.appendChild(el("label", "comfy-model-label", slot.input_name));
    if (!slot.options || !slot.options.length) {
      // No live choices to render - a <select> with zero <option>s would show
      // blank and unusable. Name the workflow's own current value instead so
      // the user knows exactly what file to install.
      row.appendChild(el("span", "comfy-model-missing-value", `${slot.current} (not installed)`));
      wrap.appendChild(row);
      continue;
    }
    const sel = document.createElement("select");
    sel.className = "comfy-model-select";
    const chosen = overrides[slot.node_id]?.[slot.input_name] ?? slot.current;
    for (const opt of slot.options) {
      const o = document.createElement("option");
      o.value = opt;
      o.textContent = opt;
      if (opt === chosen) o.selected = true;
      sel.appendChild(o);
    }
    sel.onchange = () => {
      (overrides[slot.node_id] ??= {})[slot.input_name] = sel.value;
    };
    row.appendChild(sel);
    wrap.appendChild(row);
  }
  return wrap;
}

export function workflowRow(media, name, label, active, deletable) {
  const row = el("div", "workflow-row" + (active ? " active" : ""));
  const pick = el("button", "workflow-pick");
  pick.appendChild(iconEl(active ? "dot" : "ring", "btn-ic"));
  pick.appendChild(document.createTextNode(label));
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
  if (r.ok) {
    // A different workflow can use different node IDs entirely - stale
    // overrides keyed by the old graph must not silently apply to the new one.
    modelOverrides[media] = {};
    toast("Workflow selected");
    refreshWorkflowPanel(media);
  } else toast((await r.json().catch(() => ({}))).detail || "Failed", true);
}

export function deleteWorkflow(media, name) {
  confirmDanger(`Delete workflow "${name}"?`, "This can't be undone.",
    "Delete", async () => {
      const r = await fetch(`/api/${media}/workflows/${encodeURIComponent(name)}`, {
        method: "DELETE", headers: authHeaders(),
      });
      if (r.ok) { toast("Deleted"); refreshWorkflowPanel(media); }
      else toast((await r.json().catch(() => ({}))).detail || "Failed", true);
    });
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
    modelOverrides[media] = {};   // new graph - see selectWorkflow
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

