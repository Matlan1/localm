// SPDX-License-Identifier: AGPL-3.0-or-later
/* localm GUI - per-plugin workflow management (Image/Music/Video pages). */
"use strict";

// --- ES module imports ---
import { $, authHeaders, confirmDanger, el, toast } from "../app/helpers.js";
import { emptyState } from "../app/icons.js";
import { loginWithKey } from "../app/models-sidebar.js";
import { MEDIA_PLUGIN_ORDER } from "./settings.js";

/* ================================================================ */
/*  Per-plugin workflow management (on the Image/Music/Video pages)   */
/* ================================================================ */

// The image plugin splits its routes across two prefixes: workflow management
// under /api/image/..., generation and ComfyUI plumbing under /api/imagine/...
// Music and video use one prefix for both. comfy-models and comfy-launch live
// under the generation prefix in all three plugins.
const GENERATE_PREFIX = { image: "imagine", music: "music", video: "video" };

// Per-media selected model overrides ({node_id: {input_name: value}}), set by
// the dropdowns below and read by each page's Generate handler. Reset only when
// the ACTIVE workflow changes (selectWorkflow/uploadWorkflow), never on a plain
// panel refresh.
export const modelOverrides = { image: {}, music: {}, video: {} };

/** Render the workflow panel for a media plugin: the built-in default plus each
 *  uploaded workflow, with select + delete, and an upload control. */
export async function refreshWorkflowPanel(media) {
  // Allowlist the media type before it reaches a selector or a URL.
  if (!MEDIA_PLUGIN_ORDER.includes(media)) return;
  // Query by data-media, not id: the image page's ids use the "img-" prefix
  // while its media type is "image".
  const box = document.querySelector(`[data-media="${media}"]`);
  if (!box) return;
  let data;
  try {
    const r = await fetch(`/api/${media}/workflows`, { headers: authHeaders() });
    if (!r.ok) throw new Error(r.statusText);
    data = await r.json();
  } catch (e) {
    box.replaceChildren(emptyState("warning", "Could not load workflows", e.message));
    return;
  }
  box.replaceChildren();
  const list = el("div", "workflow-list");
  // "Built-in default" = no selection, falling back to the bundled template.
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
 *  this media plugin without running a generation, then opens it in a new tab
 *  on success and refreshes the panel. */
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

/** One dropdown per model-file slot the active workflow exposes, resolved
 *  against the live ComfyUI /object_info. Selections are stored in
 *  modelOverrides[media] for the page's Generate handler to send along. */
async function comfyModelPicker(media) {
  const wrap = el("div", "comfy-model-picker");
  let data;
  try {
    const r = await fetch(`/api/${GENERATE_PREFIX[media]}/comfy-models`,
      { headers: authHeaders() });
    data = await r.json();
  } catch (e) {
    wrap.appendChild(emptyState("warning", "Could not check ComfyUI models", e.message));
    return wrap;
  }
  if (!data.reachable) {
    wrap.appendChild(el("div", "sub", data.message
      || "ComfyUI is not running - launch it to pick models."));
    // Show what the workflow needs and what is registered on this machine,
    // from localm's own registry.
    appendRegistryFallback(wrap, data);
    return wrap;
  }
  if (!data.slots || !data.slots.length) {
    wrap.appendChild(el("div", "sub", "This workflow has no selectable model files."));
    return wrap;
  }
  wrap.appendChild(el("h5", "comfy-model-picker-head", "Models"));
  // A slot with zero live options means ComfyUI has none of that file type
  // installed.
  const missing = data.slots.filter((s) => !s.options || !s.options.length);
  if (missing.length) {
    wrap.appendChild(el("div", "sub comfy-model-missing",
      `${missing.length} required model file${missing.length === 1 ? "" : "s"} `
      + "not found in ComfyUI - see below."));
  }
  const overrides = (modelOverrides[media] ??= {});
  const roleById = new Map((data.roles || []).map((r) => [r.role_id, r]));
  for (const slot of data.slots) {
    const row = el("div", "comfy-model-row");
    // The role label captions the field, with the raw input_name kept
    // alongside it. The server pairs roles to slots positionally within a
    // model type.
    const label = el("label", "comfy-model-label",
      slot.role_label ? `${slot.role_label} (${slot.input_name})` : slot.input_name);
    row.appendChild(label);
    if (!slot.options || !slot.options.length) {
      // No live choices: name the workflow's own current value instead of
      // rendering an empty select.
      row.appendChild(el("span", "comfy-model-missing-value job-state st-error", `${slot.current} (not installed)`));
      wrap.appendChild(row);
      // Before the continue, so a slot with no options still gets the hint.
      appendRegistryOnlyHint(wrap, roleById.get(slot.role_id), slot.current);
      continue;
    }
    const sel = document.createElement("select");
    sel.className = "comfy-model-select";
    const chosen = overrides[slot.node_id]?.[slot.input_name] ?? slot.current;
    if (!slot.options.includes(chosen)) {
      // The workflow names a file ComfyUI does not have, while offering others
      // of the same kind. Render that value as a disabled, selected option: a
      // select with no matching option displays the first one instead.
      const cur = document.createElement("option");
      cur.value = chosen;
      cur.textContent = `${chosen} (not installed)`;
      cur.disabled = true;
      cur.selected = true;
      sel.appendChild(cur);
    }
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
    appendRegistryOnlyHint(wrap, roleById.get(slot.role_id), slot.current);
  }
  appendUnusedRoles(wrap, data.roles);
  return wrap;
}

/** "You have one of these registered, ComfyUI is not offering it", shown on a
 *  slot ComfyUI could not serve. Leads with the exact file the workflow names
 *  when it is among them. */
function appendRegistryOnlyHint(wrap, role, current) {
  const extra = (role && role.registry_only) || [];
  if (!extra.length) return;
  const exact = extra.find((m) => m.filename === current);
  wrap.appendChild(el("div", "sub comfy-model-registry-hint", exact
    ? `${exact.filename} IS registered in localm (as "${exact.name}") but is not `
      + "in a folder ComfyUI reads - copy it into ComfyUI's models folder."
    : "Registered in localm but not offered by ComfyUI: "
      + extra.map((m) => m.filename).join(", ")
      + " - copy one into ComfyUI's models folder to use it here."));
}

/** Roles the plugin declares that the ACTIVE workflow has no slot for. Shown
 *  only for in_workflow === false; null means ComfyUI could not be asked and is
 *  handled by the unreachable branch. */
function appendUnusedRoles(wrap, roles) {
  const absent = (roles || []).filter((r) => r.in_workflow === false && r.required);
  if (!absent.length) return;
  wrap.appendChild(el("div", "sub comfy-model-missing",
    `This workflow has no slot for: ${absent.map((r) => r.label).join(", ")}.`));
}

/** With ComfyUI unreachable, report the workflow's declared roles and which of
 *  this box's registered models could fill each, from localm's own registry. */
function appendRegistryFallback(wrap, data) {
  const roles = data.roles || [];
  if (!roles.length) return;
  wrap.appendChild(el("h5", "comfy-model-picker-head", "Models this needs"));
  for (const role of roles) {
    const row = el("div", "comfy-model-row");
    row.appendChild(el("label", "comfy-model-label", role.label));
    const known = role.registry_models || [];
    row.appendChild(el("span", "comfy-model-known", known.length
      ? `${known.length} registered: ${known.map((m) => m.name).join(", ")}`
      : "none registered in localm"));
    wrap.appendChild(row);
  }
}

export function workflowRow(media, name, label, active, deletable) {
  const row = el("div", "workflow-row" + (active ? " active" : ""));
  const pick = el("button", "btn-secondary workflow-pick");
  // The row's accent bar (.workflow-row.active) marks the selection visually;
  // aria-current carries it for assistive tech.
  pick.appendChild(document.createTextNode(label));
  pick.type = "button";
  if (active) pick.setAttribute("aria-current", "true");
  pick.title = active ? "In use" : "Use this workflow";
  pick.onclick = () => selectWorkflow(media, name);
  row.appendChild(pick);
  if (deletable) {
    const del = el("button", "btn-secondary btn-danger workflow-del", "Delete");
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
    // A different workflow can use different node IDs, so drop the overrides
    // keyed by the old graph.
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
    // Record the successful login for the boot check after the reload.
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

