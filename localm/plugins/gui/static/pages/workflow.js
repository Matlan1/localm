// SPDX-License-Identifier: AGPL-3.0-or-later
/* localm GUI - per-plugin workflow management (Image/Music/Video pages) (split from pages.js). Classic script: it
   shares the one global lexical environment with app.js and the other
   page scripts, so the helpers it uses ($, el, authHeaders, toast, ...)
   resolve by bare name exactly as before. */
"use strict";

// --- ES module imports (auto-generated boundary; bodies unchanged) ---
import { $, authHeaders, confirmDanger, el, toast } from "../app/helpers.js";
import { emptyState } from "../app/icons.js";
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
    box.replaceChildren(emptyState("warning", "Could not load workflows", e.message));
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
    wrap.appendChild(emptyState("warning", "Could not check ComfyUI models", e.message));
    return wrap;
  }
  if (!data.reachable) {
    wrap.appendChild(el("div", "sub", data.message
      || "ComfyUI is not running - launch it to pick models."));
    // Not a dead end any more: the roles and this box's own registered models
    // come from localm's registry and need no ComfyUI at all, so show what the
    // workflow needs and what is already on this machine.
    appendRegistryFallback(wrap, data);
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
  const roleById = new Map((data.roles || []).map((r) => [r.role_id, r]));
  for (const slot of data.slots) {
    const row = el("div", "comfy-model-row");
    // The role label is a friendlier name for the same field, never a
    // replacement for it: the server pairs roles to slots POSITIONALLY within a
    // model type (a ComfyUI graph carries no role names), so on a hand-exported
    // graph the caption can be off by one. Keeping the raw input_name visible
    // means a wrong caption is cosmetic and never hides which field this is.
    const label = el("label", "comfy-model-label",
      slot.role_label ? `${slot.role_label} (${slot.input_name})` : slot.input_name);
    row.appendChild(label);
    if (!slot.options || !slot.options.length) {
      // No live choices to render - a <select> with zero <option>s would show
      // blank and unusable. Name the workflow's own current value instead so
      // the user knows exactly what file to install.
      row.appendChild(el("span", "comfy-model-missing-value job-state st-error", `${slot.current} (not installed)`));
      wrap.appendChild(row);
      // Deliberately BEFORE the continue: a slot ComfyUI has nothing for is
      // exactly where "you already have one registered" matters most, and
      // hanging the hint off the happy path only would skip it there.
      appendRegistryOnlyHint(wrap, roleById.get(slot.role_id), slot.current);
      continue;
    }
    const sel = document.createElement("select");
    sel.className = "comfy-model-select";
    const chosen = overrides[slot.node_id]?.[slot.input_name] ?? slot.current;
    if (!slot.options.includes(chosen)) {
      // The workflow names a file ComfyUI does not have, but it HAS others of
      // the same kind. With no matching <option> the browser silently displays
      // the first one, so the row read as "ae.safetensors is selected" while
      // generation would still use the workflow's own missing value - a
      // dropdown lying about what will run. Show the real value, disabled, so
      // the truth is on screen and picking a live one is a deliberate act.
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

/** "You have one of these registered, ComfyUI is not offering it" - shown only
 *  on a slot ComfyUI could not serve, which is the one state where the registry
 *  has something useful to add. The server already gates that; this leads with
 *  the exact file the workflow names when it is among them, because "you HAVE
 *  this file" and "you have other files of this kind" are different messages. */
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

/** Roles the plugin declares that the ACTIVE workflow has no slot for. Only
 *  shown for a role that is genuinely absent (in_workflow === false); a null
 *  means ComfyUI could not be asked, which is a different statement and is
 *  handled by the unreachable branch instead. */
function appendUnusedRoles(wrap, roles) {
  const absent = (roles || []).filter((r) => r.in_workflow === false && r.required);
  if (!absent.length) return;
  wrap.appendChild(el("div", "sub comfy-model-missing",
    `This workflow has no slot for: ${absent.map((r) => r.label).join(", ")}.`));
}

/** With ComfyUI unreachable, report what the workflow's declared roles are and
 *  which of this box's registered models could fill each - all from localm's own
 *  registry, so it stays true whether or not ComfyUI ever comes up. */
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
  // No radio dot/ring: the row's own accent bar (.workflow-row.active) already
  // marks the selection, and a second glyph saying the same thing just competed
  // with it. The non-visual signal is NOT dropped with the glyph - aria-current
  // carries it for assistive tech, which the decorative icon never did.
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

