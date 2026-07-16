// SPDX-License-Identifier: AGPL-3.0-or-later
/* localm GUI - Settings page (split from pages.js). Classic script: it
   shares the one global lexical environment with app.js and the other
   page scripts, so the helpers it uses ($, el, authHeaders, toast, ...)
   resolve by bare name exactly as before. */
"use strict";

// --- ES module imports (auto-generated boundary; bodies unchanged) ---
import { pickDirectory, pickFile } from "../app/picker.js";
import { $, authHeaders, confirmDanger, el, streamJob, toast } from "../app/helpers.js";
import { emptyState } from "../app/icons.js";
import { caps } from "../app/settings-perf.js";

/* ================================================================ */
/*  Settings page                                                    */
/* ================================================================ */

// The settings form is now schema-driven: it fetches /v1/config/schema (the
// typed CORE_FIELDS metadata with each non-secret field's current value as its
// `default`) and renders the right control per field - a <select> for a fixed
// choice set, a checkbox for a bool, a number input with min/max, a masked
// input for a secret, a comma-edited LIST sent back as a JSON array. This kills
// the old blind text-dumper (and its _CONFIG_SKIP hack for list keys: lists are
// now real LIST inputs that round-trip as arrays). plugins_enabled / plugins
// stay HIDDEN (the schema marks them widget=hidden) - they are plugin STATE
// managed by the Plugins page, not settings. On save we PATCH native types
// (numbers/bools/arrays), which validate_update accepts.

// The schema field list from the last successful fetch, keyed by field for the
// save pass. Each entry mirrors a control: { field, read() }.
export let _settingsControls = [];
// Monotonic token so overlapping refreshes don't both render (the old text
// dumper doubled every field when two refreshes raced; we keep the guard).
export let _settingsRenderToken = 0;
// The top-level GROUP the user is on (a group id below). Survives re-renders so
// saving a section keeps you on its group. Null = use the default (first) group.
export let _activeSettingsGroup = null;

// Top-level settings groups, in nav order. Every .settings-section is assigned to
// one of these via its data-group attribute (static cards carry it in index.html;
// schema + media sections get it set when rendered). A group nav link shows all of
// its sections stacked; conditionally-hidden cards (Updates/Issues via the `hidden`
// attribute, the owner-gated keys card via .sec-hidden) simply do not appear inside
// their group until they apply. This replaces the old one-tab-per-section sprawl.
export const SETTINGS_GROUPS = [
  { id: "model",    label: "Model" },
  { id: "server",   label: "Server & network" },
  { id: "security", label: "Security" },
  { id: "plugins",  label: "Plugins" },
  { id: "media",    label: "Media" },
  { id: "privacy",  label: "Privacy & data" },
  { id: "system",   label: "System" },
];

// Per-section icon + category-colour class for the settings nav (icon names from
// app/icons.js; the `cat-*` class drives the hue via --nav-cat in style.css). Color
// tells you which area at a glance, matching the planning mockup.
export const SETTINGS_NAV_META = {
  model:    { icon: "sliders",  cat: "cat-blue" },
  server:   { icon: "web",      cat: "cat-cyan" },
  security: { icon: "key",      cat: "cat-amber" },
  plugins:  { icon: "plugins",  cat: "cat-violet" },
  media:    { icon: "studio",   cat: "cat-teal" },
  privacy:  { icon: "memory",   cat: "cat-green" },
  system:   { icon: "settings", cat: "cat-slate" },
};

/** A .card-head (category icon + the section's h3) for a schema/plugin settings
 *  section, so it reads like the static cards. The h3 keeps its
 *  .settings-section-head class + text (its divider now comes from .card-head). The
 *  icon + hue follow the section's top-level group (SETTINGS_NAV_META). */
export function settingsSectionHead(heading, groupId) {
  const head = el("div", "card-head");
  const nav = SETTINGS_NAV_META[groupId] || {};
  head.appendChild(iconEl(nav.icon || "settings", "ic cat-ic " + (nav.cat || "cat-slate")));
  const txt = el("div", "card-head-text");
  txt.appendChild(el("h3", "settings-section-head", heading));
  head.appendChild(txt);
  return head;
}

// Which top-level group a core schema `group` string belongs to. Plugin (owner)
// sections go to "plugins"; the Media section is its own top-level group (built
// directly with dataset.group = "media", not looked up here). Anything unmapped
// falls back to "system" (the residual app drawer), so a new core group never
// vanishes.
export const CORE_GROUP_TO_TOP = {
  Engine: "model", Models: "model", Sampling: "model",
  Server: "server", Security: "security", Privacy: "privacy",
  Plugins: "plugins", General: "system", "Bug reports": "system",
};

// Friendlier per-section headings once grouped (the schema `group` string is left
// unchanged, so section ids + validation are untouched - this is display only). An
// empty string means render NO heading: used for the lone require_auth toggle and
// the Privacy persistence block, which are the primary content of their group and
// would only repeat the group name.
export const CORE_SECTION_HEADING = {
  Engine: "Runtime & GPU", Models: "Library", Sampling: "Generation",
  Server: "Network", Security: "", Privacy: "", Plugins: "Plugin management",
};

/** The top-level group id a section element belongs to (defaults to "system"). */
export function sectionTopGroup(sec) {
  return (sec && sec.dataset && sec.dataset.group) || "system";
}

// R10: track which setting inputs the user edited so we can warn before leaving
// with unsaved changes. A control that is re-rendered (full settings refresh, or
// the per-subsection media re-render in R12) disconnects its old node from the
// DOM; the isConnected check below makes those stop counting, so the signal stays
// honest without any per-save bookkeeping.
export const _dirtySettings = new Set();
export function markSettingDirty(input) { _dirtySettings.add(input); }
export function settingsDirty() {
  for (const n of _dirtySettings) if (n.isConnected) return true;
  return false;
}
window.settingsDirty = settingsDirty;

/** R09: Ctrl+S on the Settings page saves the section the user is working in.
 *  A group shows several sections stacked, so target the one holding the focused
 *  control; otherwise fall back to the first savable active section. Reuses the
 *  section's own Save button (all validation/PATCH logic). Returns true on save. */
export function saveActiveSettingsSection() {
  const content = $("settings-content");
  if (!content) return false;
  const savable = (sec) => sec && sec.classList.contains("active")
    && !sec.classList.contains("sec-hidden") && !sec.hidden
    && (sec.querySelector(".settings-section-save") || sec.querySelector(".actions .btn-primary"));
  let sec = document.activeElement && document.activeElement.closest
    ? document.activeElement.closest(".settings-section") : null;
  if (!savable(sec)) {
    sec = [...content.querySelectorAll(".settings-section.active")].find(savable) || null;
  }
  if (!sec) return false;
  const save = sec.querySelector(".settings-section-save") ||
               sec.querySelector(".actions .btn-primary");
  if (!save) return false;
  save.click();
  return true;
}
window.saveActiveSettingsSection = saveActiveSettingsSection;

/** Build one labelled control for a schema field. Returns { field, read } or
 *  null for HIDDEN fields (never rendered). */
export function buildSettingControl(field) {
  if (field.widget === "hidden") return null;
  const value = field.default;     // current value (omitted for secrets)

  const wrap = el("div");
  wrap.dataset.fieldKey = field.key;   // so cross-field wiring can find a control
  const label = el("label", "", field.label || field.key);
  label.title = field.key;
  wrap.appendChild(label);

  let input;
  let read;
  switch (field.widget) {
    case "select": {
      input = document.createElement("select");
      for (const opt of field.options || []) {
        const o = document.createElement("option");
        o.value = opt;
        o.textContent = opt === "" ? "(inherit)" : opt;
        input.appendChild(o);
      }
      input.value = value == null ? "" : String(value);
      read = () => (input.value === "" ? null : input.value);
      break;
    }
    case "toggle": {
      input = document.createElement("input");
      input.type = "checkbox";
      input.checked = !!value;
      input.style.width = "auto";
      read = () => input.checked;
      break;
    }
    case "number": {
      input = document.createElement("input");
      input.type = "number";
      if (field.min != null) input.min = field.min;
      if (field.max != null) input.max = field.max;
      input.step = field.step != null
        ? field.step
        : (Number.isInteger(value) ? "1" : "0.05");
      if (value != null) input.value = value;
      read = () => (input.value.trim() === "" ? undefined : Number(input.value));
      break;
    }
    case "secret": {
      input = document.createElement("input");
      input.type = "password";
      input.value = "";                 // never prefill a real secret
      input.placeholder = "unchanged";
      // Only send a secret when the user actually typed one.
      read = () => (input.value === "" ? undefined : input.value);
      break;
    }
    case "list": {
      input = document.createElement("input");
      input.type = "text";
      input.value = Array.isArray(value) ? value.join(", ") : (value ?? "");
      input.placeholder = "comma-separated";
      read = () => input.value.split(",").map((s) => s.trim()).filter(Boolean);
      break;
    }
    case "pathlist": {
      // A list of server-disk FOLDERS, edited as add/remove rows each with a
      // Browse button (the shared folder picker). Like the FOLDER/PATH controls,
      // it is hidden from a caller without host filesystem access - they cannot
      // browse or set server paths; the server still enforces on write.
      if (caps.fsAccess !== "host") return null;
      const list = el("div", "pathlist");
      // The list container is the dirty anchor: a removed row's <input> is
      // disconnected (so settingsDirty ignores it), but `list` stays in the DOM,
      // so marking IT keeps the unsaved-changes signal honest across add/remove.
      const dirty = () => markSettingDirty(list);
      const addRow = (path = "") => {
        const row = el("div", "dir-picker-row pathlist-row");
        const inp = document.createElement("input");
        inp.type = "text";
        inp.value = path;
        inp.placeholder = "folder path";
        inp.dataset.key = field.key;
        inp.addEventListener("input", dirty);
        inp.addEventListener("change", dirty);
        const browse = el("button", "btn-secondary dir-picker-btn", "Browse...");
        browse.type = "button";
        browse.onclick = async () => {
          const picked = await pickDirectory("Pick a folder to allow", inp.value.trim());
          if (picked) { inp.value = picked; dirty(); }
        };
        const rm = el("button", "btn-secondary pathlist-rm");
        rm.type = "button";
        rm.title = "Remove this folder";
        rm.appendChild(iconEl("trash", "ic"));
        rm.onclick = () => { row.remove(); dirty(); };
        row.append(inp, browse, rm);
        list.appendChild(row);
        return inp;
      };
      const initial = Array.isArray(value) ? value : [];
      for (const p of initial) addRow(p);
      const add = el("button", "btn-secondary pathlist-add");
      add.type = "button";
      add.appendChild(iconEl("plus", "ic"));
      add.appendChild(document.createTextNode("Add folder"));
      add.onclick = async () => {
        // Open the picker straight away (the common case); if the user cancels,
        // still add an empty row they can type into.
        const picked = await pickDirectory("Pick a folder to allow", "");
        addRow(picked || "");
        dirty();
      };
      read = () => [...list.querySelectorAll("input")]
        .map((i) => i.value.trim()).filter(Boolean);
      const write = (v) => {
        list.replaceChildren();
        (Array.isArray(v) ? v : []).forEach((p) => addRow(p));
        dirty();
      };
      wrap.appendChild(list);
      wrap.appendChild(add);
      if (field.help) wrap.appendChild(el("div", "sub", field.help));
      return { field, node: wrap, read, write };
    }
    case "textarea": {   // free-form multi-line (e.g. the default system prompt)
      input = document.createElement("textarea");
      input.rows = 4;
      input.spellcheck = false;
      input.value = value ?? "";
      // Blank is a real, savable value here ("Empty = no default system
      // prompt" - the field's own default IS ""), not a leave-unchanged
      // sentinel like a SECRET field's blank box. Preserve the text's own
      // line breaks otherwise.
      read = () => input.value;
      break;
    }
    default: {   // text / folder / path
      input = document.createElement("input");
      input.type = "text";
      const stored = value ?? "";
      const auto = field.auto || "";   // resolved path for a blank auto-detect field
      if (!stored && auto) {
        // "Blank = auto-detect" used to leave the box EMPTY, hiding what was
        // actually in use. Show the auto-detected path (greyed) so the field is
        // never blank; read() returns null while it is unchanged, so saving
        // keeps the value dynamic (auto) instead of pinning a stale path.
        input.value = auto;
        input.classList.add("auto-detected");
        input.dataset.auto = auto;
        input.addEventListener("input", () => {
          input.classList.toggle("auto-detected", input.value === auto);
        });
        read = () => {
          const v = input.value.trim();
          // Unchanged auto (or cleared back to blank) -> omit from the PATCH, so
          // the field stays dynamic (auto-detect) instead of being pinned.
          if (v === auto || v === "") return undefined;
          return v;
        };
      } else {
        input.value = stored;
        read = () => (input.value.trim() === "" ? null : input.value.trim());
      }
      break;
    }
  }
  input.dataset.key = field.key;
  // R10: editing any control marks the page dirty (programmatic value-setting
  // above does not fire these events, so building a control stays clean).
  input.addEventListener("input", () => markSettingDirty(input));
  input.addEventListener("change", () => markSettingDirty(input));
  // R11: set this control's value (used by the media "Copy from" prefill). Mirrors
  // each widget's read() format, drops the greyed auto-detect look, and fires
  // change so the dirty tracker (R10) and the per-field save diff both see it.
  const write = (v) => {
    if (input.type === "checkbox") input.checked = !!v;
    else if (Array.isArray(v)) input.value = v.join(", ");
    else input.value = v == null ? "" : String(v);
    input.classList.remove("auto-detected");
    input.dispatchEvent(new Event("change", { bubbles: true }));
  };
  // FOLDER / PATH fields get a "Browse..." button wired to the existing
  // directory picker, so the user does not have to type a path by hand (U10).
  const lbl = (field.label || field.key).toLowerCase();
  // An explicit schema flag (widget:"path"/"folder" or accepts_path/accepts_dir)
  // forces the Browse button for a path field whose key/label match none of the
  // naming tokens below - so tagging beats guessing (NEW-M-BROWSE).
  const isPath = field.widget === "path" || field.accepts_path || field.key.endsWith("_path") || field.key.endsWith("_file") || lbl.includes("file") || lbl.includes("path") || lbl.includes("cmd");
  const isDir = field.widget === "folder" || field.accepts_dir || field.key.endsWith("_dir") || lbl.includes("folder") || lbl.includes("dir");
  // A host-path / folder field is server-side config: hide it entirely from a
  // caller without host filesystem access - they cannot (and should not) browse
  // or set paths on the server disk. The server still enforces on /api/fs/* and
  // the config write; this just avoids rendering a dead, confusing field.
  if ((isPath || isDir) && caps.fsAccess !== "host") return null;
  if (isPath || isDir) {
    const row = el("div", "dir-picker-row");
    const browse = el("button", "btn-secondary dir-picker-btn", "Browse...");
    browse.type = "button";                 // never submit the settings form
    browse.dataset.browse = field.key;
    browse.onclick = async () => {
      let picked;
      if (isPath) {
        picked = await pickFile("Pick a file", input.value.trim());
      } else {
        picked = await pickDirectory("Pick a directory", input.value.trim());
      }
      if (picked) { input.value = picked; input.classList.remove("auto-detected"); }
    };
    row.append(input, browse);
    wrap.appendChild(row);
  } else {
    wrap.appendChild(input);
  }
  if (field.help) wrap.appendChild(el("div", "sub", field.help));
  if (field.action) {
    const actRow = el("div", "");
    actRow.style.marginTop = "0.5rem";
    const btn = el("button", "btn-secondary", field.action.label);
    btn.type = "button";
    btn.onclick = async () => {
      const r = await fetch(field.action.endpoint, { method: "POST", headers: authHeaders() });
      if (r.ok) {
        toast(field.action.success_msg || "Action completed");
        refreshSettingsPage();
      } else {
        const err = await r.json().catch(() => ({}));
        toast(err.error || "Action failed", true);
      }
    };
    actRow.appendChild(btn);
    wrap.appendChild(actRow);
  }
  return { field, node: wrap, read, write };
}

// Fetch the server-rendered key QR (owner-scope) and show the "Pair a phone"
// block. Hidden in open mode / when no key is configured (the endpoint 404s).
export async function refreshPairingQR() {
  const wrap = $("pairing"), box = $("pairing-qr");
  if (!wrap || !box) return;
  try {
    const r = await fetch("/api/pairing/qr", { headers: authHeaders() });
    if (!r.ok) { wrap.style.display = "none"; return; }
    const svg = await r.text();   // server-rendered (qrcode) SVG, same-origin
    // Sanitize even though it is our own endpoint (defense in depth, SVG profile).
    box.innerHTML = DOMPurify.sanitize(svg, { USE_PROFILES: { svg: true, svgFilters: true } });
    wrap.style.display = "block";
  } catch (e) {
    wrap.style.display = "none";
  }
}

// Decide what the Companion-app card shows from the server's address info
// (/api/companion) and the current page location. Returns
// { urls: [{kind, url}], hint }:
//   - urls : phone-reachable address(es) - LAN, then Tailscale - each built from
//            THIS page's own scheme + port (the server listens on one port for
//            every interface), so the card never shows the loopback address
//            (127.0.0.1 on a phone is the phone itself).
//   - hint : a one-line note when there is no reachable address to show - on the
//            default loopback bind, how to bind to the network instead.
// Pure + exported so the branches are unit-tested without a live server.
export function companionView(info, loc) {
  info = info || {};
  loc = loc || {};
  const proto = loc.protocol || "https:";
  const port = loc.port ? ":" + loc.port : "";
  const mk = (ip, kind) => ({ kind, url: proto + "//" + ip + port + "/" });
  const urls = [];
  if (info.network_bind) {
    if (info.lan) urls.push(mk(info.lan, "Wi-Fi / LAN"));
    if (info.tailscale) urls.push(mk(info.tailscale, "Tailscale"));
  }
  let hint = "";
  if (!urls.length) {
    hint = info.network_bind
      ? "Could not detect this machine's network address - open its LAN or Tailscale address (with this port) on the phone."
      : "Reachable only on this computer right now. To use it from a phone, restart bound to your network: localm gui -H 0.0.0.0 (set an API key first). See docs/phone.md.";
  }
  return { urls, hint };
}

// Fill the Companion-app card with the phone-reachable address(es) from
// /api/companion, or a hint when there is none yet. Best-effort: a failed fetch
// falls through to companionView's loopback-bind hint.
export async function refreshCompanion() {
  const list = $("companion-addrs"), hintEl = $("companion-hint");
  if (!list || !hintEl) return;
  let info = {};
  try {
    const r = await fetch("/api/companion", { headers: authHeaders() });
    if (r.ok) info = await r.json();
  } catch (e) { /* offline / no endpoint - show the generic hint */ }
  const view = companionView(info, window.location);
  list.replaceChildren();
  for (const u of view.urls) {
    const li = el("li", "companion-addr");
    const a = el("a", "companion-addr-url", u.url);
    a.href = u.url; a.target = "_blank"; a.rel = "noopener";
    li.appendChild(a);
    li.appendChild(el("span", "sub companion-addr-kind", u.kind));
    list.appendChild(li);
  }
  list.style.display = view.urls.length ? "block" : "none";
  hintEl.textContent = view.hint;
  hintEl.style.display = view.hint ? "block" : "none";
}

// Scopes offered in the GUI key minter (label per scope). Privileged scopes
// (coder:full, admin) are shown but OWNER-ONLY: the /v1/keys API refuses them for
// a non-owner key, so a keys:admin device cannot hand out shell / admin access.
export const KEY_SCOPES = [
  ["coder", "Coder agent - restricted: read + edit this project (no shell)"],
  ["coder:full", "Coder agent - FULL: shell + edit (owner-only, dangerous)"],
  ["models:read", "List and inspect models"],
  ["models:write", "Load, download, or remove models"],
  ["rag", "Knowledge (RAG)"],
  ["chat", "Chat history & memory (saved conversations, personas) - NOT needed to chat"],
  ["image", "Image generation"],
  ["music", "Music generation"],
  ["video", "Video generation"],
  ["voice", "Voice"],
  ["web", "Web access"],
  ["mcp", "MCP"],
  ["config:read", "Read settings"],
  ["admin", "Full admin - owner-equivalent (dangerous, owner-only)"],
];

// Quick-select preset buttons. Presets + owner-flag come from the /v1/keys envelope
// (so a keys:admin device that lacks config:read still sees the bundles). Clicking a
// preset sets the scope checkboxes; the OWNER can also save the current pick as a
// preset or delete one (persisted via PATCH /v1/config, which needs config:write).
export function buildKeyPresets(presets, isOwner) {
  const box = $("key-presets");
  if (!box) return;
  box.replaceChildren();
  if (presets && presets.length) {
    box.appendChild(el("span", "sub key-presets-label", "Presets:"));
    for (const p of presets) {
      const b = el("button", "btn-secondary key-preset-btn", p.name);
      b.type = "button";
      b.onclick = () => applyKeyPreset(p.scopes || []);
      if (isOwner) {
        const x = el("span", "key-preset-del", "×");
        x.title = `Delete preset "${p.name}"`;
        x.onclick = (ev) => { ev.stopPropagation(); deleteKeyPreset(p.name, presets); };
        b.appendChild(x);
      }
      box.appendChild(b);
    }
  }
  if (isOwner) {
    const save = el("button", "btn-secondary key-preset-save", "+ Save as preset");
    save.type = "button";
    save.onclick = () => saveCurrentAsPreset(presets || []);
    box.appendChild(save);
  }
}

export function applyKeyPreset(want) {
  const set = new Set(want), box = $("key-scopes");
  if (!box) return;
  for (const cb of box.querySelectorAll(".key-scope-cb")) {
    cb.checked = !cb.disabled && set.has(cb.value);   // never check a disabled (owner-only) scope
  }
}

// Disable + dim the owner-only (privileged) scopes for a non-owner key minter, so a
// keys:admin device cannot even try to mint a coder:full / admin key (the API would
// 403 anyway; this avoids the confusing failed-submit round-trip).
export function applyOwnerGate(isOwner) {
  for (const cb of document.querySelectorAll("#key-scopes .key-scope-cb")) {
    const ownerOnly = cb.value === "admin" || cb.value.endsWith(":full");
    cb.disabled = ownerOnly && !isOwner;
    if (cb.disabled) cb.checked = false;
    const lab = cb.closest(".key-scope");
    if (lab) lab.classList.toggle("key-scope-disabled", cb.disabled);
  }
}

// Owner-only: persist an edited preset list (PATCH /v1/config needs config:write).
export async function saveKeyPresets(presets) {
  const r = await fetch("/v1/config", {
    method: "PATCH",
    headers: { ...authHeaders(), "Content-Type": "application/json" },
    body: JSON.stringify({ key_presets: presets }),
  });
  if (r.ok) { toast("Presets saved"); refreshKeysPanel(); }
  else {
    const e = await r.json().catch(() => ({}));
    toast(e.detail || "Could not save presets", true);
  }
}

export function saveCurrentAsPreset(presets) {
  const scopes = [...document.querySelectorAll("#key-scopes .key-scope-cb")]
    .filter((c) => c.checked).map((c) => c.value);
  if (!scopes.length) { toast("Check the scopes for the preset first"); return; }
  const name = (prompt("Preset name:") || "").trim();
  if (!name) return;
  const next = presets.filter((p) => p.name !== name);   // replace an existing name
  next.push({ name, scopes });
  saveKeyPresets(next);
}

export function deleteKeyPreset(name, presets) {
  confirmDanger(`Delete preset "${name}"?`, "This can't be undone.", "Delete", () => {
    saveKeyPresets(presets.filter((p) => p.name !== name));
  });
}

// Server-rendered pairing QR for a freshly-minted SCOPED key: scan it in localm
// on the other device to pair it with exactly these capabilities (no typing).
export async function renderKeyQR(box, key) {
  try {
    const r = await fetch("/api/pairing/qr", {
      method: "POST",
      headers: { ...authHeaders(), "Content-Type": "application/json" },
      body: JSON.stringify({ key }),
    });
    if (!r.ok) return;                        // owner-only / older server: skip the QR
    const svg = await r.text();
    const wrap = el("div", "key-qr");
    wrap.appendChild(el("div", "sub",
      "Or scan to pair a phone (open localm on the phone, tap Scan QR code):"));
    const holder = document.createElement("div");
    holder.className = "key-qr-img";
    // Same-origin server-rendered SVG; sanitized (SVG profile) exactly like
    // refreshPairingQR - defense in depth before assigning to innerHTML.
    holder.innerHTML = DOMPurify.sanitize(svg, { USE_PROFILES: { svg: true, svgFilters: true } });
    wrap.appendChild(holder);
    box.appendChild(wrap);
  } catch (e) { /* QR is best-effort; the copyable secret is the fallback */ }
}

export function keyExpiryLabel(expires) {
  if (!expires) return "never expires";
  const ms = expires * 1000 - Date.now();
  if (ms <= 0) return "expired";
  const days = Math.floor(ms / 86400000);
  if (days >= 1) return `expires in ${days}d`;
  return `expires in ${Math.max(1, Math.floor(ms / 3600000))}h`;
}

export function keyLastUsedLabel(ts) {
  if (!ts) return "unused";
  const ms = Date.now() - ts * 1000;
  if (ms < 0) return "used just now";
  const days = Math.floor(ms / 86400000);
  if (days >= 1) return `used ${days}d ago`;
  const hrs = Math.floor(ms / 3600000);
  if (hrs >= 1) return `used ${hrs}h ago`;
  return "used recently";
}

// Settings -> API keys: mint named, scope-limited keys, list them, revoke them.
// Owner-gated (/v1/keys needs keys:admin); the card hides for a non-owner key.
export async function refreshKeysPanel() {
  const card = $("keys-card"), list = $("keys-list"), scopesBox = $("key-scopes");
  if (!card || !list || !scopesBox) return;

  if (!scopesBox.childElementCount) {           // render the checkboxes once
    for (const [scope, label] of KEY_SCOPES) {
      const danger = scope === "admin" || scope.endsWith(":full");
      const lab = el("label", "key-scope" + (danger ? " key-scope-danger" : ""));
      const cb = document.createElement("input");
      cb.type = "checkbox"; cb.value = scope; cb.className = "key-scope-cb";
      lab.appendChild(cb);
      lab.appendChild(document.createTextNode(" " + label));
      scopesBox.appendChild(lab);
    }
    // Chat is the baseline: a key needs NO scope to chat (the "chat" scope above
    // only gates server-saved history/personas). Scopes add EXTRA capabilities.
    if (!$("key-scope-note")) {
      const note = el("div", "sub");
      note.id = "key-scope-note";
      note.textContent = "Chatting needs no scope - any key can chat. These add "
        + "capabilities; leave all unchecked for a chat-only key.";
      scopesBox.insertAdjacentElement("afterend", note);
    }
  }

  // The keys card is a settings SECTION; hide/show it via a class so the section
  // nav (built by buildSettingsNav) drops/re-adds its link for non-owners,
  // rather than an inline display style that would fight the section show/hide.
  const setHidden = (hidden) => {
    card.classList.toggle("sec-hidden", hidden);
    if (typeof buildSettingsNav === "function") buildSettingsNav();
  };
  let keys = [], isOwner = false, presets = [];
  try {
    const r = await fetch("/v1/keys", { headers: authHeaders() });
    if (!r.ok) { setHidden(true); return; }   // 401/403 -> not a key minter
    setHidden(false);
    const data = await r.json();
    keys = data.keys || [];
    isOwner = !!data.is_owner;
    presets = data.presets || [];
  } catch (e) { setHidden(true); return; }
  applyOwnerGate(isOwner);           // hide owner-only scopes from a keys:admin device
  buildKeyPresets(presets, isOwner);

  list.replaceChildren();
  if (!keys.length) {
    list.appendChild(emptyState("key", "No named keys yet",
      "Mint a scope-limited key above to pair a device or person."));
  }
  for (const k of keys) {
    const row = el("div", "key-row");
    row.appendChild(el("span", "name", k.name || k.id));
    row.appendChild(el("span", "mono key-scope-tags", (k.scopes || []).join(", ")));
    row.appendChild(el("span", "sub key-expiry-tag", keyExpiryLabel(k.expires)));
    row.appendChild(el("span", "sub key-lastused-tag", keyLastUsedLabel(k.last_used)));
    const rm = el("button", "btn-secondary", "Revoke");
    rm.onclick = () => {
      confirmDanger(`Revoke key "${k.name || k.id}"?`,
        "Anything using this key immediately loses access.", "Revoke", async () => {
          const d = await fetch(`/v1/keys/${encodeURIComponent(k.id)}`,
                                { method: "DELETE", headers: authHeaders() });
          if (d.ok) { toast("Key revoked"); refreshKeysPanel(); }
          else { toast("Revoke failed"); }
        });
    };
    row.appendChild(rm);
    list.appendChild(row);
  }

  $("key-create").onclick = async () => {
    const name = ($("key-name").value || "").trim();
    if (!name) { toast("Enter a key name"); return; }
    const scopes = [...scopesBox.querySelectorAll(".key-scope-cb")]
      .filter((c) => c.checked).map((c) => c.value);
    // A zero-scope key is valid: it can still chat (chat is baseline). Confirm so
    // an empty pick is intentional rather than a forgotten checkbox.
    if (!scopes.length
        && !confirm("Create a chat-only key (no extra capabilities)?")) return;
    const body = { name, scopes };
    const ttl = Number(($("key-expiry") || {}).value || 0);
    if (ttl > 0) body.expires_in = ttl;   // server computes the deadline (its own clock)
    let r;
    try {
      r = await fetch("/v1/keys", {
        method: "POST",
        headers: { ...authHeaders(), "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
    } catch (e) { toast("Create failed"); return; }
    if (!r.ok) {
      const e = await r.json().catch(() => ({}));
      toast(e.detail || "Create failed"); return;
    }
    const made = await r.json();
    const box = $("key-secret");
    box.replaceChildren();
    box.appendChild(el("div", "sub", `New key "${made.name}" `
      + `(${(made.scopes || []).join(", ")}) - copy it now, it is shown only once:`));
    const secret = document.createElement("input");
    secret.type = "text"; secret.readOnly = true; secret.value = made.key;
    secret.className = "key-secret-value";
    box.appendChild(secret);
    const copy = el("button", "btn-secondary", "Copy");
    copy.onclick = () => {
      secret.select();
      if (navigator.clipboard) navigator.clipboard.writeText(made.key);
      toast("Copied");
    };
    box.appendChild(copy);
    await renderKeyQR(box, made.key);
    box.style.display = "";
    $("key-name").value = "";
    scopesBox.querySelectorAll(".key-scope-cb").forEach((c) => { c.checked = false; });
    refreshKeysPanel();
  };
}

// Friendly section label per plugin owner (falls back to the capitalized scope).
export const PLUGIN_SECTION_LABEL = {
  image: "Image", web: "Web access", voice: "Voice", coder: "Coder",
  music: "Music", video: "Video", rag: "Knowledge",
  mcp: "MCP", chat: "Chat",
};

/** Which settings section a field belongs to: each core `group` is its own
 *  section; each plugin (owner != core) is its own section (its own tab). */
export function settingsSectionOf(field) {
  if (field.owner && field.owner !== "core") {
    return {
      id: "plugin-" + field.owner,
      label: PLUGIN_SECTION_LABEL[field.owner]
        || (field.owner.charAt(0).toUpperCase() + field.owner.slice(1)),
      plugin: true,
    };
  }
  return { id: "core-" + field.group, label: field.group, plugin: false };
}

/** Show one top-level GROUP: activate every section assigned to it and hide the
 *  rest, then highlight its nav link. Sections stay `.active` in DOM order, so a
 *  group renders its members stacked. Conditionally-hidden members (Updates/Issues
 *  via the `hidden` attribute, the owner-gated keys card via `.sec-hidden`) are
 *  still activated but kept invisible by their display:none !important rule, so
 *  they self-reveal the moment they apply without another nav rebuild. */
export function showSettingsGroup(groupId) {
  const content = $("settings-content");
  if (!content) return;
  for (const sec of content.querySelectorAll(".settings-section")) {
    sec.classList.toggle("active", sectionTopGroup(sec) === groupId);
  }
  const nav = $("settings-nav");
  if (nav) {
    for (const link of nav.querySelectorAll(".settings-nav-link")) {
      link.classList.toggle("active", link.dataset.target === groupId);
    }
  }
}

/** Jump to the GROUP that owns section *secId* and scroll that section into view,
 *  remembering the group so an async rebuild (e.g. the owner-only keys card
 *  resolving) cannot bounce the selection. Used by the command palette to reach
 *  "Keys & devices" (inside the Security group) directly. Accepts a raw section id
 *  ("keys-card") or a schema section id; falls back to treating the arg as a group
 *  id so older callers still work. */
export function gotoSettingsSection(secId) {
  const sec = document.getElementById(secId)
    || document.getElementById("settings-sec-" + secId);
  const groupId = sec ? sectionTopGroup(sec)
    : (SETTINGS_GROUPS.some((g) => g.id === secId) ? secId : null);
  if (!groupId) return;
  _activeSettingsGroup = groupId;
  showSettingsGroup(groupId);
  if (sec && typeof sec.scrollIntoView === "function") {
    sec.scrollIntoView({ block: "start" });
  }
}
window.gotoSettingsSection = gotoSettingsSection;

/** Whether a group has at least one section that is NOT gate-hidden (owner-only
 *  keys card via .sec-hidden, or a proxy-gated Updates/Issues card via [hidden]).
 *  A group with only gated-away sections drops out of the nav until one applies. */
function groupHasVisibleSection(content, groupId) {
  for (const sec of content.querySelectorAll(".settings-section")) {
    if (sectionTopGroup(sec) !== groupId) continue;
    if (sec.classList.contains("sec-hidden") || sec.hidden) continue;
    return true;
  }
  return false;
}

/** (Re)build the left nav as ONE link per top-level group (SETTINGS_GROUPS order),
 *  listing only groups that currently have a visible section. The active group is
 *  the user's explicit choice if still present, else the first group - chosen
 *  deterministically so a stray rebuild (e.g. the owner-only keys panel resolving)
 *  can never bounce the selection. */
export function buildSettingsNav() {
  const nav = $("settings-nav"), content = $("settings-content");
  if (!nav || !content) return;
  const present = SETTINGS_GROUPS.filter((g) => groupHasVisibleSection(content, g.id));
  nav.replaceChildren();
  for (const g of present) {
    const meta = SETTINGS_NAV_META[g.id] || {};
    const link = el("button", "settings-nav-link" + (meta.cat ? " " + meta.cat : ""));
    link.dataset.target = g.id;
    link.appendChild(iconEl(meta.icon || "settings", "nav-ic"));
    link.appendChild(document.createTextNode(g.label));
    link.onclick = () => { _activeSettingsGroup = g.id; showSettingsGroup(g.id); };
    nav.appendChild(link);
  }
  let target = null;
  if (_activeSettingsGroup && present.some((g) => g.id === _activeSettingsGroup)) {
    target = _activeSettingsGroup;                 // the user's chosen group, still present
  } else if (present.length) {
    target = present[0].id;                         // default: the first group (Model)
  }
  if (target) showSettingsGroup(target);
}

/** Save just one section: PATCH only the keys whose controls live in it. */
export async function saveSettingsSection(secId) {
  const panel = $("settings-sec-" + secId);
  if (!panel) return;
  const updates = {};
  for (const { field, node, read } of _settingsControls) {
    if (!node || !panel.contains(node)) continue;
    const value = read();
    if (value === undefined) continue;     // untouched secret / blank number
    updates[field.key] = value;
  }
  if (!Object.keys(updates).length) { toast("Nothing changed"); return; }
  const r = await fetch("/v1/config", {
    method: "PATCH", headers: authHeaders(),
    body: JSON.stringify(updates),
  });
  const data = await r.json().catch(() => ({}));
  if (r.ok) {
    toast("Saved - engine values apply on the next model load");
    refreshSettingsPage();   // re-render to reflect server-normalized values
  } else {
    toast(data.detail || "Save failed", true);
  }
}

export async function refreshSettingsPage() {
  const myToken = ++_settingsRenderToken;
  _dirtySettings.clear();   // R10: a fresh render is a clean baseline
  $("gui-api-key").value = "";   // HttpOnly key is unreadable; field is for entry only
  const form = $("config-form");
  let fields;
  try {
    const r = await fetch("/v1/config/schema", { headers: authHeaders() });
    if (!r.ok) throw new Error(r.statusText);
    fields = (await r.json()).fields || [];
  } catch (e) {
    if (myToken === _settingsRenderToken) {
      form.replaceChildren(el("div", "sub", "Could not load settings: " + e.message));
    }
    return;
  }
  if (myToken !== _settingsRenderToken) return;  // a newer refresh superseded us

  // One section per core group and per plugin. Core sections first (in field
  // order), then plugin sections (each its own tab with its own Save button).
  const controls = [];
  const sections = new Map();        // id -> { id, label, plugin, ctrls: [] }
  for (const field of fields) {
    // Media (ComfyUI) config is rendered in its own Media section below, one
    // subsection per plugin (image/music/video), edited per-plugin via the
    // /v1/media/config endpoint - not as flat keys here.
    if (field.group === "Media") continue;
    const ctrl = buildSettingControl(field);
    if (!ctrl) continue;             // HIDDEN
    controls.push(ctrl);
    const s = settingsSectionOf(field);
    if (!sections.has(s.id)) sections.set(s.id, { ...s, ctrls: [] });
    sections.get(s.id).ctrls.push(ctrl);
  }
  _settingsControls = controls;

  const ordered = [...sections.values()]
    .sort((a, b) => (a.plugin ? 1 : 0) - (b.plugin ? 1 : 0));   // core first, stable

  form.replaceChildren();
  for (const sec of ordered) {
    // Heading: plugin sections keep "<name> plugin"; core sections use the
    // friendlier grouped heading (schema `group` unchanged), and a blank heading
    // (the lone require_auth toggle, the Privacy block) renders no <h3> at all so
    // it does not just echo its group name.
    const heading = sec.plugin ? (sec.label + " plugin")
      : (sec.label in CORE_SECTION_HEADING ? CORE_SECTION_HEADING[sec.label] : sec.label);
    const topGroup = sec.plugin ? "plugins" : (CORE_GROUP_TO_TOP[sec.label] || "system");
    const panel = el("section", "card settings-section");
    panel.id = "settings-sec-" + sec.id;
    panel.dataset.sec = sec.id;
    panel.dataset.group = topGroup;
    panel.dataset.secLabel = heading || sec.label;
    if (heading) panel.appendChild(settingsSectionHead(heading, topGroup));
    const grid = el("div", "settings-fields");
    for (const c of sec.ctrls) grid.appendChild(c.node);
    panel.appendChild(grid);
    const actions = el("div", "actions");
    const save = el("button", "btn-primary settings-section-save", "Save " + (heading || sec.label));
    save.dataset.sec = sec.id;
    save.onclick = () => saveSettingsSection(sec.id);
    actions.appendChild(save);
    panel.appendChild(actions);
    form.appendChild(panel);
  }

  // Per-plugin Media (ComfyUI) config: one "Media" section (its own top-level
  // nav group) with a compact managed-ComfyUI panel on top and one subsection
  // per plugin (image/music/video) below. Appended after the core schema
  // sections; fields is passed through so the two managed-ComfyUI schema
  // fields (group="Media", skipped from the flat loop above) can be rendered
  // here instead.
  await buildMediaSection(form, fields);
  if (myToken !== _settingsRenderToken) return;  // a newer refresh superseded us

  // Build the nav now that the schema sections exist, so the first config
  // section (not a static card) is the default tab. The owner-gated panels then
  // refresh: each may rebuild the nav, but they preserve the active section.
  buildSettingsNav();
  syncRagIndexingModeHint();
  refreshPairingQR();
  refreshCompanion();
  refreshKeysPanel();
}

/** Mark which folder list the current RAG indexing MODE actually uses (Allowed in
 *  whitelist mode, Denied in blacklist mode) with a small "in use" tag, while
 *  keeping BOTH lists visible and editable - you can curate both without flipping
 *  the mode. The lists are stored separately, so the mode never reinterprets your
 *  entries. No-op when the owner-only Knowledge fields are absent (a non-owner
 *  never receives them). */
export function syncRagIndexingModeHint() {
  const sel = document.querySelector('select[data-key="rag_indexing_mode"]');
  const allow = document.querySelector('[data-field-key="rag_allowed_roots"]');
  const deny = document.querySelector('[data-field-key="rag_denied_roots"]');
  if (!sel || !allow || !deny) return;
  const mark = (wrap, on) => {
    const label = wrap.querySelector("label");
    if (!label) return;
    let tag = label.querySelector(".rag-inuse");
    if (on && !tag) label.appendChild(el("span", "rag-inuse", " · in use"));
    else if (!on && tag) tag.remove();
  };
  const apply = () => {
    const whitelist = sel.value !== "blacklist";
    mark(allow, whitelist);
    mark(deny, !whitelist);
  };
  sel.addEventListener("change", apply);
  apply();
}

// Media plugins, in display order, that the Media section configures.
export const MEDIA_PLUGIN_ORDER = ["image", "music", "video"];

/** Did a media control's value change from what was displayed? Treats
 *  null/undefined/"" as the same "empty", so saving an untouched inherited field
 *  does not pin it as an override. */
export function _mediaChanged(cur, orig) {
  const empty = (v) => v === null || v === undefined || v === "";
  if (empty(cur) && empty(orig)) return false;
  return cur !== orig;
}

/** Build the "Media" settings section (its own top-level nav group): a compact
 *  "localm's own ComfyUI" panel on top (its own little box - the set-up/remove
 *  action plus the two coexistence fields), then one subsection per media
 *  plugin (image/music/video) below, laid out as a responsive grid of three
 *  boxes. Each subsection edits that plugin's own ComfyUI config block via
 *  /v1/media/config; a field left at its inherited value is not sent, so the
 *  plugin keeps falling back to the shared default until the user overrides it.
 *  *fields* is the full core schema list, so the two managed-ComfyUI toggle
 *  fields (group="Media", skipped from the flat core loop above) can be pulled
 *  out and rendered here instead of vanishing. */
export async function buildMediaSection(form, fields) {
  let data;
  try {
    const r = await fetch("/v1/media/config", { headers: authHeaders() });
    if (!r.ok) throw new Error(r.statusText);
    data = await r.json();
  } catch (e) {
    // Surface, do not hide (binding rule 5): the group=Media fields are skipped
    // from the flat form, so if this fetch fails we must SHOW that the media
    // settings could not load - silently dropping the section would make those
    // settings vanish with no clue why.
    const fail = el("section", "card settings-section");
    fail.id = "settings-sec-media";
    fail.dataset.sec = "media";
    fail.dataset.group = "media";
    fail.dataset.secLabel = "Media";
    fail.appendChild(settingsSectionHead("Media", "media"));
    fail.appendChild(el("div", "sub",
      "Could not load media settings (" + e.message + "). The image/music/video "
      + "config is unavailable - check the server logs."));
    form.appendChild(fail);
    return;
  }
  const byName = {};
  for (const p of (data.plugins || [])) byName[p.plugin] = p;

  const panel = el("section", "card settings-section");
  panel.id = "settings-sec-media";
  panel.dataset.sec = "media";
  panel.dataset.group = "media";
  panel.dataset.secLabel = "Media";
  panel.appendChild(settingsSectionHead("Media", "media"));
  panel.appendChild(el("div", "sub",
    "ComfyUI settings for image, music, and video, each configured "
    + "independently. A blank field uses the shared default."));

  // S5: localm's OWN managed ComfyUI (set up / status / remove), plus the S1
  // coexistence field (comfy_target) it needs to be useful - a compact box of
  // its own, ahead of the three per-plugin boxes.
  const targetField = (fields || []).find(f => f.key === "comfy_target");
  const managed = el("div", "media-comfy-box");
  panel.appendChild(managed);
  renderManagedComfyPanel(managed, { targetField });

  // R11/R12: register every subsection's node + label first (empty), so that when
  // we render each, its "Copy from <other>" buttons can see the other subsections,
  // and a later single-subsection re-render (R12) can find its node.
  _mediaSubs = {};
  _mediaControls = {};
  const grid = el("div", "media-grid");
  panel.appendChild(grid);
  for (const name of MEDIA_PLUGIN_ORDER) {
    const p = byName[name];
    if (!p) continue;
    const sub = el("div", "media-subsection");
    sub.dataset.plugin = name;
    _mediaSubs[name] = { sub, label: p.label, fields: p.fields || [] };
    grid.appendChild(sub);
  }
  for (const name of MEDIA_PLUGIN_ORDER) {
    if (_mediaSubs[name]) renderMediaSubsection(name);
  }

  form.appendChild(panel);
}

/** (Re)render the compact "localm's own ComfyUI" box in *host*: read
 *  /api/comfy/managed-status, then show either a Set-up button (not installed
 *  yet, so the coexistence field below would be inert - progressive
 *  disclosure keeps the box small) or, once installed, the install path plus
 *  the coexistence control (comfy_target, from *toggleFields*) with its own
 *  small Save, and a Remove button. Set-up
 *  POSTs /api/comfy/setup and streams the install job; Remove POSTs
 *  /api/comfy/remove. Re-renders itself in place after either finishes, so the
 *  view always matches what is on disk. Off by default: nothing here runs
 *  until the user clicks Set up. */
export async function renderManagedComfyPanel(host, toggleFields) {
  host.replaceChildren();
  const head = el("div", "media-comfy-head");
  head.appendChild(el("h4", "media-sub-head", "localm's own ComfyUI"));
  const pill = el("span", "comfy-pill", "checking...");
  head.appendChild(pill);
  host.appendChild(head);

  let st;
  try {
    const r = await fetch("/api/comfy/managed-status", { headers: authHeaders() });
    if (!r.ok) throw new Error(r.statusText);
    st = await r.json();
  } catch (e) {
    // Surface, do not hide (rule 5): if the managed state cannot be read, say so
    // rather than defaulting to a possibly-wrong (e.g. Set-up) view.
    pill.textContent = "unknown";
    host.appendChild(el("div", "sub",
      "Could not read the managed ComfyUI status (" + e.message + ")."));
    return;
  }

  if (st.installed) {
    // Durable disclosure of whether media generation ACTUALLY routes here
    // right now (matches `localm comfy status`'s "Target now" line) - not
    // just a one-time setup toast, so it stays visible on every later visit
    // to this page too, whichever way comfy_target is currently set.
    pill.textContent = st.managed_active ? "installed - in use" : "installed - not in use";
    if (st.managed_active) pill.classList.add("ok");
    const info = el("div", "sub");
    info.append("Installed at ");
    info.appendChild(el("code", null, st.path || ""));
    if (!st.managed_active) {
      info.append(" - image/music/video generation is using your OWN ComfyUI "
        + "instead (set \"ComfyUI to use\" to \"own\" below to switch).");
    }
    host.appendChild(info);

    // The S1 coexistence control only matters once an instance exists; render
    // it here (rather than earlier as an inert field) and save it with the
    // Media section's own generic save path (it is an ordinary core field).
    // Built independently of the actions row below: Remove must always be
    // offered once installed, even if the schema fetch is missing this field
    // for some reason (rule 5 - a partial schema must not hide Remove).
    const row = el("div", "media-comfy-row");
    const ctrls = [];
    for (const field of [toggleFields.targetField]) {
      if (!field) continue;
      const ctrl = buildSettingControl(field);
      if (!ctrl) continue;
      ctrls.push(ctrl);
      row.appendChild(ctrl.node);
    }
    if (ctrls.length) {
      host.appendChild(row);
      // This panel can re-render in place (after Set up / Remove / Save)
      // without a full page refresh, which would otherwise leave stale,
      // detached-node entries for these two keys piling up in the shared
      // control list every time - drop any prior entries for them first.
      const keys = new Set(ctrls.map(c => c.field.key));
      _settingsControls = _settingsControls.filter(c => !keys.has(c.field.key));
      _settingsControls.push(...ctrls);
    }

    const actions = el("div", "actions");
    if (ctrls.length) {
      const save = el("button", "btn-primary", "Save");
      save.type = "button";
      save.onclick = () => saveSettingsSection("media");
      actions.appendChild(save);
    }
    const remove = el("button", "btn-secondary comfy-managed-remove-btn", "Remove");
    remove.type = "button";
    remove.onclick = () => confirmDanger(
      "Remove localm's ComfyUI?",
      "This deletes localm's own ComfyUI under the data folder. Your own ComfyUI is "
      + "not touched, and downloaded models are kept.",
      "Remove",
      async () => {
        remove.disabled = true;
        try {
          const r = await fetch("/api/comfy/remove",
                                { method: "POST", headers: authHeaders() });
          const d = await r.json().catch(() => ({}));
          toast(r.ok ? "Removed localm's ComfyUI" : (d.detail || "Remove failed"), !r.ok);
        } catch (e) {
          toast("Remove failed", true);
        }
        renderManagedComfyPanel(host, toggleFields);
      });
    actions.appendChild(remove);
    host.appendChild(actions);
    return;
  }

  if (st.state === "corrupt") {
    // An earlier setup attempt was abandoned (a crashed process, a closed
    // browser tab mid-setup) before the completion marker was written - the
    // checkout dir exists but is_managed_comfy_installed() correctly says no.
    // Offer Repair instead of a dead end: Set up would just hit the route's
    // own "already exists" 409, and there is no Remove button in THIS branch
    // (that only appears once genuinely installed) - a user would otherwise
    // have no way out short of the CLI.
    pill.textContent = "incomplete - needs repair";
    host.appendChild(el("div", "sub",
      "A previous setup attempt did not finish (the folder exists at "
      + (st.path || "the data folder") + " but the install is incomplete) - your "
      + "downloaded models and settings are untouched either way."));
    const actions = el("div", "actions");
    const repair = el("button", "btn-primary comfy-managed-repair-btn", "Repair");
    repair.type = "button";
    actions.appendChild(repair);
    host.appendChild(actions);
    const log = el("pre", "comfy-managed-log");
    log.style.display = "none";
    host.appendChild(log);
    repair.onclick = () => confirmDanger(
      "Repair localm's ComfyUI?",
      "This clears the incomplete install folder and sets it up again from "
      + "scratch. Your downloaded models and localm settings live outside it "
      + "and are not touched.",
      "Repair",
      async () => {
        repair.disabled = true;
        repair.textContent = "Repairing...";
        log.style.display = "";
        log.textContent = "";
        let jobId;
        try {
          const r = await fetch("/api/comfy/repair",
                                { method: "POST", headers: authHeaders() });
          const d = await r.json().catch(() => ({}));
          if (!r.ok) {
            toast(d.detail || "Repair failed", true);
            repair.disabled = false;
            repair.textContent = "Repair";
            log.style.display = "none";
            return;
          }
          jobId = d.job_id;
        } catch (e) {
          toast("Repair failed", true);
          repair.disabled = false;
          repair.textContent = "Repair";
          return;
        }
        const end = await streamJob(jobId, (line) => {
          log.textContent += line + "\n";
          log.scrollTop = log.scrollHeight;
        });
        const ok = !!(end && end.status === "done");
        toast(ok ? "localm's ComfyUI is ready" : "Repair did not finish (see the log)", !ok);
        if (ok) {
          renderManagedComfyPanel(host, toggleFields);
        } else {
          // Same reasoning as the Set-up flow below: do not re-render over a
          // failure log the toast just pointed the user at.
          repair.disabled = false;
          repair.textContent = "Repair";
        }
      });
    return;
  }

  // "installing": a setup job is genuinely running right now (checked by
  // reading the actual job registry, not inferred) - most likely another
  // browser tab/session started it, or this page was reloaded mid-setup, so
  // there is no local job id to re-attach a live log to here. Say so plainly
  // rather than offering a Set-up button that would just 409.
  pill.textContent = st.state === "installing" ? "installing..." : "not set up";
  host.appendChild(el("div", "sub",
    st.state === "installing"
      ? "A setup is currently running (started from another tab or session) - "
        + "this will update once it finishes."
      : "Optional: let localm run its OWN ComfyUI under the localm data folder so it "
        + "can pin a known-good version and carry fixes. Off by default; your own "
        + "ComfyUI is never modified."));
  if (st.state === "installing") return;

  // Not installed: offer to set it up. Provisioning is long (multi-GB), so it runs
  // as a background job whose log streams into <pre> below - the request never blocks.
  const actions = el("div", "actions");
  const setup = el("button", "btn-primary comfy-managed-setup-btn",
                   "Set up localm's own ComfyUI");
  setup.type = "button";
  actions.appendChild(setup);
  host.appendChild(actions);
  const log = el("pre", "comfy-managed-log");
  log.style.display = "none";
  host.appendChild(log);
  const reset = () => {
    setup.disabled = false;
    setup.textContent = "Set up localm's own ComfyUI";
  };
  setup.onclick = async () => {
    setup.disabled = true;
    setup.textContent = "Setting up...";
    log.style.display = "";
    log.textContent = "";
    let jobId;
    try {
      const r = await fetch("/api/comfy/setup",
                            { method: "POST", headers: authHeaders() });
      const d = await r.json().catch(() => ({}));
      if (!r.ok) {
        toast(d.detail || "Setup failed", true);
        reset();
        log.style.display = "none";
        return;
      }
      jobId = d.job_id;
    } catch (e) {
      toast("Setup failed", true);
      reset();
      return;
    }
    const end = await streamJob(jobId, (line) => {
      log.textContent += line + "\n";
      log.scrollTop = log.scrollHeight;
    });
    const ok = !!(end && end.status === "done");
    toast(ok ? "localm's ComfyUI is ready" : "Setup did not finish (see the log)", !ok);
    if (ok) {
      // Re-read status: swaps to the coexistence-fields + Remove view.
      renderManagedComfyPanel(host, toggleFields);
    } else {
      // Do NOT re-render on failure: renderManagedComfyPanel starts with
      // host.replaceChildren(), which would destroy the log this toast just
      // told the user to check, the instant it appears - the log window
      // "vanishes when it errors" with no way to read the real reason.
      // Leave it visible; re-enable the button so a retry is one click away
      // (its own onclick clears the log itself, before starting fresh).
      reset();
    }
  };
}

// R11/R12: live media subsection registry, so we can re-render just the saved one
// (R12) and prefill fields between subsections client-side (R11).
export let _mediaSubs = {};        // name -> { sub: <div.media-subsection>, label, fields }
export let _mediaControls = {};    // name -> controls[] (each {field,node,read,write,orig})

/** (Re)build one media subsection's body in place (head + grid + Copy-from + Save)
 *  from its registered fields. Used for the initial build and the R12 single-
 *  subsection re-render after a save. */
export function renderMediaSubsection(name) {
  const entry = _mediaSubs[name];
  if (!entry) return;
  const { sub, label, fields } = entry;
  sub.replaceChildren();
  const head = el("h4", "media-sub-head", label);
  sub.appendChild(head);
  if (["image", "music", "video"].includes(name)) {
    const badge = el("span", "sub comfy-status-badge", "ComfyUI: checking...");
    badge.style.marginLeft = "12px";
    badge.style.padding = "2px 6px";
    badge.style.borderRadius = "4px";
    badge.style.backgroundColor = "var(--bg-input)";
    head.appendChild(badge);
    
    // NEW-STOPCOMFY: Stop/Restart controls. Stop shows whenever ComfyUI is up
    // (it always aborts the render + clears the queue; it terminates the process
    // only if localm launched it). Restart shows only for a localm-launched one.
    const stopBtn = el("button", "btn-secondary comfy-stop-btn", "Stop");
    const restartBtn = el("button", "btn-secondary comfy-restart-btn", "Restart");
    for (const b of [stopBtn, restartBtn]) {
      b.type = "button";
      b.style.marginLeft = "8px";
      b.style.display = "none";
    }
    head.appendChild(stopBtn);
    head.appendChild(restartBtn);

    // ComfyUI status is checked here (Settings page opened), on app start, on
    // the Image/Music/Video page being opened, before the first task
    // submission, and after Stop/Restart below - not on a recurring timer.
    // A 5-second setInterval used to poll this badge continuously for as
    // long as Settings stayed open; ComfyUI does not appear/disappear on its
    // own between requests, so that was pure unnecessary traffic (see
    // comfy_client.py's readiness-cache docstring for the backend half).
    const checkComfy = () => {
      fetch("/v1/comfy/status", { headers: authHeaders() })
        .then(r => r.json())
        .then(d => {
          badge.textContent = d.alive ? "ComfyUI: Running" : "ComfyUI: Stopped";
          // Real theme tokens (matches the app's actual light/dark palette),
          // not the fabricated --success-color/--error-color fallbacks this
          // used before - those were never defined anywhere, so the badge
          // always rendered the same hardcoded hex regardless of theme.
          badge.style.color = d.alive ? "var(--green)" : "var(--red)";
          stopBtn.style.display = d.alive ? "" : "none";
          restartBtn.style.display = d.launched_by_localm ? "" : "none";
        }).catch(() => {
          badge.textContent = "ComfyUI: Unknown";
          badge.style.color = "";   // drop a stale Running/Stopped color from a prior check
          stopBtn.style.display = "none";
          restartBtn.style.display = "none";
        });
    };

    const comfyAction = (path, btn, busyLabel) => {
      const prev = btn.textContent;
      btn.disabled = true; btn.textContent = busyLabel;
      fetch(path, { method: "POST", headers: authHeaders() })
        .then(r => r.json())
        .then(d => { if (d && d.message) toast(d.message, !d.ok); })
        .catch(() => toast("ComfyUI control failed", true))
        .finally(() => { btn.disabled = false; btn.textContent = prev; checkComfy(); });
    };
    stopBtn.onclick = () => comfyAction("/v1/comfy/stop", stopBtn, "Stopping...");
    restartBtn.onclick = () => comfyAction("/v1/comfy/restart", restartBtn, "Restarting...");

    checkComfy();
  }
  
  const grid = el("div", "settings-fields");
  const controls = [];
  for (const f of (fields || [])) {
    const ctrl = buildSettingControl({
      key: f.key, widget: f.widget, label: f.label, help: f.help,
      default: f.value, options: f.options,
    });
    if (!ctrl) continue;
    ctrl.orig = f.value;
    if (!f.is_override) ctrl.node.classList.add("media-inherited");
    controls.push(ctrl);
    grid.appendChild(ctrl.node);
  }
  _mediaControls[name] = controls;
  sub.appendChild(grid);

  const actions = el("div", "actions");
  // R11/R13: client-side "Copy from <other>" - prefills this subsection's shared
  // fields from another subsection's CURRENT in-DOM values. It is a one-shot
  // SNAPSHOT copy (reads the source once on click, writes here), with NO live
  // binding between subsections, so an A->B then B->A sequence cannot loop. Never
  // shown on the subsection's own row (no self-copy).
  for (const other of MEDIA_PLUGIN_ORDER) {
    if (other === name || !_mediaSubs[other]) continue;
    const copy = el("button", "btn-secondary media-copy-from",
                    "Copy from " + _mediaSubs[other].label);
    copy.type = "button";
    copy.dataset.from = other;
    copy.onclick = () => copyMediaFields(other, name);
    actions.appendChild(copy);
  }
  const save = el("button", "btn-primary media-save", "Save " + label);
  save.onclick = () => saveMediaPlugin(name);
  actions.appendChild(save);
  sub.appendChild(actions);
}

/** R11: prefill the *to* subsection's shared fields from the *from* subsection's
 *  current in-DOM values. Prefill only (no server call); the user still presses
 *  Save. Only fields present in BOTH subsections are copied. */
export function copyMediaFields(from, to) {
  const src = _mediaControls[from] || [];
  const dst = _mediaControls[to] || [];
  const byKey = {};
  for (const c of src) byKey[c.field.key] = c;
  let n = 0;
  for (const c of dst) {
    const s = byKey[c.field.key];
    if (!s || !c.write) continue;     // only fields both subsections have
    c.write(s.read());
    c.node.classList.remove("media-inherited");
    n += 1;
  }
  const label = (_mediaSubs[from] || {}).label || from;
  toast(n ? `Copied ${n} field${n > 1 ? "s" : ""} from ${label} - review, then Save`
          : "No shared fields to copy", !n);
}

/** Save one media plugin's block: POST only the fields the user changed (so an
 *  untouched inherited field is not pinned), then re-render JUST this subsection
 *  (R12) so unsaved edits in the other subsections are preserved. */
export async function saveMediaPlugin(name) {
  const controls = _mediaControls[name] || [];
  const updates = {};
  for (const c of controls) {
    const cur = c.read();
    if (_mediaChanged(cur, c.orig)) updates[c.field.key] = cur === undefined ? "" : cur;
  }
  if (!Object.keys(updates).length) { toast("Nothing changed"); return; }
  const r = await fetch("/v1/media/config/" + encodeURIComponent(name), {
    method: "POST", headers: authHeaders(), body: JSON.stringify(updates),
  });
  const data = await r.json().catch(() => ({}));
  if (r.ok) {
    toast("Saved");
    // R12: re-render ONLY this subsection from the server's normalised fields. A
    // full refreshSettingsPage() here wiped unsaved edits in the other two.
    if (data && Array.isArray(data.fields) && _mediaSubs[name]) {
      _mediaSubs[name].fields = data.fields;
      renderMediaSubsection(name);
    } else {
      refreshSettingsPage();   // fallback if the response shape is unexpected
    }
  } else {
    toast(data.detail || "Save failed", true);
  }
}

