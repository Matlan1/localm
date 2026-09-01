// SPDX-License-Identifier: AGPL-3.0-or-later
/* localm GUI - Settings page. */
"use strict";

// --- ES module imports ---
import { pickDirectory, pickFile } from "../app/picker.js";
import { $, applyChatBackground, authHeaders, clearImageProxyCache, confirmDanger, el, fileToAvatarDataUri, fileToBackgroundDataUri, openModal, promptText, safeAvatarImageSrc, streamJob, toast } from "../app/helpers.js";
import { t } from "../app/i18n.js";
import { emptyState } from "../app/icons.js";
import { applyServerTtsConfig, browserVoiceOverride, caps, capsReady, clearBrowserVoiceOverride } from "../app/settings-perf.js";

/* ================================================================ */
/*  Settings page                                                    */
/* ================================================================ */

// The settings form is schema-driven: it fetches /v1/config/schema (the typed
// CORE_FIELDS metadata with each non-secret field's current value as its
// `default`) and renders the right control per field - a <select> for a fixed
// choice set, a checkbox for a bool, a number input with min/max, a masked
// input for a secret, a comma-edited LIST sent back as a JSON array. Fields the
// schema marks widget=hidden (plugins_enabled / plugins) are never rendered:
// they are plugin STATE managed by the Plugins page. Save PATCHes native types
// (numbers/bools/arrays).

// The schema field list from the last successful fetch, keyed by field for the
// save pass. Each entry mirrors a control: { field, read() }.
export let _settingsControls = [];
// Monotonic token so overlapping refreshes do not both render.
export let _settingsRenderToken = 0;
// The top-level GROUP the user is on (a group id below). Survives re-renders so
// saving a section keeps you on its group. Null = use the default (first) group.
export let _activeSettingsGroup = null;

// Top-level settings groups, in nav order. Every .settings-section is assigned to
// one of these via its data-group attribute (static cards carry it in index.html;
// schema + media sections get it set when rendered). A group nav link shows all of
// its sections stacked; conditionally-hidden cards (Updates/Issues via the `hidden`
// attribute, the owner-gated keys card via .sec-hidden) do not appear inside their
// group until they apply.
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
// app/icons.js; the `cat-*` class drives the hue via --nav-cat in style.css).
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
 *  section. The h3 keeps its .settings-section-head class and text; the icon and
 *  hue follow the section's top-level group (SETTINGS_NAV_META). */
export function settingsSectionHead(heading, groupId) {
  const head = el("div", "card-head");
  const nav = SETTINGS_NAV_META[groupId] || {};
  head.appendChild(iconEl(nav.icon || "settings", "ic cat-ic " + (nav.cat || "cat-slate")));
  const txt = el("div", "card-head-text");
  txt.appendChild(el("h3", "settings-section-head", heading));
  head.appendChild(txt);
  return head;
}

/** A smaller .card-head for a sub-panel nested one level down inside a settings
 *  section (a "Shared"/"Advanced"/"Experimental" box, or the managed-ComfyUI
 *  panel): a category-hued icon plus the sub-head's own .media-sub-head <h4>. */
function subCardHead(heading, iconName, cat) {
  const head = el("div", "card-head");
  head.appendChild(iconEl(iconName, "ic cat-ic " + cat));
  const txt = el("div", "card-head-text");
  txt.appendChild(el("h4", "media-sub-head", heading));
  head.appendChild(txt);
  return head;
}

// Which top-level group a core schema `group` string belongs to. The Media
// section is its own top-level group (built directly with dataset.group =
// "media", not looked up here). This map is TOTAL: every `group` the schema can
// produce is listed, and an unmapped one warns before falling back to "system"
// (see settingsTopGroupFor).
export const CORE_GROUP_TO_TOP = {
  Engine: "model", Timeouts: "model", Chat: "model", Models: "model",
  Embeddings: "model",
  Server: "server", Network: "server",
  Security: "security",
  Plugins: "plugins", Coder: "plugins", Knowledge: "plugins", Voice: "plugins",
  Privacy: "privacy", Memory: "privacy", Diagnostics: "privacy",
  Updates: "system", "Bug reports": "system", General: "system",
  Media: "media", Desktop: "system",
};

/** The nav group for a core section, complaining loudly about an unmapped group
 *  instead of quietly parking it in System. A new schema group with no entry above
 *  is a bug in this file, and it should be visible the first time it renders. */
export function settingsTopGroupFor(group) {
  const top = CORE_GROUP_TO_TOP[group];
  if (top) return top;
  console.warn(`[settings] group "${group}" has no CORE_GROUP_TO_TOP entry; `
    + `showing it under System. Add it to CORE_GROUP_TO_TOP.`);
  return "system";
}

// Friendlier per-section headings once grouped (the schema `group` string is left
// unchanged, so section ids + validation are untouched - this is display only). An
// empty string means render NO heading: used for the lone require_auth toggle and
// the Privacy persistence block, which are the primary content of their group and
// would only repeat the group name.
// Every group gets a REAL heading. The two empty strings that used to live here
// (Security, Privacy) meant settings.js:1133 appended no .card-head at all, so the
// canon surface violated its own design rule 4 ("no card shows a bare grey title")
// in exactly the two sections this overhaul splits. An empty heading is no longer
// representable: each group below is named for what is under it.
//
// Note Server's heading is no longer "Network". That collided with the group
// literally CALLED Network (the net_* egress policy), so the page had two different
// panels a user could reasonably call Network and the search box indexed both.
export const CORE_SECTION_HEADING = {
  Engine: "Runtime & GPU",
  Timeouts: "Timeouts & limits",
  Chat: "Generation defaults",
  Models: "Library",
  Embeddings: "Embeddings",
  Server: "Server",
  Network: "Outbound access",
  Security: "Access",
  Plugins: "Plugin management",
  Privacy: "Session persistence",
  Memory: "Memory",
  Diagnostics: "Diagnostics",
  Updates: "Updates",
  "Bug reports": "Bug reports",
  General: "Appearance",
  Coder: "Coder",
  Knowledge: "Knowledge (RAG)",
  Voice: "Voice",
  Desktop: "Desktop app",
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
  // True when the current value is STILL the factory default (never saved, or
  // saved back to exactly what it already was) - see settings_schema.py's
  // schema_json() for why this needs its OWN field instead of comparing
  // `value` to some remembered constant: `default` is the CURRENT value, which
  // after a save IS the user's override, so only a separately-sourced
  // `shipped_default` (always DEFAULT_CONFIG) can tell the two apart
  // (NEW-DEFAULT-VALUE-PLACEHOLDER). Media per-plugin fields never set
  // shipped_default (renderMediaSubsection builds a plain object literal
  // without it), so this is always false for them - their own
  // `.media-inherited` mechanism is untouched by this.
  const isShippedDefault = field.shipped_default !== undefined
    && value === field.shipped_default;

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
      // A dropdown always shows SOME selected option - there is no blank state
      // to hide the default behind the way a number/text box has - so the
      // GREY CUE is visual-only (reusing/extending .auto-detected, see
      // style.css). The SAVE PAYLOAD still needs the same omit-when-unchanged
      // contract NUMBER/TEXT get, though, or the grey styling would be lying:
      // without this, selecting nothing and merely saving some OTHER field in
      // the same section would silently pin this untouched default as an
      // explicit override (still displayed grey, but no longer actually
      // inheriting - a future DEFAULT_CONFIG change would then leave this
      // install stuck on the old value while its own UI keeps claiming
      // "default"). Compare against the option shown at RENDER time, not a
      // live "still equals shipped_default" check - correct either way here
      // since isShippedDefault was already true, but this is the same shape
      // as the number/text branches: a value genuinely unchanged from what
      // was shown is a no-op, independent of whether the user's mouse ever
      // touched the control.
      if (isShippedDefault) input.classList.add("auto-detected");
      const shownAsDefault = isShippedDefault ? input.value : null;
      read = () => {
        const v = input.value;
        if (shownAsDefault !== null && v === shownAsDefault) return undefined;
        return v === "" ? null : v;
      };
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
      // Still the factory default: leave the box BLANK with the value shown as
      // a native placeholder (exactly the chat-drawer / image-gen pattern),
      // rather than a solid value indistinguishable from something the user
      // typed. read() below already omits a blank box from the save payload -
      // that contract predates this fix and is unchanged.
      if (value != null && isShippedDefault) input.placeholder = "default (" + value + ")";
      else if (value != null) input.value = value;
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
      } else if (isShippedDefault && stored) {
        // Still the shipped default (e.g. mdns_name="localm",
        // embedding_model="bge-small-en-v1.5") - same treatment as the
        // auto-detect branch above: show it as a placeholder, not a solid
        // value, and treat "still blank or still exactly the default text" as
        // no change. Safe by construction: this branch only runs when the
        // CURRENT value already equals the shipped default, so there is
        // nothing customized to lose - a genuinely customized value (the
        // common "clear it back to empty" case) always takes the plain `else`
        // branch below instead, whose null-on-blank explicit-clear semantics
        // are completely untouched by this branch's existence.
        input.placeholder = "default (" + stored + ")";
        read = () => {
          const v = input.value.trim();
          if (v === "" || v === stored) return undefined;
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
  // TAGGING ONLY - never guess from the label or the key spelling (NEW-M-BROWSE
  // said "tagging beats guessing"; the guessing was left in as a fallback and
  // over-matched into six fields that are not paths at all).
  //
  // The old heuristic also matched `lbl.includes("file"/"folder"/"dir"/"cmd")` and
  // the key suffixes `_path`/`_file`/`_dir`. Measured 2026-08-13, that flagged
  // import_max_depth (number, "Folder import depth"), autoprune_missing_models
  // (toggle, "...missing files"), coder_grep_max_per_file (number - matched the
  // KEY suffix as well as the label), coder_grep_max_file_bytes (number),
  // rag_indexing_mode (select, "Indexing folder rule") and
  // rag_classify_unknown_files (toggle). Each got a "Browse..." button it has no
  // use for, and - because the gate below hides a path field until capabilities
  // load - each vanished entirely from a cold render, taking the whole Knowledge
  // section with it.
  //
  // A widget/accepts_* flag is the field's own declaration and cannot drift with
  // wording, so the set is exactly right by construction: only path, folder and
  // pathlist widgets browse. Verified this loses no control that has a picker
  // today (binary_dir, rag_allowed_roots, rag_denied_roots, comfy_workdir,
  // comfy_output_dir all declare a path widget).
  const isPath = field.widget === "path" || !!field.accepts_path;
  const isDir = field.widget === "folder" || !!field.accepts_dir;
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
  // The server port: show the LIVE bound port when it is known and differs
  // from the persisted default above (an explicit -p override, or an
  // auto-bump onto a different free port never gets written back to disk).
  if (field.key === "port" && field.live_port != null && field.live_port !== value) {
    const saved = value == null ? "The saved value" : `The saved value (${value})`;
    wrap.appendChild(el("div", "sub",
      `Currently running on port ${field.live_port}. ${saved} takes effect on the next restart.`));
  }
  if (field.link) {
    const link = el("a", "settings-field-link", field.link.label);
    link.href = field.link.url;
    link.target = "_blank";
    link.rel = "noopener";
    wrap.appendChild(link);
  }
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
  if (info.bind_fallback) {
    // The server REFUSED a configured network bind at startup (no strong API
    // key / TLS unavailable) and stayed on loopback. That reason outranks the
    // generic hints below: without it, setting Bind address and restarting
    // would look like it silently did nothing.
    hint = info.bind_fallback;
  } else if (!urls.length) {
    hint = info.network_bind
      ? "Could not detect this machine's network address - open its LAN or Tailscale address (with this port) on the phone."
      : "Reachable only on this computer right now. To use it from a phone: set an API key, set Server > Bind address to 0.0.0.0, then Restart server (or run: localm gui -H 0.0.0.0). See docs/phone.md.";
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
// are shown but OWNER-ONLY: the /v1/keys API refuses them for a non-owner key
// (create_key's allow_privileged gate), so a keys:admin device cannot hand
// itself a broader grant through this form.
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
  ["config:write", "Change settings (owner-only, dangerous)"],
  ["plugins:admin", "Enable, disable, install, or uninstall plugins (owner-only, dangerous)"],
  ["keys:admin", "Create, scope, and revoke API keys (owner-only, dangerous)"],
  ["admin", "Full admin - owner-equivalent (dangerous, owner-only)"],
];

// Mirrors localm.scopes.PRIVILEGED_SCOPES exactly, and is bound to it by
// tests/test_gui_key_scope_options.py, which reads this file: a scope here
// that drifted from the Python set would either dim a safe scope for no
// reason or, worse, leave a real privileged one un-dimmed - the /v1/keys API
// would still refuse the mint, but only after the confusing failed-submit
// round trip this list exists to avoid.
export const PRIVILEGED_KEY_SCOPES = new Set([
  "admin", "coder:full", "keys:admin", "plugins:admin", "config:write",
]);

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
// keys:admin device cannot even try to mint a coder:full / admin / keys:admin /
// plugins:admin / config:write key (the API would 403 anyway; this avoids the
// confusing failed-submit round-trip).
export function applyOwnerGate(isOwner) {
  for (const cb of document.querySelectorAll("#key-scopes .key-scope-cb")) {
    const ownerOnly = PRIVILEGED_KEY_SCOPES.has(cb.value);
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

export async function saveCurrentAsPreset(presets) {
  const scopes = [...document.querySelectorAll("#key-scopes .key-scope-cb")]
    .filter((c) => c.checked).map((c) => c.value);
  if (!scopes.length) { toast("Check the scopes for the preset first"); return; }
  const name = (await promptText("Preset name:") || "").trim();
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
      const danger = PRIVILEGED_KEY_SCOPES.has(scope);
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

// Settings -> Owner key: roll or set the ONE key that grants full access. The GUI
// form of `localm key generate` and `localm key set <key>`.
//
// Hidden from a non-owner the same way the keys card is, but the gate that MATTERS is
// the server's: POST /api/auth/key/rotate requires the owner scope, because setting a
// key the caller CHOSE is a promotion to owner. Hiding this card is a courtesy so a
// keys:admin device is not shown a control it cannot use; it is never the control.
export async function refreshOwnerKeyPanel() {
  const card = $("owner-key-card"), box = $("owner-key-secret");
  if (!card || !box) return;

  // A settings SECTION, so show/hide via the class the section nav reads, exactly as
  // refreshKeysPanel does - an inline display style fights the section show/hide.
  const setHidden = (hidden) => {
    card.classList.toggle("sec-hidden", hidden);
    if (typeof buildSettingsNav === "function") buildSettingsNav();
  };
  // Same probe the keys card uses. /v1/keys reports is_owner, and a non-ok answer
  // means this caller is not even a key minter.
  try {
    const r = await fetch("/v1/keys", { headers: authHeaders() });
    if (!r.ok) { setHidden(true); return; }
    const data = await r.json();
    if (!data.is_owner) { setHidden(true); return; }
  } catch (e) { setHidden(true); return; }
  setHidden(false);

  const rotate = async (body, verb) => {
    let r;
    try {
      r = await fetch("/api/auth/key/rotate", {
        method: "POST",
        headers: { ...authHeaders(), "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
    } catch (e) { toast(`${verb} failed`, true); return; }
    if (!r.ok) {
      const e = await r.json().catch(() => ({}));
      toast(e.detail || `${verb} failed`, true);
      return;
    }
    const out = await r.json();
    box.replaceChildren();
    // READ `rotated`; never infer success from the 200. The server answers 200 with
    // rotated:false when the key reached disk but is NOT the credential it accepts
    // (LOCALM_API_KEY outranks the stored key). A green "key updated" there is exactly
    // the lie the route's shape exists to prevent: someone rotating a leaked key would
    // be told the leaked one was dead while it still authenticates.
    if (out.rotated) {
      toast("Owner key updated");
      box.appendChild(el("div", "sub",
        "New owner key - copy it now, it is shown only once:"));
    } else {
      toast("Key saved but NOT in effect", true);
      for (const w of out.warnings || []) box.appendChild(el("div", "key-warn", w));
      box.appendChild(el("div", "sub",
        "The key that was written (not currently in effect):"));
    }
    const secret = document.createElement("input");
    secret.type = "text"; secret.readOnly = true; secret.value = out.key;
    secret.className = "key-secret-value";
    box.appendChild(secret);
    const copy = el("button", "btn-secondary", "Copy");
    copy.onclick = () => {
      secret.select();
      if (navigator.clipboard) navigator.clipboard.writeText(out.key);
      toast("Copied");
    };
    box.appendChild(copy);
    box.style.display = "";
    $("owner-key-value").value = "";
  };

  // Confirm both paths: this cuts off every other device holding the old key, which is
  // not obvious from a button labelled "Generate".
  const WARN = "Every other device holding the current key loses access until you give "
    + "it the new one. This browser stays signed in.";
  $("owner-key-roll").onclick = () => {
    confirmDanger("Generate a new owner key?", WARN, "Generate",
                  () => rotate({}, "Generate"));
  };
  $("owner-key-set").onclick = () => {
    const chosen = ($("owner-key-value").value || "").trim();
    // Guard here as well as server-side: an empty value GENERATES a random key on the
    // server, which is not what a user pressing "Set this key" asked for.
    if (!chosen) { toast("Paste a key, or use Generate new key", true); return; }
    confirmDanger("Set this as the owner key?", WARN, "Set key",
                  () => rotate({ key: chosen }, "Set"));
  };
}

// Friendly section label per plugin owner (falls back to the capitalized scope).
export const PLUGIN_SECTION_LABEL = {
  image: "Image", web: "Web access", voice: "Voice", coder: "Coder",
  music: "Music", video: "Video", rag: "Knowledge",
  mcp: "MCP", chat: "Chat",
};

/** Which settings section a field belongs to: its `group`, always.
 *
 *  This used to route by `owner` whenever owner != "core", so a plugin's whole
 *  config landed on one tab and the schema's `group` tag stopped predicting
 *  anything. Measured 2026-08-13: 49 of the 84 visible fields rendered in a nav
 *  group their group label did not predict. Worse, the two are not equivalent -
 *  `owner` says who OWNS the code, `group` says what the setting is ABOUT, and a
 *  user searches by the second. The Privacy group was the casualty: its nine
 *  fields have four different owners, so seven of them left the Privacy panel
 *  entirely and a user could not find "does memory still recall in privacy mode"
 *  under Privacy & data.
 *
 *  `owner` keeps its real jobs - which plugin's config a key migrates to, and the
 *  scope gate. Both are server-side. Placement is a display concern and is now
 *  decided by the display tag alone.
 *
 *  A group may legitimately hold fields from several owners (Privacy holds core,
 *  chat, coder and memory keys). That is safe for saving: every CORE_FIELDS key
 *  lives in the flat config - DEFAULT_CONFIG and CORE_FIELDS are the same 99 keys -
 *  so one PATCH /v1/config saves a mixed-owner section exactly like any other. */
export function settingsSectionOf(field) {
  return { id: "core-" + field.group, label: field.group, plugin: false };
}

/* ================================================================ */
/*  Settings search                                                  */
/* ================================================================ */

// One box over the WHOLE page. The form is ~80 schema fields plus the hand-built
// rows, behind a 7-group nav, so finding a setting used to mean knowing (or
// guessing) its group. With a query, matches from every group show at once, each
// still inside its own section so you can see where a setting lives; clearing it
// restores the ordinary one-group-at-a-time view.
export let _settingsFilterQuery = "";

/** Is this element invisible to the user right now? Gate-hidden content must not
 *  be findable: the Main GPU / Split rows on a single-GPU box, the keys card for
 *  a non-owner. Surfacing a control that is not offered would send the user
 *  hunting through a card that then shows nothing. */
function filterHiddenFromUser(node) {
  return !!(node.hidden || (node.classList && node.classList.contains("sec-hidden"))
    || (node.style && node.style.display === "none"));
}

/** The text of *root* for matching, skipping anything gate-hidden and, when
 *  *skipFields* is set, the schema controls (each is indexed on its own). What
 *  remains is the section's OWN text: its heading plus every hand-built row that
 *  has no schema field behind it (Main GPU, Split across GPUs, the logo picker,
 *  the key presets, the server buttons). Indexing those by EXCLUSION rather than
 *  from a hand-listed set of custom rows is deliberate - a list goes stale, which
 *  is exactly how three media settings ended up rendered nowhere (2026-07-22
 *  settings-exposure audit); this way a row added later is searchable already. */
function filterTextOf(root, skipFields, parts = []) {
  for (const node of root.childNodes) {
    if (node.nodeType === 3) { parts.push(node.nodeValue); continue; }   // text
    if (node.nodeType !== 1) continue;                                    // element
    if (filterHiddenFromUser(node)) continue;
    if (skipFields && node.dataset && node.dataset.fieldKey) continue;
    filterTextOf(node, skipFields, parts);
  }
  return parts;
}

/** Every term must appear (AND), case-insensitively, so "gpu split" narrows. */
function filterMatches(text, terms) {
  const hay = text.toLowerCase();
  return terms.every((t) => hay.includes(t));
}

/** Apply the search box's current query to the whole page.
 *
 *  A section matches either through its own text (heading, hand-built rows), in
 *  which case it is shown whole, or through individual schema fields, in which
 *  case only those fields stay visible inside it. Sections that match are shown
 *  regardless of group; gate-hidden ones never match. An empty query restores the
 *  grouped view. */
export function applySettingsFilter() {
  const content = $("settings-content");
  if (!content) return;
  const note = $("settings-filter-empty");
  const terms = _settingsFilterQuery.toLowerCase().split(/\s+/).filter(Boolean);

  if (!terms.length) {
    for (const n of content.querySelectorAll(".filter-hidden")) n.classList.remove("filter-hidden");
    if (note) note.hidden = true;
    buildSettingsNav();          // back to one group at a time
    return;
  }

  let shown = 0;
  for (const sec of content.querySelectorAll(".settings-section")) {
    const gated = filterHiddenFromUser(sec);
    const secText = filterTextOf(sec, true).join(" ") + " " + (sec.dataset.secLabel || "");
    const secHit = !gated && filterMatches(secText, terms);
    let fieldHits = 0;
    for (const field of sec.querySelectorAll("[data-field-key]")) {
      const hit = !gated && (secHit || filterMatches(
        filterTextOf(field, false).join(" ") + " " + field.dataset.fieldKey, terms));
      field.classList.toggle("filter-hidden", !hit);
      if (hit) fieldHits += 1;
    }
    const show = !gated && (secHit || fieldHits > 0);
    sec.classList.toggle("active", show);
    if (show) shown += 1;
  }

  const nav = $("settings-nav");
  if (nav) {
    // Results span groups, so no single group is the selected one while filtering.
    for (const link of nav.querySelectorAll(".settings-nav-link")) link.classList.remove("active");
  }
  if (note) {
    note.textContent = shown ? "" : `No settings match "${_settingsFilterQuery.trim()}".`;
    note.hidden = shown > 0;
  }
}

/** Empty the search box and restore the grouped view (no-op when not filtering,
 *  so an ordinary group click does not trigger a pointless nav rebuild). */
export function clearSettingsFilter() {
  const box = $("settings-filter");
  if (box) box.value = "";
  if (!_settingsFilterQuery) return;
  _settingsFilterQuery = "";
  applySettingsFilter();
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
  // While a search is live the page shows matches from every group at once. An
  // async rebuild (the owner-only keys panel resolving, a post-save re-render)
  // must not yank the user back to a single group mid-search.
  if (_settingsFilterQuery) { applySettingsFilter(); return; }
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
  clearSettingsFilter();     // a deep link lands on a group, so leave the search
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
    link.appendChild(document.createTextNode(t("settings.group." + g.id)));
    link.onclick = () => {
      clearSettingsFilter();     // picking a group means leaving the search view
      _activeSettingsGroup = g.id;
      showSettingsGroup(g.id);
    };
    nav.appendChild(link);
  }
  const target = settingsTargetGroup(content);
  if (target) showSettingsGroup(target);
}

// The nav labels are built from the catalog, so they are redrawn when the
// interface language changes.
document.addEventListener("localm:language", () => buildSettingsNav());

/** Which group the settings page should be SHOWING: the user's explicit choice if it
 *  still has a visible section, else the first group that does (Model). Extracted from
 *  buildSettingsNav so refreshSettingsPage can apply it to freshly built sections
 *  IMMEDIATELY, before the awaited plugin builders run - see the why-comment there. */
function settingsTargetGroup(content) {
  const present = SETTINGS_GROUPS.filter((g) => groupHasVisibleSection(content, g.id));
  if (_activeSettingsGroup && present.some((g) => g.id === _activeSettingsGroup)) {
    return _activeSettingsGroup;                   // the user's chosen group, still present
  }
  return present.length ? present[0].id : null;    // default: the first group (Model)
}

/** In-page confirm before a PATCH /v1/config that would switch embedding_model
 *  and invalidate existing collections' semantic search - mirrors
 *  knowledge.js's kbConfirmEmbeddingSwitch for the RAG picker's identical gate
 *  (NEW-RAG-DIM-NO-REEMBED: PATCH /v1/config is the second writer of this
 *  key and needed the same pre-switch warning). */
function confirmEmbeddingModelSwitch(model, report) {
  return new Promise((resolve) => {
    openModal(`Switch embedding model to '${model}'?`, (body) => {
      body.appendChild(el("p", "", report.note));
      const list = el("ul", "");
      for (const c of report.collections || []) {
        list.appendChild(el("li", "",
          c.name + (c.built_with ? ` (built with ${c.built_with})` : "")
          + (c.n_chunks != null ? ` - ${c.n_chunks} chunks` : "")));
      }
      body.appendChild(list);
      body.appendChild(el("p", "sub",
        "Re-embed each one afterward on the Knowledge page to restore semantic "
        + "search if it does turn out to need it."));
      const row = el("div", "actions");
      const cancel = el("button", "btn-secondary", "Cancel");
      cancel.onclick = () => { $("modal").style.display = "none"; resolve(false); };
      const ok = el("button", "btn-primary", "Switch anyway");
      ok.onclick = () => { $("modal").style.display = "none"; resolve(true); };
      row.append(cancel, ok);
      body.appendChild(row);
    });
  });
}

/** Save just one section: PATCH only the keys whose controls live in it.
 *
 *  NEW-RAG-DIM-NO-REEMBED: when the section's updates include an
 *  embedding_model change that would invalidate existing RAG collections,
 *  PATCH /v1/config answers with a needs_confirm dry-run report instead of
 *  writing (no config write happened yet). Show it, and only if the user
 *  proceeds, re-PATCH the exact same body plus confirm:true - so the whole
 *  section's edits (not just embedding_model) land together, same as the
 *  single-step case, once confirmed. */
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
  let r = await fetch("/v1/config", {
    method: "PATCH", headers: authHeaders(),
    body: JSON.stringify(updates),
  });
  let data = await r.json().catch(() => ({}));
  if (r.ok && data.needs_confirm) {
    const proceed = await confirmEmbeddingModelSwitch(data.model, data);
    if (!proceed) { toast("Cancelled"); return; }
    r = await fetch("/v1/config", {
      method: "PATCH", headers: authHeaders(),
      body: JSON.stringify({ ...updates, confirm: true }),
    });
    data = await r.json().catch(() => ({}));
  }
  if (r.ok) {
    toast("Saved - engine values apply on the next model load");
    // A save may have moved "Show remote images in replies" (off / ask / on).
    // The route re-decides immediately, but an already-fetched image is held in
    // a page-lifetime blob cache and the reader's per-origin answers are held
    // beside it, so without this the change appears to do nothing for every
    // image already on screen. Both are cleared unconditionally: it costs one
    // refetch and at most one re-ask, and reading which key moved would put a
    // security decision behind a diff.
    clearImageProxyCache();
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

  // The live bound port: GET /v1/config/schema only ever carries the PERSISTED
  // "port" default, never what the server actually bound (an explicit -p
  // override, or an auto-bump onto a different free port, never gets written
  // back to disk). Best-effort - a failed fetch just leaves live_port unset,
  // and the field renders its persisted value only, same as before this.
  const portField = fields.find((f) => f.key === "port");
  if (portField) {
    try {
      const cr = await fetch("/v1/config", { headers: authHeaders() });
      if (cr.ok) portField.live_port = (await cr.json()).instance_port ?? null;
    } catch (e) { /* live_port stays unset */ }
  }
  if (myToken !== _settingsRenderToken) return;

  // Host-path fields (folder/path/pathlist) are hidden from a caller without host
  // filesystem access, and that decision reads caps.fsAccess - which starts at the
  // SAFE default until /api/capabilities lands, and is only populated by a
  // fire-and-forget call in init.js. Rendering before it resolves cannot tell "may
  // not" from "do not know yet", so a cold load silently dropped every path field
  // and the whole Knowledge section with it. Wait for the ANSWER, then decide.
  await capsReady;
  if (myToken !== _settingsRenderToken) return;  // awaiting is a suspension point

  // One section per `group`, in schema order. There is no longer a separate
  // per-plugin section: a plugin-owned field renders under the category it
  // declares, so Coder/Knowledge/Voice still get their own panels (their group IS
  // that name) while Privacy finally keeps its memory and per-surface toggles.
  //
  // The old crossFiledInto map, and the "Related settings also live on plugin
  // tabs: ..." line it produced, are GONE. That line existed only to paper over
  // owner-first routing: it told a user browsing Privacy that the rest of Privacy
  // was somewhere else. With the fields back where their group says they belong
  // there is nothing to point at, and keeping it would have every section
  // cheerfully cross-referencing itself.
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

  // Schema order, which is the display order the schema already documents.
  const ordered = [...sections.values()];

  form.replaceChildren();
  // The two update-behavior toggles (update_allow_prerelease,
  // update_ignore_net_policy, both admin_only) render INTO the static
  // "Updates" card (index.html's #update-toggles-block) rather than getting
  // their own section below - that second panel headed "Updates" is exactly
  // the duplication S1's card merge removes. Reset to hidden first so a caller
  // without those fields (or a schema fetch that dropped them) leaves it
  // empty rather than showing a stale render from an earlier refresh.
  const updateToggleBlock = $("update-toggles-block");
  if (updateToggleBlock) updateToggleBlock.hidden = true;
  for (const sec of ordered) {
    if (sec.id === "core-Updates" && updateToggleBlock) {
      const grid = $("settings-sec-core-Updates");
      grid.replaceChildren(...sec.ctrls.map((c) => c.node));
      const save = $("update-toggles-save");
      if (save) save.onclick = () => saveSettingsSection(sec.id);
      updateToggleBlock.hidden = false;
      continue;
    }
    // Every group has a real heading now (CORE_SECTION_HEADING is total, and an
    // empty string is no longer representable), so a section can never render as
    // a bare grey block again - design rule 4.
    const heading = (sec.label in CORE_SECTION_HEADING)
      ? CORE_SECTION_HEADING[sec.label] : sec.label;
    const topGroup = settingsTopGroupFor(sec.label);
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

  // SHOW what was just built, NOW, before the three awaited builders below.
  //
  // Every .settings-section is `display: none` until it carries `.active`
  // (style.css), and the only thing that adds `.active` is showSettingsGroup -
  // which used to be reached ONLY from buildSettingsNav(), after all three
  // awaited fetches. So a re-render inserted the whole form invisible and left
  // it that way across three network round-trips: the page collapsed to less
  // than one viewport, the browser clamped scrollTop to 0 because there was no
  // longer anything to scroll, and the content came back 20-50 ms later with the
  // position already gone. That is the settings half of the scroll jump - saving
  // any section threw you back to the top. MEASURED before this line existed:
  // scrollHeight 3157 -> 889 (== clientHeight) -> 3157, scrollTop 966 -> 0, with
  // zero scroll calls from our own code and no reload.
  //
  // buildSettingsNav() still runs at the end and re-applies the same group; this
  // is deliberately idempotent rather than a replacement for it, because the
  // sections the builders below append need the nav rebuilt once they exist.
  const activeNow = settingsTargetGroup($("settings-content"));
  if (activeNow) showSettingsGroup(activeNow);

  // Per-plugin Media (ComfyUI) config: one "Media" section (its own top-level
  // nav group) with a compact managed-ComfyUI panel on top and one subsection
  // per plugin (image/music/video) below. Appended after the core schema
  // sections; fields is passed through so the two managed-ComfyUI schema
  // fields (group="Media", skipped from the flat loop above) can be rendered
  // here instead.
  await buildMediaSection(form, fields);
  if (myToken !== _settingsRenderToken) return;  // a newer refresh superseded us

  // user_avatar / model_avatar_default / model_avatar_overrides: HIDDEN
  // schema fields (skipped from the flat loop above, same as Media's own
  // fields), rendered here with their own picker UI instead.
  await buildAvatarsSection(form, fields);
  if (myToken !== _settingsRenderToken) return;  // a newer refresh superseded us

  // The tts plugin's own settings block (its own section in the Plugins group).
  // Not part of the core schema: those keys live under config["plugins"]["tts"]
  // and are edited through /v1/tts/config, like the media blocks above.
  await buildTtsSection(form);
  if (myToken !== _settingsRenderToken) return;  // a newer refresh superseded us

  // Any OTHER active plugin's own contributed fields (host.add_settings()),
  // one section per plugin - the generic counterpart to the tts block above
  // for a plugin the core has no bespoke schema for.
  await buildPluginSettingsSections(form);
  if (myToken !== _settingsRenderToken) return;  // a newer refresh superseded us

  // Build the nav now that the schema sections exist, so the first config
  // section (not a static card) is the default tab. The owner-gated panels then
  // refresh: each may rebuild the nav, but they preserve the active section.
  buildSettingsNav();
  syncRagIndexingModeHint();
  syncEmbeddingWarmupButton();
  refreshPairingQR();
  refreshCompanion();
  refreshKeysPanel();
  refreshOwnerKeyPanel();
  // Wires the wallpaper picker (a static card, never rebuilt above) only from
  // this auth-gated point, never at module load. See keygate.test.mjs.
  setupChatBackgroundPicker();
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

/** ADR-0004 Unit B: a "Warm up now" action next to the embedding_model field, so
 *  a user can pay the (possibly minute-long) first-load cost explicitly instead
 *  of it happening silently on their first real memory/RAG/embeddings call -
 *  measured up to two 300s timeout windows on a cold server. Coarse STAGE text
 *  (not a spinner), streamed via the same job/SSE mechanism model pull already
 *  uses. Runs every refreshSettingsPage() rebuild, appending to the freshly
 *  built field wrapper - no de-dup needed, the whole form is rebuilt each time.
 *  No-op when the field is absent (a non-owner never receives it - admin_only). */
export function syncEmbeddingWarmupButton() {
  const wrap = document.querySelector('[data-field-key="embedding_model"]');
  if (!wrap) return;
  const row = el("div", "embedding-warmup-row");
  // btn-SECONDARY, not "btn": there is no `.btn` rule in style.css, so that
  // class rendered a fully native light-grey/black/square button inside the
  // dark app (gui-design.md rules 3 and 8). Only visible in a real browser -
  // jsdom parses but never paints, so every DOM test passed on it.
  const btn = el("button", "btn-secondary", "Warm up now");
  btn.type = "button";
  const status = el("span", "embedding-warmup-status sub");
  row.appendChild(btn);
  row.appendChild(status);
  wrap.appendChild(row);

  btn.onclick = async () => {
    btn.disabled = true;
    status.textContent = "Starting...";
    try {
      const r = await fetch("/api/embedding/warmup",
                            { method: "POST", headers: authHeaders() });
      const data = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(data.detail || r.statusText);
      const end = await streamJob(data.job_id, (line) => { status.textContent = line; });
      if (end.status !== "done") {
        toast("Warm-up did not finish cleanly - see the status line above", true);
      }
    } catch (e) {
      status.textContent = "Warm-up failed: " + e.message;
      toast("Warm-up failed: " + e.message, true);
    } finally {
      btn.disabled = false;
    }
  };
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

  // SHARED media settings: every remaining visible group="Media" schema field
  // that is neither explicitly placed here (comfy_target above,
  // comfy_gpu_placement below) nor per-plugin-mapped (media_per_plugin, from
  // MEDIA_PLUGIN_FIELDS - those render inside the per-plugin boxes instead).
  // Rendered by EXCLUSION, not an allowlist, on purpose: a name allowlist is
  // exactly how comfy_launch_timeout / comfy_disable_auto_launch /
  // comfy_func_shim ended up rendered NOWHERE in the GUI (schema-visible,
  // CLI-only in practice; 2026-07-22 settings-exposure audit) - this way a
  // future Media field is visible by default and cannot silently vanish.
  // Saved through the Media section's generic PATCH /v1/config path.
  const sharedFields = (fields || []).filter(f =>
    f.group === "Media" && !f.media_per_plugin
    && f.key !== "comfy_target" && f.key !== "comfy_gpu_placement");
  const sharedCtrls = sharedFields.map(f => buildSettingControl(f)).filter(Boolean);
  if (sharedCtrls.length) {
    const box = el("div", "media-comfy-box");
    box.appendChild(subCardHead("Shared", "sliders", "cat-teal"));
    box.appendChild(el("div", "sub",
      "Applies to every media plugin (whichever ComfyUI is used)."));
    for (const ctrl of sharedCtrls) {
      box.appendChild(ctrl.node);
      // Same de-dup-then-register dance as the boxes below: a re-render must
      // not pile up stale detached entries for these keys in the shared list.
      _settingsControls = _settingsControls.filter(c => c.field.key !== ctrl.field.key);
      _settingsControls.push(ctrl);
    }
    const actions = el("div", "actions");
    const save = el("button", "btn-primary", "Save");
    save.type = "button";
    save.onclick = () => saveSettingsSection("media");
    actions.appendChild(save);
    box.appendChild(actions);
    panel.appendChild(box);
  }

  // EXPERIMENTAL per-component GPU placement toggle (comfy_gpu_placement). Unlike
  // the managed box's comfy_target - which only matters once a managed instance is
  // installed - this applies to a user's OWN ComfyUI too, so it renders
  // UNCONDITIONALLY (not gated on the managed-status fetch). Saved through the
  // Media section's generic PATCH /v1/config path, exactly like comfy_target; the
  // control shows its own help (why it is experimental and default-off).
  const placementField = (fields || []).find(f => f.key === "comfy_gpu_placement");
  if (placementField) {
    const box = el("div", "media-comfy-box");
    box.appendChild(subCardHead("Experimental", "warning", "cat-teal"));
    const ctrl = buildSettingControl(placementField);
    if (ctrl) {
      box.appendChild(ctrl.node);
      // Same de-dup-then-register dance as renderManagedComfyPanel: a re-render
      // must not pile up stale detached entries for this key in the shared list.
      _settingsControls = _settingsControls.filter(c => c.field.key !== placementField.key);
      _settingsControls.push(ctrl);
      const actions = el("div", "actions");
      const save = el("button", "btn-primary", "Save");
      save.type = "button";
      save.onclick = () => saveSettingsSection("media");
      actions.appendChild(save);
      box.appendChild(actions);
    }
    panel.appendChild(box);
  }

  // R11/R12: register every subsection's node + label first (empty), so that when
  // we render each, its "Copy from <other>" buttons can see the other subsections,
  // and a later single-subsection re-render (R12) can find its node.
  _mediaSubs = {};
  _mediaControls = {};
  const grid = el("div", "media-settings-grid");
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

/* ---------------- Avatars (user_avatar / model_avatar_default /
   model_avatar_overrides - HIDDEN schema fields, bespoke picker UI) -------- */

/** A small avatar-picker widget: preview + emoji/glyph text input + upload +
 * clear. The value is always either "" or a data:image/... URI (never a URL
 * - fileToAvatarDataUri only ever produces one from a local file). Reused for
 * user_avatar, model_avatar_default, and each per-model override row. */
function buildAvatarPicker(initial) {
  let value = initial || "";
  const wrap = el("div", "avatar-picker");
  const preview = el("div", "avatar-picker-preview");
  const glyphInput = document.createElement("input");
  glyphInput.type = "text";
  glyphInput.placeholder = "emoji or short text";
  glyphInput.maxLength = 16;
  glyphInput.className = "avatar-picker-glyph";
  const fileInput = document.createElement("input");
  fileInput.type = "file";
  fileInput.accept = "image/png,image/jpeg,image/gif,image/webp";
  fileInput.hidden = true;
  const uploadBtn = el("button", "btn-secondary", "Upload image");
  uploadBtn.type = "button";
  const clearBtn = el("button", "btn-secondary", "Clear");
  clearBtn.type = "button";

  function renderPreview() {
    preview.replaceChildren();
    const safeSrc = safeAvatarImageSrc(value);
    if (safeSrc) {
      const img = document.createElement("img");
      img.src = safeSrc;
      preview.appendChild(img);
      glyphInput.value = "";
    } else {
      preview.textContent = value;
      glyphInput.value = value;
    }
  }

  glyphInput.oninput = () => { value = glyphInput.value; renderPreview(); };
  uploadBtn.onclick = () => fileInput.click();
  fileInput.onchange = async () => {
    const f = fileInput.files[0];
    fileInput.value = "";
    if (!f) return;
    try {
      value = await fileToAvatarDataUri(f);
      renderPreview();
    } catch (e) {
      toast(e.message, true);
    }
  };
  clearBtn.onclick = () => { value = ""; renderPreview(); };

  renderPreview();
  wrap.append(preview, glyphInput, uploadBtn, clearBtn, fileInput);
  return { node: wrap, getValue: () => value };
}

/** One row of the per-model override list: a model-id <select> (populated from
 * *installedNames*, the real registry keys /api/models returns) beside its own
 * avatar picker, and a remove button that detaches the row.
 *
 * modelId is always kept as a selectable option even when it is not in
 * installedNames, so an override saved for a model that is currently
 * uninstalled (or temporarily missing) is never silently dropped or
 * rewritten to a different model just because the picker rendered. */
function buildAvatarOverrideRow(modelId, iconValue, installedNames) {
  const row = el("div", "avatar-override-row");
  const idSelect = document.createElement("select");
  idSelect.className = "avatar-override-id";
  const names = [...new Set(installedNames || [])];
  if (modelId && !names.includes(modelId)) names.push(modelId);
  if (!names.length) {
    const placeholder = document.createElement("option");
    placeholder.value = "";
    placeholder.textContent = "No models installed";
    placeholder.disabled = true;
    idSelect.appendChild(placeholder);
  }
  for (const name of names) {
    const opt = document.createElement("option");
    opt.value = name;
    opt.textContent = (installedNames || []).includes(name) ? name : `${name} (not installed)`;
    idSelect.appendChild(opt);
  }
  idSelect.value = modelId || "";
  const picker = buildAvatarPicker(iconValue);
  const removeBtn = el("button", "btn-secondary", "Remove");
  removeBtn.type = "button";
  removeBtn.onclick = () => row.remove();
  row.append(idSelect, picker.node, removeBtn);
  return { node: row, getModelId: () => idSelect.value, getValue: picker.getValue };
}

/** The Avatars section: user_avatar / model_avatar_default /
 * model_avatar_overrides are Widget.HIDDEN (buildSettingControl skips them,
 * same as logo_style/gpu_split_ratios), so they get their own bespoke picker
 * UI here instead of the generic text/number grid - a dict and an image blob
 * do not fit any existing Widget shape. Registers into _settingsControls so
 * the existing saveSettingsSection("avatars") PATCH /v1/config flow applies
 * unchanged. Returns null (renders nothing) on a schema/fetch mismatch,
 * matching the flat form's own tolerance for a field it cannot find. */
export async function buildAvatarsSection(form, fields) {
  const userField = (fields || []).find((f) => f.key === "user_avatar");
  const nameField = (fields || []).find((f) => f.key === "user_name");
  const modelField = (fields || []).find((f) => f.key === "model_avatar_default");
  const overridesField = (fields || []).find((f) => f.key === "model_avatar_overrides");
  if (!userField || !nameField || !modelField || !overridesField) return;

  let current;
  try {
    const r = await fetch("/v1/config", { headers: authHeaders() });
    if (!r.ok) throw new Error(r.statusText);
    current = await r.json();
  } catch (e) {
    return;   // best-effort: skip this refresh rather than show a broken panel
  }

  // Real installed model names, for the per-model override picker below.
  // Best-effort exactly like the /v1/config fetch above: an empty list just
  // means every override row falls back to its own preserved current value.
  let installedModels = [];
  try {
    const mr = await fetch("/api/models?type=llm", { headers: authHeaders() });
    if (mr.ok) {
      const md = await mr.json();
      installedModels = Array.isArray(md.models) ? md.models.map((m) => m.name) : [];
    }
  } catch (e) { /* ignored - rows fall back to their own current value */ }

  const panel = el("section", "card settings-section");
  panel.id = "settings-sec-avatars";
  panel.dataset.sec = "avatars";
  panel.dataset.group = "model";
  panel.dataset.secLabel = "Avatars";
  panel.appendChild(settingsSectionHead("Avatars", "model"));
  panel.appendChild(el("div", "sub",
    "A short emoji or a small uploaded image next to a turn. Always local - "
    + "never a URL or a remote fetch. A model with nothing set here falls "
    + "back to a generated monogram."));

  const userPicker = buildAvatarPicker(current.user_avatar || "");
  const userRow = el("div", "avatar-field-row");
  const nameInput = document.createElement("input");
  nameInput.type = "text";
  nameInput.placeholder = "Your name (shown instead of \"You\")";
  nameInput.value = current.user_name || "";
  nameInput.className = "avatar-name-input";
  nameInput.addEventListener("input", () => markSettingDirty(nameInput));
  nameInput.addEventListener("change", () => markSettingDirty(nameInput));
  userRow.append(el("div", "avatar-field-label", "Your icon"), userPicker.node, nameInput);
  panel.appendChild(userRow);

  const modelPicker = buildAvatarPicker(current.model_avatar_default || "");
  const modelRow = el("div", "avatar-field-row");
  modelRow.append(el("div", "avatar-field-label", "Model icon (default)"), modelPicker.node);
  panel.appendChild(modelRow);

  const overridesBox = el("div", "avatar-overrides-box");
  overridesBox.appendChild(el("div", "avatar-field-label", "Per-model icons"));
  const overridesList = el("div", "avatar-overrides-list");
  overridesBox.appendChild(overridesList);
  const rows = [];
  for (const [mid, icon] of Object.entries(current.model_avatar_overrides || {})) {
    const row = buildAvatarOverrideRow(mid, icon, installedModels);
    rows.push(row);
    overridesList.appendChild(row.node);
  }
  const addBtn = el("button", "btn-secondary", "Add override");
  addBtn.type = "button";
  addBtn.onclick = () => {
    const row = buildAvatarOverrideRow("", "", installedModels);
    rows.push(row);
    overridesList.appendChild(row.node);
  };
  overridesBox.appendChild(addBtn);
  panel.appendChild(overridesBox);

  _settingsControls = _settingsControls.filter((c) =>
    !["user_avatar", "user_name", "model_avatar_default", "model_avatar_overrides"].includes(c.field.key));
  _settingsControls.push({ field: userField, node: userRow, read: () => userPicker.getValue() });
  _settingsControls.push({ field: nameField, node: userRow, read: () => nameInput.value.trim() });
  _settingsControls.push({ field: modelField, node: modelRow, read: () => modelPicker.getValue() });
  _settingsControls.push({
    field: overridesField, node: overridesBox,
    read: () => {
      const out = {};
      for (const row of rows) {
        if (!row.node.isConnected) continue;   // a removed row
        const mid = row.getModelId();
        const v = row.getValue();
        if (mid && v) out[mid] = v;
      }
      return out;
    },
  });

  const actions = el("div", "actions");
  const save = el("button", "btn-primary", "Save Avatars");
  save.type = "button";
  save.onclick = () => saveSettingsSection("avatars");
  actions.appendChild(save);
  panel.appendChild(actions);

  form.appendChild(panel);
}

/* ---------------- Chat background (Settings -> System -> Appearance) ------ */

/** Wire the "Chat background" upload/preview/clear controls in the static
 *  Appearance card (index.html #sec-appearance, alongside the logo picker).
 *  Self-contained: its own GET /v1/config to seed the preview, each action
 *  PATCHes /v1/config immediately, with the same optimistic-apply-then-PATCH
 *  shape setupResidencyControls in settings-perf.js uses - but unlike that
 *  function, this one projects the new value onto other live surfaces
 *  (applyChatBackground's CSS var, the preview swatch) before the PATCH
 *  resolves, so a failed save must roll both back to the last
 *  server-confirmed value, not just report the failure. No-op if the card is
 *  absent. */
export function setupChatBackgroundPicker() {
  const preview = $("chat-bg-preview"), fileInput = $("chat-bg-file"),
        uploadBtn = $("chat-bg-upload"), clearBtn = $("chat-bg-clear");
  if (!preview || !fileInput || !uploadBtn || !clearBtn) return;

  let current = "";   // last value confirmed persisted on the server

  const renderPreview = (value) => {
    const src = safeAvatarImageSrc(value);
    preview.style.backgroundImage = src ? `url("${src}")` : "";
    preview.classList.toggle("empty", !src);
  };

  const save = async (value) => {
    applyChatBackground(value);
    try {
      const r = await fetch("/v1/config", {
        method: "PATCH", headers: authHeaders(),
        body: JSON.stringify({ chat_background: value }),
      });
      if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
      current = value;
      return true;
    } catch (e) {
      applyChatBackground(current);
      renderPreview(current);
      toast("Could not save background: " + e.message, true);
      return false;
    }
  };

  fetch("/v1/config", { headers: authHeaders() })
    .then((r) => (r.ok ? r.json() : {}))
    .then((cfg) => {
      current = cfg.chat_background || "";
      renderPreview(current);
    })
    .catch(() => { /* server unreachable - stays on the empty placeholder */ });

  uploadBtn.onclick = () => fileInput.click();
  fileInput.onchange = async () => {
    const f = fileInput.files[0];
    fileInput.value = "";
    if (!f) return;
    let dataUri;
    try {
      dataUri = await fileToBackgroundDataUri(f);
    } catch (e) { toast(e.message, true); return; }
    renderPreview(dataUri);
    if (await save(dataUri)) toast("Background saved");
  };
  clearBtn.onclick = async () => {
    renderPreview("");
    if (await save("")) toast("Background cleared");
  };
}

/* ---------------- Text-to-speech (the tts plugin's own block) -------------- */

// The controls currently rendered in the tts section, for the save pass. Kept
// out of _settingsControls: those are PATCH /v1/config keys, and these are the
// plugin block's own fields (POSTed to /v1/tts/config).
export let _ttsControls = [];

/** Build the "Text-to-speech" settings section: the tts plugin's own config
 *  block (voice / speed / model, plus an Advanced box for device + precision),
 *  edited through /v1/tts/config. Skipped when the plugin is not active - those
 *  settings would do nothing - but a FAILED fetch still renders a visible
 *  failure rather than silently vanishing (binding rule 5).
 *
 *  It also names the split the settings-exposure audit found: the server-side
 *  voice is the DEFAULT, while the chat picker stores a per-browser override.
 *  When this browser has one, the section says so and offers to clear it -
 *  otherwise changing the default here would look like it did nothing. */
export async function buildTtsSection(form) {
  let data;
  _ttsControls = [];               // reset first: a failed fetch renders no controls
  try {
    const r = await fetch("/v1/tts/config", { headers: authHeaders() });
    if (!r.ok) throw new Error(r.statusText);
    data = await r.json();
  } catch (e) {
    const fail = el("section", "card settings-section");
    fail.id = "settings-sec-tts";
    fail.dataset.sec = "tts";
    fail.dataset.group = "plugins";
    fail.dataset.secLabel = "Text-to-speech";
    fail.appendChild(settingsSectionHead("Text-to-speech", "plugins"));
    fail.appendChild(el("div", "sub",
      "Could not load the text-to-speech settings (" + e.message + ")."));
    form.appendChild(fail);
    return;
  }
  if (!data.active) return;        // plugin not installed/enabled: nothing to set

  const fields = (data.fields || []).filter(f => f.gui);
  const panel = el("section", "card settings-section");
  panel.id = "settings-sec-tts";
  panel.dataset.sec = "tts";
  panel.dataset.group = "plugins";
  panel.dataset.secLabel = "Text-to-speech";
  panel.appendChild(settingsSectionHead("Text-to-speech plugin", "plugins"));
  panel.appendChild(el("div", "sub",
    "Replies are spoken by the Kokoro voice model running in your browser. "
    + "These are the server-side defaults, shared by every browser."));

  const mkControl = (f) => {
    // A SELECT can only round-trip a value that is one of its options: a value
    // outside the list reads back as "" (= clear this override), so saving any
    // OTHER field in the section would silently wipe it. Two guards:
    //   - no options at all (the shipped voice list could not be read - the
    //     server then falls back to a shape check) -> render a text box, so the
    //     setting stays usable instead of an empty, value-destroying dropdown;
    //   - a current value outside the list (hand-edited config, or a list that
    //     changed) -> keep it as an option so it displays and survives a save.
    // "(inherit)" is offered whenever the field IS overridden, so a select-backed
    // override can be cleared from the GUI, not only through the API.
    const hasOptions = !!(f.options || []).length;
    const widget = (f.widget === "select" && !hasOptions) ? "text" : f.widget;
    let options = hasOptions ? [...f.options] : f.options;
    let labels = f.option_labels ? [...f.option_labels] : null;
    if (options && f.value != null && f.value !== "" && !options.includes(f.value)) {
      options.unshift(f.value);
      if (labels) labels.unshift(f.value);
    }
    if (options && f.is_override) {
      options.unshift("");                   // buildSettingControl labels it "(inherit)"
      if (labels) labels.unshift("(inherit)");
    }
    const ctrl = buildSettingControl({
      key: f.key, widget, label: f.label, help: f.help,
      default: f.value, options, min: f.min, max: f.max, step: f.step,
    });
    if (!ctrl) return null;
    // Show the friendly voice names ("Heart (en-us, Female, A)") the chat picker
    // uses, while the option VALUES stay the ids the server validates.
    if (labels) {
      const sel = ctrl.node.querySelector("select");
      if (sel) for (const [i, o] of [...sel.options].entries()) {
        if (labels[i]) o.textContent = labels[i];
      }
    }
    // model gets one extra, CLIENT-ONLY option: picking it reveals a free-text
    // box for any other Kokoro-compatible repo id. The sentinel value never
    // reaches the save payload - read() substitutes the text box's value (or
    // the unchanged original, while it is still empty) instead.
    if (f.key === "model") {
      const sel = ctrl.node.querySelector("select");
      if (sel) {
        const CUSTOM = "__custom__";
        const custom = document.createElement("option");
        custom.value = CUSTOM;
        custom.textContent = "Custom (type your own)...";
        sel.appendChild(custom);
        const box = document.createElement("input");
        box.type = "text";
        box.placeholder = "owner/name";
        box.hidden = true;
        box.style.marginTop = "0.5rem";
        box.dataset.key = f.key;
        ctrl.node.appendChild(box);
        sel.addEventListener("change", () => {
          box.hidden = sel.value !== CUSTOM;
          if (!box.hidden) box.focus();
        });
        box.addEventListener("input", () => markSettingDirty(box));
        box.addEventListener("change", () => markSettingDirty(box));
        const baseRead = ctrl.read;
        ctrl.read = () => {
          if (sel.value !== CUSTOM) return baseRead();
          const typed = box.value.trim();
          return typed || ctrl.orig;    // still empty: report unchanged, not a clear
        };
      }
    }
    ctrl.orig = f.value;
    if (!f.is_override) ctrl.node.classList.add("media-inherited");
    _ttsControls.push(ctrl);
    return ctrl;
  };

  const grid = el("div", "settings-fields");
  for (const f of fields.filter(f => !f.advanced)) {
    const ctrl = mkControl(f);
    if (ctrl) grid.appendChild(ctrl.node);
  }
  panel.appendChild(grid);

  // This browser's own voice override, if any: without this the user changes
  // the default above, hears the old voice, and concludes the setting is fake.
  const voiceField = fields.find(f => f.key === "voice") || {};
  const serverVoice = voiceField.value || "";
  const voiceName = (id) => {
    const i = (voiceField.options || []).indexOf(id);
    return (i >= 0 && (voiceField.option_labels || [])[i]) || id;
  };
  const override = browserVoiceOverride();
  if (override && override !== serverVoice) {
    const note = el("div", "sub tts-browser-override");
    note.append("This browser plays " + voiceName(override) + ", picked in chat. "
      + "That overrides the default above, here only. ");
    const clear = el("button", "btn-secondary tts-clear-override",
                     "Use the server default in this browser");
    clear.type = "button";
    clear.onclick = () => {
      // Never report a clear that did not happen (rule 5): this button exists
      // only because the value was READ back, so a failure is a real one and
      // the override is still in force.
      if (!clearBrowserVoiceOverride(serverVoice)) {
        toast("Could not clear it: this browser is blocking storage, so its own "
              + "voice is still in use here", true);
        return;
      }
      toast("This browser now follows the server default voice");
      refreshSettingsPage();
    };
    note.appendChild(clear);
    panel.appendChild(note);
  }

  const advanced = fields.filter(f => f.advanced);
  if (advanced.length) {
    const box = el("div", "media-comfy-box");
    box.appendChild(subCardHead("Advanced", "sliders", "cat-violet"));
    box.appendChild(el("div", "sub",
      "How the browser runs the voice model. Changes here (and to the model "
      + "above) apply the next time the page is loaded."));
    const advGrid = el("div", "settings-fields");
    for (const f of advanced) {
      const ctrl = mkControl(f);
      if (ctrl) advGrid.appendChild(ctrl.node);
    }
    box.appendChild(advGrid);
    panel.appendChild(box);
  }

  const actions = el("div", "actions");
  const save = el("button", "btn-primary settings-section-save", "Save Text-to-speech");
  save.type = "button";
  save.dataset.sec = "tts";
  save.onclick = () => saveTtsSettings();
  actions.appendChild(save);
  panel.appendChild(actions);
  form.appendChild(panel);
}

/** Save the tts block: POST only the fields the user changed (so an untouched
 *  field is never pinned as an override), then apply the new voice/speed to the
 *  RUNNING provider so the change is audible without a reload. */
export async function saveTtsSettings() {
  const updates = {};
  for (const c of _ttsControls) {
    const cur = c.read();
    if (_mediaChanged(cur, c.orig)) updates[c.field.key] = cur === undefined ? "" : cur;
  }
  if (!Object.keys(updates).length) { toast("Nothing changed"); return; }
  const r = await fetch("/v1/tts/config", {
    method: "POST", headers: authHeaders(), body: JSON.stringify(updates),
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) { toast(data.detail || "Save failed", true); return; }
  if (!Array.isArray(data.fields)) {
    // A 200 whose body we cannot read means the save probably landed but we
    // cannot say what is now in effect - do not claim a clean "Saved".
    toast("Saved, but the server's reply could not be read - reloading the "
          + "settings to show what is actually stored", true);
    refreshSettingsPage();
    return;
  }
  const saved = {};
  for (const f of data.fields) saved[f.key] = f.value;
  const live = applyServerTtsConfig({ voice: saved.voice, speed: saved.speed });
  // Say what actually took effect, per case - never a flat "Saved" that leaves
  // the user waiting to hear a change that cannot happen yet. The model, device
  // and precision are baked into the loaded model; this browser's own voice pick
  // deliberately still wins until it is cleared.
  const voiceChanged = "voice" in updates;
  let msg = "Saved";
  if (["model", "device", "dtype"].some(k => k in updates)) {
    msg = "Saved - the voice model reloads on the next page load";
  } else if (voiceChanged && browserVoiceOverride()) {
    msg = "Saved - this browser keeps its own voice until you clear it below";
  } else if (voiceChanged && !live) {
    msg = "Saved - the new voice applies on the next page load";
  }
  toast(msg);
  refreshSettingsPage();
}

// ------------------------------------------------------------------ //
//  Generic plugin-contributed settings (host.add_settings())          //
// ------------------------------------------------------------------ //

// Controls per plugin section, keyed by plugin name, mirroring _ttsControls -
// saving section "myplug" reads only its own entry and POSTs to
// /v1/plugins/myplug/settings.
export let _pluginSettingsControls = {};

/** Build one settings section per ACTIVE plugin that called host.add_settings()
 *  (the generic seam for a plugin the core has no bespoke schema for - e.g.
 *  the Open WebUI Valves interop path, see docs/plugin-interop.md). Unlike the
 *  tts/media sections above, the field LIST is not known ahead of time: it
 *  comes from GET /v1/plugins/settings, which reflects whatever the
 *  currently-loaded plugins registered. A plugin that is inactive, or that
 *  registered no fields (or only ones this caller cannot see), contributes no
 *  section here. A FAILED fetch still renders a visible failure rather than
 *  silently vanishing (rule 5), mirroring buildTtsSection. */
export async function buildPluginSettingsSections(form) {
  _pluginSettingsControls = {};        // reset first: a failed fetch renders no controls
  let data;
  try {
    const r = await fetch("/v1/plugins/settings", { headers: authHeaders() });
    if (!r.ok) throw new Error(r.statusText);
    data = await r.json();
  } catch (e) {
    const fail = el("section", "card settings-section");
    fail.id = "settings-sec-plugin-settings";
    fail.dataset.sec = "plugin-settings";
    fail.dataset.group = "plugins";
    fail.dataset.secLabel = "Plugin settings";
    fail.appendChild(settingsSectionHead("Plugin settings", "plugins"));
    fail.appendChild(el("div", "sub",
      "Could not load plugin-contributed settings (" + e.message + ")."));
    form.appendChild(fail);
    return;
  }
  for (const sec of data.plugins || []) {
    const ctrls = [];
    _pluginSettingsControls[sec.plugin] = ctrls;
    const panel = el("section", "card settings-section");
    panel.id = "settings-sec-plugin-" + sec.plugin;
    panel.dataset.sec = "plugin-" + sec.plugin;
    panel.dataset.group = "plugins";
    panel.dataset.secLabel = sec.label;
    panel.appendChild(settingsSectionHead(sec.label, "plugins"));
    const grid = el("div", "settings-fields");
    for (const f of sec.fields || []) {
      // Same SELECT handling as buildTtsSection's mkControl: a value outside
      // the option list (hand-edited config) must stay selectable so it
      // survives a save, no options at all falls back to a text box rather
      // than an empty value-destroying dropdown, and an overridden field
      // offers "(inherit)" so the override is clearable from the GUI.
      const hasOptions = !!(f.options || []).length;
      const widget = (f.widget === "select" && !hasOptions) ? "text" : f.widget;
      let options = hasOptions ? [...f.options] : f.options;
      if (options && f.value != null && f.value !== "" && !options.includes(f.value)) {
        options.unshift(f.value);
      }
      if (options && f.is_override) {
        options.unshift("");                 // buildSettingControl labels it "(inherit)"
      }
      const ctrl = buildSettingControl({
        key: f.key, widget, label: f.label, help: f.help,
        default: f.value, options, min: f.min, max: f.max, step: f.step,
      });
      if (!ctrl) continue;             // HIDDEN
      ctrl.orig = f.value;
      if (!f.is_override) ctrl.node.classList.add("media-inherited");
      ctrls.push(ctrl);
      grid.appendChild(ctrl.node);
    }
    panel.appendChild(grid);
    const actions = el("div", "actions");
    const save = el("button", "btn-primary settings-section-save", "Save " + sec.label);
    save.type = "button";
    save.dataset.sec = "plugin-" + sec.plugin;
    save.onclick = () => savePluginSettings(sec.plugin);
    actions.appendChild(save);
    panel.appendChild(actions);
    form.appendChild(panel);
  }
}

/** Save one plugin's add_settings() block: POST only the fields the user
 *  actually changed (mirrors saveTtsSettings), so an untouched field is never
 *  pinned as an override. */
export async function savePluginSettings(name) {
  const ctrls = _pluginSettingsControls[name] || [];
  const updates = {};
  for (const c of ctrls) {
    const cur = c.read();
    if (_mediaChanged(cur, c.orig)) updates[c.field.key] = cur === undefined ? "" : cur;
  }
  if (!Object.keys(updates).length) { toast("Nothing changed"); return; }
  const r = await fetch("/v1/plugins/" + encodeURIComponent(name) + "/settings", {
    method: "POST", headers: authHeaders(), body: JSON.stringify(updates),
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) { toast(data.detail || "Save failed", true); return; }
  toast("Saved");
  refreshSettingsPage();
}

/** Starts a "still working" indicator for a long-running comfy job (Set up,
 *  Update, Repair, or a reattach to one already running after a reload): a
 *  small spinner appended to *anchorEl* (the disabled button, or the status
 *  pill when there is no button to disable) plus a live "still working"
 *  readout inserted just above *logEl*. Without this, a quiet stretch in the
 *  streamed log - a large git clone, a slow pip install, a multi-GB download
 *  with no per-line progress - looks identical to a hung job, because the
 *  only signal so far was a disabled button and a static log.
 *
 *  Shared by every call site that starts or attaches to one of these jobs
 *  (both a fresh click and the reattach-after-reload path) so the start/stop
 *  bookkeeping - and the interval it owns - lives in one place. Returns
 *  { onLine, stop }: call onLine() whenever a log line arrives (it resets
 *  the "ago" clock so the readout reflects real silence, not wall-clock
 *  runtime) and stop() once the job settles, success or failure - every call
 *  site must call stop() on every exit path or the interval leaks. */
function startComfyJobIndicator(anchorEl, logEl) {
  const spinner = el("span", "comfy-managed-spinner");
  anchorEl.appendChild(spinner);
  const readout = el("div", "sub comfy-managed-elapsed");
  logEl.insertAdjacentElement("beforebegin", readout);
  let lastOutput = Date.now();
  const tick = () => {
    const secs = Math.max(0, Math.round((Date.now() - lastOutput) / 1000));
    readout.textContent = secs < 2
      ? "Still working..."
      : `Still working... (no new output for ${secs}s)`;
  };
  tick();
  const timer = setInterval(tick, 1000);
  return {
    onLine: () => { lastOutput = Date.now(); },
    stop: () => {
      clearInterval(timer);
      spinner.remove();
      readout.remove();
    },
  };
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
  const head = subCardHead("localm's own ComfyUI", "download", "cat-teal");
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
    // Update: move the managed checkout to the pin localm ships. Long (a re-checkout,
    // optionally a dependency reinstall), so it streams a job exactly like Set up.
    // ALWAYS rendered once installed - an unreadable/unknown update status must never
    // silently remove the only way to update (rule 5); it is disabled only for the one
    // case where updating is genuinely impossible, and then it says why.
    const update = el("button", "btn-secondary comfy-managed-update-btn", "Update");
    update.type = "button";
    if (st.updatable === false) {
      update.disabled = true;
      update.title = st.update_blocked_reason || "This install cannot be updated.";
    }
    actions.appendChild(update);

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

    // Say WHICH version is installed and which localm ships, so "Update" is a
    // decision rather than a mystery button. update_available is deliberately
    // tri-state: null means the marker could not be read, which is NOT "up to date".
    const vers = el("div", "sub comfy-managed-version");
    if (st.update_available === true) {
      vers.textContent = "Installed ComfyUI " + (st.installed_version || st.installed_commit || "")
        + " - localm ships " + (st.pinned_version || "a newer pin") + ". Update to move to it.";
    } else if (st.update_available === false) {
      vers.textContent = "Up to date with the ComfyUI localm ships ("
        + (st.pinned_version || "") + ").";
    } else if (st.installed) {
      vers.textContent = "Could not read which ComfyUI version is installed, so it is "
        + "unknown whether an update is due. Update is safe to run either way: it rolls "
        + "back if it fails.";
    }
    if (vers.textContent) host.appendChild(vers);

    if (st.updatable === false) {
      // The blocking reason as TEXT, not only a hover title - a disabled button with
      // no visible explanation is exactly the opaque dead end this route exists to fix.
      host.appendChild(el("div", "sub", st.update_blocked_reason));
    }

    const deps = el("label", "comfy-managed-reinstall");
    const depsBox = el("input");
    depsBox.type = "checkbox";
    depsBox.className = "comfy-managed-reinstall-box";
    deps.appendChild(depsBox);
    deps.append(" Also reinstall ComfyUI's dependencies (needed when the new version "
                + "changed them; slower)");
    if (st.updatable !== false) host.appendChild(deps);

    // Advanced/testing knob, matching `localm comfy update --commit` exactly - update
    // to a specific ComfyUI commit instead of the shipped pin. Left blank (the common
    // case) it is omitted entirely, so update_managed_comfy() falls back to its own
    // COMFYUI_PINNED_COMMIT default rather than the GUI silently pinning "".
    const commitRow = el("label", "comfy-managed-commit");
    commitRow.append("Advanced: update to a specific commit ");
    const commitBox = el("input");
    commitBox.type = "text";
    commitBox.className = "comfy-managed-commit-box";
    commitBox.style.maxWidth = "220px";
    commitBox.placeholder = "leave blank for the shipped pin";
    commitRow.appendChild(commitBox);
    if (st.updatable !== false) host.appendChild(commitRow);

    const ulog = el("pre", "comfy-managed-log");
    ulog.style.display = "none";
    host.appendChild(ulog);

    update.onclick = async () => {
      update.disabled = true;
      update.textContent = "Updating...";
      ulog.style.display = "";
      ulog.textContent = "";
      // Drop any error block from a PREVIOUS attempt, or a retry stacks a stale
      // reason above the fresh one and the user reads the wrong failure.
      for (const old of host.querySelectorAll(".comfy-managed-update-error")) old.remove();
      const resetUpdate = () => { update.disabled = false; update.textContent = "Update"; };
      const indicator = startComfyJobIndicator(update, ulog);
      let jobId;
      try {
        const params = new URLSearchParams();
        if (depsBox.checked) params.set("reinstall_requirements", "true");
        const commitVal = commitBox.value.trim();
        if (commitVal) params.set("commit", commitVal);
        const q = params.toString() ? "?" + params.toString() : "";
        const r = await fetch("/api/comfy/update" + q,
                              { method: "POST", headers: authHeaders() });
        const d = await r.json().catch(() => ({}));
        if (!r.ok) {
          toast(d.detail || "Update failed", true);
          indicator.stop();
          resetUpdate();
          ulog.style.display = "none";
          return;
        }
        jobId = d.job_id;
      } catch (e) {
        toast("Update failed", true);
        indicator.stop();
        resetUpdate();
        return;
      }
      const tail = [];
      const end = await streamJob(jobId, (line) => {
        indicator.onLine();
        ulog.textContent += line + "\n";
        // Keep a short TAIL, not just the last line: update_managed_comfy's failure
        // messages are long sentences and the console wraps them, so the final line
        // alone is often a fragment ("...comfy setup'.") with the actual reason lost.
        if (line && line.trim()) { tail.push(line.trim()); if (tail.length > 6) tail.shift(); }
        ulog.scrollTop = ulog.scrollHeight;
      });
      indicator.stop();
      const ok = !!(end && end.status === "done");
      const reason = tail.join(" ").trim();
      if (!ok) {
        // Rule 5: update_managed_comfy distinguishes states a user MUST be able to
        // tell apart - rolled back cleanly, rolled back but the patch re-apply
        // failed, and the rollback ITSELF failed (a genuinely mixed install). A GUI
        // that renders all of them as "update failed" swallows exactly the
        // distinction that module goes to trouble to make, so show its own words.
        const why = el("div", "sub comfy-managed-update-error");
        why.textContent = reason || "The update did not finish. See the log below.";
        host.insertBefore(why, ulog);
      }
      toast(ok ? "localm's ComfyUI is up to date"
               : (reason || "Update did not finish (see the log)"), !ok);
      // Same reasoning as Set up: never re-render over a failure log the toast just
      // pointed at. On success re-render so the version line and pill refresh.
      if (ok) renderManagedComfyPanel(host, toggleFields);
      else resetUpdate();
    };
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
        const indicator = startComfyJobIndicator(repair, log);
        let jobId;
        try {
          const r = await fetch("/api/comfy/repair",
                                { method: "POST", headers: authHeaders() });
          const d = await r.json().catch(() => ({}));
          if (!r.ok) {
            toast(d.detail || "Repair failed", true);
            indicator.stop();
            repair.disabled = false;
            repair.textContent = "Repair";
            log.style.display = "none";
            return;
          }
          jobId = d.job_id;
        } catch (e) {
          toast("Repair failed", true);
          indicator.stop();
          repair.disabled = false;
          repair.textContent = "Repair";
          return;
        }
        const end = await streamJob(jobId, (line) => {
          indicator.onLine();
          log.textContent += line + "\n";
          log.scrollTop = log.scrollHeight;
        });
        indicator.stop();
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
  // reading the actual job registry, not inferred). This page has no local
  // job id for it - it may have been started from this very tab moments ago
  // (a navigate-away-and-back resets the local `jobId` variable the click
  // handler below uses) just as easily as from another tab/session, so the
  // message must not guess which. Look the running job up via /api/activity
  // (ADR-0008 - the one route that finds a job without already holding its
  // id) and re-attach a live log to it, same as the Setup button's own flow,
  // instead of leaving the user with a static "please wait".
  pill.textContent = st.state === "installing" ? "installing..." : "not set up";
  if (st.state === "installing") {
    host.appendChild(el("div", "sub",
      "A setup is already in progress - showing its live output below."));
    const log = el("pre", "comfy-managed-log");
    host.appendChild(log);
    let jobId = null;
    try {
      const r = await fetch("/api/activity", { headers: authHeaders() });
      if (r.ok) {
        const d = await r.json();
        const op = (d.operations || [])
          .find((o) => o.kind === "comfy-setup" && o.status === "running");
        if (op) jobId = op.id;
      }
    } catch (e) { /* fall through to the not-found message below */ }
    if (!jobId) {
      // Not found is not necessarily stale: /api/activity is owner-filtered
      // (KEY-SCOPE-2), so on a keyed server with distinct principals this is
      // the expected, PERSISTENT result for a job another key started - state
      // will keep reading "installing" on every future check too. Do NOT
      // recurse into another render here: state staying "installing" forever
      // would turn that into an unbounded fetch loop. Say plainly that the
      // live log is unavailable and stop; the pill above already shows
      // "installing...", and revisiting this page later re-checks fresh.
      log.textContent = "(its live output is not available from here - check back "
        + "later, or from wherever the setup was started)";
      return;
    }
    // No button exists in this branch (an "installing" state deliberately
    // offers no action button - see the test asserting that), so the
    // indicator's spinner anchors to the status pill instead.
    const indicator = startComfyJobIndicator(pill, log);
    const end = await streamJob(jobId, (line) => {
      indicator.onLine();
      log.textContent += line + "\n";
      log.scrollTop = log.scrollHeight;
    });
    indicator.stop();
    const ok = !!(end && end.status === "done");
    toast(ok ? "localm's ComfyUI is ready" : "Setup did not finish (see the log)", !ok);
    // Same reasoning as the Set-up flow below: only re-render on success. On
    // failure, renderManagedComfyPanel's host.replaceChildren() would destroy
    // the log this toast just told the user to check, the instant it appears.
    if (ok) renderManagedComfyPanel(host, toggleFields);
    return;
  }
  host.appendChild(el("div", "sub",
    "Optional: let localm run its OWN ComfyUI under the localm data folder so it "
    + "can pin a known-good version and carry fixes. Off by default; your own "
    + "ComfyUI is never modified."));

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
    const indicator = startComfyJobIndicator(setup, log);
    let jobId;
    try {
      const r = await fetch("/api/comfy/setup",
                            { method: "POST", headers: authHeaders() });
      const d = await r.json().catch(() => ({}));
      if (!r.ok) {
        toast(d.detail || "Setup failed", true);
        indicator.stop();
        reset();
        log.style.display = "none";
        return;
      }
      jobId = d.job_id;
    } catch (e) {
      toast("Setup failed", true);
      indicator.stop();
      reset();
      return;
    }
    const end = await streamJob(jobId, (line) => {
      indicator.onLine();
      log.textContent += line + "\n";
      log.scrollTop = log.scrollHeight;
    });
    indicator.stop();
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
  const wrap = el("div", "card-head");
  wrap.appendChild(iconEl(name, "ic cat-ic cat-teal"));
  const txt = el("div", "card-head-text");
  txt.appendChild(head);
  wrap.appendChild(txt);
  sub.appendChild(wrap);
  if (["image", "music", "video"].includes(name)) {
    // docs/gui-design.md rule 6: state renders as a .job-state pill, not an
    // inline-JS-styled span. checking -> base pill (neutral, matches the old
    // no-color-set look); Running/Stopped/Unknown are set in checkComfy() below.
    const badge = el("span", "comfy-status-badge job-state", "ComfyUI: checking...");
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
          badge.className = "comfy-status-badge job-state " + (d.alive ? "st-ok" : "st-error");
          stopBtn.style.display = d.alive ? "" : "none";
          restartBtn.style.display = d.launched_by_localm ? "" : "none";
        }).catch(() => {
          badge.textContent = "ComfyUI: Unknown";
          badge.className = "comfy-status-badge job-state st-unknown";
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

/** Wire the search box (top level, like the other page controls). Typing filters
 *  live; Escape clears without leaving the page. */
const _settingsFilterBox = $("settings-filter");
if (_settingsFilterBox) {
  _settingsFilterBox.addEventListener("input", () => {
    _settingsFilterQuery = _settingsFilterBox.value || "";
    applySettingsFilter();
  });
  _settingsFilterBox.addEventListener("keydown", (e) => {
    if (e.key === "Escape") { e.preventDefault(); clearSettingsFilter(); }
  });
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


/* ================================================================ */
/*  Settings > Diagnostics                                           */
/* ================================================================ */
/* The ACTIVE self-checks from `localm doctor`, run in the app. The five probes
   live server-side in localm/diagnostics.py and are shared with the CLI, so the
   two surfaces cannot drift; this only renders them.

   THE WORDING IS THE DESIGN DECISION HERE. A terminal shows a transcript and the
   reader draws their own conclusion; a card has to state one. So the verdict line
   always says WHAT was checked ("5 active checks"), never "your system is fine" -
   these five probes are a real but narrow slice, and a card that overclaims is
   worse than no card, because the next person trusts it. */

// Rendered pill per check status. `skipped` is deliberately the NEUTRAL pill
// (no st- modifier): an absent optional backend is the common case and is not a
// fault, so painting it yellow would put a warning on every ordinary box.
const _DOCTOR_PILL = { ok: "st-ok", warn: "st-warn", fail: "st-error",
                       error: "st-error", skipped: "" };
const _DOCTOR_WORD = { ok: "ok", warn: "warning", fail: "failed",
                       error: "error", skipped: "not run" };

// Set while a run is being polled, so entering Settings twice does not start a
// second poll loop against the same job.
let _doctorPolling = false;

/** One check as a row: name + status pill, its sentence, and any extra lines
 *  the check produced (an ABI mismatch lists the fields that differ). */
function renderDoctorCheck(check) {
  const box = el("div", "doctor-check");
  const head = el("div", "job-head");
  head.appendChild(el("span", "job-name", check.label));
  const pill = el("span", "job-state " + (_DOCTOR_PILL[check.status] || ""),
                  _DOCTOR_WORD[check.status] || check.status);
  head.appendChild(pill);
  box.appendChild(head);
  if (check.summary) box.appendChild(el("div", "sub", check.summary));
  // `summary` is already the finding that carries the check's verdict (the
  // server picks it, so both surfaces agree on which line leads). Show the
  // OTHERS underneath - for a library that was found and then failed its BLAS
  // kernel check, the "found it" line is the context that makes the failure
  // readable - plus every finding's hints, which is where an ABI mismatch lists
  // the fields that actually differ.
  const findings = check.findings || [];
  const lead = findings.find((f) => f.status === check.status);
  for (const f of findings) {
    if (f !== lead) box.appendChild(el("div", "doctor-check-hint", f.text));
    for (const h of (f.hints || [])) box.appendChild(el("div", "doctor-check-hint", h));
  }
  return box;
}

/** Paint the card from one GET /api/doctor body. */
export function renderDoctorReport(body) {
  const status = $("doctor-status"), list = $("doctor-checks"), btn = $("doctor-run");
  if (!list) return;
  list.textContent = "";
  const report = body.report;
  const covers = body.covers || [];

  if (btn) btn.disabled = !!body.running;
  if (status) {
    status.hidden = false;
    if (body.running) {
      const p = body.progress || {};
      // "3 of 5" counts what has FINISHED and names what is running now, which
      // is what the server sends - never a percentage invented here.
      status.textContent = p.phase
        ? `Running: ${p.phase} (${p.done || 0} of ${p.total || covers.length} done)`
        : "Running the checks...";
    } else if (!report) {
      status.textContent = "Not run yet. These checks take about half a minute.";
    } else if (report.verdict === "error") {
      // The run did not happen. This must never render as a clean result.
      status.textContent = "The checks could not be run: " + (report.error || "no reason reported");
    } else {
      const checks = report.checks || [];
      const bad = checks.filter((c) => c.status === "fail" || c.status === "warn");
      const ran = checks.filter((c) => c.status !== "skipped").length;
      status.textContent = bad.length
        ? `${bad.length} of ${ran} active checks need attention.`
        : `All ${ran} active checks passed. This covers the active probes only, `
          + "not everything about your system.";
    }
  }

  // Before the first run, still show the rows - so the card names what it is
  // about to check rather than presenting an unexplained button.
  //
  // MID-RUN, each row says where IT is, from the server's `done` count. Saying
  // "waiting" on all five while the line above reads "4 of 5 done" contradicts
  // itself, and the fix is not to guess a verdict for the four that finished:
  // the browser genuinely does not have their results until the run ends (they
  // arrive as one report). So a finished row says it was checked and that the
  // result is coming, which is exactly what is true.
  const done = (body.progress || {}).done || 0;
  const placeholder = (i) => {
    if (!body.running) return "not run yet";
    if (i < done) return "checked - result when the run finishes";
    return i === done ? "checking now..." : "waiting...";
  };
  const rows = (report && report.checks && report.checks.length)
    ? report.checks
    : covers.map((c, i) => ({ key: c.key, label: c.label, status: "skipped",
                              summary: placeholder(i), findings: [] }));
  for (const c of rows) list.appendChild(renderDoctorCheck(c));
}

/** Fetch and paint. Also resumes polling when a run started elsewhere (another
 *  tab, or this one before a reload) is still in flight - ADR-0008. */
export async function refreshDiagnosticsCard() {
  if (!$("doctor-checks")) return;
  try {
    const r = await fetch("/api/doctor", { headers: authHeaders() });
    if (!r.ok) return;
    const body = await r.json();
    renderDoctorReport(body);
    if (body.running) pollDiagnostics();
  } catch (e) { /* a card that cannot reach the server just stays as it was */ }
}

/** Poll until the run finishes, then paint the result. */
export async function pollDiagnostics() {
  if (_doctorPolling) return;
  _doctorPolling = true;
  try {
    for (;;) {
      await new Promise((res) => setTimeout(res, 1000));
      const r = await fetch("/api/doctor", { headers: authHeaders() });
      if (!r.ok) return;
      const body = await r.json();
      renderDoctorReport(body);
      if (!body.running) return;
    }
  } catch (e) {
    // Say so rather than leaving the card frozen mid-run: a stalled poll and a
    // still-running check look identical from the outside.
    const status = $("doctor-status");
    if (status) { status.hidden = false; status.textContent = "Lost contact with the server while the checks were running."; }
    const btn = $("doctor-run");
    if (btn) btn.disabled = false;
  } finally { _doctorPolling = false; }
}

export async function runDiagnostics() {
  const btn = $("doctor-run"), status = $("doctor-status");
  if (btn) btn.disabled = true;
  if (status) { status.hidden = false; status.textContent = "Starting the checks..."; }
  try {
    const r = await fetch("/api/doctor/run", { method: "POST", headers: authHeaders() });
    const d = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(d.detail || r.statusText);
  } catch (e) {
    if (status) status.textContent = "Could not start the checks: " + e.message;
    if (btn) btn.disabled = false;
    return;
  }
  pollDiagnostics();
}
if ($("doctor-run")) $("doctor-run").onclick = runDiagnostics;
