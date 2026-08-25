// SPDX-License-Identifier: AGPL-3.0-or-later
/* localm GUI - shared gallery engine for the three Studio pages.

   Images, Music and Video present the same library: a grid of generated
   artifacts you can select in bulk, open for detail, and act on (download,
   copy path, rename, move, delete, reuse the settings that made it). The grid
   and its interactions live here; each page passes its own preview builders.

   Object-URL lifetime: fetchImageURL() mints a blob: URL per call and never
   revokes one. Every URL minted here is revoked once the element that needed
   it has loaded, or when the element goes away. */

"use strict";

import { lsSetScoped } from "./chat.js";
import { $, authHeaders, confirmDanger, el, fetchImageURL, openModal, promptText, toast } from "./helpers.js";
import { emptyState, iconEl } from "./icons.js";
import { pickDirectory } from "./picker.js";

/* Every fetch below spells its path as a template literal starting with "/"
   (`/api/${cfg.slug}/...`, never `${cfg.api}/...`). */

export const GRID_DEFAULT = 24;

/** Mint a blob: URL and revoke it once *node* has loaded it (or failed).
 *  `event` is whichever load-ish event that element type actually fires. */
async function urlFor(path, node, event) {
  const url = await fetchImageURL(path);
  if (node) {
    const done = () => URL.revokeObjectURL(url);
    node.addEventListener(event, done, { once: true });
    node.addEventListener("error", done, { once: true });
  }
  return url;
}

/* Toast the reason when a media element fails to load, and run onFail.

   Call this on any media element BEFORE assigning src. */
export function reportMediaLoadFailure(player, what, onFail) {
  player.addEventListener("error", () => {
    const err = player.error;
    const why = {
      1: "loading was aborted",
      2: "a network error",
      3: "the file could not be decoded",
      4: "this browser refused the source",
    }[err && err.code] || "an unknown error";
    toast(`Could not play ${what}: ${why}.`, true);
    if (onFail) onFail();
  });
}

/** Build one gallery. Returns { refresh, render, state } - `refresh` is what
 *  the page and the tab dispatcher call. */
export function createGallery(cfg) {
  const state = { items: [], selected: new Set(), showAll: false };

  const fileURL = (name, node, event) =>
    urlFor(`/api/${cfg.slug}/file/${encodeURIComponent(name)}`, node, event);

  async function apiDelete(name) {
    const r = await fetch(`/api/${cfg.slug}/file/${encodeURIComponent(name)}`, {
      method: "DELETE", headers: authHeaders() });
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  }

  async function apiMove(name, dest) {
    const r = await fetch(`/api/${cfg.slug}/file/${encodeURIComponent(name)}/move`, {
      method: "POST", headers: authHeaders(), body: JSON.stringify({ dest }) });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || r.statusText);
    return data.path;
  }

  async function apiRename(name, newName) {
    const r = await fetch(`/api/${cfg.slug}/file/${encodeURIComponent(name)}/rename`, {
      method: "POST", headers: authHeaders(),
      body: JSON.stringify({ new_name: newName }) });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || r.statusText);
    return data.name;
  }

  async function download(name) {
    // Revoked on a 30s timer rather than on an element's load event: the
    // browser reads the URL asynchronously after the synchronous click().
    const url = await fetchImageURL(`/api/${cfg.slug}/file/${encodeURIComponent(name)}`);
    const a = document.createElement("a");
    a.href = url;
    a.download = name;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 30000);
  }

  const ctx = { slug: cfg.slug, fileURL, apiDelete, apiMove, apiRename,
                download, refresh, closeModal, reportMediaLoadFailure };

  function closeModal() { $("modal").style.display = "none"; }

  async function refresh() {
    if (cfg.beforeRefresh) cfg.beforeRefresh();
    const grid = $(cfg.gridId);
    if (!grid) return;
    try {
      const r = await fetch(`/api/${cfg.slug}/history`, { headers: authHeaders() });
      if (!r.ok) {
        // Show the failure in the grid instead of leaving it blank.
        grid.replaceChildren(emptyState("warning", `Could not load ${cfg.plural}`,
          `The server returned HTTP ${r.status}. Try refreshing the page.`));
        return;
      }
      state.items = (await r.json())[cfg.listKey] || [];
    } catch (e) {
      grid.replaceChildren(emptyState("warning", `Could not load ${cfg.plural}`,
        `${(e && e.message) || e}`));
      return;
    }
    // Drop selections for items that no longer exist
    const names = new Set(state.items.map((i) => i.name));
    for (const n of state.selected) if (!names.has(n)) state.selected.delete(n);
    render();
  }

  function render() {
    const grid = $(cfg.gridId);
    if (!grid) return;
    releaseDetailURL();
    grid.replaceChildren();
    renderBulkBar();
    if (!state.items.length) {
      grid.appendChild(emptyState(cfg.emptyIcon, cfg.emptyTitle, cfg.emptyHint));
      return;
    }
    const shown = state.showAll ? state.items : state.items.slice(0, GRID_DEFAULT);
    for (const item of shown) grid.appendChild(buildCard(item));
    if (state.items.length > GRID_DEFAULT) {
      const toggle = el("button", "btn-secondary media-show-all",
        state.showAll ? "show fewer" : `show all (${state.items.length})`);
      toggle.onclick = () => { state.showAll = !state.showAll; render(); };
      grid.appendChild(toggle);
    }
  }

  function buildCard(item) {
    const thumb = el("div", "thumb" + (cfg.cardClass ? " " + cfg.cardClass : ""));

    thumb.appendChild(cfg.buildPreview(item, ctx));

    // selection checkbox (top-left) - selected cards stay marked
    const sel = document.createElement("input");
    sel.type = "checkbox";
    sel.className = "thumb-sel";
    sel.checked = state.selected.has(item.name);
    thumb.classList.toggle("selected", sel.checked);
    sel.onclick = (e) => {
      e.stopPropagation();
      if (sel.checked) state.selected.add(item.name);
      else state.selected.delete(item.name);
      thumb.classList.toggle("selected", sel.checked);
      renderBulkBar();
    };
    thumb.appendChild(sel);

    // hover quick actions (top-right)
    const acts = el("div", "thumb-acts");
    const dl = el("button", "download");
    dl.appendChild(iconEl("download"));
    dl.title = "Download";
    dl.onclick = (e) => {
      e.stopPropagation();
      download(item.name).catch((err) => toast("Download failed: " + err.message, true));
    };
    const del = el("button", "danger");
    del.appendChild(iconEl("trash"));
    del.title = "Delete from disk";
    del.onclick = (e) => {
      e.stopPropagation();
      confirmDelete(item);
    };
    acts.append(dl, del);
    thumb.appendChild(acts);

    thumb.appendChild(el("div", "cap", cfg.caption(item)));
    thumb.onclick = () => showDetail(item);
    return thumb;
  }

  function confirmDelete(item, alsoClose) {
    confirmDanger(`Delete "${item.name}"?`, "This removes the file from disk.",
      "Delete", async () => {
        try {
          await apiDelete(item.name);
          toast("Deleted " + item.name);
          if (alsoClose) closeModal();
          refresh();
        } catch (err) { toast("Delete failed: " + err.message, true); }
      });
  }

  /** Bulk actions bar - appears above the grid while a selection exists. */
  function renderBulkBar() {
    const bar = $(cfg.bulkId);
    if (!bar) return;
    const n = state.selected.size;
    bar.style.display = n ? "flex" : "none";
    bar.replaceChildren();
    if (!n) return;
    bar.appendChild(el("span", "count", `${n} selected`));

    const move = el("button", "btn-secondary", "move to folder…");
    move.onclick = async () => {
      const dest = await pickDirectory(`Move ${n} ${n === 1 ? cfg.noun : cfg.plural} to…`,
        localStorage.getItem(cfg.moveDestKey) || "");
      if (!dest) return;
      let moved = 0, failed = 0;
      for (const name of [...state.selected]) {
        try { await apiMove(name, dest); moved++; }
        catch (e) { failed++; toast(`${name}: ${e.message}`, true); }
      }
      lsSetScoped(cfg.moveDestKey, dest);
      toast(`Moved ${moved} ${cfg.plural}` + (failed ? `, ${failed} failed` : ""));
      state.selected.clear();
      refresh();
    };

    const del = el("button", "btn-secondary btn-danger", "delete");
    del.onclick = () => {
      confirmDanger(`Delete ${n} ${n === 1 ? cfg.noun : cfg.plural}?`,
        "This removes the files from disk.", "Delete", async () => {
          let deleted = 0, failed = 0;
          for (const name of [...state.selected]) {
            try { await apiDelete(name); deleted++; }
            catch (e) { failed++; toast(`${name}: ${e.message}`, true); }
          }
          toast(`Deleted ${deleted} ${cfg.plural}` + (failed ? `, ${failed} failed` : ""));
          state.selected.clear();
          refresh();
        });
    };

    const clear = el("button", "btn-secondary", "clear selection");
    clear.onclick = () => { state.selected.clear(); render(); };

    bar.append(move, del, clear);
  }

  function showDetail(item) {
    openModal(item.name, (body) => {
      cfg.buildDetailPreview(item, body, ctx);

      const actions = el("div", "actions");
      for (const btn of (cfg.extraActions ? cfg.extraActions(item, ctx) : [])) {
        actions.appendChild(btn);
      }

      if (cfg.reuse && item.meta && Object.keys(item.meta).length) {
        const reuse = el("button", "btn-secondary", "reuse settings");
        reuse.title = `Fill the generation form with this ${cfg.noun}'s settings`;
        reuse.onclick = () => {
          cfg.reuse(item);
          closeModal();
          toast("Settings restored - tweak and generate");
        };
        actions.appendChild(reuse);
      }

      const dl = el("button", "btn-secondary", "download");
      dl.onclick = () =>
        download(item.name).catch((e) => toast("Download failed: " + e.message, true));
      actions.appendChild(dl);

      const copyPath = el("button", "btn-secondary", "copy path");
      copyPath.title = item.path || "";
      copyPath.onclick = async () => {
        try {
          await navigator.clipboard.writeText(item.path || item.name);
          toast("Path copied");
        } catch (e) { toast("Copy failed: " + e.message, true); }
      };
      actions.appendChild(copyPath);

      const rename = el("button", "btn-secondary", "rename…");
      rename.onclick = async () => {
        // promptText() reuses the shared #modal and replaces this detail
        // view's content, so every path except success re-opens the view.
        const newName = await promptText("New name:", item.name);
        if (!newName || newName.trim() === item.name) { showDetail(item); return; }
        try {
          const name = await apiRename(item.name, newName.trim());
          toast("Renamed to " + name);
          closeModal();
          refresh();
        } catch (e) {
          toast(e.message || "Rename failed", true);
          showDetail(item);
        }
      };
      actions.appendChild(rename);

      const move = el("button", "btn-secondary", "move to folder…");
      move.onclick = async () => {
        const dest = await pickDirectory(`Move ${cfg.noun} to…`,
          localStorage.getItem(cfg.moveDestKey) || "");
        if (!dest) return;
        try {
          const path = await apiMove(item.name, dest);
          lsSetScoped(cfg.moveDestKey, dest);
          toast("Moved to " + path);
          closeModal();
          refresh();
        } catch (e) { toast("Move failed: " + e.message, true); }
      };
      actions.appendChild(move);

      const del = el("button", "btn-secondary btn-danger", "delete");
      del.onclick = () => confirmDelete(item, true);
      actions.appendChild(del);

      body.appendChild(actions);

      if (item.meta && Object.keys(item.meta).length) {
        for (const [k, v] of Object.entries(item.meta)) {
          const row = el("div", "log-entry");
          row.appendChild(el("span", "t", k));
          row.appendChild(document.createTextNode(
            typeof v === "string" ? v : JSON.stringify(v)));
          body.appendChild(row);
        }
      }
    });
  }

  return { refresh, render, showDetail, state, ctx };
}

/* ---- preview builders, one per medium ---------------------------------- */

/* Replace an unshowable preview with a warning icon and a reason. The card
   itself stays, keeping its checkbox, metadata and actions. */
function previewFailed(wrap, why) {
  wrap.replaceChildren();
  wrap.classList.add("thumb-failed");
  wrap.appendChild(iconEl("warning", "audio-ic"));
  wrap.appendChild(el("div", "audio-tags", why));
}

/** Image: the file IS the thumbnail. */
export function imagePreview(item, ctx) {
  const wrap = el("div", "thumb-face");
  const img = document.createElement("img");
  img.addEventListener("error", () => previewFailed(wrap, "Preview unavailable"));
  ctx.fileURL(item.name, img, "load")
    .then((url) => (img.src = url))
    .catch(() => previewFailed(wrap, "Could not load this image"));
  wrap.appendChild(img);
  return wrap;
}

/** Video: a first frame painted by the browser. `preload="metadata"` fetches
 *  enough to decode, then seeks just past zero. Muted and playsinline. */
export function videoPreview(item, ctx) {
  const wrap = el("div", "thumb-face");
  const v = document.createElement("video");
  v.preload = "metadata";
  v.muted = true;
  v.playsInline = true;
  v.addEventListener("loadedmetadata", () => {
    // Assigning a non-finite currentTime throws, so only seek on a finite,
    // non-zero duration.
    if (Number.isFinite(v.duration) && v.duration > 0) {
      try { v.currentTime = Math.min(0.1, v.duration / 2); } catch (e) { /* keep frame 0 */ }
    }
  });
  v.addEventListener("error", () => previewFailed(wrap, "Preview unavailable"));
  ctx.fileURL(item.name, v, "loadeddata")
    .then((url) => (v.src = url))
    .catch(() => previewFailed(wrap, "Could not load this clip"));
  wrap.appendChild(v);
  wrap.appendChild(el("span", "thumb-badge", durationLabel(item) || "clip"));
  return wrap;
}

/** Music: the card shows the track's tags plus a play control that plays it
 *  inline. */
export function musicPreview(item, ctx) {
  const wrap = el("div", "thumb-face thumb-audio");
  wrap.appendChild(iconEl("music", "audio-ic"));
  const tags = item.meta?.tags || item.name;
  wrap.appendChild(el("div", "audio-tags", tags));

  let player = null;
  const play = el("button", "audio-play");
  play.type = "button";
  play.title = "Play this track";
  play.appendChild(iconEl("play", "btn-ic"));
  const stop = () => {
    if (!player) return;
    player.pause();
    if (player.dataset.url) {
      URL.revokeObjectURL(player.dataset.url);
      _cardURLs.delete(player.dataset.url);
    }
    player.remove();
    player = null;
    play.replaceChildren(iconEl("play", "btn-ic"));
    play.title = "Play this track";
  };
  play.onclick = async (e) => {
    e.stopPropagation();               // the card itself opens the detail view
    if (player) { stop(); return; }
    play.disabled = true;
    try {
      const url = await fetchImageURL(
        `/api/${ctx.slug}/file/${encodeURIComponent(item.name)}`);
      player = document.createElement("audio");
      player.controls = true;
      player.autoplay = true;
      player.dataset.url = url;
      _cardURLs.add(url);
      player.className = "audio-inline";
      player.onclick = (ev) => ev.stopPropagation();
      reportMediaLoadFailure(player, "the track", stop);
      player.src = url;
      wrap.appendChild(player);
      play.replaceChildren(iconEl("stop", "btn-ic"));
      play.title = "Stop";
    } catch (err) {
      toast("Could not load the track: " + err.message, true);
    } finally {
      play.disabled = false;
    }
  };
  wrap.appendChild(play);

  const badge = durationLabel(item);
  if (badge) wrap.appendChild(el("span", "thumb-badge", badge));
  return wrap;
}

/** "2:05" from a track's duration, "5s" from a clip's, else "". */
export function durationLabel(item) {
  const m = item.meta || {};
  const secs = Number(m.duration_seconds ?? m.seconds);
  if (!Number.isFinite(secs) || secs <= 0) return "";
  if (secs < 60) return `${Math.round(secs)}s`;
  return `${Math.floor(secs / 60)}:${String(Math.round(secs % 60)).padStart(2, "0")}`;
}

/* The detail modal's blob: URL. At most one is live: it is released when the
   next one is minted and when any gallery re-renders. */
let _detailURL = null;

/* blob: URLs of the inline card players, released on the next render. */
const _cardURLs = new Set();

export function releaseDetailURL() {
  if (_detailURL) { URL.revokeObjectURL(_detailURL); _detailURL = null; }
  for (const u of _cardURLs) URL.revokeObjectURL(u);
  _cardURLs.clear();
}

/** Modal preview for a playable medium: a real, controllable player. */
export function playerDetail(tag, what) {
  return (item, body, ctx) => {
    const player = document.createElement(tag);
    player.controls = true;
    player.style.width = "100%";
    ctx.reportMediaLoadFailure(player, what);
    fetchImageURL(`/api/${ctx.slug}/file/${encodeURIComponent(item.name)}`)
      .then((url) => {
        releaseDetailURL();
        _detailURL = url;
        player.src = url;
      })
      .catch((e) => toast(`Could not load the ${what}: ${e.message}`, true));
    body.appendChild(player);
  };
}

/* ---- per-plugin "reload chat model after generating" toggle ------------- */

/* POST /v1/media/config/<plugin> {"reload_after": bool} writes the per-plugin
   key, which takes precedence over the legacy global reload_llm_after_imagine.
   A plugin with no per-plugin value falls back to that global. */

/* Binding makes no network request: it only attaches the change handler. The
   value is read in each gallery's beforeRefresh, once its page is opened and
   bootAuthProbe has confirmed this client is authed. */
export function bindReloadToggle(plugin, checkboxId) {
  const box = $(checkboxId);
  if (!box) return;

  box.onchange = async () => {
    const value = box.checked;
    try {
      const r = await fetch(`/v1/media/config/${plugin}`, {
        method: "POST", headers: authHeaders(),
        body: JSON.stringify({ reload_after: value }),
      });
      if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
      toast(value ? "Chat model reloads after each generation"
                  : "This backend stays loaded - chat model reloads on next message");
    } catch (e) {
      toast("Could not save setting: " + e.message, true);
      box.checked = !value;      // revert: the write failed
    }
  };
}

/** Re-read the RESOLVED value (per-plugin override, else the legacy global). */
export async function refreshReloadToggle(plugin, checkboxId) {
  const box = $(checkboxId);
  if (!box) return;
  try {
    const r = await fetch("/v1/media/config", { headers: authHeaders() });
    if (!r.ok) return;                       // leave the last known state alone
    const data = await r.json();
    const entry = (data.plugins || []).find((p) => p.plugin === plugin);
    const field = (entry?.fields || []).find((f) => f.key === "reload_after");
    if (field && typeof field.value === "boolean") box.checked = field.value;
  } catch (e) { /* ignored - keep the current state */ }
}
