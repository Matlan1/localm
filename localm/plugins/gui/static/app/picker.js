// SPDX-License-Identifier: AGPL-3.0-or-later
/* localm GUI - shared file/folder path picker.

   One pickPath() modal powers the RAG add-docs flow and every "Browse..." /
   "Move to..." button in the app. pickDirectory() and pickFile() are thin
   wrappers over it.

   pickDirectory and pickFile must stay `function` declarations so the global
   can be reassigned to a stub. */
"use strict";

// --- ES module imports ---
import { $, authHeaders, el, openModal, toast } from "./helpers.js";
import { t, tn } from "./i18n.js";

/** GET the listing for `path`. include_files adds files; meta adds the
 *  entries[] array of {name, is_dir, size, mtime}; includeHidden asks the
 *  server for dot-prefixed entries too. The picker always passes includeHidden
 *  and decides visibility itself. */
export async function fetchDirs(path, includeFiles = false, meta = false, includeHidden = false) {
  let url = "/api/fs/dirs?path=" + encodeURIComponent(path || "");
  if (includeFiles) url += "&include_files=true";
  if (meta) url += "&meta=true";
  if (includeHidden) url += "&include_hidden=true";
  const r = await fetch(url, { headers: authHeaders() });
  const data = await r.json();
  if (!r.ok) throw new Error(data.detail || r.statusText);
  return data;
}

/** GET the quick-access Places rail (home + standard folders + drives). */
export async function fetchPlaces() {
  const r = await fetch("/api/fs/places", { headers: authHeaders() });
  if (!r.ok) throw new Error(r.statusText);
  return r.json();
}

/** Create a folder named `name` inside `path`. Resolves to
 *  {path, parent, name}, or throws with the server's reason. The server 409s
 *  rather than overwriting an existing file or folder. */
export async function mkdirFs(path, name) {
  const r = await fetch("/api/fs/mkdir", {
    method: "POST", headers: authHeaders(),
    body: JSON.stringify({ path, name }),
  });
  const data = await r.json();
  if (!r.ok) throw new Error(data.detail || r.statusText);
  return data;
}

/** Rename the file/folder at `path` to `newName`, in the same parent
 *  directory. Resolves to {path, parent, name}, or throws. The server 409s
 *  rather than overwriting an existing target. */
export async function renameFs(path, newName) {
  const r = await fetch("/api/fs/rename", {
    method: "POST", headers: authHeaders(),
    body: JSON.stringify({ path, new_name: newName }),
  });
  const data = await r.json();
  if (!r.ok) throw new Error(data.detail || r.statusText);
  return data;
}

function fmtSize(n) {
  if (n === null || n === undefined) return "";
  if (n < 1024) return n + " B";
  const u = ["KB", "MB", "GB", "TB"];
  let v = n / 1024, i = 0;
  while (v >= 1024 && i < u.length - 1) { v /= 1024; i++; }
  return (v >= 10 ? Math.round(v) : v.toFixed(1)) + " " + u[i];
}
function fmtDate(mtime) {
  if (!mtime) return "";
  try { return new Date(mtime * 1000).toISOString().slice(0, 10); }
  catch (_) { return ""; }
}
function extOf(name) {
  const i = name.lastIndexOf(".");
  return i > 0 ? name.slice(i).toLowerCase() : "";
}
/** Join a child name onto a directory. "/" works as the separator on Windows
 *  too. An empty dir means `name` is itself a root (a drive like "C:\"). */  /* hygiene-ok */
function joinPath(dir, name) {
  if (!dir) return name;
  return dir.replace(/[\\/]+$/, "") + "/" + name;
}

function svg(paths) {
  return '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" ' +
    'stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" ' +
    'aria-hidden="true">' + paths + "</svg>";
}
const ICONS = {
  folder: svg('<path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>'),
  file: svg('<path d="M14 3v5h5"/><path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/>'),
  chevron: svg('<path d="M9 6l6 6-6 6"/>'),
  up: svg('<path d="M12 5v14"/><path d="M6 11l6-6 6 6"/>'),
  back: svg('<path d="M15 6l-6 6 6 6"/>'),
  search: svg('<path d="M11 4a7 7 0 1 0 0 14 7 7 0 0 0 0-14z"/><path d="M20 20l-3.5-3.5"/>'),
  edit: svg('<path d="M4 20h4l10-10-4-4L4 16z"/>'),
  home: svg('<path d="M3 11l9-8 9 8"/><path d="M5 10v10h14V10"/>'),
  desktop: svg('<path d="M3 5h18v11H3z"/><path d="M8 20h8"/><path d="M12 16v4"/>'),
  document: svg('<path d="M14 3v5h5"/><path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/><path d="M9 13h6"/><path d="M9 17h4"/>'),
  download: svg('<path d="M12 4v10"/><path d="M8 12l4 4 4-4"/><path d="M5 20h14"/>'),
  drive: svg('<path d="M4 6h16v6H4z"/><path d="M4 12v6h16v-6"/><path d="M8 15h.01"/><path d="M8 9h.01"/>'),
  newfolder: svg('<path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><path d="M12 11v6M9 14h6"/>'),
  check: svg('<path d="M5 12l4 4 10-10"/>'),
  close: svg('<path d="M6 6l12 12M18 6L6 18"/>'),
};
function iconSpan(cls, name) {
  const s = el("span", cls);
  s.innerHTML = ICONS[name] || ICONS.file;   // static, trusted markup only
  return s;
}

/**
 * Modal file/folder browser.
 *
 * opts:
 *   title        modal title
 *   mode         "dir" (default) | "file" | "multi"
 *   startPath    initial directory
 *   exts         array of selectable file extensions (".pdf", ...); other files
 *                are shown greyed and cannot be selected. Omit to allow any file.
 *   hint         footer hint line
 *   confirmLabel primary-button label override
 *
 * Resolves:
 *   "dir"/"file" -> a path string, or null if dismissed.
 *   "multi"      -> an array of >=1 path strings, or null if dismissed.
 */
export function pickPath(opts = {}) {
  const mode = opts.mode || "dir";
  const multi = mode === "multi";
  const exts = opts.exts ? new Set(opts.exts.map((e) => e.toLowerCase())) : null;

  return new Promise((resolve) => {
    let settled = false;
    let cleanup = () => {};
    const finish = (value) => {
      if (settled) return;
      settled = true;
      cleanup();
      $("modal").style.display = "none";
      resolve(value);
    };

    openModal(opts.title || t("picker.title"), (body) => {
      const modalEl = body.closest(".modal");
      if (modalEl) modalEl.classList.add("picker-modal");

      // ---- state ----
      let current = "";          // resolved current directory ("" = drive list)
      let data = null;           // last good listing
      let filterText = "";
      let showHidden = false;    // dot-prefixed entries hidden until toggled on
      let activeIdx = -1;        // keyboard highlight into visibleRows
      let visibleRows = [];
      let reqSeq = 0;
      const selected = new Map();// full path -> { is_dir }
      const history = [];        // for Back

      // ---- chrome ----
      const root = el("div", "picker");

      const nav = el("div", "picker-nav");
      const backBtn = el("button", "picker-icon-btn");
      backBtn.type = "button"; backBtn.title = t("picker.back");
      backBtn.appendChild(iconSpan("picker-svg", "back"));
      const upBtn = el("button", "picker-icon-btn");
      upBtn.type = "button"; upBtn.title = t("picker.up");
      upBtn.appendChild(iconSpan("picker-svg", "up"));
      const newFolderBtn = el("button", "picker-icon-btn");
      newFolderBtn.type = "button"; newFolderBtn.title = t("picker.newFolder");
      newFolderBtn.appendChild(iconSpan("picker-svg", "newfolder"));
      newFolderBtn.disabled = true;   // enabled once a real directory is loaded
      const pathWrap = el("div", "picker-path");
      nav.append(backBtn, upBtn, newFolderBtn, pathWrap);

      const mainRow = el("div", "picker-main");
      const rail = el("div", "picker-places");
      const panel = el("div", "picker-panel");
      const tools = el("div", "picker-tools");
      const filterWrap = el("div", "picker-filter");
      filterWrap.appendChild(iconSpan("picker-svg picker-filter-ic", "search"));
      const filterIn = document.createElement("input");
      filterIn.type = "text";
      filterIn.placeholder = t("picker.filterPlaceholder");
      filterIn.setAttribute("aria-label", t("picker.filterPlaceholder"));
      filterWrap.appendChild(filterIn);

      let showAllFiles = false;
      let typeSelect = null;
      if (exts) {
        typeSelect = el("select", "picker-type-select");
        typeSelect.setAttribute("aria-label", t("picker.typeFilterAria"));
        const optSupported = el("option", "", t("picker.supportedFiles"));
        optSupported.value = "supported";
        const optAll = el("option", "", t("picker.allFiles"));
        optAll.value = "all";
        typeSelect.append(optSupported, optAll);
        typeSelect.onchange = () => {
          showAllFiles = typeSelect.value === "all";
          renderList();
          updateCount();
        };
      }

      const hiddenLabel = el("label", "picker-hidden-toggle");
      const hiddenCb = document.createElement("input");
      hiddenCb.type = "checkbox";
      hiddenCb.setAttribute("aria-label", t("picker.showHiddenAria"));
      hiddenCb.onchange = () => {
        showHidden = hiddenCb.checked;
        renderList();
        updateCount();
      };
      const hiddenText = document.createTextNode(t("picker.hidden"));
      hiddenLabel.append(hiddenCb, hiddenText);

      const countEl = el("span", "picker-count");
      if (typeSelect) {
        tools.append(filterWrap, hiddenLabel, typeSelect, countEl);
      } else {
        tools.append(filterWrap, hiddenLabel, countEl);
      }
      const listEl = el("div", "picker-list");
      listEl.tabIndex = 0;
      panel.append(tools, listEl);
      mainRow.append(rail, panel);

      const foot = el("div", "picker-foot");
      const hintEl = opts.hint ? el("div", "picker-hint", opts.hint) : null;
      const cancelBtn = el("button", "btn-secondary", t("picker.cancel"));
      cancelBtn.type = "button";
      const okLabel = opts.confirmLabel
        || (multi ? t("picker.confirmAdd") : mode === "dir" ? t("picker.confirmDir") : t("picker.title"));
      const okBtn = el("button", "btn-primary", okLabel);
      okBtn.type = "button";
      if (mode === "file") okBtn.style.display = "none";   // file mode: click a file
      foot.append(cancelBtn, okBtn);

      root.append(nav, mainRow);
      if (hintEl) root.append(hintEl);
      root.append(foot);
      body.append(root);

      // ---- selection / counts ----
      function selectable(entry) {
        if (entry.is_dir) return multi;      // folders are selectable in multi only
        if (mode === "dir") return false;    // dir mode: files are context, not picks
        if (!exts || showAllFiles) return true;
        return exts.has(extOf(entry.name));
      }
      function updateCount() {
        if (multi) {
          const n = selected.size;
          countEl.textContent = tn("picker.selectedCount", n);
          okBtn.textContent = tn("picker.addItems", n,
            { label: opts.confirmLabel || t("picker.confirmAdd") });
          okBtn.disabled = n === 0;
        } else if (mode === "dir") {
          okBtn.disabled = !current;
          countEl.textContent = "";
        }
      }
      function toggleSelect(full, entry, on) {
        if (on) selected.set(full, { is_dir: entry.is_dir });
        else selected.delete(full);
        updateCount();
      }

      // ---- path bar: clickable breadcrumbs, editable on demand ----
      function renderPath() {
        pathWrap.replaceChildren();
        const crumbs = el("div", "picker-crumbs");
        const items = pathSegments(current);
        items.forEach((it, i) => {
          const isLast = i === items.length - 1;
          const c = el("span",
            "picker-crumb" + (isLast ? " picker-crumb-cur" : ""), it.label);
          if (!isLast) { const dest = it.path; c.onclick = () => navigate(dest); }
          crumbs.appendChild(c);
          if (!isLast) crumbs.appendChild(el("span", "picker-crumb-sep", "›"));
        });
        crumbs.onclick = (e) => { if (e.target === crumbs) editPath(); };
        const edit = el("button", "picker-path-edit");
        edit.type = "button"; edit.title = t("picker.editPathTitle");
        edit.appendChild(iconSpan("picker-svg", "edit"));
        edit.onclick = editPath;
        pathWrap.append(crumbs, edit);
      }
      function editPath() {
        pathWrap.replaceChildren();
        const inp = document.createElement("input");
        inp.type = "text";
        inp.className = "picker-path-input";
        inp.value = current;
        inp.setAttribute("aria-label", t("picker.pathAria"));
        const go = el("button", "picker-path-go", t("picker.go"));
        go.type = "button";
        pathWrap.append(inp, go);
        inp.focus(); inp.select();
        const commit = () => { const v = inp.value.trim(); if (v) navigate(v); else renderPath(); };
        go.onmousedown = (e) => { e.preventDefault(); commit(); };
        inp.onkeydown = (e) => {
          if (e.key === "Enter") { e.preventDefault(); commit(); }
          else if (e.key === "Escape") { e.preventDefault(); renderPath(); listEl.focus(); }
        };
        // Revert to breadcrumbs on blur unless the Go button took the click.
        inp.onblur = () => setTimeout(() => {
          if (pathWrap.contains(inp)) renderPath();
        }, 100);
      }

      // ---- listing ----
      function renderList() {
        listEl.replaceChildren();
        visibleRows = [];
        activeIdx = -1;
        const entries = (data && data.entries) ? data.entries : [];
        const dirs = entries.filter((e) => e.is_dir);
        const files = entries.filter((e) => !e.is_dir);
        const ordered = dirs.concat(files);   // folders first
        const q = filterText.toLowerCase();
        for (const entry of ordered) {
          if (!showHidden && entry.name.startsWith(".")) continue;
          if (q && !entry.name.toLowerCase().includes(q)) continue;
          const row = buildRow(entry);
          listEl.appendChild(row);
          visibleRows.push(row);
        }
        if (!visibleRows.length) {
          const onlyHidden = !showHidden && entries.length > 0
            && entries.every((e) => e.name.startsWith("."));
          listEl.appendChild(el("div", "picker-empty",
            q ? t("picker.noMatches") : onlyHidden
              ? t("picker.onlyHidden")
              : (data && data.parent === null)
                ? t("picker.noDrives") : t("picker.emptyFolder")));
        }
        // The server caps very large listings; say so when the list is partial.
        if (data && data.truncated) {
          listEl.appendChild(el("div", "picker-trunc", t("picker.truncated")));
        }
      }
      function buildRow(entry) {
        const canSelect = selectable(entry);
        const full = joinPath(current, entry.name);
        const row = el("div", "picker-row " +
          (entry.is_dir ? "is-dir" : "is-file") + (canSelect ? "" : " unselectable"));
        row.dataset.path = full;
        row._entry = entry;
        if (multi) {
          if (canSelect) {
            const cb = document.createElement("input");
            cb.type = "checkbox";
            cb.className = "picker-cb";
            cb.checked = selected.has(full);
            cb.setAttribute("aria-label", t("picker.selectAria", { name: entry.name }));
            cb.onclick = (e) => { e.stopPropagation(); toggleSelect(full, entry, cb.checked); };
            row.appendChild(cb);
          } else {
            row.appendChild(el("span", "picker-cb-spacer"));
          }
        }
        row.appendChild(iconSpan("picker-ic", entry.is_dir ? "folder" : "file"));
        row.appendChild(el("span", "picker-name", entry.name));
        if (current) {
          // Only entries inside a real directory are renamable, not the
          // synthetic drive-letter rows at the drive-list root (current === "").
          const renameBtn = el("button", "picker-row-rename");
          renameBtn.type = "button"; renameBtn.title = t("picker.renameTitle");
          renameBtn.appendChild(iconSpan("picker-svg", "edit"));
          renameBtn.onclick = (e) => { e.stopPropagation(); startRename(row); };
          row.appendChild(renameBtn);
        }
        const meta = el("span", "picker-meta");
        if (entry.is_dir) {
          meta.appendChild(iconSpan("picker-svg picker-chev", "chevron"));
        } else {
          const bits = [];
          if (entry.size !== null && entry.size !== undefined) bits.push(fmtSize(entry.size));
          if (entry.mtime) bits.push(fmtDate(entry.mtime));
          meta.appendChild(document.createTextNode(bits.join("  ·  ")));
          if (exts && !canSelect) meta.appendChild(el("span", "picker-tag", t("picker.unsupported")));
        }
        row.appendChild(meta);
        row.onclick = () => onRowActivate(row);
        return row;
      }
      function onRowActivate(row) {
        const entry = row._entry;
        const full = row.dataset.path;
        if (entry.is_dir) { navigate(full); return; }   // navigate into folder
        if (!selectable(entry)) return;
        if (mode === "file") { finish(full); return; }
        if (multi) {
          const on = !selected.has(full);
          const cb = row.querySelector(".picker-cb");
          if (cb) cb.checked = on;
          toggleSelect(full, entry, on);
        }
      }

      // ---- create folder / rename ----
      function startCreateFolder() {
        if (!current) return;   // nothing to create into at the drive-list root
        const row = el("div", "picker-row picker-row-editing");
        row.appendChild(iconSpan("picker-ic", "folder"));
        const inp = document.createElement("input");
        inp.type = "text";
        inp.className = "picker-inline-input";
        inp.value = t("picker.newFolderDefault");
        inp.setAttribute("aria-label", t("picker.newFolderNameAria"));
        const okIc = el("button", "picker-inline-btn");
        okIc.type = "button"; okIc.title = t("picker.create");
        okIc.appendChild(iconSpan("picker-svg", "check"));
        const cancelIc = el("button", "picker-inline-btn");
        cancelIc.type = "button"; cancelIc.title = t("picker.cancel");
        cancelIc.appendChild(iconSpan("picker-svg", "close"));
        row.append(inp, okIc, cancelIc);
        listEl.prepend(row);
        inp.focus(); inp.select();

        let done = false;
        const cancel = () => { if (done) return; done = true; renderList(); };
        const commit = () => {
          if (done) return;
          const name = inp.value.trim();
          if (!name) { cancel(); return; }
          done = true;
          mkdirFs(current, name).then((created) => {
            navigate(created.path);
          }).catch((e) => {
            toast(t("picker.createFolderFailed", { message: e.message }), true);
            done = false;
            renderList();
          });
        };
        okIc.onmousedown = (e) => { e.preventDefault(); commit(); };
        cancelIc.onmousedown = (e) => { e.preventDefault(); cancel(); };
        inp.onkeydown = (e) => {
          if (e.key === "Enter") { e.preventDefault(); commit(); }
          else if (e.key === "Escape") { e.preventDefault(); cancel(); }
        };
        // Revert on blur unless a button's mousedown already set `done`.
        inp.onblur = () => setTimeout(() => {
          if (!done && listEl.contains(row)) cancel();
        }, 100);
      }
      newFolderBtn.onclick = startCreateFolder;

      function startRename(row) {
        const entry = row._entry;
        const full = row.dataset.path;
        const nameSpan = row.querySelector(".picker-name");
        if (!nameSpan) return;
        const inp = document.createElement("input");
        inp.type = "text";
        inp.className = "picker-inline-input";
        inp.value = entry.name;
        inp.setAttribute("aria-label", t("picker.renameAria", { name: entry.name }));
        nameSpan.replaceWith(inp);
        inp.focus();
        if (entry.is_dir) {
          inp.select();
        } else {
          // Select the stem only, leaving the extension unselected.
          const ext = extOf(entry.name);
          const stopAt = ext ? entry.name.length - ext.length : entry.name.length;
          inp.setSelectionRange(0, stopAt);
        }

        let done = false;
        const restore = () => { if (done) return; done = true; renderList(); };
        const commit = () => {
          if (done) return;
          const newName = inp.value.trim();
          if (!newName || newName === entry.name) { restore(); return; }
          done = true;
          renameFs(full, newName).then((renamed) => {
            if (selected.has(full)) {
              const meta2 = selected.get(full);
              selected.delete(full);
              selected.set(renamed.path, meta2);
            }
            navigate(current, { push: false });   // refresh the listing in place
          }).catch((e) => {
            toast(t("picker.renameFailed", { message: e.message }), true);
            done = false;
            renderList();
          });
        };
        inp.onkeydown = (e) => {
          e.stopPropagation();   // don't let listEl's arrow-key navigation fire
          if (e.key === "Enter") { e.preventDefault(); commit(); }
          else if (e.key === "Escape") { e.preventDefault(); restore(); }
        };
        inp.onclick = (e) => e.stopPropagation();
        inp.onblur = () => setTimeout(() => { if (!done) restore(); }, 100);
      }

      // ---- navigation ----
      function navigate(path, { push = true } = {}) {
        const seq = ++reqSeq;
        setLoading(true);
        return fetchDirs(path, true, true, true).then((d) => {
          if (seq !== reqSeq) return;                 // superseded by a newer nav
          if (push && current !== d.path) history.push(current);
          current = d.path;
          data = d;
          // Landing in a dot-directory turns the Hidden toggle on.
          const base = (current || "").replace(/[\\/]+$/, "").split(/[\\/]/).pop() || "";
          if (base.startsWith(".") && !showHidden) {
            showHidden = true;
            hiddenCb.checked = true;
          }
          setLoading(false);
          renderPath();
          renderList();
          highlightPlace();
          updateCount();
          backBtn.disabled = history.length === 0;
          newFolderBtn.disabled = !current;
          listEl.scrollTop = 0;
        }).catch((e) => {
          if (seq !== reqSeq) return;
          setLoading(false);
          toast(t("picker.openFailed",
            { path: path || t("picker.location"), message: e.message }), true);
          if (!data) navigate("", { push: false });   // first load failed -> drives
        });
      }
      function setLoading(on) { listEl.classList.toggle("picker-loading", on); }
      function goUp() {
        if (data && data.parent !== null && data.parent !== undefined) navigate(data.parent);
        else navigate("");
      }
      function goBack() {
        if (history.length) navigate(history.pop(), { push: false });
      }
      backBtn.onclick = goBack;
      upBtn.onclick = goUp;

      // ---- places rail ----
      const PLACE_ICON = { home: "home", desktop: "desktop", documents: "document",
        downloads: "download", drive: "drive" };
      function loadPlaces() {
        fetchPlaces().then((pl) => {
          rail.replaceChildren();
          const group = (label, arr) => {
            if (!arr || !arr.length) return;
            rail.appendChild(el("div", "picker-rail-head", label));
            for (const it of arr) {
              const b = el("div", "picker-place");
              b.appendChild(iconSpan("picker-svg", PLACE_ICON[it.icon] || "folder"));
              b.appendChild(el("span", "picker-place-lbl", it.label));
              b.dataset.path = it.path;
              b.onclick = () => navigate(it.path);
              rail.appendChild(b);
            }
          };
          group(t("picker.places"), pl.places);
          group(t("picker.drives"), pl.drives);
          highlightPlace();
        }).catch((e) => {
          // Hide the rail and log the reason.
          rail.style.display = "none";
          try { console.debug("picker: places unavailable:", e && e.message); } catch (_) {}
        });
      }
      function highlightPlace() {
        const norm = (p) => (p || "").replace(/[\\/]+$/, "").toLowerCase();
        for (const b of rail.querySelectorAll(".picker-place")) {
          b.classList.toggle("active", !!current && norm(b.dataset.path) === norm(current));
        }
      }

      // ---- keyboard ----
      function moveActive(delta) {
        if (!visibleRows.length) return;
        activeIdx = activeIdx < 0
          ? (delta > 0 ? 0 : visibleRows.length - 1)
          : Math.max(0, Math.min(visibleRows.length - 1, activeIdx + delta));
        visibleRows.forEach((r, i) => r.classList.toggle("active", i === activeIdx));
        visibleRows[activeIdx].scrollIntoView({ block: "nearest" });
      }
      filterIn.oninput = () => { filterText = filterIn.value; renderList(); updateCount(); };
      filterIn.onkeydown = (e) => {
        if (e.key === "ArrowDown") { e.preventDefault(); listEl.focus(); moveActive(1); }
        else if (e.key === "Enter") { e.preventDefault(); if (visibleRows.length) onRowActivate(visibleRows[0]); }
        else if (e.key === "Escape") {
          if (filterText) { filterIn.value = ""; filterText = ""; renderList(); }
          else finish(null);
        }
      };
      listEl.onkeydown = (e) => {
        if (e.key === "ArrowDown") { e.preventDefault(); moveActive(1); }
        else if (e.key === "ArrowUp") { e.preventDefault(); moveActive(-1); }
        else if (e.key === "Enter") {
          e.preventDefault();
          if (activeIdx >= 0) onRowActivate(visibleRows[activeIdx]);
          else if (mode === "dir" && current) finish(current);
        }
        else if (e.key === "Backspace" || e.key === "ArrowLeft") { e.preventDefault(); goUp(); }
        else if (e.key === " " && multi && activeIdx >= 0) {
          e.preventDefault();
          const row = visibleRows[activeIdx];
          const entry = row._entry;
          if (entry && selectable(entry)) {
            const full = row.dataset.path;
            const on = !selected.has(full);
            const cb = row.querySelector(".picker-cb");
            if (cb) cb.checked = on;
            toggleSelect(full, entry, on);
          }
        }
      };

      // ---- confirm / cancel ----
      okBtn.onclick = () => {
        if (multi) { if (selected.size) finish([...selected.keys()]); }
        else if (mode === "dir") { if (current) finish(current); }
      };
      cancelBtn.onclick = () => finish(null);

      // ---- language ----
      // Re-paint the chrome this modal built once at open time. An edit in
      // progress (New Folder / Rename / the editable path bar) is left alone
      // so a language switch never discards text the user is mid-typing;
      // it picks up the new language next time it is opened.
      function retranslate() {
        backBtn.title = t("picker.back");
        upBtn.title = t("picker.up");
        newFolderBtn.title = t("picker.newFolder");
        filterIn.placeholder = t("picker.filterPlaceholder");
        filterIn.setAttribute("aria-label", t("picker.filterPlaceholder"));
        if (typeSelect) {
          typeSelect.setAttribute("aria-label", t("picker.typeFilterAria"));
          const optS = typeSelect.querySelector('option[value="supported"]');
          const optA = typeSelect.querySelector('option[value="all"]');
          if (optS) optS.textContent = t("picker.supportedFiles");
          if (optA) optA.textContent = t("picker.allFiles");
        }
        hiddenCb.setAttribute("aria-label", t("picker.showHiddenAria"));
        hiddenText.nodeValue = t("picker.hidden");
        cancelBtn.textContent = t("picker.cancel");
        if (!multi) {
          okBtn.textContent = opts.confirmLabel
            || (mode === "dir" ? t("picker.confirmDir") : t("picker.title"));
        }
        updateCount();
        if (!pathWrap.querySelector(".picker-path-input")) renderPath();
        if (!listEl.querySelector(".picker-inline-input")) renderList();
        loadPlaces();
      }
      const languageHandler = () => retranslate();
      document.addEventListener("localm:language", languageHandler);

      // Dismissing via the shared modal chrome (x / backdrop) sets
      // display:none; poll for that and resolve null.
      const watch = setInterval(() => {
        if ($("modal").style.display === "none") { clearInterval(watch); finish(null); }
      }, 200);
      const escHandler = (e) => {
        if (e.key !== "Escape") return;
        const a = document.activeElement;
        if (a && a.tagName === "INPUT" && root.contains(a)) return;   // let inputs handle Esc
        e.preventDefault();
        finish(null);
      };
      document.addEventListener("keydown", escHandler, true);
      cleanup = () => {
        clearInterval(watch);
        document.removeEventListener("keydown", escHandler, true);
        document.removeEventListener("localm:language", languageHandler);
        if (modalEl) modalEl.classList.remove("picker-modal");
      };

      // ---- go ----
      renderPath();
      updateCount();
      loadPlaces();
      navigate(opts.startPath || "", { push: false });
      setTimeout(() => { try { filterIn.focus(); } catch (_) {} }, 0);
    });
  });
}

/** Break a resolved path into clickable breadcrumb segments {label, path}. */
function pathSegments(current) {
  if (!current) return [{ label: t("picker.thisPc"), path: "" }];
  const win = /^([A-Za-z]:)[\\/]?/.exec(current);
  const out = [];
  let segs, acc;
  if (win) {
    out.push({ label: win[1], path: win[1] + "\\" });
    acc = win[1] + "\\";
    segs = current.slice(win[0].length).split(/[\\/]+/).filter(Boolean);
  } else if (current.startsWith("/")) {
    out.push({ label: "/", path: "/" });
    acc = "/";
    segs = current.split("/").filter(Boolean);
  } else {
    acc = "";
    segs = current.split(/[\\/]+/).filter(Boolean);
  }
  for (const s of segs) { acc = joinPath(acc, s); out.push({ label: s, path: acc }); }
  return out;
}

/** Pick a single directory. Resolves the chosen path, or null if dismissed. */
export function pickDirectory(title, startPath = "") {
  return pickPath({ mode: "dir", title: title || t("picker.pickDirTitle"), startPath });
}

/** Pick a single file. Resolves the chosen file path, or null if dismissed. */
export function pickFile(title, startPath = "") {
  return pickPath({ mode: "file", title: title || t("picker.pickFileTitle"), startPath });
}
