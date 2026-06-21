// SPDX-License-Identifier: AGPL-3.0-or-later
/* localm GUI - Models / Images / Plugins / Settings pages.
   Relies on helpers from app.js ($, el, authHeaders, toast, streamJob,
   fetchImageURL, openModal, refreshModels, modelCache, switchModel).
   Untrusted strings only ever reach the DOM via textContent. */

"use strict";

/* ================================================================ */
/*  View refresh dispatcher                                          */
/* ================================================================ */

window.onViewShown = (name) => {
  // Re-sync the plugin command catalog on entering a composer so a plugin
  // toggled elsewhere (CLI, another tab) updates the slash hints without a
  // full reload (refreshPluginCommands lives in app.js, shared global scope).
  if (name === "chat" || name === "coder") refreshPluginCommands();
  if (name === "coder") { populateSetupModels(); presetCoderMode(); }
  if (name === "models") refreshModelsPage();
  if (name === "images") { refreshImageHistory(); refreshWorkflowPanel("image"); }
  if (name === "music") { refreshMusicHistory(); refreshWorkflowPanel("music"); }
  if (name === "video") { refreshVideoHistory(); refreshWorkflowPanel("video"); }
  if (name === "knowledge") refreshKnowledgePage();
  if (name === "plugins") { renderCatalogPlugins(); refreshPluginsPage(); }
  if (name === "settings") refreshSettingsPage();
};

/** Pre-select the configured coder session mode in the setup form. */
async function presetCoderMode() {
  try {
    const r = await fetch("/v1/config", { headers: authHeaders() });
    if (!r.ok) return;
    const cfg = await r.json();
    const sel = $("setup-mode");
    if (sel && cfg.effective_coder_mode) sel.value = cfg.effective_coder_mode;
  } catch (e) { /* keep form default */ }
}

/* ================================================================ */
/*  Models page                                                      */
/* ================================================================ */

function fmtSize(bytes) {
  if (bytes == null) return "";
  return (bytes / GIB).toFixed(2) + " GB";   // binary GiB, labelled GB (see app.js)
}

async function refreshModelsPage() {
  await refreshModels();
  const box = $("models-table");
  box.replaceChildren();
  if (!modelCache.models.length) {
    box.appendChild(el("div", "sub", "No models registered yet - pull one above."));
    return;
  }
  const table = el("table", "data-table");
  const thead = el("thead");
  const hr = el("tr");
  for (const h of ["Name", "Source", "Size", ""]) hr.appendChild(el("th", "", h));
  thead.appendChild(hr);
  table.appendChild(thead);
  const tbody = el("tbody");
  for (const m of modelCache.models) {
    const tr = el("tr");
    const nameTd = el("td");
    nameTd.appendChild(el("span", "name", m.name));
    if (m.active) nameTd.appendChild(el("span", "active-tag", "active"));
    tr.appendChild(nameTd);
    tr.appendChild(el("td", "mono", m.source || ""));
    tr.appendChild(el("td", "mono", fmtSize(m.size_bytes)));

    const actions = el("td");
    actions.style.textAlign = "right";
    const detail = el("button", "", "info");
    detail.onclick = () => showModelDetail(m.name);
    actions.appendChild(detail);
    if (!m.active) {
      const use = el("button", "", "use");
      use.onclick = async () => {
        use.disabled = true;
        try {
          await switchModel(m.name);
          toast("Model switched to " + m.name);
          refreshModelsPage();
        } catch (e) {
          toast("Load failed: " + e.message, true);
        } finally { use.disabled = false; }
      };
      actions.appendChild(use);
    }
    const alias = el("button", "", "alias");
    alias.onclick = async () => {
      const name = prompt(`New alias for '${m.name}':`);
      if (!name) return;
      const r = await fetch("/api/models/alias", {
        method: "POST", headers: authHeaders(),
        body: JSON.stringify({ model: m.name, alias: name.trim() }),
      });
      const data = await r.json();
      if (r.ok) { toast(`Aliased as '${name.trim()}'`); refreshModelsPage(); }
      else toast(data.detail || "Alias failed", true);
    };
    actions.appendChild(alias);
    if (!m.active) {
      const rm = el("button", "danger", "remove");
      rm.onclick = async () => {
        if (!confirm(`Remove '${m.name}'? The file is deleted only when this is its last name and it lives in ~/.localm.`)) return;
        const r = await fetch("/api/models/remove", {
          method: "POST", headers: authHeaders(),
          body: JSON.stringify({ model: m.name }),
        });
        const data = await r.json();
        if (!r.ok) { toast(data.detail || "Remove failed", true); return; }
        const end = await streamJob(data.job_id, null);
        toast(end.status === "done" ? `Removed '${m.name}'` : "Remove failed",
              end.status !== "done");
        refreshModelsPage();
      };
      actions.appendChild(rm);
    }
    tr.appendChild(actions);
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
  box.appendChild(table);
}

async function showModelDetail(name) {
  const r = await fetch(`/v1/models/${encodeURIComponent(name)}`, {
    headers: authHeaders() });
  const data = await r.json();
  if (!r.ok) { toast(data.detail || "Lookup failed", true); return; }
  openModal("Model - " + name, (body) => {
    const rows = [
      ["Path", data.path],
      ["Source", data.source],
      ["Size", fmtSize(data.size_bytes)],
      ["SHA256", data.sha256 || "(not computed yet - hashes lazily on use)"],
      ["Aliases", data.aliases.length ? data.aliases.join(", ") : "(none)"],
      ["Status", data.active ? (data.loaded ? "active, loaded" : "active, not loaded") : "registered"],
    ];
    for (const [k, v] of rows) {
      const row = el("div", "log-entry");
      row.appendChild(el("span", "t", k));
      row.appendChild(document.createTextNode(String(v)));
      body.appendChild(row);
    }
  });
}

/* ---- model discovery (HuggingFace search + VRAM fit badges) ---- */

function fmtCount(n) {
  if (n == null) return "0";
  if (n >= 1e6) return (n / 1e6).toFixed(1) + "M";
  if (n >= 1e3) return (n / 1e3).toFixed(1) + "k";
  return String(n);
}

const FIT_TEXT = { "fits": "fits your VRAM", "tight": "tight fit",
                   "too-big": "needs partial CPU offload" };

async function discoverSearch() {
  const box = $("disc-results");
  box.replaceChildren(el("div", "sub", "searching HuggingFace…"));
  $("disc-search").disabled = true;
  try {
    const q = $("disc-query").value.trim();
    const r = await fetch("/api/discover/search?q=" + encodeURIComponent(q),
                          { headers: authHeaders() });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || r.statusText);
    $("disc-vram").textContent = data.vram.total
      ? `Badges compare each file against your ${(data.vram.total / GIB).toFixed(0)} GB total VRAM (weights + ~1.5 GB overhead).`
      : "No GPU VRAM detected - sizes shown without fit badges.";
    box.replaceChildren();
    if (!data.results.length) {
      box.appendChild(el("div", "sub", "(no GGUF repos found)"));
      return;
    }
    for (const m of data.results) {
      const row = el("div", "disc-repo");
      const head = el("div", "head");
      head.appendChild(el("span", "name", m.id));
      head.appendChild(el("span", "meta",
        `⬇ ${fmtCount(m.downloads)}  ♥ ${fmtCount(m.likes)}`));
      const btn = el("button", "", "files");
      const filesBox = el("div", "files");
      btn.onclick = () => discoverFiles(m.id, filesBox, btn);
      head.appendChild(btn);
      row.appendChild(head);
      row.appendChild(filesBox);
      box.appendChild(row);
    }
  } catch (e) {
    box.replaceChildren(el("div", "sub", "Search failed: " + e.message));
  } finally {
    $("disc-search").disabled = false;
  }
}

async function discoverFiles(repo, filesBox, btn) {
  if (filesBox.childElementCount) {            // toggle collapse
    filesBox.replaceChildren();
    return;
  }
  btn.disabled = true;
  filesBox.replaceChildren(el("div", "sub", "loading file list…"));
  try {
    const r = await fetch("/api/discover/files?repo=" + encodeURIComponent(repo),
                          { headers: authHeaders() });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || r.statusText);
    filesBox.replaceChildren();
    for (const f of data.files) {
      const row = el("div", "disc-file");
      row.appendChild(el("span", "quant", f.quant || "?"));
      const desc = `${(f.size_bytes / GIB).toFixed(1)} GB` +
        (f.n_parts > 1 ? ` (${f.n_parts} parts)` : "");
      row.appendChild(el("span", "mono", desc));
      if (f.fit) row.appendChild(el("span", "fit " + f.fit, FIT_TEXT[f.fit]));
      row.appendChild(el("span", "fname", f.file));
      const pull = el("button", "", "pull");
      pull.onclick = () => {
        // Prefill the pull form - the user confirms (and can set an alias)
        // before anything downloads. The suggested alias mirrors the
        // server's default name (file name without .gguf).
        $("pull-spec").value = `${repo}:${f.file}`;
        $("pull-name").value = f.file.replace(/\.gguf$/i, "");
        const nameInput = $("pull-name");
        nameInput.scrollIntoView({ behavior: "smooth", block: "center" });
        nameInput.focus();
        nameInput.select();
        toast("Review the alias, then click Pull to start the download");
      };
      row.appendChild(pull);
      filesBox.appendChild(row);
    }
  } catch (e) {
    filesBox.replaceChildren(el("div", "sub", "Failed: " + e.message));
  } finally {
    btn.disabled = false;
  }
}

$("disc-search").onclick = discoverSearch;
$("disc-query").addEventListener("keydown", (e) => {
  if (e.key === "Enter") discoverSearch();
});

$("pull-start").onclick = async () => {
  const spec = $("pull-spec").value.trim();
  if (!spec) { toast("Enter a model spec", true); return; }
  if (spec.startsWith("-")) {
    toast("A model spec can't start with '-'. Use owner/repo, " +
          "owner/repo:file.gguf, or an https URL.", true);
    return;
  }
  const name = $("pull-name").value.trim();
  $("pull-start").disabled = true;
  const log = $("pull-log");
  log.style.display = "block";
  log.textContent = "";
  const prog = $("pull-progress");
  const bar = $("pull-bar");
  const pct = $("pull-pct");
  // Show a live (indeterminate) bar from the start so a job that fails before
  // emitting any progress is visibly running rather than a blank panel.
  prog.style.display = "block";
  bar.classList.remove("failed");
  bar.classList.add("indeterminate");
  bar.style.width = "35%";
  pct.textContent = "starting…";
  const samples = [];   // rolling {t, downloaded} window for speed/ETA (U4)
  try {
    const r = await fetch("/api/models/pull", {
      method: "POST", headers: authHeaders(),
      body: JSON.stringify({ spec, name: name || null }),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || r.statusText);
    const end = await streamJob(data.job_id, (line) => {
      log.textContent += line + "\n";
      log.scrollTop = log.scrollHeight;
    }, (ev) => {
      if (ev.pct != null && ev.total) {
        bar.classList.remove("indeterminate");
        bar.style.width = ev.pct + "%";
        // Smoothed speed + ETA from a rolling ~10-sample window (U4).
        samples.push({ t: Date.now(), downloaded: ev.downloaded });
        if (samples.length > 10) samples.shift();
        const { bytesPerSec, etaSec } = downloadRate(samples, ev.total);
        let extra = "";
        if (bytesPerSec) extra += `  ·  ${fmtBytes(bytesPerSec)}/s`;
        if (etaSec != null) extra += `  ·  ETA ${fmtDuration(etaSec)}`;
        pct.textContent =
          `${ev.pct.toFixed(0)}%  ·  ${fmtBytes(ev.downloaded)} / ${fmtBytes(ev.total)}${extra}`;
      } else {
        // Unknown total - busy bar with a running byte count
        bar.classList.add("indeterminate");
        bar.style.width = "100%";
        pct.textContent = "downloading…  " + fmtBytes(ev.downloaded);
      }
    });
    if (end.status === "done") {
      bar.classList.remove("indeterminate");
      bar.style.width = "100%";
      pct.textContent = "done";
      toast("Pull finished");
      $("pull-spec").value = "";
      $("pull-name").value = "";
      refreshModelsPage();
    } else {
      // Surface the failure: red bar, exit code, and keep the inputs so the
      // user can see what they typed and retry.
      bar.classList.remove("indeterminate");
      bar.classList.add("failed");
      const code = end.returncode != null ? `, exit ${end.returncode}` : "";
      pct.textContent = `failed${code} - see log`;
      toast(`Pull failed (${end.status}${code}) - see log`, true);
    }
  } catch (e) {
    bar.classList.remove("indeterminate");
    bar.classList.add("failed");
    pct.textContent = "failed - see log";
    toast("Pull failed: " + e.message, true);
  } finally {
    $("pull-start").disabled = false;
  }
};

/* ================================================================ */
/*  Images page                                                      */
/* ================================================================ */

const imgState = {
  items: [],
  selected: new Set(),   // names selected for bulk actions
  showAll: false,        // grid shows 24 by default
};

const IMG_GRID_DEFAULT = 24;

async function imgApiDelete(name) {
  const r = await fetch(`/api/imagine/file/${encodeURIComponent(name)}`, {
    method: "DELETE", headers: authHeaders() });
  if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
}

async function imgApiMove(name, dest) {
  const r = await fetch(`/api/imagine/file/${encodeURIComponent(name)}/move`, {
    method: "POST", headers: authHeaders(),
    body: JSON.stringify({ dest }),
  });
  const data = await r.json();
  if (!r.ok) throw new Error(data.detail || r.statusText);
  return data.path;
}

async function imgDownload(name) {
  const url = await fetchImageURL(`/api/imagine/file/${encodeURIComponent(name)}`);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
}

async function refreshImageHistory() {
  refreshReloadToggle();
  try {
    const r = await fetch("/api/imagine/history", { headers: authHeaders() });
    if (!r.ok) return;
    imgState.items = (await r.json()).images;
  } catch (e) { return; /* server unreachable */ }
  // Drop selections for images that no longer exist
  const names = new Set(imgState.items.map((i) => i.name));
  for (const n of imgState.selected) if (!names.has(n)) imgState.selected.delete(n);
  renderImageGrid();
}

function renderImageGrid() {
  const grid = $("img-history");
  grid.replaceChildren();
  renderImgBulkBar();
  if (!imgState.items.length) {
    grid.appendChild(el("div", "sub", "No generated images yet."));
    return;
  }
  const shown = imgState.showAll
    ? imgState.items : imgState.items.slice(0, IMG_GRID_DEFAULT);
  for (const item of shown) {
    const thumb = el("div", "thumb");
    const img = document.createElement("img");
    fetchImageURL(`/api/imagine/file/${encodeURIComponent(item.name)}`)
      .then((url) => (img.src = url))
      .catch(() => thumb.remove());
    thumb.appendChild(img);

    // selection checkbox (top-left) - selected thumbs stay marked
    const sel = document.createElement("input");
    sel.type = "checkbox";
    sel.className = "thumb-sel";
    sel.checked = imgState.selected.has(item.name);
    thumb.classList.toggle("selected", sel.checked);
    sel.onclick = (e) => {
      e.stopPropagation();
      if (sel.checked) imgState.selected.add(item.name);
      else imgState.selected.delete(item.name);
      thumb.classList.toggle("selected", sel.checked);
      renderImgBulkBar();
    };
    thumb.appendChild(sel);

    // hover quick actions (top-right)
    const acts = el("div", "thumb-acts");
    const dl = el("button", "", "⤓");
    dl.title = "Download";
    dl.onclick = (e) => {
      e.stopPropagation();
      imgDownload(item.name).catch((err) => toast("Download failed: " + err.message, true));
    };
    const del = el("button", "danger", "🗑");
    del.title = "Delete from disk";
    del.onclick = async (e) => {
      e.stopPropagation();
      if (!confirm(`Delete ${item.name}? This removes the file from disk.`)) return;
      try {
        await imgApiDelete(item.name);
        toast("Deleted " + item.name);
        refreshImageHistory();
      } catch (err) { toast("Delete failed: " + err.message, true); }
    };
    acts.append(dl, del);
    thumb.appendChild(acts);

    const capText = item.meta?.prompt
      ? item.meta.prompt.slice(0, 60) : item.name;
    thumb.appendChild(el("div", "cap", capText));
    thumb.onclick = () => showImageDetail(item);
    grid.appendChild(thumb);
  }
  if (imgState.items.length > IMG_GRID_DEFAULT) {
    const toggle = el("button", "btn-quiet img-show-all",
      imgState.showAll ? "show fewer"
                       : `show all (${imgState.items.length})`);
    toggle.onclick = () => { imgState.showAll = !imgState.showAll; renderImageGrid(); };
    grid.appendChild(toggle);
  }
}

/** Bulk actions bar - appears above the grid while a selection exists. */
function renderImgBulkBar() {
  const bar = $("img-bulk");
  const n = imgState.selected.size;
  bar.style.display = n ? "flex" : "none";
  if (!n) { bar.replaceChildren(); return; }
  bar.replaceChildren();
  bar.appendChild(el("span", "count", `${n} selected`));

  const move = el("button", "btn-quiet", "move to folder…");
  move.onclick = async () => {
    const dest = await pickDirectory(`Move ${n} image(s) to…`,
      localStorage.getItem("localm.imgMoveDest") || "");
    if (!dest) return;
    let moved = 0, failed = 0;
    for (const name of [...imgState.selected]) {
      try { await imgApiMove(name, dest); moved++; }
      catch (e) { failed++; toast(`${name}: ${e.message}`, true); }
    }
    if (!chat.privacy) localStorage.setItem("localm.imgMoveDest", dest);
    toast(`Moved ${moved} image(s)` + (failed ? `, ${failed} failed` : ""));
    imgState.selected.clear();
    refreshImageHistory();
  };

  const del = el("button", "danger", "delete");
  del.onclick = async () => {
    if (!confirm(`Delete ${n} image(s) from disk?`)) return;
    let deleted = 0, failed = 0;
    for (const name of [...imgState.selected]) {
      try { await imgApiDelete(name); deleted++; }
      catch (e) { failed++; toast(`${name}: ${e.message}`, true); }
    }
    toast(`Deleted ${deleted} image(s)` + (failed ? `, ${failed} failed` : ""));
    imgState.selected.clear();
    refreshImageHistory();
  };

  const clear = el("button", "btn-quiet", "clear selection");
  clear.onclick = () => {
    imgState.selected.clear();
    renderImageGrid();
  };

  bar.append(move, del, clear);
}

function closeModal() { $("modal").style.display = "none"; }

function showImageDetail(item) {
  openModal(item.name, (body) => {
    const img = document.createElement("img");
    img.style.maxWidth = "100%";
    img.style.borderRadius = "8px";
    fetchImageURL(`/api/imagine/file/${encodeURIComponent(item.name)}`)
      .then((url) => (img.src = url));
    body.appendChild(img);

    const actions = el("div", "actions");

    const useInput = el("button", "btn-quiet", "use as input");
    useInput.title = "Use this image as the img2img input";
    useInput.onclick = () => {
      $("img-input").value = item.path || item.name;
      closeModal();
      toast("Set as img2img input - adjust denoise and generate");
    };
    actions.appendChild(useInput);

    const toChat = el("button", "btn-quiet", "send to chat");
    toChat.title = "Attach this image to the chat composer";
    toChat.onclick = async () => {
      try {
        const url = await fetchImageURL(
          `/api/imagine/file/${encodeURIComponent(item.name)}`);
        const blob = await (await fetch(url)).blob();
        const dataUri = await new Promise((resolve, reject) => {
          const reader = new FileReader();
          reader.onload = () => resolve(reader.result);
          reader.onerror = reject;
          reader.readAsDataURL(blob);
        });
        chat.attachments.push({ name: item.name, dataUri });
        renderAttachChips();
        closeModal();
        showView("chat");
        toast("Image attached - type your message");
      } catch (e) {
        toast("Attach failed: " + e.message, true);
      }
    };
    actions.appendChild(toChat);

    if (item.meta?.prompt) {
      const reuse = el("button", "btn-quiet", "reuse settings");
      reuse.title = "Fill the generation form with this image's prompt, seed, and settings";
      reuse.onclick = () => {
        $("img-prompt").value = item.meta.prompt || "";
        $("img-negative").value = item.meta.negative_prompt || "";
        $("img-seed").value = item.meta.seed ?? "";
        $("img-guidance").value = item.meta.guidance ?? "";
        $("img-denoise").value = item.meta.denoise ?? "";
        $("img-input").value = item.meta.input_image || "";
        closeModal();
        toast("Settings restored - tweak and generate");
      };
      actions.appendChild(reuse);
    }

    const dl = el("button", "btn-quiet", "download");
    dl.onclick = () =>
      imgDownload(item.name).catch((e) => toast("Download failed: " + e.message, true));
    actions.appendChild(dl);

    const copyImg = el("button", "btn-quiet", "copy image");
    copyImg.title = "Copy the image to the clipboard";
    copyImg.onclick = async () => {
      try {
        const url = await fetchImageURL(
          `/api/imagine/file/${encodeURIComponent(item.name)}`);
        const blob = await (await fetch(url)).blob();
        await navigator.clipboard.write([new ClipboardItem({ "image/png": blob })]);
        toast("Image copied to clipboard");
      } catch (e) {
        toast("Copy failed: " + e.message, true);
      }
    };
    actions.appendChild(copyImg);

    const copyPath = el("button", "btn-quiet", "copy path");
    copyPath.title = item.path || "";
    copyPath.onclick = async () => {
      try {
        await navigator.clipboard.writeText(item.path || item.name);
        toast("Path copied");
      } catch (e) { toast("Copy failed: " + e.message, true); }
    };
    actions.appendChild(copyPath);

    const rename = el("button", "btn-quiet", "rename…");
    rename.onclick = async () => {
      const newName = prompt("New name:", item.name);
      if (!newName || newName.trim() === item.name) return;
      const r = await fetch(
        `/api/imagine/file/${encodeURIComponent(item.name)}/rename`, {
          method: "POST", headers: authHeaders(),
          body: JSON.stringify({ new_name: newName.trim() }),
        });
      const data = await r.json();
      if (r.ok) {
        toast("Renamed to " + data.name);
        closeModal();
        refreshImageHistory();
      } else {
        toast(data.detail || "Rename failed", true);
      }
    };
    actions.appendChild(rename);

    const move = el("button", "btn-quiet", "move to folder…");
    move.onclick = async () => {
      const dest = await pickDirectory("Move image to…",
        localStorage.getItem("localm.imgMoveDest") || "");
      if (!dest) return;
      try {
        const path = await imgApiMove(item.name, dest);
        if (!chat.privacy) localStorage.setItem("localm.imgMoveDest", dest);
        toast("Moved to " + path);
        refreshImageHistory();
      } catch (e) {
        toast("Move failed: " + e.message, true);
      }
    };
    actions.appendChild(move);

    const del = el("button", "danger", "delete");
    del.onclick = async () => {
      if (!confirm(`Delete ${item.name}? This removes the file from disk.`)) return;
      try {
        await imgApiDelete(item.name);
        toast("Deleted " + item.name);
        closeModal();
        refreshImageHistory();
      } catch (e) {
        toast("Delete failed: " + e.message, true);
      }
    };
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

/* reload-after-generation toggle - mirrors the reload_llm_after_imagine
   server config key */

async function refreshReloadToggle() {
  try {
    const r = await fetch("/v1/config", { headers: authHeaders() });
    if (!r.ok) return;
    const cfg = await r.json();
    $("img-reload-llm").checked = cfg.reload_llm_after_imagine !== false;
  } catch (e) { /* server unreachable */ }
}

$("img-reload-llm").onchange = async () => {
  const value = $("img-reload-llm").checked;
  const r = await fetch("/v1/config", {
    method: "PATCH", headers: authHeaders(),
    body: JSON.stringify({ reload_llm_after_imagine: value }),
  });
  if (r.ok) {
    toast(value ? "Chat model reloads after each generation"
                : "ComfyUI stays loaded - chat model reloads on next message");
  } else {
    toast("Could not save setting", true);
    $("img-reload-llm").checked = !value;
  }
};

// Media-generation Stop button: reveal it while a job runs and wire it to
// cancel that job; hide it again when the job ends. Shared by image/music/video.
function showStop(btnId, jobId) {
  const btn = $(btnId);
  if (!btn) return;
  btn.style.display = "inline-block";
  btn.disabled = false;
  btn.onclick = () => { btn.disabled = true; btn.textContent = "Stopping…"; cancelJob(jobId); };
}
function hideStop(btnId) {
  const btn = $(btnId);
  if (!btn) return;
  btn.style.display = "none";
  btn.disabled = false;
  btn.textContent = "Stop";
  btn.onclick = null;
}

$("img-generate").onclick = async () => {
  const promptText = $("img-prompt").value.trim();
  if (!promptText) { toast("Enter a prompt", true); return; }
  const num = (id) => {
    const v = $(id).value.trim();
    return v === "" ? null : Number(v);
  };
  const body = {
    prompt: promptText,
    negative_prompt: $("img-negative").value.trim() || null,
    seed: num("img-seed"),
    guidance: num("img-guidance"),
    denoise: num("img-denoise"),
    input_image: $("img-input").value.trim() || null,
  };
  $("img-generate").disabled = true;
  const log = $("img-log");
  log.style.display = "block";
  log.textContent = "";
  $("img-result").replaceChildren();
  try {
    const r = await fetch("/api/imagine", {
      method: "POST", headers: authHeaders(), body: JSON.stringify(body),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || r.statusText);
    showStop("img-stop", data.job_id);
    const end = await streamJob(data.job_id, (line) => {
      log.textContent += line + "\n";
      log.scrollTop = log.scrollHeight;
    });
    if (end.status === "done" && end.result) {
      const img = document.createElement("img");
      img.src = await fetchImageURL(
        `/api/imagine/file/${encodeURIComponent(end.result)}`);
      $("img-result").appendChild(img);
      toast("Image generated");
      refreshImageHistory();
    } else {
      toast("Generation " + end.status, end.status !== "cancelled");
    }
  } catch (e) {
    toast("Generation failed: " + e.message, true);
  } finally {
    $("img-generate").disabled = false;
    hideStop("img-stop");
  }
};

/* ================================================================ */
/*  Plugins page                                                     */
/* ================================================================ */

/* First-party catalog plugins, driven by the engine (/api/plugins). Each row
   shows install/enable/disable/uninstall depending on its state; chat is the
   protected #0 and has no actions. After any change the catalog table AND the
   slash-command hint cache re-derive from one fetch (nav follows in Phase 4). */
// Each call supersedes the previous one (a newer refresh or leaving the page);
// the staggered populate below checks the token so a stale render stops. The
// per-row delay is a module var so tests can drop it to 0.
let _catalogRenderToken = 0;
let _catalogStaggerMs = 24;

function _catalogRow(p) {
  const tr = el("tr");
  const nameTd = el("td");
  nameTd.appendChild(el("span", "name", p.label || p.name));
  tr.appendChild(nameTd);
  const status = p.protected ? "protected"
    : p.active ? "active"
    : p.installed ? "installed (off)"
    : "available";
  tr.appendChild(el("td", "mono", status));
  const descTd = el("td", "", p.description);
  // Warn when a plugin needs other plugins that are not installed (B15).
  const missing = Array.isArray(p.missing_requires) ? p.missing_requires : [];
  if (missing.length) {
    descTd.appendChild(el("div", "sub plugin-missing-req",
      `requires ${(p.requires || []).join(", ")} (missing: ${missing.join(", ")})`));
  }
  tr.appendChild(descTd);
  const actions = el("td");
  actions.style.textAlign = "right";
  if (!p.protected) {
    if (!p.installed) {
      actions.appendChild(_catalogBtn("install", p.name, "btn-primary", "Install"));
    } else {
      actions.appendChild(p.enabled
        ? _catalogBtn("disable", p.name, "", "Disable")
        : _catalogBtn("enable", p.name, "btn-primary", "Enable"));
      // Re-copy this builtin from the bundled store if a localm upgrade shipped
      // newer code (the installed copy would otherwise keep shadowing it). Only
      // builtins refresh from the store; a third-party install is never a target.
      if (p.builtin) actions.appendChild(_catalogBtn("refresh", p.name, "", "Refresh"));
      actions.appendChild(_catalogBtn("uninstall", p.name, "danger", "Uninstall"));
    }
  }
  // One-click install of any missing requirements (B15).
  if (missing.length) {
    const req = el("button", "btn-primary", "Install requirements");
    req.style.marginLeft = "6px";
    req.dataset.reqfor = p.name;
    req.onclick = () => installRequirements(p.name, missing);
    actions.appendChild(req);
  }
  tr.appendChild(actions);
  return tr;
}

async function renderCatalogPlugins() {
  const box = $("catalog-table");
  if (!box) return;
  const myToken = ++_catalogRenderToken;
  box.replaceChildren();
  let data;
  try {
    const r = await fetch("/api/plugins", { headers: authHeaders() });
    if (!r.ok) throw new Error(r.statusText);
    data = await r.json();
  } catch (e) {
    box.appendChild(el("div", "sub", "Could not load plugins: " + e.message));
    return;
  }
  if (myToken !== _catalogRenderToken) return;   // superseded while fetching
  const plugins = (data.plugins || []).filter((p) => p.builtin);

  // A status line that tells the user what is loading / what loaded (U2).
  const status = el("div", "sub catalog-status");
  box.appendChild(status);

  const table = el("table", "data-table");
  const thead = el("thead");
  const hr = el("tr");
  for (const h of ["Name", "Status", "Description", ""]) hr.appendChild(el("th", "", h));
  thead.appendChild(hr);
  table.appendChild(thead);
  const tbody = el("tbody");
  table.appendChild(tbody);
  box.appendChild(table);

  // Populate the rows one after another so the catalog visibly fills in instead
  // of flashing all at once (U2). The token guard cancels a stale populate if a
  // newer refresh started or the user left the page mid-fill.
  for (let i = 0; i < plugins.length; i++) {
    if (myToken !== _catalogRenderToken) return;
    tbody.appendChild(_catalogRow(plugins[i]));
    status.textContent = `Loading plugins… (${i + 1}/${plugins.length})`;
    await new Promise((r) => setTimeout(r, _catalogStaggerMs));
  }
  if (myToken !== _catalogRenderToken) return;

  const active = plugins.filter((p) => p.active).length;
  const failed = plugins.filter((p) => p.error).length;
  status.textContent =
    `${active}/${plugins.length} plugins active` + (failed ? ` · ${failed} failed` : "");

  if (data.errors) {
    for (const [name, err] of Object.entries(data.errors)) {
      box.appendChild(el("div", "sub", `⚠ ${name}: ${err}`));
    }
  }
}

function _catalogBtn(action, name, cls, label) {
  const b = el("button", cls, label);
  b.style.marginLeft = "6px";
  b.onclick = () => pluginCatalogAction(action, name);
  return b;
}

// Install every plugin a given plugin requires but that is not yet installed
// (B15). Best-effort and sequential; each result is toasted, then the catalog
// re-renders so the warnings clear.
async function installRequirements(name, missing) {
  for (const dep of missing) {
    try {
      const r = await fetch(`/api/plugins/${encodeURIComponent(dep)}/install`,
                            { method: "POST", headers: authHeaders() });
      const d = await r.json().catch(() => ({}));
      if (!r.ok) {
        toast(`install ${dep}: ${d.detail || "failed"}`, true);
      } else {
        toast(`${dep}: ${d.status || "installed"}`);
      }
    } catch (e) {
      toast(`install ${dep} failed: ${e.message}`, true);
    }
  }
  renderCatalogPlugins();
  refreshPluginCommands();
}

async function pluginCatalogAction(action, name) {
  if (action === "uninstall" &&
      !confirm(`Uninstall '${name}'? Its plugin files are removed (your data is kept).`)) return;
  try {
    const r = await fetch(`/api/plugins/${encodeURIComponent(name)}/${action}`,
                          { method: "POST", headers: authHeaders() });
    const d = await r.json().catch(() => ({}));
    if (!r.ok) {
      const msg = d.detail || (r.status === 404 ? "no such plugin"
                 : r.status === 409 ? "not allowed in the current state"
                 : "failed");
      toast(`${action} ${name}: ${msg}`, true);
      return;
    }
    toast(`${name}: ${d.status || action}`);
  } catch (e) {
    toast(`${action} failed: ${e.message}`, true);
    return;
  }
  // Re-derive from one place: the catalog table and the slash-hint cache.
  renderCatalogPlugins();
  refreshPluginCommands();
}

async function refreshPluginsPage() {
  const box = $("plugins-table");
  box.replaceChildren();
  try {
    const r = await fetch("/v1/plugins", { headers: authHeaders() });
    if (!r.ok) throw new Error(r.statusText);
    const data = await r.json();
    if (!data.plugins.length) {
      box.appendChild(el("div", "sub", "No external plugins installed."));
    } else {
      const table = el("table", "data-table");
      const thead = el("thead");
      const hr = el("tr");
      for (const h of ["Name", "Version", "Description", "Tools", ""])
        hr.appendChild(el("th", "", h));
      thead.appendChild(hr);
      table.appendChild(thead);
      const tbody = el("tbody");
      for (const p of data.plugins) {
        const tr = el("tr");
        const nameTd = el("td");
        nameTd.appendChild(el("span", "name", p.name));
        tr.appendChild(nameTd);
        tr.appendChild(el("td", "mono", p.version));
        tr.appendChild(el("td", "", p.description));
        tr.appendChild(el("td", "mono",
          p.tool_exports.length ? p.tool_exports.join(", ") : ""));
        const actions = el("td");
        actions.style.textAlign = "right";
        const rm = el("button", "danger", "remove");
        rm.onclick = async () => {
          if (!confirm(`Remove plugin '${p.name}'? Its folder under ~/.localm/plugins/ is deleted.`)) return;
          const rr = await fetch(`/v1/plugins/${encodeURIComponent(p.name)}`, {
            method: "DELETE", headers: authHeaders() });
          const dd = await rr.json();
          if (rr.ok) { toast(`Removed '${p.name}'`); refreshPluginsPage(); }
          else toast(dd.detail || "Remove failed", true);
        };
        actions.appendChild(rm);
        tr.appendChild(actions);
        tbody.appendChild(tr);
      }
      table.appendChild(tbody);
      box.appendChild(table);
    }
    for (const err of data.errors) {
      box.appendChild(el("div", "sub", "⚠ " + err));
    }
  } catch (e) {
    box.appendChild(el("div", "sub", "Could not load plugins: " + e.message));
  }
}

$("plugin-install").onclick = async () => {
  const source = $("plugin-source").value.trim();
  if (!source) { toast("Enter the plugin folder path", true); return; }
  const r = await fetch("/v1/plugins/install", {
    method: "POST", headers: authHeaders(),
    body: JSON.stringify({ source }),
  });
  const data = await r.json();
  if (r.ok) {
    toast(`Installed '${data.name}' ${data.version} - restart localm gui to load its command`);
    $("plugin-source").value = "";
    refreshPluginsPage();
  } else {
    toast(data.detail || "Install failed", true);
  }
};

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
let _settingsControls = [];
// Monotonic token so overlapping refreshes don't both render (the old text
// dumper doubled every field when two refreshes raced; we keep the guard).
let _settingsRenderToken = 0;
// The section the user explicitly navigated to (a section element id). Survives
// re-renders so saving a section keeps you on it. Null = use the default tab.
let _activeSettingsSection = null;

/** Build one labelled control for a schema field. Returns { field, read } or
 *  null for HIDDEN fields (never rendered). */
function buildSettingControl(field) {
  if (field.widget === "hidden") return null;
  const value = field.default;     // current value (omitted for secrets)

  const wrap = el("div");
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
  // FOLDER / PATH fields get a "Browse..." button wired to the existing
  // directory picker, so the user does not have to type a path by hand (U10).
  if (field.widget === "folder" || field.widget === "path") {
    const row = el("div", "dir-picker-row");
    const browse = el("button", "btn-secondary dir-picker-btn", "Browse...");
    browse.type = "button";                 // never submit the settings form
    browse.dataset.browse = field.key;
    browse.onclick = async () => {
      const picked = await pickDirectory(
        field.widget === "path" ? "Pick a location" : "Pick a directory",
        input.value.trim());
      if (picked) { input.value = picked; input.classList.remove("auto-detected"); }
    };
    row.append(input, browse);
    wrap.appendChild(row);
  } else {
    wrap.appendChild(input);
  }
  if (field.help) wrap.appendChild(el("div", "sub", field.help));
  return { field, node: wrap, read };
}

// Fetch the server-rendered key QR (owner-scope) and show the "Pair a phone"
// block. Hidden in open mode / when no key is configured (the endpoint 404s).
async function refreshPairingQR() {
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

// Non-privileged scopes offered in the GUI key minter (label per scope). The
// /v1/keys API refuses privileged scopes (admin/keys:admin/plugins:admin/
// config:write) for a non-owner anyway; mint those from the CLI if ever needed.
const KEY_SCOPES = [
  ["coder", "Coder agent - restricted: read + edit this project (no shell)"],
  ["models:read", "List and inspect models"],
  ["models:write", "Load, download, or remove models"],
  ["rag", "Knowledge (RAG)"],
  ["chat", "Chat history"],
  ["image", "Image generation"],
  ["music", "Music generation"],
  ["video", "Video generation"],
  ["voice", "Voice"],
  ["web", "Web access"],
  ["mcp", "MCP"],
];

// Settings -> API keys: mint named, scope-limited keys, list them, revoke them.
// Owner-gated (/v1/keys needs keys:admin); the card hides for a non-owner key.
async function refreshKeysPanel() {
  const card = $("keys-card"), list = $("keys-list"), scopesBox = $("key-scopes");
  if (!card || !list || !scopesBox) return;

  if (!scopesBox.childElementCount) {           // render the checkboxes once
    for (const [scope, label] of KEY_SCOPES) {
      const lab = el("label", "key-scope");
      const cb = document.createElement("input");
      cb.type = "checkbox"; cb.value = scope; cb.className = "key-scope-cb";
      lab.appendChild(cb);
      lab.appendChild(document.createTextNode(" " + label));
      scopesBox.appendChild(lab);
    }
  }

  // The keys card is a settings SECTION; hide/show it via a class so the section
  // nav (built by buildSettingsNav) drops/re-adds its link for non-owners,
  // rather than an inline display style that would fight the section show/hide.
  const setHidden = (hidden) => {
    card.classList.toggle("sec-hidden", hidden);
    if (typeof buildSettingsNav === "function") buildSettingsNav();
  };
  let keys = [];
  try {
    const r = await fetch("/v1/keys", { headers: authHeaders() });
    if (!r.ok) { setHidden(true); return; }   // 401/403 -> not the owner
    setHidden(false);
    keys = (await r.json()).keys || [];
  } catch (e) { setHidden(true); return; }

  list.replaceChildren();
  if (!keys.length) {
    list.appendChild(el("div", "sub", "No named keys yet."));
  }
  for (const k of keys) {
    const row = el("div", "key-row");
    row.appendChild(el("span", "name", k.name || k.id));
    row.appendChild(el("span", "mono key-scope-tags", (k.scopes || []).join(", ")));
    const rm = el("button", "btn-quiet", "Revoke");
    rm.onclick = async () => {
      if (!confirm(`Revoke key "${k.name || k.id}"?`)) return;
      const d = await fetch(`/v1/keys/${encodeURIComponent(k.id)}`,
                            { method: "DELETE", headers: authHeaders() });
      if (d.ok) { toast("Key revoked"); refreshKeysPanel(); }
      else { toast("Revoke failed"); }
    };
    row.appendChild(rm);
    list.appendChild(row);
  }

  $("key-create").onclick = async () => {
    const name = ($("key-name").value || "").trim();
    if (!name) { toast("Enter a key name"); return; }
    const scopes = [...scopesBox.querySelectorAll(".key-scope-cb")]
      .filter((c) => c.checked).map((c) => c.value);
    if (!scopes.length) { toast("Pick at least one capability"); return; }
    let r;
    try {
      r = await fetch("/v1/keys", {
        method: "POST",
        headers: { ...authHeaders(), "Content-Type": "application/json" },
        body: JSON.stringify({ name, scopes }),
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
    const copy = el("button", "btn-quiet", "Copy");
    copy.onclick = () => {
      secret.select();
      if (navigator.clipboard) navigator.clipboard.writeText(made.key);
      toast("Copied");
    };
    box.appendChild(copy);
    box.style.display = "";
    $("key-name").value = "";
    scopesBox.querySelectorAll(".key-scope-cb").forEach((c) => { c.checked = false; });
    refreshKeysPanel();
  };
}

// Friendly section label per plugin owner (falls back to the capitalized scope).
const PLUGIN_SECTION_LABEL = {
  image: "Image", web: "Web access", voice: "Voice", coder: "Coder",
  abliterate: "Abliterate", music: "Music", video: "Video", rag: "Knowledge",
  mcp: "MCP", chat: "Chat",
};

/** Which settings section a field belongs to: each core `group` is its own
 *  section; each plugin (owner != core) is its own section (its own tab). */
function settingsSectionOf(field) {
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

/** Show one settings section (others hidden) and highlight its nav link. */
function showSettingsSection(secId) {
  const content = $("settings-content");
  if (!content) return;
  for (const sec of content.querySelectorAll(".settings-section")) {
    sec.classList.toggle("active", sec.id === secId);
  }
  const nav = $("settings-nav");
  if (nav) {
    for (const link of nav.querySelectorAll(".settings-nav-link")) {
      link.classList.toggle("active", link.dataset.target === secId);
    }
  }
}

/** (Re)build the left nav from every section currently in the content area,
 *  skipping any hidden by their own gating (e.g. the owner-only API keys card).
 *  The active tab is the user's explicit choice if still present, else the first
 *  config section - chosen deterministically so a stray rebuild (e.g. the
 *  owner-only keys panel resolving) can never leave a static card selected. */
function buildSettingsNav() {
  const nav = $("settings-nav"), content = $("settings-content");
  if (!nav || !content) return;
  const secs = [...content.querySelectorAll(".settings-section")]
    .filter((s) => !s.classList.contains("sec-hidden"));
  nav.replaceChildren();
  for (const sec of secs) {
    const label = sec.dataset.secLabel || sec.querySelector("h3")?.textContent || sec.id;
    const link = el("button", "settings-nav-link", label);
    link.dataset.target = sec.id;
    link.onclick = () => { _activeSettingsSection = sec.id; showSettingsSection(sec.id); };
    nav.appendChild(link);
  }
  const schema = secs.filter((s) => s.id.startsWith("settings-sec-"));
  let target = null;
  if (_activeSettingsSection && secs.some((s) => s.id === _activeSettingsSection)) {
    target = _activeSettingsSection;             // the user's chosen tab, still present
  } else if (schema.length) {
    target = schema[0].id;                        // default: first config section
  }
  if (target) showSettingsSection(target);
  else if (!content.querySelector(".settings-section.active:not(.sec-hidden)") && secs.length) {
    showSettingsSection(secs[0].id);             // nothing config-y yet: pick something
  }
}

/** Save just one section: PATCH only the keys whose controls live in it. */
async function saveSettingsSection(secId) {
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

async function refreshSettingsPage() {
  const myToken = ++_settingsRenderToken;
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
    const panel = el("section", "card settings-section");
    panel.id = "settings-sec-" + sec.id;
    panel.dataset.sec = sec.id;
    panel.dataset.secLabel = sec.label + (sec.plugin ? " plugin" : "");
    panel.appendChild(el("h3", "settings-section-head",
                         sec.label + (sec.plugin ? " plugin" : "")));
    const grid = el("div", "settings-fields");
    for (const c of sec.ctrls) grid.appendChild(c.node);
    panel.appendChild(grid);
    const actions = el("div", "actions");
    const save = el("button", "btn-primary settings-section-save", "Save " + sec.label);
    save.dataset.sec = sec.id;
    save.onclick = () => saveSettingsSection(sec.id);
    actions.appendChild(save);
    panel.appendChild(actions);
    form.appendChild(panel);
  }

  // Per-plugin Media (ComfyUI) config: one "Media" section with image/music/video
  // subsections, each editing that plugin's own block independently. Appended
  // after the core schema sections so it sits among the plugin tabs.
  await buildMediaSection(form);
  if (myToken !== _settingsRenderToken) return;  // a newer refresh superseded us

  // Build the nav now that the schema sections exist, so the first config
  // section (not a static card) is the default tab. The owner-gated panels then
  // refresh: each may rebuild the nav, but they preserve the active section.
  buildSettingsNav();
  refreshPairingQR();
  refreshKeysPanel();
}

// Media plugins, in display order, that the Media section configures.
const MEDIA_PLUGIN_ORDER = ["image", "music", "video"];

/** Did a media control's value change from what was displayed? Treats
 *  null/undefined/"" as the same "empty", so saving an untouched inherited field
 *  does not pin it as an override. */
function _mediaChanged(cur, orig) {
  const empty = (v) => v === null || v === undefined || v === "";
  if (empty(cur) && empty(orig)) return false;
  return cur !== orig;
}

/** Build the "Media" settings section: one subsection per media plugin
 *  (image/music/video), each editing that plugin's own ComfyUI config block via
 *  /v1/media/config. A field left at its inherited value is not sent, so the
 *  plugin keeps falling back to the shared default until the user overrides it. */
async function buildMediaSection(form) {
  let data;
  try {
    const r = await fetch("/v1/media/config", { headers: authHeaders() });
    if (!r.ok) throw new Error(r.statusText);
    data = await r.json();
  } catch (e) {
    return;   // media config unavailable; skip the section (do not break settings)
  }
  const byName = {};
  for (const p of (data.plugins || [])) byName[p.plugin] = p;

  const panel = el("section", "card settings-section");
  panel.id = "settings-sec-media";
  panel.dataset.sec = "media";
  panel.dataset.secLabel = "Media";
  panel.appendChild(el("h3", "settings-section-head", "Media"));
  panel.appendChild(el("div", "sub",
    "ComfyUI settings for image, music, and video, each configured "
    + "independently. A blank field uses the shared default."));

  for (const name of MEDIA_PLUGIN_ORDER) {
    const p = byName[name];
    if (!p) continue;
    const sub = el("div", "media-subsection");
    sub.appendChild(el("h4", "media-sub-head", p.label));
    const grid = el("div", "settings-fields");
    const controls = [];
    for (const f of (p.fields || [])) {
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
    sub.appendChild(grid);
    const actions = el("div", "actions");
    const save = el("button", "btn-primary", "Save " + p.label);
    save.onclick = () => saveMediaPlugin(p.plugin, controls);
    actions.appendChild(save);
    sub.appendChild(actions);
    panel.appendChild(sub);
  }
  form.appendChild(panel);
}

/** Save one media plugin's block: POST only the fields the user changed (so an
 *  untouched inherited field is not pinned), then re-render. */
async function saveMediaPlugin(name, controls) {
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
    refreshSettingsPage();
  } else {
    toast(data.detail || "Save failed", true);
  }
}

/* ================================================================ */
/*  Per-plugin workflow management (on the Image/Music/Video pages)   */
/* ================================================================ */

/** Render the workflow panel for a media plugin: the built-in default plus each
 *  uploaded workflow, with select + delete, and an upload control. */
async function refreshWorkflowPanel(media) {
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

function workflowRow(media, name, label, active, deletable) {
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

async function selectWorkflow(media, name) {
  const r = await fetch(`/api/${media}/workflows/select`, {
    method: "POST", headers: authHeaders(), body: JSON.stringify({ name }),
  });
  if (r.ok) { toast("Workflow selected"); refreshWorkflowPanel(media); }
  else toast((await r.json().catch(() => ({}))).detail || "Failed", true);
}

async function deleteWorkflow(media, name) {
  if (!confirm(`Delete workflow "${name}"?`)) return;
  const r = await fetch(`/api/${media}/workflows/${encodeURIComponent(name)}`, {
    method: "DELETE", headers: authHeaders(),
  });
  if (r.ok) { toast("Deleted"); refreshWorkflowPanel(media); }
  else toast((await r.json().catch(() => ({}))).detail || "Failed", true);
}

async function uploadWorkflow(media, fileInput) {
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

$("gui-key-save").onclick = async () => {
  const key = $("gui-api-key").value.trim();
  if (key) {
    await loginWithKey(key);   // POST /api/session -> server sets the HttpOnly cookie
  } else {
    // Empty -> sign out (clear the session cookie).
    try {
      await fetch("/api/session/logout", { method: "POST", headers: authHeaders() });
    } catch (e) { /* offline / already cleared */ }
  }
  toast("Key saved - reloading");
  setTimeout(() => location.reload(), 600);
};

/* ================================================================ */
/*  Music page                                                       */
/* ================================================================ */

$("music-generate").onclick = async () => {
  const tags = $("music-tags").value.trim();
  if (!tags) { toast("Enter style tags first", true); return; }
  const body = { tags };
  const lyrics = $("music-lyrics").value.trim();
  if (lyrics) body.lyrics = lyrics;
  const duration = Number($("music-duration").value);
  if (duration > 0) body.duration_seconds = duration;
  for (const [field, id] of [["seed", "music-seed"], ["steps", "music-steps"],
                             ["cfg", "music-cfg"]]) {
    const v = $(id).value.trim();
    if (v !== "" && !Number.isNaN(Number(v))) body[field] = Number(v);
  }

  $("music-generate").disabled = true;
  const log = $("music-log");
  log.style.display = "block";
  log.textContent = "";
  $("music-result").replaceChildren();
  try {
    const r = await fetch("/api/music", {
      method: "POST", headers: authHeaders(), body: JSON.stringify(body),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || r.statusText);
    showStop("music-stop", data.job_id);
    const end = await streamJob(data.job_id, (line) => {
      log.textContent += line + "\n";
      log.scrollTop = log.scrollHeight;
    });
    if (end.status === "done" && end.result) {
      toast("Track finished");
      const player = document.createElement("audio");
      player.controls = true;
      player.style.width = "100%";
      const url = await fetchImageURL(
        "/api/music/file/" + encodeURIComponent(end.result));
      player.src = url;
      $("music-result").appendChild(player);
      refreshMusicHistory();
    } else {
      toast("Generation " + end.status, end.status !== "cancelled");
    }
  } catch (e) {
    toast("Music generation failed: " + e.message, true);
  } finally {
    $("music-generate").disabled = false;
    hideStop("music-stop");
  }
};

/* ================================================================ */
/*  Video page                                                       */
/* ================================================================ */

$("video-generate").onclick = async () => {
  const promptText = $("video-prompt").value.trim();
  if (!promptText) { toast("Enter a prompt first", true); return; }
  const body = { prompt: promptText };
  const negative = $("video-negative").value.trim();
  if (negative) body.negative_prompt = negative;
  const image = $("video-image").value.trim();
  if (image) body.input_image = image;
  for (const [field, id] of [["seconds", "video-seconds"], ["fps", "video-fps"],
                             ["width", "video-width"], ["height", "video-height"],
                             ["seed", "video-seed"], ["steps", "video-steps"],
                             ["cfg", "video-cfg"]]) {
    const v = $(id).value.trim();
    if (v !== "" && !Number.isNaN(Number(v))) body[field] = Number(v);
  }

  $("video-generate").disabled = true;
  const log = $("video-log");
  log.style.display = "block";
  log.textContent = "";
  $("video-result").replaceChildren();
  try {
    const r = await fetch("/api/video", {
      method: "POST", headers: authHeaders(), body: JSON.stringify(body),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || r.statusText);
    showStop("video-stop", data.job_id);
    const end = await streamJob(data.job_id, (line) => {
      log.textContent += line + "\n";
      log.scrollTop = log.scrollHeight;
    });
    if (end.status === "done" && end.result) {
      toast("Clip finished");
      const player = document.createElement("video");
      player.controls = true;
      player.style.width = "100%";
      const url = await fetchImageURL(
        "/api/video/file/" + encodeURIComponent(end.result));
      player.src = url;
      $("video-result").appendChild(player);
      refreshVideoHistory();
    } else {
      toast("Generation " + end.status, end.status !== "cancelled");
    }
  } catch (e) {
    toast("Video generation failed: " + e.message, true);
  } finally {
    $("video-generate").disabled = false;
    hideStop("video-stop");
  }
};

async function refreshVideoHistory() {
  const box = $("video-history");
  box.replaceChildren();
  let data;
  try {
    const r = await fetch("/api/video/history", { headers: authHeaders() });
    if (!r.ok) throw new Error(r.statusText);
    data = await r.json();
  } catch (e) {
    box.appendChild(el("div", "sub", "Could not load history: " + e.message));
    return;
  }
  if (!data.videos.length) {
    box.appendChild(el("div", "sub", "No clips yet - generate one above."));
    return;
  }
  for (const item of data.videos) {
    const row = el("div", "disc-repo");
    const head = el("div", "head");
    head.appendChild(el("span", "name", item.name));
    const bits = [];
    if (item.meta?.prompt) bits.push(item.meta.prompt.slice(0, 60));
    if (item.meta?.seconds) bits.push(`${item.meta.seconds}s`);
    bits.push(`${(item.size_bytes / MIB).toFixed(1)} MB`);
    head.appendChild(el("span", "meta", bits.join(" · ")));

    const play = el("button", "", "play");
    let player = null;
    play.onclick = async () => {
      if (player) { player.remove(); player = null; play.textContent = "play"; return; }
      play.disabled = true;
      try {
        const url = await fetchImageURL(
          "/api/video/file/" + encodeURIComponent(item.name));
        player = document.createElement("video");
        player.controls = true;
        player.autoplay = true;
        player.style.width = "100%";
        player.src = url;
        row.appendChild(player);
        play.textContent = "hide";
      } catch (e) {
        toast("Could not load clip: " + e.message, true);
      } finally {
        play.disabled = false;
      }
    };
    head.appendChild(play);

    const move = el("button", "", "move…");
    move.onclick = async () => {
      const dest = prompt("Destination folder:",
        localStorage.getItem("localm.videoMoveDest") || "");
      if (!dest) return;
      const r = await fetch(
        `/api/video/file/${encodeURIComponent(item.name)}/move`, {
          method: "POST", headers: authHeaders(),
          body: JSON.stringify({ dest: dest.trim() }),
        });
      const data2 = await r.json();
      if (r.ok) {
        if (!chat.privacy) localStorage.setItem("localm.videoMoveDest", dest.trim());
        toast("Moved to " + data2.path);
        refreshVideoHistory();
      } else {
        toast(data2.detail || "Move failed", true);
      }
    };
    head.appendChild(move);

    const del = el("button", "danger", "delete");
    del.onclick = async () => {
      if (!confirm(`Delete ${item.name}?`)) return;
      const r = await fetch("/api/video/file/" + encodeURIComponent(item.name),
                            { method: "DELETE", headers: authHeaders() });
      if (r.ok) { toast("Deleted"); refreshVideoHistory(); }
      else toast("Delete failed", true);
    };
    head.appendChild(del);

    row.appendChild(head);
    box.appendChild(row);
  }
}

async function refreshMusicHistory() {
  const box = $("music-history");
  box.replaceChildren();
  let data;
  try {
    const r = await fetch("/api/music/history", { headers: authHeaders() });
    if (!r.ok) throw new Error(r.statusText);
    data = await r.json();
  } catch (e) {
    box.appendChild(el("div", "sub", "Could not load history: " + e.message));
    return;
  }
  if (!data.tracks.length) {
    box.appendChild(el("div", "sub", "No tracks yet - generate one above."));
    return;
  }
  for (const item of data.tracks) {
    const row = el("div", "disc-repo");
    const head = el("div", "head");
    head.appendChild(el("span", "name", item.name));
    const bits = [];
    if (item.meta?.tags) bits.push(item.meta.tags.slice(0, 60));
    if (item.meta?.duration_seconds) bits.push(`${item.meta.duration_seconds}s`);
    bits.push(`${(item.size_bytes / MIB).toFixed(1)} MB`);
    head.appendChild(el("span", "meta", bits.join(" · ")));

    const play = el("button", "", "play");
    let player = null;
    play.onclick = async () => {
      if (player) { player.remove(); player = null; play.textContent = "play"; return; }
      play.disabled = true;
      try {
        const url = await fetchImageURL(
          "/api/music/file/" + encodeURIComponent(item.name));
        player = document.createElement("audio");
        player.controls = true;
        player.autoplay = true;
        player.style.width = "100%";
        player.src = url;
        row.appendChild(player);
        play.textContent = "hide";
      } catch (e) {
        toast("Could not load track: " + e.message, true);
      } finally {
        play.disabled = false;
      }
    };
    head.appendChild(play);

    const move = el("button", "", "move…");
    move.onclick = async () => {
      const dest = prompt("Destination folder:",
        localStorage.getItem("localm.musicMoveDest") || "");
      if (!dest) return;
      const r = await fetch(
        `/api/music/file/${encodeURIComponent(item.name)}/move`, {
          method: "POST", headers: authHeaders(),
          body: JSON.stringify({ dest: dest.trim() }),
        });
      const data2 = await r.json();
      if (r.ok) {
        if (!chat.privacy) localStorage.setItem("localm.musicMoveDest", dest.trim());
        toast("Moved to " + data2.path);
        refreshMusicHistory();
      } else {
        toast(data2.detail || "Move failed", true);
      }
    };
    head.appendChild(move);

    const del = el("button", "danger", "delete");
    del.onclick = async () => {
      if (!confirm(`Delete ${item.name}?`)) return;
      const r = await fetch("/api/music/file/" + encodeURIComponent(item.name),
                            { method: "DELETE", headers: authHeaders() });
      if (r.ok) { toast("Deleted"); refreshMusicHistory(); }
      else toast("Delete failed", true);
    };
    head.appendChild(del);

    row.appendChild(head);
    box.appendChild(row);
  }
}

/* ================================================================ */
/*  Knowledge page                                                   */
/* ================================================================ */

async function refreshKnowledgePage() {
  refreshKbSelect();   // keep the chat drawer selector in sync
  const box = $("kb-table");
  box.replaceChildren();
  let data;
  try {
    const r = await fetch("/api/rag/collections", { headers: authHeaders() });
    if (!r.ok) throw new Error(r.statusText);
    data = await r.json();
  } catch (e) {
    box.appendChild(el("div", "sub", "Could not load collections: " + e.message));
    return;
  }
  if (!data.collections.length) {
    box.appendChild(el("div", "sub",
      "No collections yet - create one above, then add files or folders to it."));
    return;
  }
  const table = el("table", "data-table");
  const thead = el("thead");
  const hr = el("tr");
  for (const h of ["Name", "Docs", "Chunks", "Retrieval", ""]) {
    hr.appendChild(el("th", "", h));
  }
  thead.appendChild(hr);
  table.appendChild(thead);
  const tbody = el("tbody");
  for (const c of data.collections) {
    const tr = el("tr");
    const nameTd = el("td");
    nameTd.appendChild(el("span", "name", c.name));
    tr.appendChild(nameTd);
    tr.appendChild(el("td", "mono", String(c.n_docs)));
    tr.appendChild(el("td", "mono", String(c.n_chunks)));
    tr.appendChild(el("td", "mono", c.has_vectors ? "hybrid" : "BM25"));
    const actions = el("td");
    actions.style.textAlign = "right";

    const add = el("button", "", "add docs");
    add.onclick = () => kbAddDocs(c.name);
    actions.appendChild(add);

    const search = el("button", "", "search");
    search.onclick = () => kbSearchModal(c.name);
    actions.appendChild(search);

    const info = el("button", "", "info");
    info.onclick = () => kbInfoModal(c.name);
    actions.appendChild(info);

    const del = el("button", "danger", "delete");
    del.onclick = async () => {
      if (!confirm(`Delete collection '${c.name}'? Only the index is removed - ` +
                   "your original files are untouched.")) return;
      const r = await fetch(
        "/api/rag/collections/" + encodeURIComponent(c.name), {
          method: "DELETE", headers: authHeaders() });
      if (r.ok) { toast("Deleted " + c.name); refreshKnowledgePage(); }
      else toast("Delete failed", true);
    };
    actions.appendChild(del);
    tr.appendChild(actions);
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
  box.appendChild(table);
}

$("kb-create").onclick = async () => {
  const name = $("kb-name").value.trim();
  if (!name) { toast("Enter a collection name", true); return; }
  const r = await fetch("/api/rag/collections", {
    method: "POST", headers: authHeaders(),
    body: JSON.stringify({ name }),
  });
  const data = await r.json();
  if (!r.ok) { toast(data.detail || "Create failed", true); return; }
  $("kb-name").value = "";
  toast(`Collection '${data.name}' created - now add files or folders`);
  refreshKnowledgePage();
};

async function kbAddDocs(name) {
  const path = prompt(
    `Add documents to '${name}' - file or folder path on this machine\n` +
    "(folders are indexed recursively; txt/md/pdf/docx/html/code):",
    localStorage.getItem("localm.kbAddPath") || "");
  if (!path || !path.trim()) return;
  if (!chat.privacy) localStorage.setItem("localm.kbAddPath", path.trim());
  const log = $("kb-log");
  log.style.display = "block";
  log.textContent = `Indexing ${path.trim()} into '${name}'…\n`;
  try {
    const r = await fetch(
      `/api/rag/collections/${encodeURIComponent(name)}/add`, {
        method: "POST", headers: authHeaders(),
        body: JSON.stringify({ paths: [path.trim()] }),
      });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || r.statusText);
    const end = await streamJob(data.job_id, (line) => {
      log.textContent += line + "\n";
      log.scrollTop = log.scrollHeight;
    });
    toast(end.status === "done" ? "Indexing finished"
                                : "Indexing " + end.status,
          end.status !== "done");
    refreshKnowledgePage();
  } catch (e) {
    log.textContent += "failed: " + e.message + "\n";
    toast("Indexing failed: " + e.message, true);
  }
}

async function kbInfoModal(name) {
  const r = await fetch("/api/rag/collections/" + encodeURIComponent(name),
                        { headers: authHeaders() });
  const data = await r.json();
  if (!r.ok) { toast(data.detail || "Load failed", true); return; }
  openModal("Collection - " + name, (body) => {
    body.appendChild(el("div", "sub",
      `${data.n_docs} documents · ${data.n_chunks} chunks · ` +
      (data.has_vectors ? "hybrid retrieval (BM25 + embeddings)"
                        : "lexical retrieval (BM25)")));
    for (const d of data.docs) {
      const row = el("div", "log-entry");
      row.appendChild(el("span", "t", `${d.chunks} chunks`));
      row.appendChild(document.createTextNode(d.path + " "));
      const rm = el("button", "action", "remove");
      rm.onclick = async () => {
        const rr = await fetch(
          `/api/rag/collections/${encodeURIComponent(name)}/remove-doc`, {
            method: "POST", headers: authHeaders(),
            body: JSON.stringify({ path: d.path }),
          });
        if (rr.ok) { toast("Removed"); kbInfoModal(name); refreshKnowledgePage(); }
        else toast("Remove failed", true);
      };
      row.appendChild(rm);
      body.appendChild(row);
    }
    if (!data.docs.length) {
      body.appendChild(el("div", "sub", "(empty - use add docs)"));
    }
  });
}

function kbSearchModal(name) {
  openModal("Search - " + name, (body) => {
    const row = el("div", "row");
    const input = document.createElement("input");
    input.type = "text";
    input.placeholder = "test query…";
    input.style.flex = "1";
    const go = el("button", "btn-primary", "Search");
    row.appendChild(input);
    row.appendChild(go);
    body.appendChild(row);
    const results = el("div");
    body.appendChild(results);
    const run = async () => {
      const q = input.value.trim();
      if (!q) return;
      results.replaceChildren(el("div", "sub", "searching…"));
      try {
        const r = await fetch(
          `/api/rag/collections/${encodeURIComponent(name)}/query`, {
            method: "POST", headers: authHeaders(),
            body: JSON.stringify({ query: q, k: 5 }),
          });
        const data = await r.json();
        if (!r.ok) throw new Error(data.detail || r.statusText);
        results.replaceChildren();
        if (!data.hits.length) {
          results.appendChild(el("div", "sub", "(no matches)"));
          return;
        }
        for (const h of data.hits) {
          const entry = el("div", "log-entry");
          entry.appendChild(el("span", "t",
            `${h.score}  ${h.source.split(/[\\/]/).pop()}:${h.pos}`));
          entry.appendChild(document.createTextNode(h.text.slice(0, 400)));
          results.appendChild(entry);
        }
      } catch (e) {
        results.replaceChildren(el("div", "sub", "failed: " + e.message));
      }
    };
    go.onclick = run;
    input.onkeydown = (e) => { if (e.key === "Enter") run(); };
    input.focus();
  });
}

$("gui-clear-convs").onclick = async () => {
  const where = chat.persist
    ? "from this browser AND the server store" : "from this browser";
  if (!confirm(`Delete all saved conversations ${where}?`)) return;
  if (chat.persist) {
    // Server store is the source of truth - clear it too or they come back.
    await Promise.allSettled(chat.conversations.map((c) =>
      fetch("/api/conversations/" + encodeURIComponent(c.id), {
        method: "DELETE", headers: authHeaders(),
      })));
  }
  localStorage.removeItem("localm.conversations");
  location.reload();
};
