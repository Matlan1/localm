// SPDX-License-Identifier: AGPL-3.0-or-later
/* localm GUI - Models page (split from pages.js). Classic script: it
   shares the one global lexical environment with app.js and the other
   page scripts, so the helpers it uses ($, el, authHeaders, toast, ...)
   resolve by bare name exactly as before. */
"use strict";

// --- ES module imports (auto-generated boundary; bodies unchanged) ---
import { pickDirectory } from "../app/picker.js";
import { $, GIB, authHeaders, confirmDanger, downloadRate, el, fmtBytes, fmtDuration, openModal, renderMarkdown, streamJob, toast } from "../app/helpers.js";
import { onServerUnreachable } from "../app/init.js";
import { emptyState } from "../app/icons.js";
import { modelCache, refreshModels, switchModel } from "../app/models-sidebar.js";
import { refreshPerfEstimate } from "../app/settings-perf.js";

/* ================================================================ */
/*  Models page                                                      */
/* ================================================================ */

export function fmtSize(bytes) {
  if (bytes == null) return "";
  return (bytes / GIB).toFixed(2) + " GB";   // binary GiB, labelled GB (see app.js)
}

let currentTypeFilter = "all";

export async function refreshModelsPage() {
  await refreshModels();
  
  // Toggle the search box vs placeholder based on selected tab type
  const searchBox = $("disc-search-box");
  const placeholder = $("disc-placeholder");
  if (searchBox && placeholder) {
    if (currentTypeFilter === "all" || currentTypeFilter === "llm") {
      searchBox.style.display = "block";
      placeholder.style.display = "none";
    } else {
      searchBox.style.display = "none";
      placeholder.style.display = "block";
    }
  }

  // Show scan row on media model tabs
  const scanRow = $("models-scan-row");
  if (scanRow) {
    scanRow.style.display = (currentTypeFilter !== "all" && currentTypeFilter !== "llm") ? "flex" : "none";
  }

  const box = $("models-table");
  box.replaceChildren();

  const typeParam = currentTypeFilter === "all" ? "" : "?type=" + currentTypeFilter;
  let models = [];
  try {
    const r = await fetch("/api/models" + typeParam, { headers: authHeaders() });
    const data = await r.json();
    models = (data && Array.isArray(data.models)) ? data.models : [];
  } catch (e) {
    box.appendChild(el("div", "sub", "Error loading models: " + e.message));
    return;
  }

  if (!models.length) {
    box.appendChild(emptyState("models", "No models yet",
      "Pull a model above, or search HuggingFace to add your first one."));
    return;
  }

  const table = el("table", "data-table");
  const thead = el("thead");
  const hr = el("tr");
  for (const h of ["Name", "Role", "Source", "Size", ""]) hr.appendChild(el("th", "", h));
  thead.appendChild(hr);
  table.appendChild(thead);
  const tbody = el("tbody");
  // Selectable model types - mirrors localm.model_manager.registry.MODEL_TYPES
  // (keep in sync). Powers the per-row one-click set-type control.
  const MODEL_TYPE_OPTIONS =
    ["llm", "mmproj", "diffusion-unet", "text-encoder", "vae", "lora", "unknown"];
  for (const m of models) {
    const tr = el("tr");
    const nameTd = el("td", "name-cell");
    nameTd.appendChild(iconEl("models", "ic ic-model"));
    nameTd.appendChild(el("span", "name", m.name));
    if (m.active) nameTd.appendChild(el("span", "active-tag", "active"));
    // Independent of "active": a model can sit resident in VRAM without being
    // the one currently serving requests - surfaced so a background-loaded
    // model is never invisible/indistinguishable from one that was never
    // loaded at all.
    else if (m.loaded) nameTd.appendChild(el("span", "active-tag loaded-tag", "loaded"));
    tr.appendChild(nameTd);
    
    // Role column (using job-state badge layout). An 'unknown'-type model (its
    // type could not be determined) is highlighted so it stands out as needing a
    // type set via the control below.
    const roleTd = el("td", "mono");
    const roleClass = "job-state" + (m.model_type === "unknown" ? " st-unknown" : "");
    roleTd.appendChild(el("span", roleClass, m.model_type || "llm"));
    tr.appendChild(roleTd);
    
    tr.appendChild(el("td", "mono", m.source || ""));
    tr.appendChild(el("td", "mono", fmtSize(m.size_bytes)));

    const actions = el("td");
    actions.style.textAlign = "right";
    const detail = el("button", "", "info");
    detail.onclick = () => showModelDetail(m.name);
    actions.appendChild(detail);

    // One-click set-type control on every row (outside the LLM-only gate below):
    // an 'unknown'/media model otherwise has no controls, and this is how a
    // mis-detected model is reclassified without the CLI. Changing it POSTs the
    // chosen type, then re-renders (so an unknown->llm switch reveals use/alias).
    const typeSel = el("select", "model-type-select");
    typeSel.title = "Change this model's type";
    for (const t of MODEL_TYPE_OPTIONS) {
      const opt = el("option", "", t);
      opt.value = t;
      if ((m.model_type || "llm") === t) opt.selected = true;
      typeSel.appendChild(opt);
    }
    typeSel.onchange = async () => {
      typeSel.disabled = true;
      try {
        const r = await fetch("/api/models/type", {
          method: "POST", headers: authHeaders(),
          body: JSON.stringify({ model: m.name, model_type: typeSel.value }),
        });
        const data = await r.json().catch(() => ({}));
        if (!r.ok) { toast(data.detail || "Set type failed", true); return; }
        toast(`Set '${m.name}' type to ${typeSel.value}`);
        refreshModelsPage();
        refreshPerfEstimate();
      } finally { typeSel.disabled = false; }
    };
    actions.appendChild(typeSel);

    // Only LLMs support use/alias/remove in Phase 1
    const isLlm = !m.model_type || m.model_type === "llm";
    if (isLlm) {
      if (!m.active) {
        const use = el("button", "primary", "use");
        use.onclick = async () => {
          use.disabled = true;
          try {
            const res = await switchModel(m.name);
            if (!res || res.status !== "superseded") {
              toast("Model switched to " + m.name);
              refreshModelsPage();
              // Keep the Settings "Live tuning" VRAM estimate (which defaults to
              // the active model) in sync with a switch made from this page too.
              refreshPerfEstimate();
            }
          } catch (e) {
            toast("Load failed: " + e.message, true);
          } finally { use.disabled = false; }
        };
        actions.appendChild(use);
      }
      const aliasBtn = el("button", "", "alias");
      aliasBtn.onclick = async () => {
        const name = prompt(`New alias for '${m.name}':`);
        if (!name) return;
        const r = await fetch("/api/models/alias", {
          method: "POST", headers: authHeaders(),
          body: JSON.stringify({ model: m.name, alias: name.trim() }),
        });
        const data = await r.json().catch(() => ({}));
        if (r.ok) { toast(`Aliased as '${name.trim()}'`); refreshModelsPage(); }
        else toast(data.detail || "Alias failed", true);
      };
      actions.appendChild(aliasBtn);
      if (m.loaded) {
        const unload = el("button", "", "unload");
        unload.title = "Release this model from GPU/CPU memory (it reloads "
          + "automatically on the next chat request)";
        unload.onclick = async () => {
          unload.disabled = true;
          try {
            const r = await fetch("/api/models/unload", {
              method: "POST", headers: authHeaders(),
              body: JSON.stringify({ model: m.name }),
            });
            const data = await r.json().catch(() => ({}));
            if (!r.ok) { toast(data.detail || "Unload failed", true); return; }
            toast(`Unloaded '${m.name}'`);
            refreshModelsPage();
            refreshPerfEstimate();
          } finally { unload.disabled = false; }
        };
        actions.appendChild(unload);
      }
      if (!m.active) {
        const rm = el("button", "danger", "remove");
        rm.onclick = async () => {
          if (!confirm(`Remove '${m.name}'? The file is deleted only when this is its last name and it lives in the data directory.`)) return;
          const r = await fetch("/api/models/remove", {
            method: "POST", headers: authHeaders(),
            body: JSON.stringify({ model: m.name }),
          });
          const data = await r.json().catch(() => ({}));
          if (!r.ok) { toast(data.detail || "Remove failed", true); return; }
          const end = await streamJob(data.job_id, null);
          toast(end.status === "done" ? `Removed '${m.name}'` : "Remove failed",
                end.status !== "done");
          refreshModelsPage();
        };
        actions.appendChild(rm);
      }
    }
    tr.appendChild(actions);
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
  box.appendChild(table);
}

export async function showModelDetail(name) {
  const r = await fetch(`/v1/models/${encodeURIComponent(name)}`, {
    headers: authHeaders() });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) { toast(data.detail || "Lookup failed", true); return; }
  openModal("Model - " + name, (body) => {
    const rows = [
      ["Path", data.path],
      ["Type", data.model_type || "llm"],
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

export function fmtCount(n) {
  if (n == null) return "0";
  if (n >= 1e6) return (n / 1e6).toFixed(1) + "M";
  if (n >= 1e3) return (n / 1e3).toFixed(1) + "k";
  return String(n);
}

export const FIT_TEXT = { "fits": "fits your VRAM", "tight": "tight fit",
                   "too-big": "needs partial CPU offload" };

export const FMT_LABEL = { gguf: "GGUF", hf: "HF" };

export function discFormats() {
  // The search-page model-type toggles -> the formats list. Defaults to gguf
  // when the checkboxes are absent (older DOM) so search never silently returns
  // nothing. Empty (both unchecked) is a real state the caller handles.
  const g = $("disc-fmt-gguf"), h = $("disc-fmt-hf");
  if (!g && !h) return ["gguf"];
  const out = [];
  if (!g || g.checked) out.push("gguf");
  if (h && h.checked) out.push("hf");
  return out;
}

function showHfHint() {
  const hint = $("disc-hf-hint");
  if (!hint) return;
  // Non-blocking: HF (transformers) models still DOWNLOAD, they just cannot RUN
  // until torch + transformers are installed. State that, do not block the pull.
  hint.textContent = "No transformers runtime detected. HF models will download, "
    + "but need the [gpu] extra (torch + transformers) installed to run.";
  hint.style.display = "block";
}
function hideHfHint() {
  const hint = $("disc-hf-hint");
  if (hint) hint.style.display = "none";
}

// A bare owner/repo (no :file.gguf) tells `localm pull` to fetch the WHOLE
// transformers repo -> the HF backend. Prefill the Add box and let the user
// confirm (they can still just download the files, backend or not).
function prefillHfPull(repo) {
  $("pull-spec").value = repo;
  $("pull-name").value = repo.split("/").pop();
  const mmprojSelect = $("pull-mmproj");
  if (mmprojSelect) { mmprojSelect.replaceChildren(); mmprojSelect.style.display = "none"; }
  const nameInput = $("pull-name");
  // scrollIntoView is absent in some environments (e.g. jsdom); guard so the
  // prefill never throws where it is unimplemented. Browsers have it.
  if (typeof nameInput.scrollIntoView === "function") {
    nameInput.scrollIntoView({ behavior: "smooth", block: "center" });
  }
  nameInput.focus();
  nameInput.select();
  toast("Review the alias, then click Add to download the full HF model");
}

// One search-result repo: name + per-format badge(s) + the right pull affordance
// (GGUF -> a per-quant file list; HF -> a whole-repo add). A repo tagged with
// both formats gets both.
function discRepoRow(m) {
  const row = el("div", "disc-repo");
  const head = el("div", "head");
  head.appendChild(iconEl("models", "ic ic-model"));
  head.appendChild(el("span", "name", m.id));
  const fmts = Array.isArray(m.formats) ? m.formats : ["gguf"];
  for (const f of fmts) head.appendChild(el("span", "fmt-badge fmt-" + f, FMT_LABEL[f] || f));
  // HF repos pull whole, so show total size + a VRAM fit badge inline (from the
  // server's safetensors param estimate). When the estimate is unknown (no
  // safetensors metadata) say so - never guess a fit. GGUF results are sized
  // per-quant in the files expander instead.
  if (fmts.includes("hf")) {
    if (m.size_bytes) {
      head.appendChild(el("span", "disc-hf-size", fmtSize(m.size_bytes)));
      if (m.fit) head.appendChild(el("span", "fit " + m.fit, FIT_TEXT[m.fit]));
    } else {
      head.appendChild(el("span", "disc-hf-size sub", "size unknown"));
    }
  }
  // Downloads + likes as inline SVGs (no emoji glyphs on the shipping surface).
  const meta = el("span", "meta disc-stats");
  meta.appendChild(iconEl("download", "meta-ic"));
  meta.appendChild(el("span", "", fmtCount(m.downloads)));
  meta.appendChild(iconEl("heart", "meta-ic"));
  meta.appendChild(el("span", "", fmtCount(m.likes)));
  head.appendChild(meta);
  const filesBox = el("div", "files");
  if (fmts.includes("gguf")) {
    const btn = el("button", "btn-secondary", "files");
    btn.onclick = () => discoverFiles(m.id, filesBox, btn);
    head.appendChild(btn);
  }
  if (fmts.includes("hf")) {
    const btn = el("button", "btn-secondary", "add full repo");
    btn.onclick = () => prefillHfPull(m.id);
    head.appendChild(btn);
  }
  row.appendChild(head);
  row.appendChild(filesBox);
  return row;
}

export async function discoverSearch() {
  const box = $("disc-results");
  const formats = discFormats();
  if (!formats.length) {
    box.replaceChildren(el("div", "sub",
      "Select at least one model type (GGUF or HF) to search."));
    hideHfHint();
    return;
  }
  box.replaceChildren(el("div", "sub", "searching HuggingFace…"));
  $("disc-search").disabled = true;
  try {
    const q = $("disc-query").value.trim();
    const r = await fetch("/api/discover/search?q=" + encodeURIComponent(q)
                          + "&formats=" + encodeURIComponent(formats.join(",")),
                          { headers: authHeaders() });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || r.statusText);
    $("disc-vram").textContent = data.vram.total
      ? `Badges compare each file against your ${(data.vram.total / GIB).toFixed(0)} GB total VRAM (weights + ~1.5 GB overhead).`
      : "No GPU VRAM detected - sizes shown without fit badges.";
    // Show the HF-runtime hint only when HF is actually being searched and the
    // runtime is missing (backend flag comes from the server probe).
    if (formats.includes("hf") && data.hf_backend_available === false) showHfHint();
    else hideHfHint();
    box.replaceChildren();
    if (!data.results.length) {
      box.appendChild(el("div", "sub", "(no matching repos found)"));
      return;
    }
    for (const m of data.results) box.appendChild(discRepoRow(m));
  } catch (e) {
    box.replaceChildren(el("div", "sub", "Search failed: " + e.message));
  } finally {
    $("disc-search").disabled = false;
  }
}

export async function discoverFiles(repo, filesBox, btn) {
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
    
    const showMmproj = localStorage.getItem("localm.showMmprojFiles") === "true";
    let filesToShow = data.files;
    if (showMmproj && data.mmprojs && data.mmprojs.length > 0) {
      filesToShow = filesToShow.concat(data.mmprojs);
    }
    
    for (const f of filesToShow) {
      const row = el("div", "disc-file");
      row.appendChild(el("span", "quant", f.quant || "?"));
      const desc = `${(f.size_bytes / GIB).toFixed(1)} GB` +
        (f.n_parts > 1 ? ` (${f.n_parts} parts)` : "");
      row.appendChild(el("span", "mono", desc));
      if (f.fit) row.appendChild(el("span", "fit " + f.fit, FIT_TEXT[f.fit]));
      row.appendChild(el("span", "fname", f.file));
      const pull = el("button", "btn-secondary", "pull");
      pull.onclick = () => {
        // Prefill the pull form - the user confirms (and can set an alias)
        // before anything downloads. The suggested alias mirrors the
        // server's default name (file name without .gguf).
        $("pull-spec").value = `${repo}:${f.file}`;
        $("pull-name").value = f.file.replace(/\.gguf$/i, "");
        
        // Populate the mmproj dropdown
        const mmprojSelect = $("pull-mmproj");
        mmprojSelect.replaceChildren();
        const noOption = el("option", "", "No vision projector");
        noOption.value = "";
        mmprojSelect.appendChild(noOption);
        
        if (data.mmprojs && data.mmprojs.length > 0) {
          mmprojSelect.style.display = "inline-block";
          
          // Guess the best mmproj by filename-token overlap.
          let bestMatch = "";
          let bestScore = 0;
          
          for (const m of data.mmprojs) {
            const opt = el("option", "", m.file);
            opt.value = `${repo}:${m.file}`;
            mmprojSelect.appendChild(opt);
            
            const fTokens = f.file.toLowerCase().replace(".gguf", "").split("-");
            const mTokens = m.file.toLowerCase().replace(".gguf", "").split("-");
            let score = 0;
            for (const t of fTokens) {
              if (mTokens.includes(t)) score++;
            }
            if (score > bestScore) {
              bestScore = score;
              bestMatch = opt.value;
            }
          }
          if (bestMatch && !f.file.toLowerCase().includes("mmproj")) {
            mmprojSelect.value = bestMatch;
          }
        } else {
          mmprojSelect.style.display = "none";
        }

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

// Restore + persist the search format toggles (both default on). Re-run the
// search on change when results are already showing, so a toggle reflects live.
function _bindDiscToggle(id, key) {
  const box = $(id);
  if (!box) return;
  const saved = localStorage.getItem(key);
  if (saved !== null) box.checked = saved === "true";
  box.addEventListener("change", () => {
    localStorage.setItem(key, box.checked ? "true" : "false");
    const results = $("disc-results");
    if (results && results.childElementCount) discoverSearch();
  });
}
_bindDiscToggle("disc-fmt-gguf", "localm.discFmtGguf");
_bindDiscToggle("disc-fmt-hf", "localm.discFmtHf");

// add-models-disk: make adding a model already on disk discoverable - pick a
// folder on this machine and drop its path into the spec field (the /api/models/
// pull endpoint already accepts a local folder/file path). The user no longer has
// to guess that pasting a path works.
document.addEventListener("click", async (e) => {
  if (e.target && e.target.id === "pull-browse") {
    const spec = $("pull-spec");
    if (!spec) return;
    const dir = await pickDirectory("Pick a folder that holds the model(s)",
                                    spec.value.trim());
    if (dir) spec.value = dir;
  }
});

// R18: restart the server in place from Settings - the backend unloads the model,
// then re-execs the same process, so it comes back on the same port. The reconnect
// overlay polls and auto-reconnects once the fresh process is up.
if ($("server-restart")) {
  $("server-restart").onclick = () => {
    confirmDanger("Restart the server?",
      "This restarts the LocaLM server (the model is unloaded first, then reloaded). " +
      "It will be briefly unavailable, then reconnect automatically.",
      "Restart", async () => {
        try {
          const r = await fetch("/v1/server/restart",
                                { method: "POST", headers: authHeaders() });
          if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
          toast("Server restarting…");
          // The server briefly goes away and comes back; the reconnect overlay
          // polls and reconnects automatically once the new process is up.
          if (window.onServerUnreachable) setTimeout(() => onServerUnreachable(), 800);
        } catch (e) { toast("Could not restart: " + e.message, true); }
      });
  };
}

// R18: shut the server down cleanly from Settings (the backend unloads the model
// before exit) instead of force-closing the window. Start it again from the launcher.
if ($("server-shutdown")) {
  $("server-shutdown").onclick = () => {
    confirmDanger("Shut down the server?",
      "This stops the LocaLM server (the model is unloaded first). You will need to " +
      "start it again from your launcher or terminal.",
      "Shut down", async () => {
        try {
          const r = await fetch("/v1/server/shutdown",
                                { method: "POST", headers: authHeaders() });
          if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
          toast("Server shutting down…");
          // The server is going away; show the reconnect overlay rather than a dead app.
          if (window.onServerUnreachable) setTimeout(() => onServerUnreachable(), 800);
        } catch (e) { toast("Could not shut down: " + e.message, true); }
      });
  };
}

// R47: file a bug report from Settings. "Save report" writes an editable markdown
// report to the data folder (safe snapshot, optional log tail - never secrets or
// chat). "Send to maintainer" (shown only when an upload endpoint is configured,
// via capabilities.bugreport_upload) ALSO files it as a GitHub issue through the
// proxy, so a tester needs no GitHub account. A failed upload is reported honestly
// (the file is still saved), never as success.
export async function submitBugReport(upload, isRetry = false) {
  const desc = ($("bug-desc").value || "").trim();
  if (!desc) { toast("Describe the problem first", true); return; }
  const includeLog = !!($("bug-include-log") && $("bug-include-log").checked);
  const saveBtn = $("bug-send"), upBtn = $("bug-upload");
  if (saveBtn) saveBtn.disabled = true;
  if (upBtn) upBtn.disabled = true;
  // Browser context so the report carries what actually broke in the page (env
  // snapshot + server state are added server-side). Sanitized + capped on the
  // server; rendered as plain text, never executed.
  const client = {
    userAgent: navigator.userAgent,
    page: location.hash || location.pathname,
    viewport: window.innerWidth + "x" + window.innerHeight,
    console: (window.__localmClientLog || []).slice(-40),
  };
  const out = $("bug-result");
  try {
    const r = await fetch("/api/bug-report", {
      method: "POST",
      headers: { "Content-Type": "application/json", ...authHeaders() },
      body: JSON.stringify({ description: desc, include_log: includeLog, client,
        upload: !!upload }),
    });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(data.detail || r.statusText);
    // Rate limited: if auto-retry is on (default), keep the user's text, count down,
    // and retry the send ONCE; if the toggle is off, fall through to a plain notice.
    const autoRetry = !$("bug-autoretry") || $("bug-autoretry").checked;
    if (upload && data.rate_limited && !isRetry && autoRetry) {
      const secs = Math.max(1, parseInt(data.retry_after, 10) || 30);
      await countdownRetryBugReport(secs, out);
      return;   // the retry (and the finally below) re-enable the buttons
    }
    const where = data.path || data.filename || "report";
    const uploadFailed = upload && data.upload_error;
    if (out) {
      out.hidden = false;
      if (upload && data.uploaded) {
        out.textContent = "Sent." +
          (data.issue_url ? " Tracking issue: " + data.issue_url : "");
      } else if (data.rate_limited) {
        out.textContent = "Saved: " + where +
          "  -  rate limited; wait a bit and click Send again.";
      } else if (uploadFailed) {
        out.textContent = "Saved: " + where + "  -  could not send (" +
          data.upload_error + "); email it to " + (data.maintainer || "the maintainer");
      } else {
        out.textContent = "Saved: " + where +
          (data.maintainer ? "  -  send it to " + data.maintainer : "");
      }
    }
    // Keep the description if a rate-limited send might still be retried by hand.
    if (!data.rate_limited) $("bug-desc").value = "";
    if (upload && data.uploaded) toast("Bug report sent");
    else if (data.rate_limited) toast("Rate limited; wait a bit and click Send again", true);
    else if (uploadFailed) toast("Saved, but could not send: " + data.upload_error, true);
    else toast("Bug report saved");
  } catch (e) {
    toast("Could not file report: " + e.message, true);
  } finally {
    if (saveBtn) saveBtn.disabled = false;
    if (upBtn) upBtn.disabled = false;
  }
}

// Show a live countdown in the bug-report result line, then auto-retry the send once
// (isRetry=true, so it never loops). The caller's buttons stay disabled throughout,
// so this cannot be double-fired.
async function countdownRetryBugReport(secs, out) {
  for (let s = secs; s > 0; s--) {
    if (out) {
      out.hidden = false;
      out.textContent = "Rate limited - retrying in " + s + "s...";
    }
    await new Promise((res) => setTimeout(res, 1000));
  }
  if (out) out.textContent = "Retrying...";
  await submitBugReport(true, true);
}

if ($("bug-send")) $("bug-send").onclick = () => submitBugReport(false);
if ($("bug-upload")) $("bug-upload").onclick = () => submitBugReport(true);

// Updates: check-only auto-surface (a throttled startup check in app.js calls
// __localmUpdateCheck) + an explicit "Update now". localm never self-updates; the
// apply runs only on this click and the server rolls back + reports honestly on
// failure.
export async function updateCheck() {
  const out = $("update-status"), applyBtn = $("update-apply");
  try {
    const r = await fetch("/api/update/check", { headers: authHeaders() });
    const d = await r.json().catch(() => ({}));
    if (out) out.hidden = false;
    if (!r.ok) throw new Error(d.detail || r.statusText);
    if (d.error) { if (out) out.textContent = "Could not check: " + d.error; return; }
    if (!d.available) { if (out) out.textContent = "The updater is not configured."; return; }
    if (d.newer && d.asset && d.asset.id) {
      if (out) out.textContent = "Update available: " + d.latest + " (you have " + d.current +
        ")." + (d.notes ? "  " + String(d.notes).slice(0, 200) : "");
      if (applyBtn) applyBtn.hidden = false;
    } else if (d.newer) {
      if (out) out.textContent = "Update " + d.latest + " is available but has no build attached.";
      if (applyBtn) applyBtn.hidden = true;
    } else {
      if (out) out.textContent = "localm is up to date (running " + d.current + ").";
      if (applyBtn) applyBtn.hidden = true;
    }
  } catch (e) { if (out) { out.hidden = false; out.textContent = "Could not check: " + e.message; } }
}
window.__localmUpdateCheck = updateCheck;

export async function updateApply() {
  const out = $("update-status"), btn = $("update-apply");
  if (btn) btn.disabled = true;
  if (out) { out.hidden = false; out.textContent = "Downloading and applying ..."; }
  try {
    const r = await fetch("/api/update/apply", { method: "POST", headers: authHeaders() });
    const d = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(d.detail || r.statusText);
    if (d.applied) {
      if (out) out.textContent = "Updated to " + (d.version || "") +
        (d.restarting ? ". Restarting ..." :
          (d.klass === "setup" ? ". Re-run setup.bat to finish." : "."));
      if (btn) btn.hidden = true;
    } else if (d.error) {
      if (out) out.textContent = "Update failed (rolled back): " + d.error;
    } else {
      if (out) out.textContent = d.reason || "Nothing to apply.";
    }
  } catch (e) { if (out) out.textContent = "Update failed: " + e.message; }
  finally { if (btn) btn.disabled = false; }
}
if ($("update-check")) $("update-check").onclick = updateCheck;
if ($("update-apply")) $("update-apply").onclick = updateApply;

// Changelog: show the full release history (newest first) in the shared modal.
// Fetched from /api/changelog and rendered via renderMarkdown - the same
// DOMPurify(marked) path chat uses, so it is XSS-safe. Always available (it ships
// in every build); a missing/failed fetch is shown in the modal, never left blank.
export async function showChangelog() {
  openModal("Changelog", (body) => {
    const md = el("div", "changelog-md");
    md.textContent = "Loading the changelog...";
    body.appendChild(md);
    fetch("/api/changelog", { headers: authHeaders() })
      .then(async (r) => {
        const d = await r.json().catch(() => ({}));
        if (!r.ok) throw new Error(d.detail || r.statusText);
        if (!d.available) { md.textContent = "No changelog is available in this build."; return; }
        md.textContent = "";
        renderMarkdown(md, d.markdown || "");
      })
      .catch((e) => { md.textContent = "Could not load the changelog: " + e.message; });
  });
}
if ($("changelog-show")) $("changelog-show").onclick = showChangelog;

// Issues: read-only list (textContent only - never raw innerHTML for proxy data).
export async function issuesRefresh() {
  const out = $("issues-list");
  if (out) out.textContent = "Loading ...";
  try {
    const r = await fetch("/api/issues", { headers: authHeaders() });
    const d = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(d.detail || r.statusText);
    if (!out) return;
    if (d.error) { out.textContent = "Could not load: " + d.error; return; }
    const list = d.issues || [];
    if (!list.length) { out.textContent = "No issues."; return; }
    out.textContent = "";
    for (const it of list) {
      const row = document.createElement("div");
      row.textContent = "#" + it.number + " " + (it.state || "?") + "  " + (it.title || "");
      if (it.html_url) {
        const a = document.createElement("a");
        a.href = it.html_url; a.target = "_blank"; a.rel = "noopener";
        a.textContent = " (open)";
        row.appendChild(a);
      }
      out.appendChild(row);
    }
  } catch (e) { if (out) out.textContent = "Could not load issues: " + e.message; }
}
if ($("issues-refresh")) $("issues-refresh").onclick = issuesRefresh;

// R30: export all logs of this instance to a folder the user picks (reuses the
// shared directory-picker modal), then POSTs the chosen path to /api/logs/export.
if ($("logs-export")) {
  $("logs-export").onclick = async () => {
    const dest = await pickDirectory("Choose a folder for the exported logs");
    if (!dest) return;                       // dismissed
    const btn = $("logs-export");
    btn.disabled = true;
    try {
      const r = await fetch("/api/logs/export", {
        method: "POST",
        headers: { "Content-Type": "application/json", ...authHeaders() },
        body: JSON.stringify({ dest }),
      });
      const data = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(data.detail || r.statusText);
      const out = $("logs-result");
      if (out) {
        out.hidden = false;
        out.textContent = data.copied
          ? `Exported ${data.copied} log file(s) to: ${data.dest}`
          : (data.message || "No logs found to export.");
      }
      toast(data.copied ? `Exported ${data.copied} log file(s)` : "No logs to export", !data.copied);
    } catch (e) {
      toast("Could not export logs: " + e.message, true);
    } finally {
      btn.disabled = false;
    }
  };
}

// R37: upload files from this device or a phone into <home>/uploads/ so models and
// tools can read them (beyond transient chat attachments). The POST is multipart;
// we strip the JSON Content-Type so the browser sets multipart/form-data with its
// own boundary, but keep the auth + CSRF headers. CONFIG_WRITE-gated server-side.
// The list is built with safe DOM nodes (textContent), never innerHTML, so a
// crafted file name cannot inject markup.
export async function refreshUploadsList() {
  const list = $("upload-list");
  if (!list) return;
  try {
    const r = await fetch("/api/uploads", { headers: authHeaders() });
    if (!r.ok) { list.replaceChildren(); return; }   // e.g. a read-only key: hide
    const data = await r.json().catch(() => ({ items: [] }));
    list.replaceChildren();
    for (const it of (data.items || [])) {
      const li = document.createElement("li");
      const span = document.createElement("span");
      span.className = "upload-name";
      span.textContent = `${it.name}  ·  ${fmtBytes(it.bytes)}`;
      const del = document.createElement("button");
      del.className = "btn-secondary upload-del";
      del.textContent = "Remove";
      del.onclick = () => deleteUpload(it.name);
      li.appendChild(span);
      li.appendChild(del);
      list.appendChild(li);
    }
  } catch (e) { /* leave the list as-is on a transient error */ }
}
window.refreshUploadsList = refreshUploadsList;

export async function deleteUpload(name) {
  try {
    const r = await fetch("/api/uploads/" + encodeURIComponent(name),
      { method: "DELETE", headers: authHeaders() });
    if (!r.ok) {
      const d = await r.json().catch(() => ({}));
      throw new Error(d.detail || r.statusText);
    }
    toast("Removed " + name);
    refreshUploadsList();
  } catch (e) {
    toast("Could not remove: " + e.message, true);
  }
}
window.deleteUpload = deleteUpload;

if ($("upload-send")) {
  $("upload-send").onclick = async () => {
    const input = $("upload-input");
    const files = input && input.files ? Array.from(input.files) : [];
    if (!files.length) { toast("Choose a file to upload first", true); return; }
    const fd = new FormData();
    for (const f of files) fd.append("file", f, f.name);
    const headers = authHeaders();
    delete headers["Content-Type"];   // let the browser set the multipart boundary
    const btn = $("upload-send");
    btn.disabled = true;
    try {
      const r = await fetch("/api/upload", { method: "POST", headers, body: fd });
      const data = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(data.detail || r.statusText);
      const n = (data.uploaded || []).length;
      const out = $("upload-result");
      if (out) {
        out.hidden = false;
        out.textContent = `Uploaded ${n} file(s) to: ${data.dir || "uploads"}`;
      }
      input.value = "";
      toast(`Uploaded ${n} file(s)`);
      refreshUploadsList();
    } catch (e) {
      toast("Upload failed: " + e.message, true);
    } finally {
      btn.disabled = false;
    }
  };
}

$("pull-start").onclick = async () => {
  const spec = $("pull-spec").value.trim();
  if (!spec) { toast("Enter a model spec", true); return; }
  if (spec.startsWith("-")) {
    toast("A model spec can't start with '-'. Use owner/repo, " +
          "owner/repo:file.gguf, or an https URL.", true);
    return;
  }
  const name = $("pull-name").value.trim();
  const mmprojInput = $("pull-mmproj");
  const mmproj = mmprojInput && mmprojInput.style.display !== "none" ? mmprojInput.value : null;
  const storeInput = $("pull-store");
  const store = storeInput && storeInput.value ? storeInput.value : null;

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
  if ($("pull-file")) { $("pull-file").hidden = true; $("pull-file").textContent = ""; }
  const samples = [];   // rolling {t, downloaded} window for speed/ETA (U4)
  try {
    const payload = { spec, name: name || null };
    if (mmproj) payload.mmproj = mmproj;
    if (store) payload.store = store;

    const r = await fetch("/api/models/pull", {
      method: "POST", headers: authHeaders(),
      body: JSON.stringify(payload),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || r.statusText);
    const end = await streamJob(data.job_id, (line) => {
      log.textContent += line + "\n";
      log.scrollTop = log.scrollHeight;
    }, (ev) => {
      // R06: for a multi-file (split GGUF) download, show which file is in flight.
      const fileLine = $("pull-file");
      if (fileLine) {
        if (ev.count > 1) {
          fileLine.hidden = false;
          fileLine.textContent = ev.name
            ? `file ${ev.index} of ${ev.count}: ${ev.name}`
            : `file ${ev.index} of ${ev.count}`;
        } else {
          fileLine.hidden = true;
        }
      }
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
      if (storeInput) storeInput.value = "";
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


// Bind tab click handlers
const tabNav = $("models-tab-nav");
if (tabNav) {
  for (const btn of tabNav.querySelectorAll(".tab-btn")) {
    btn.onclick = () => {
      for (const b of tabNav.querySelectorAll(".tab-btn")) {
        b.classList.remove("active");
      }
      btn.classList.add("active");
      currentTypeFilter = btn.dataset.type;
      refreshModelsPage();
    };
  }
}

// Turn a ComfyUI ScanResult (added / skipped / method) into a human toast. The
// scanner's `method` field carries WHY an empty scan found nothing ("none
// (comfy_workdir not configured)", "none (models folder not found under
// {path})"); surfacing that reason instead of a bare "Added 0" keeps a
// misconfigured scan from failing silently (AGENTS.md rule 5). The internal
// "folder-walk" / "hybrid" method jargon stays out of the visible text.
export function scanResultMessage(data) {
  const added = data.added || 0;
  const skipped = data.skipped || 0;
  const method = String(data.method || "");
  if (method.startsWith("none")) {
    if (method.includes("comfy_workdir not configured")) {
      return "No ComfyUI workdir configured - set it in Settings.";
    }
    const under = "models folder not found under ";
    const i = method.indexOf(under);
    if (i !== -1) {
      const path = method.slice(i + under.length).replace(/\)\s*$/, "");
      return `ComfyUI models folder not found at ${path}.`;
    }
    // Any other "none (...)" reason: still say something specific, never hide it.
    return "Scan found no ComfyUI models. Check the ComfyUI setup in Settings.";
  }
  if (added === 0 && skipped === 0) {
    // A real scan (folder-walk / hybrid) that turned up nothing: the folders
    // exist but held no models. Not a misconfig, but clearer than "Added 0".
    return "Scan complete. No new ComfyUI models found.";
  }
  return `Scan complete. Added ${added} models, skipped ${skipped} existing.`;
}

// Bind Scan button click handler
const scanBtn = $("models-scan-btn");
if (scanBtn) {
  scanBtn.onclick = async () => {
    scanBtn.disabled = true;
    toast("Scanning ComfyUI model folders...");
    try {
      const r = await fetch("/api/models/scan", {
        method: "POST",
        headers: authHeaders(),
      });
      const data = await r.json().catch(() => ({}));
      if (r.ok) {
        toast(scanResultMessage(data));
        refreshModelsPage();
      } else {
        toast(data.detail || "Scan failed", true);
      }
    } catch (e) {
      toast("Scan failed: " + e.message, true);
    } finally {
      scanBtn.disabled = false;
    }
  };
}

// Bind Unload-all button click handler
const unloadAllBtn = $("models-unload-all-btn");
if (unloadAllBtn) {
  unloadAllBtn.onclick = async () => {
    unloadAllBtn.disabled = true;
    try {
      const r = await fetch("/api/models/unload", {
        method: "POST", headers: authHeaders(), body: JSON.stringify({}),
      });
      const data = await r.json().catch(() => ({}));
      if (!r.ok) { toast(data.detail || "Unload failed", true); return; }
      const n = (data.unloaded_models || []).length;
      toast(n ? `Unloaded ${n} model(s)` : "Nothing was loaded");
      refreshModelsPage();
      refreshPerfEstimate();
    } catch (e) {
      toast("Unload failed: " + e.message, true);
    } finally {
      unloadAllBtn.disabled = false;
    }
  };
}

