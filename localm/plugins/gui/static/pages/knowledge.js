// SPDX-License-Identifier: AGPL-3.0-or-later
/* localm GUI - Knowledge page. */
"use strict";

// --- ES module imports ---
import { chat, lsSetScoped } from "../app/chat.js";
import { $, authHeaders, confirmDanger, el, jobStatusWord, openModal, streamJob, toast } from "../app/helpers.js";
import { t, tn } from "../app/i18n.js";
import { emptyState, iconEl } from "../app/icons.js";
import { pickPath } from "../app/picker.js";
import { caps, refreshKbSelect } from "../app/settings-perf.js";

// Selectable file types, matching rag/extract.py EXTRACTABLE_SUFFIXES. Files
// outside this set are shown greyed in the picker. Folders are indexed
// recursively, picking up any of these.
const RAG_EXTS = [
  ".txt", ".md", ".markdown", ".rst", ".log", ".csv", ".tsv", ".json", ".jsonl",
  ".yaml", ".yml", ".toml", ".ini", ".cfg", ".py", ".js", ".ts", ".jsx", ".tsx",
  ".java", ".c", ".h", ".cpp", ".hpp", ".cs", ".go", ".rs", ".rb", ".php",
  ".swift", ".kt", ".sh", ".ps1", ".bat", ".sql", ".r", ".lua", ".xml", ".css",
  ".pdf", ".docx", ".html", ".htm", ".ipynb",
  ".png", ".jpg", ".jpeg", ".webp", ".gif",
  ".zip", ".tar", ".gz", ".bz2", ".xz", ".tgz", ".tbz", ".txz",
];

/* ================================================================ */
/*  Knowledge page                                                   */
/* ================================================================ */

export async function refreshKnowledgePage() {
  refreshKbSelect();   // keep the chat drawer selector in sync
  // Fetch the embedding status once and pass it into each row.
  const embedStatus = await refreshEmbeddingPanel();
  const embedReady = !!(embedStatus && embedStatus.status === "ready");
  const box = $("kb-table");
  if (!box) return;
  box.replaceChildren();
  let data;
  try {
    const r = await fetch("/api/rag/collections", { headers: authHeaders() });
    if (!r.ok) throw new Error(r.statusText);
    data = await r.json();
  } catch (e) {
    box.appendChild(el("div", "sub", t("knowledge.loadFailed", { message: e.message })));
    return;
  }
  const collections = data.collections || [];
  if (!collections.length) {
    box.appendChild(emptyState("book", t("knowledge.empty.title"),
      t("knowledge.empty.hint")));
    return;
  }
  const table = el("table", "data-table");
  const thead = el("thead");
  const hr = el("tr");
  for (const h of [t("knowledge.col.name"), t("knowledge.col.docs"), t("knowledge.col.chunks"),
                   t("knowledge.col.retrieval"), ""]) {
    hr.appendChild(el("th", "", h));
  }
  thead.appendChild(hr);
  table.appendChild(thead);
  const tbody = el("tbody");
  for (const c of collections) {
    const tr = el("tr");
    // Flag a collection that retrieves lexically only: it holds chunks but no
    // vectors while the embedding model is ready, or its stored vectors were
    // built under a different model. dim_mismatch is null when the server
    // cannot tell, so it only ever adds the badge.
    const staleVectors = c.dim_mismatch === true;
    const needsReembed = (c.n_chunks > 0 && !c.has_vectors && embedReady) || staleVectors;

    // Drag-and-drop upload directly onto the collection row
    tr.addEventListener("dragover", (e) => {
      e.preventDefault();
      tr.classList.add("drag-over");
    });
    tr.addEventListener("dragleave", () => {
      tr.classList.remove("drag-over");
    });
    tr.addEventListener("drop", async (e) => {
      e.preventDefault();
      tr.classList.remove("drag-over");
      const files = Array.from(e.dataTransfer.files || []);
      if (files.length) {
        await uploadAndIndexFiles(c.name, files);
      }
    });

    const nameTd = el("td", "name-cell");
    // The icon/name/badge flex line goes on this inner span, never on the td.
    const nameLine = el("span", "cell-line");
    nameTd.appendChild(nameLine);
    nameLine.appendChild(iconEl("book", "ic ic-doc"));
    nameLine.appendChild(el("span", "name", c.name));
    tr.appendChild(nameTd);
    tr.appendChild(el("td", "mono shrink-cell", String(c.n_docs)));
    tr.appendChild(el("td", "mono shrink-cell", String(c.n_chunks)));
    const retrievalTd = el("td", "mono", c.has_vectors ? t("knowledge.retrieval.hybrid")
                                                        : t("knowledge.retrieval.bm25"));
    if (needsReembed) {
      const badge = el("span", "retrieval-badge job-state st-skipped", t("knowledge.badge.reembedNeeded"));
      badge.title = staleVectors
        ? t("knowledge.badge.reembed.titleStale")
        : t("knowledge.badge.reembed.titleReady")
          + (c.vector_degrade_reason ? " (" + c.vector_degrade_reason + ")" : "");
      retrievalTd.appendChild(badge);
    }
    // c.corrupt covers a corrupt meta.json, a malformed line in chunks.jsonl,
    // or an unreconstructable roots map. Only the chunks.jsonl case carries a
    // count (c.chunks_bad_lines); the badge names it when present and uses
    // generic wording otherwise.
    if (c.corrupt) {
      const badge = el("span", "corrupt-badge job-state st-error",
        c.chunks_bad_lines > 0
          ? t("knowledge.corrupt.badge.counted", { count: c.chunks_bad_lines })
          : t("knowledge.corrupt.badge.generic"));
      badge.title = (c.chunks_bad_lines > 0
        ? t("knowledge.corrupt.counted", { count: c.chunks_bad_lines })
        : t("knowledge.corrupt.generic"))
        + " " + t("knowledge.corrupt.clickRepair");
      retrievalTd.appendChild(badge);
    }
    tr.appendChild(retrievalTd);
    const actions = el("td", "actions-cell");

    if (c.corrupt) {
      const repair = el("button", "corrupt-fix", t("knowledge.repairButton"));
      repair.title = t("knowledge.repairButton.title");
      repair.onclick = () => kbRepairCollection(c.name);
      actions.appendChild(repair);
    }

    const add = el("button", "secondary", t("knowledge.addDocs"));
    add.onclick = () => kbAddDocs(c.name);
    actions.appendChild(add);

    const reembed = el("button", needsReembed ? "warn" : "secondary",
      needsReembed ? t("knowledge.badge.reembedNeeded") : t("knowledge.reembedButton"));
    reembed.title = t("knowledge.reembedButton.title");
    reembed.onclick = () => kbReembedCollection(c.name);
    actions.appendChild(reembed);

    const search = el("button", "secondary", t("knowledge.searchButton"));
    search.onclick = () => kbSearchModal(c.name);
    actions.appendChild(search);

    const info = el("button", "secondary", t("knowledge.infoButton"));
    info.onclick = () => kbInfoModal(c.name);
    actions.appendChild(info);

    const del = el("button", "danger", t("knowledge.deleteButton"));
    del.onclick = () => {
      confirmDanger(t("knowledge.deleteConfirm.title", { name: c.name }),
        t("knowledge.deleteConfirm.body"),
        t("knowledge.deleteConfirm.confirm"), async () => {
          const r = await fetch(
            "/api/rag/collections/" + encodeURIComponent(c.name), {
              method: "DELETE", headers: authHeaders() });
          if (r.ok) { toast(t("knowledge.deleted", { name: c.name })); refreshKnowledgePage(); }
          // Show the server's reason.
          else toast((await r.json().catch(() => ({}))).detail || t("knowledge.deleteFailed"),
                     true);
        });
    };
    actions.appendChild(del);
    tr.appendChild(actions);
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
  box.appendChild(table);
}

/* ---- Embedding model: status + one-click setup / model picker ---------- */
const INTERNAL_DEFAULT = "bge-small-en-v1.5";

function embedLabels() {
  return {
    "bge-small-en-v1.5": t("knowledge.embed.label.bgeSmall"),
    "nomic-embed-text-v1.5": t("knowledge.embed.label.nomicEmbed"),
  };
}

export async function refreshEmbeddingPanel() {
  const statusEl = $("kb-embed-status");
  const sel = $("kb-embed-model");
  if (!statusEl || !sel) return null;
  let st;
  try {
    const r = await fetch("/api/rag/embedding", { headers: authHeaders() });
    st = await r.json();
    if (!r.ok) throw new Error(st.detail || r.statusText);
  } catch (e) {
    statusEl.textContent = t("knowledge.embed.statusLoadFailed", { message: e.message });
    return null;
  }
  if (st.status === "ready") {
    statusEl.textContent = st.dim
      ? t("knowledge.embed.status.readyDim", { model: st.model, dim: st.dim })
      : t("knowledge.embed.status.readyNoDim", { model: st.model });
    statusEl.style.color = "var(--green)";
  } else if (st.status === "unknown") {
    // embedding_model points at a filesystem path and the server withheld its
    // status from this key.
    statusEl.textContent = t("knowledge.embed.status.unknown");
    statusEl.style.color = "var(--text-dim)";
  } else {
    statusEl.textContent = t("knowledge.embed.status.notSetUp", { model: st.model });
    statusEl.style.color = "var(--yellow)";
  }
  if (st.error) {
    statusEl.textContent += t("knowledge.embed.status.lastError", { error: st.error });
    statusEl.style.color = "var(--yellow)";
  }
  if (st.gpu_fallback_reason) {
    statusEl.textContent += "  " + st.gpu_fallback_reason;
    statusEl.style.color = "var(--yellow)";
  }
  // One-time download of the configured model (not the dropdown selection),
  // shown only when the server reports can_download. Nothing is persisted.
  const dlBtn = $("kb-embed-download");
  if (dlBtn) {
    dlBtn.style.display = st.can_download ? "" : "none";
    if (st.can_download) dlBtn.textContent = t("knowledge.embed.downloadNow", { model: st.model });
  }
  // Options: the internal keys, then the user's registered models.
  const labels = embedLabels();
  const opts = [];
  for (const key of st.internal || []) {
    opts.push([key, labels[key] || t("knowledge.embed.optionInternalFallback", { key })]);
  }
  try {
    const md = await (await fetch("/api/models", { headers: authHeaders() })).json();
    for (const m of (md.models || [])) {
      if ((st.internal || []).includes(m.name)) continue;   // already listed
      opts.push([m.name, t("knowledge.embed.optionFromModels", { name: m.name })]);
    }
  } catch (e) { /* model list is optional for the picker */ }
  // A current custom path/name not otherwise listed still shows (and stays selected).
  if (st.model && !opts.some(([v]) => v === st.model)) {
    opts.unshift([st.model, t("knowledge.embed.optionCurrent", { model: st.model })]);
  }
  sel.replaceChildren();
  for (const [val, label] of opts) {
    const o = document.createElement("option");
    o.value = val;
    o.textContent = label;
    if (val === st.model) o.selected = true;
    sel.appendChild(o);
  }
  return st;
}

async function applyEmbeddingModel(model) {
  const log = $("kb-embed-log");
  const btn = $("kb-embed-apply");
  log.style.display = "block";
  log.textContent = t("knowledge.embed.checking", { model }) + "\n";
  if (btn) btn.disabled = true;
  try {
    // Dry run first: no config write, no embedder reset. The confirm below is
    // shown only when it reports collections that would drop to BM25.
    const dry = await fetch("/api/rag/embedding", {
      method: "POST", headers: authHeaders(),
      body: JSON.stringify({ model }),
    });
    const dryData = await dry.json();
    if (!dry.ok) throw new Error(dryData.detail || dry.statusText);
    if ((dryData.collections || []).length
        && !(await kbConfirmEmbeddingSwitch(model, dryData))) {
      log.textContent += t("knowledge.cancelled");
      return;   // declined - nothing was written
    }

    log.textContent += t("knowledge.embed.settingUp", { model });
    const r = await fetch("/api/rag/embedding", {
      method: "POST", headers: authHeaders(),
      body: JSON.stringify({ model, confirm: true }),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || r.statusText);
    const end = await streamJob(data.job_id, (line) => {
      log.textContent += line + "\n";
      log.scrollTop = log.scrollHeight;
    });
    // The job's own lines carry the outcome (a leading "error:" means it failed).
    const failed = /(^|\n)error:/i.test(log.textContent) || end.status !== "done";
    toast(failed ? t("knowledge.embed.setupFailed") : t("knowledge.embed.ready"), failed);
    // On failure, offer a one-click switch to the internal default, unless
    // that is what was tried.
    if (failed && model !== INTERNAL_DEFAULT) offerInternalFallback();
    else clearInternalFallback();
    refreshEmbeddingPanel();
  } catch (e) {
    log.textContent += t("knowledge.failedPrefix", { message: e.message }) + "\n";
    toast(t("knowledge.embed.setupFailedMsg", { message: e.message }), true);
  } finally {
    if (btn) btn.disabled = false;
  }
}

/** In-page confirm before a model switch that would invalidate existing
 *  collections' semantic search. */
export function kbConfirmEmbeddingSwitch(model, report) {
  return new Promise((resolve) => {
    openModal(t("knowledge.embedSwitch.title", { model }), (body) => {
      body.appendChild(el("p", "", report.note));
      const list = el("ul", "kb-addroots");
      for (const c of report.collections) {
        list.appendChild(el("li", "",
          c.name + (c.built_with ? t("knowledge.embedSwitch.builtWith", { model: c.built_with }) : "")
          + (c.n_chunks != null ? " - " + tn("knowledge.chunksCount", c.n_chunks) : "")));
      }
      body.appendChild(list);
      body.appendChild(el("p", "sub", t("knowledge.embedSwitch.reembedHint")));
      const row = el("div", "actions");
      const cancel = el("button", "btn-secondary", t("knowledge.cancel"));
      cancel.onclick = () => { $("modal").style.display = "none"; resolve(false); };
      const ok = el("button", "btn-primary", t("knowledge.embedSwitch.confirm"));
      ok.onclick = () => { $("modal").style.display = "none"; resolve(true); };
      row.append(cancel, ok);
      body.appendChild(row);
    });
  });
}

function offerInternalFallback() {
  if ($("kb-embed-fallback")) return;
  const btn = el("button", "btn-secondary",
    t("knowledge.embed.useInternalInstead", { model: INTERNAL_DEFAULT }));
  btn.id = "kb-embed-fallback";
  btn.style.marginTop = "8px";
  btn.onclick = () => {
    const sel = $("kb-embed-model");
    if (sel) sel.value = INTERNAL_DEFAULT;
    clearInternalFallback();
    applyEmbeddingModel(INTERNAL_DEFAULT);
  };
  $("kb-embed-log").insertAdjacentElement("afterend", btn);
}

function clearInternalFallback() {
  const b = $("kb-embed-fallback");
  if (b) b.remove();
}

if ($("kb-embed-apply")) {
  $("kb-embed-apply").onclick = () => {
    const sel = $("kb-embed-model");
    const model = sel && sel.value;
    if (!model) { toast(t("knowledge.embed.pickModel"), true); return; }
    applyEmbeddingModel(model);
  };
}

/** One-time download of the configured embedding model. Writes nothing: no
 *  model switch and no config change. */
async function downloadEmbeddingModel() {
  const log = $("kb-embed-log");
  const btn = $("kb-embed-download");
  log.style.display = "block";
  log.textContent = t("knowledge.embed.download.requesting");
  if (btn) btn.disabled = true;
  try {
    const r = await fetch("/api/rag/embedding/download",
                          { method: "POST", headers: authHeaders() });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || r.statusText);
    if (data.job_id) {
      const end = await streamJob(data.job_id, (line) => {
        log.textContent += line + "\n";
        log.scrollTop = log.scrollHeight;
      });
      const failed = /(^|\n)error:/i.test(log.textContent) || end.status !== "done";
      toast(failed ? t("knowledge.embed.download.failed") : t("knowledge.embed.ready"), failed);
    } else {
      log.textContent += t("knowledge.embed.alreadyInstalled");
    }
    refreshEmbeddingPanel();
  } catch (e) {
    log.textContent += t("knowledge.failedPrefix", { message: e.message }) + "\n";
    toast(t("knowledge.embed.downloadFailedMsg", { message: e.message }), true);
  } finally {
    if (btn) btn.disabled = false;
  }
}

if ($("kb-embed-download")) {
  $("kb-embed-download").onclick = downloadEmbeddingModel;
}

$("kb-create").onclick = async () => {
  const name = $("kb-name").value.trim();
  if (!name) { toast(t("knowledge.enterName"), true); return; }
  const r = await fetch("/api/rag/collections", {
    method: "POST", headers: authHeaders(),
    body: JSON.stringify({ name }),
  });
  const data = await r.json();
  if (!r.ok) { toast(data.detail || t("knowledge.createFailed"), true); return; }
  $("kb-name").value = "";
  toast(t("knowledge.created", { name: data.name }));
  refreshKnowledgePage();
};

export async function kbAddDocs(name) {
  // Host access browses the SERVER disk through the host-gated /api/fs/*;
  // without it, upload from the caller's own device instead.
  if (caps.fsAccess !== "host") return kbUploadDocs(name);
  // In-page file/folder picker (multi-select). /add takes a paths[] array, so
  // several files and folders index in one job.
  const paths = await pickPath({
    mode: "multi",
    title: t("knowledge.addDocs.pickerTitle", { name }),
    startPath: localStorage.getItem("localm.kbAddPath") || "",
    exts: RAG_EXTS,
    hint: t("knowledge.addDocs.pickerHint"),
    confirmLabel: t("knowledge.addDocs.pickerConfirm"),
  });
  if (!paths || !paths.length) return;
  // Reopen near the last add next time (the containing folder of the first pick).
  const m = paths[0].match(/^(.*)[\\/][^\\/]+[\\/]?$/);
  lsSetScoped("localm.kbAddPath", m ? m[1] : paths[0]);
  // Embeddings are opt-out: the server defaults embed=true and indexes
  // BM25-only when unchecked.
  const embed = $("kb-embed") ? $("kb-embed").checked : true;
  const log = $("kb-log");
  log.style.display = "block";
  const label = paths.length === 1 ? paths[0] : t("knowledge.itemsCount", { count: paths.length });
  log.textContent = t("knowledge.indexing.into", { label, name })
    + (embed ? "" : t("knowledge.bm25OnlySuffix")) + "…\n";
  await kbRunAdd(name, paths, embed, log);
}

/** POST the add job. On a 409 needs_consent reply, offer to add the folders to
 *  the allowed list and retry ONCE. reindex=true forces re-embedding of files
 *  that already look unchanged (mtime/size/hash match), which a plain add
 *  skips. */
export async function kbRunAdd(name, paths, embed, log, retried = false, reindex = false) {
  try {
    const r = await fetch(
      `/api/rag/collections/${encodeURIComponent(name)}/add`, {
        method: "POST", headers: authHeaders(),
        body: JSON.stringify({ paths, embed, reindex }),
      });
    // Read the body ONCE: every reply carries JSON, and a second r.json() on
    // the same Response throws "body stream already read".
    const data = await r.json().catch(() => ({}));
    if (r.status === 409 && !retried && data.needs_consent) {
      const folders = data.addable || [];
      if (!(await kbConfirmAddRoots(folders))) {
        log.textContent += t("knowledge.addRoots.cancelled");
        return;
      }
      if (!(await kbAppendAllowedRoots(folders))) return;   // PATCH failed (toasted)
      log.textContent += t("knowledge.addRoots.addedIndexing");
      return kbRunAdd(name, paths, embed, log, true, reindex);   // retry once
    }
    if (!r.ok) throw new Error(data.detail || r.statusText);
    const end = await streamJob(data.job_id, (line) => {
      log.textContent += line + "\n";
      log.scrollTop = log.scrollHeight;
    });
    toast(end.status === "done" ? t("knowledge.indexing.finished")
                                : t("knowledge.indexing.status", { status: jobStatusWord(end.status) }),
          end.status !== "done");
    refreshKnowledgePage();
  } catch (e) {
    log.textContent += t("knowledge.failedPrefix", { message: e.message }) + "\n";
    toast(t("knowledge.indexing.failedMsg", { message: e.message }), true);
  }
}

/** Recompute this collection's vectors with the CURRENT embedding model, from
 *  the chunk text already stored in the collection.
 *
 *  POSTs to /api/rag/collections/<name>/reembed, which works from
 *  chunks.jsonl: no source file has to exist, uploaded documents are included,
 *  and an interrupted run leaves the previous index intact. There is no
 *  `embed` toggle. */
export async function kbReembedCollection(name) {
  if (!(await kbConfirmReembed(name))) return;
  const log = $("kb-log");
  log.style.display = "block";
  log.textContent = t("knowledge.reembed.log.start", { name });
  try {
    const r = await fetch(
      `/api/rag/collections/${encodeURIComponent(name)}/reembed`,
      { method: "POST", headers: authHeaders() });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(data.detail || r.statusText);
    const end = await streamJob(data.job_id, (line) => {
      log.textContent += line + "\n";
      log.scrollTop = log.scrollHeight;
    });
    toast(end.status === "done" ? t("knowledge.reembed.finished")
                                : t("knowledge.reembed.status", { status: jobStatusWord(end.status) }),
          end.status !== "done");
    refreshKnowledgePage();
  } catch (e) {
    log.textContent += t("knowledge.failedPrefix", { message: e.message }) + "\n";
    toast(t("knowledge.reembed.failedMsg", { message: e.message }), true);
  }
}

/** In-page confirm before a full re-embed. */
export function kbConfirmReembed(name) {
  return new Promise((resolve) => {
    openModal(t("knowledge.reembedConfirm.title", { name }), (body) => {
      body.appendChild(el("p", "", t("knowledge.reembedConfirm.body1", { name })));
      body.appendChild(el("p", "sub", t("knowledge.reembedConfirm.body2")));
      const row = el("div", "actions");
      const cancel = el("button", "btn-secondary", t("knowledge.cancel"));
      cancel.onclick = () => { $("modal").style.display = "none"; resolve(false); };
      const ok = el("button", "btn-primary", t("knowledge.reembedConfirm.confirm"));
      ok.onclick = () => { $("modal").style.display = "none"; resolve(true); };
      row.append(cancel, ok);
      body.appendChild(row);
    });
  });
}

/** In-page confirm before a repair, shown with the `needs_confirm` reason the
 *  repair route returns instead of starting a job. */
function kbConfirmRepair(name, detail) {
  return new Promise((resolve) => {
    openModal(t("knowledge.repairConfirm.title", { name }), (body) => {
      body.appendChild(el("p", "", detail));
      const row = el("div", "actions");
      const cancel = el("button", "btn-secondary", t("knowledge.cancel"));
      cancel.onclick = () => { $("modal").style.display = "none"; resolve(false); };
      const ok = el("button", "btn-primary", t("knowledge.repairConfirm.confirm"));
      ok.onclick = () => { $("modal").style.display = "none"; resolve(true); };
      row.append(cancel, ok);
      body.appendChild(row);
    });
  });
}

/** Whether a repair started right now would actually re-embed: a fresh read of
 *  embedder status, never the page's last-rendered snapshot. Stays true (the
 *  server decides) unless the status is a CONFIRMED "not_installed" - "ready"
 *  and "unknown" (withheld for a non-owner key) both keep the prior default,
 *  since declaring false here would force the server to skip embedding even
 *  when it is in fact available. */
async function repairWillEmbed() {
  try {
    const r = await fetch("/api/rag/embedding", { headers: authHeaders() });
    const st = await r.json().catch(() => ({}));
    return !(r.ok && st.status === "not_installed");
  } catch (e) {
    return true;
  }
}

export async function kbRepairCollection(name) {
  const log = $("kb-log");
  log.style.display = "block";
  log.textContent = t("knowledge.repair.log.start", { name });
  try {
    const post = (body) => fetch(
      `/api/rag/collections/${encodeURIComponent(name)}/repair`,
      { method: "POST", headers: authHeaders(), body: JSON.stringify(body) });
    const embed = await repairWillEmbed();
    let r = await post({ embed });
    let data = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(data.detail || r.statusText);
    if (data.needs_confirm) {
      if (!(await kbConfirmRepair(name, data.detail))) {
        log.textContent += t("knowledge.cancelled");
        return;
      }
      r = await post({ embed, confirm: true });
      data = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(data.detail || r.statusText);
    }
    const end = await streamJob(data.job_id, (line) => {
      log.textContent += line + "\n";
      log.scrollTop = log.scrollHeight;
    });
    toast(end.status === "done" ? t("knowledge.repair.finished")
                                : t("knowledge.repair.status", { status: jobStatusWord(end.status) }),
          end.status !== "done");
    refreshKnowledgePage();
  } catch (e) {
    log.textContent += t("knowledge.failedPrefix", { message: e.message }) + "\n";
    toast(t("knowledge.repair.failedMsg", { message: e.message }), true);
  }
}

/** In-page confirm asking whether to add the out-of-whitelist folders to the
 *  allowed list and continue. */
export function kbConfirmAddRoots(folders) {
  return new Promise((resolve) => {
    openModal(t("knowledge.addRoots.title"), (body) => {
      body.appendChild(el("p", "", tn("knowledge.addRoots.intro", folders.length)));
      const list = el("ul", "kb-addroots");
      for (const f of folders) list.appendChild(el("li", "", f));
      body.appendChild(list);
      body.appendChild(el("p", "sub", tn("knowledge.addRoots.confirm", folders.length)));
      const row = el("div", "actions");
      const cancel = el("button", "btn-secondary", t("knowledge.cancel"));
      cancel.onclick = () => { $("modal").style.display = "none"; resolve(false); };
      const ok = el("button", "btn-primary", t("knowledge.addRoots.confirmButton"));
      ok.onclick = () => { $("modal").style.display = "none"; resolve(true); };
      row.append(cancel, ok);
      body.appendChild(row);
    });
  });
}

/** Append *folders* to rag_allowed_roots (owner-only) via the config API, merging
 *  with the current list. Returns true on success. */
export async function kbAppendAllowedRoots(folders) {
  try {
    const cur = await fetch("/v1/config", { headers: authHeaders() });
    const cfg = cur.ok ? await cur.json() : {};
    const existing = Array.isArray(cfg.rag_allowed_roots) ? cfg.rag_allowed_roots : [];
    const merged = [...new Set([...existing, ...folders])];
    const pr = await fetch("/v1/config", {
      method: "PATCH", headers: authHeaders(),
      body: JSON.stringify({ rag_allowed_roots: merged }),
    });
    if (!pr.ok) {
      const e = await pr.json().catch(() => ({}));
      toast(e.detail || t("knowledge.addRoots.updateFailed"), true);
      return false;
    }
    return true;
  } catch (e) {
    toast(t("knowledge.addRoots.updateFailedMsg", { message: e.message }), true);
    return false;
  }
}

/** POST the bytes of files picked from the user's own device to /upload for
 *  indexing. Reads no server path. */
export async function uploadAndIndexFiles(name, files) {
  const MAX = 30 * 1024 * 1024;                 // mirror the server per-file cap
  const tooBig = files.filter((f) => f.size > MAX);
  if (tooBig.length) {
    toast(t("knowledge.upload.tooLarge", { names: tooBig.map((f) => f.name).join(", ") }), true);
    return;
  }
  const embed = $("kb-embed") ? $("kb-embed").checked : true;
  const log = $("kb-log");
  log.style.display = "block";
  const label = files.length === 1 ? files[0].name : t("knowledge.filesCount", { count: files.length });
  log.textContent = t("knowledge.uploading.to", { label, name })
    + (embed ? "" : t("knowledge.bm25OnlySuffix")) + "…\n";
  let payload;
  try {
    payload = await Promise.all(files.map(async (f) => ({
      filename: f.name, content_b64: await fileToB64(f),
    })));
  } catch (e) {
    log.textContent += t("knowledge.upload.readFailed", { message: e.message }) + "\n";
    toast(t("knowledge.upload.readFailedToast"), true);
    return;
  }
  try {
    const r = await fetch(
      `/api/rag/collections/${encodeURIComponent(name)}/upload`, {
        method: "POST", headers: authHeaders(),
        body: JSON.stringify({ files: payload, embed }),
      });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || r.statusText);
    const end = await streamJob(data.job_id, (line) => {
      log.textContent += line + "\n";
      log.scrollTop = log.scrollHeight;
    });
    toast(end.status === "done" ? t("knowledge.upload.indexed")
                                : t("knowledge.upload.status", { status: jobStatusWord(end.status) }),
          end.status !== "done");
    refreshKnowledgePage();
  } catch (e) {
    log.textContent += t("knowledge.failedPrefix", { message: e.message }) + "\n";
    toast(t("knowledge.upload.failedMsg", { message: e.message }), true);
  }
}

export async function kbUploadDocs(name) {
  // Relax picker to allow all files; let the server's sniffer decide.
  const files = await pickDeviceFiles();
  if (!files.length) return;
  await uploadAndIndexFiles(name, files);
}

/** Open the browser's native file picker (multi-select, filtered to the RAG
 *  extensions) and resolve the chosen File objects, or [] if cancelled. */
export function pickDeviceFiles(exts) {
  return new Promise((resolve) => {
    const input = document.createElement("input");
    input.type = "file";
    input.multiple = true;
    if (Array.isArray(exts) && exts.length) input.accept = exts.join(",");
    input.style.display = "none";
    let done = false;
    const finish = (files) => {
      if (done) return;
      done = true;
      input.remove();
      resolve(files);
    };
    input.addEventListener("change", () => finish(Array.from(input.files || [])));
    // A cancelled dialog fires no 'change', only a window focus: resolve empty
    // on the next focus, after a beat so a real 'change' wins.
    window.addEventListener("focus",
      () => setTimeout(() => finish([]), 300), { once: true });
    document.body.appendChild(input);
    input.click();
  });
}

/** Read a File as base64 (without the data: URL prefix). */
export function fileToB64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result).split(",", 2)[1] || "");
    reader.onerror = () => reject(new Error(t("knowledge.upload.fileReadErrorReason", { filename: file.name })));
    reader.readAsDataURL(file);
  });
}

/** A "sub" line prefixed with the warning icon, tinted `color`. */
function warnLine(text, color) {
  const warn = el("div", "sub");
  warn.appendChild(iconEl("warning", "btn-ic"));
  warn.appendChild(document.createTextNode(text));
  warn.style.color = color;
  return warn;
}

export async function kbInfoModal(name) {
  const r = await fetch("/api/rag/collections/" + encodeURIComponent(name),
                        { headers: authHeaders() });
  const data = await r.json();
  if (!r.ok) { toast(data.detail || t("knowledge.info.loadFailed"), true); return; }
  openModal(t("knowledge.info.title", { name }), (body) => {
    body.appendChild(el("div", "sub",
      tn("knowledge.docsCount", data.n_docs) + " · " + tn("knowledge.chunksCount", data.n_chunks) + " · " +
      (data.has_vectors ? t("knowledge.retrieval.hybridFull") : t("knowledge.retrieval.lexicalFull"))));
    // On-disk index damage: meta.json, chunks.jsonl, or the roots map. Names
    // the malformed-line count when the server reports one (chunks.jsonl
    // only), generic wording otherwise.
    if (data.corrupt) {
      body.appendChild(warnLine(data.chunks_bad_lines > 0
        ? t("knowledge.corrupt.counted", { count: data.chunks_bad_lines })
        : t("knowledge.corrupt.generic"),
        "var(--red)"));
      const repair = el("button", "btn-secondary btn-danger", t("knowledge.info.repairButton"));
      repair.style.marginTop = "4px";
      repair.onclick = () => kbRepairCollection(name);
      body.appendChild(repair);
    }
    // The server sets this when vectors are corrupt, stale or mismatched.
    if (data.vector_degrade_reason) {
      body.appendChild(warnLine(
        t("knowledge.info.vectorDegrade", { reason: data.vector_degrade_reason }),
        "var(--yellow)"));
    }
    // Stored vectors were built under a different embedding model than the
    // active one, the same signal as the table's "re-embed needed" badge.
    if (data.dim_mismatch === true) {
      body.appendChild(warnLine(t("knowledge.info.dimMismatch"), "var(--yellow)"));
    }
    // A re-sync flags documents whose source file has vanished rather than
    // deleting them, so the index can be ahead of the disk.
    if (data.n_missing) {
      body.appendChild(warnLine(tn("knowledge.info.missingDocs", data.n_missing), "var(--yellow)"));
    }
    for (const d of data.docs) {
      const row = el("div", "log-entry");
      row.appendChild(iconEl("file", "ic ic-doc log-ic"));
      row.appendChild(el("span", "t", tn("knowledge.chunksCount", d.chunks)));
      if (d.missing) {
        const tag = el("span", "t", t("knowledge.info.fileMissing"));
        tag.style.color = "var(--yellow)";
        row.appendChild(tag);
      }
      row.appendChild(document.createTextNode(d.path + " "));
      const rm = el("button", "action", t("knowledge.removeButton"));
      rm.onclick = async () => {
        const rr = await fetch(
          `/api/rag/collections/${encodeURIComponent(name)}/remove-doc`, {
            method: "POST", headers: authHeaders(),
            body: JSON.stringify({ path: d.path }),
          });
        if (rr.ok) { toast(t("knowledge.removed")); kbInfoModal(name); refreshKnowledgePage(); }
        else toast((await rr.json().catch(() => ({}))).detail || t("knowledge.removeFailed"),
                   true);
      };
      row.appendChild(rm);
      body.appendChild(row);
    }
    if (!data.docs.length) {
      body.appendChild(emptyState("file", t("knowledge.info.empty.title"),
        t("knowledge.info.empty.hint")));
    }
  });
}

export function kbSearchModal(name) {
  openModal(t("knowledge.search.title", { name }), (body) => {
    const row = el("div", "row");
    const input = document.createElement("input");
    input.type = "text";
    input.className = "kb-search-input";
    input.placeholder = t("knowledge.search.placeholder");
    input.style.flex = "1";
    const go = el("button", "btn-primary", t("knowledge.search.button"));
    row.appendChild(input);
    row.appendChild(go);
    body.appendChild(row);
    const results = el("div");
    body.appendChild(results);
    const run = async () => {
      const q = input.value.trim();
      if (!q) return;
      results.replaceChildren(el("div", "sub", t("knowledge.search.searching")));
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
          results.appendChild(emptyState("search", t("knowledge.search.empty.title"),
            t("knowledge.search.empty.hint")));
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
        results.replaceChildren(el("div", "sub", t("knowledge.failedPrefix", { message: e.message })));
      }
    };
    go.onclick = run;
    input.onkeydown = (e) => { if (e.key === "Enter") run(); };
    input.focus();
  });
}

$("gui-clear-convs").onclick = () => {
  const where = chat.persist
    ? t("knowledge.clearConvs.scope.persisted") : t("knowledge.clearConvs.scope.browserOnly");
  confirmDanger(t("knowledge.clearConvs.confirm.title", { where }),
    t("knowledge.clearConvs.confirm.body"), t("knowledge.deleteConfirm.confirm"), async () => {
      if (chat.persist) {
        // Clear the server store as well as this browser's copy.
        await Promise.allSettled(chat.conversations.map((c) =>
          fetch("/api/conversations/" + encodeURIComponent(c.id), {
            method: "DELETE", headers: authHeaders(),
          })));
      }
      localStorage.removeItem("localm.conversations");
      location.reload();
    });
};

// The collections table and embedding panel are painted from fetched data, so
// both are re-fetched and redrawn when the interface language changes.
document.addEventListener("localm:language", () => {
  refreshKnowledgePage();
});
