// SPDX-License-Identifier: AGPL-3.0-or-later
/* localm GUI - Knowledge page (split from pages.js). Classic script: it
   shares the one global lexical environment with app.js and the other
   page scripts, so the helpers it uses ($, el, authHeaders, toast, ...)
   resolve by bare name exactly as before. */
"use strict";

// --- ES module imports (auto-generated boundary; bodies unchanged) ---
import { chat, lsSetScoped } from "../app/chat.js";
import { $, authHeaders, confirmDanger, el, jobStatusWord, openModal, streamJob, toast } from "../app/helpers.js";
import { emptyState } from "../app/icons.js";
import { pickPath } from "../app/picker.js";
import { caps, refreshKbSelect } from "../app/settings-perf.js";

// Selectable file types, kept in step with rag/extract.py EXTRACTABLE_SUFFIXES:
// files outside this set are shown greyed in the picker (the server would refuse
// them anyway). Folders are indexed recursively, picking up any of these.
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
  // The embedding status is also what a row needs to tell "never embedded"
  // apart from "embedding is ready now but THIS collection predates it / went
  // stale" - only the latter is actionable via re-embed, so fetch it once here
  // and pass it into each row instead of re-deriving it from DOM text.
  const embedStatus = await refreshEmbeddingPanel();
  const embedReady = !!(embedStatus && embedStatus.status === "ready");
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
    box.appendChild(emptyState("book", "No collections yet",
      "Create one above, then add files or folders to it."));
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
    // The embedding model being ready is a GLOBAL fact; a collection only picks
    // it up when (re)indexed. BM25-while-ready means this specific collection
    // predates the model (or its vector index went stale) and re-embedding would
    // fix it - worth flagging inline, not just in the info-modal warning.
    // Gated on n_chunks > 0: an EMPTY collection also reports has_vectors=false
    // (there is nothing to have vectors for), which is not the same situation as
    // "was indexed without embeddings" - a freshly created, never-indexed
    // collection has nothing to re-embed and must not show the badge.
    // A collection can ALSO look fine (has_vectors=true, "hybrid" shown) while
    // its stored vectors no longer match the currently active embedding model
    // - the badge below never used to say so, and only an actual QUERY would
    // discover it, dropping silently to BM25 (#1078 post-merge review). This
    // server-computed comparison is best-effort (null, never a false "fine",
    // whenever it cannot tell - see rag_collections()'s own docstring), so it
    // only ever ADDS this badge when confident, never suppresses the paths above.
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
    nameTd.appendChild(iconEl("book", "ic ic-doc"));
    nameTd.appendChild(el("span", "name", c.name));
    tr.appendChild(nameTd);
    tr.appendChild(el("td", "mono", String(c.n_docs)));
    tr.appendChild(el("td", "mono", String(c.n_chunks)));
    const retrievalTd = el("td", "mono", c.has_vectors ? "hybrid" : "BM25");
    if (needsReembed) {
      const badge = el("span", "retrieval-badge job-state st-skipped", "re-embed needed");
      badge.title = staleVectors
        ? "This collection's stored vectors were built under a different "
          + "embedding model than the one currently active, so a query here "
          + "silently falls back to BM25 until you click 're-embed'."
        : "The embedding model is ready, but this collection was "
          + "indexed before that (or its vector index went stale), so it is "
          + "BM25/lexical-only until you click 're-embed'."
          + (c.vector_degrade_reason ? " (" + c.vector_degrade_reason + ")" : "");
      retrievalTd.appendChild(badge);
    }
    // c.corrupt covers three distinct on-disk faults (a corrupt meta.json, a
    // malformed line in chunks.jsonl, or an unreconstructable roots map) -
    // only the chunks.jsonl case carries a count (c.chunks_bad_lines), so the
    // badge names it when it can and falls back to the generic wording for
    // the other two rather than claiming "0 malformed lines" for a fault it
    // cannot count (NEW-RAG-INDEX-WARN-SPAM residual B). Separate from
    // needsReembed: a corrupt index needs Repair, not re-embed.
    if (c.corrupt) {
      const badge = el("span", "corrupt-badge job-state st-error",
        c.chunks_bad_lines > 0
          ? `${c.chunks_bad_lines} malformed line(s)`
          : "index damaged");
      badge.title = (c.chunks_bad_lines > 0
        ? `${c.chunks_bad_lines} malformed chunk line(s) in this collection's index. `
        : "Part of this collection's index is corrupt or malformed on disk. ")
        + "Click Repair to fix it.";
      retrievalTd.appendChild(badge);
    }
    tr.appendChild(retrievalTd);
    const actions = el("td");
    actions.style.textAlign = "right";

    if (c.corrupt) {
      const repair = el("button", "corrupt-fix", "repair");
      repair.title = "Rebuild this collection's index from its indexed "
        + "files. Documents added by upload cannot be rebuilt this way (there "
        + "is no server-side copy) and are left as-is.";
      repair.onclick = () => kbRepairCollection(c.name);
      actions.appendChild(repair);
    }

    const add = el("button", "", "add docs");
    add.onclick = () => kbAddDocs(c.name);
    actions.appendChild(add);

    const reembed = el("button", needsReembed ? "warn" : "",
      needsReembed ? "re-embed needed" : "re-embed");
    reembed.title = "Recompute this collection's vectors with the embedding "
      + "model currently set in Settings, from the text already stored here - "
      + "no original files needed, uploads included, nothing deleted. Use this "
      + "after switching the embedding model, or any time this shows BM25 for a "
      + "collection you expected to be hybrid. Clicking 'add docs' again does "
      + "NOT do this: unchanged files are skipped, not re-embedded.";
    reembed.onclick = () => kbReembedCollection(c.name);
    actions.appendChild(reembed);

    const search = el("button", "", "search");
    search.onclick = () => kbSearchModal(c.name);
    actions.appendChild(search);

    const info = el("button", "", "info");
    info.onclick = () => kbInfoModal(c.name);
    actions.appendChild(info);

    const del = el("button", "danger", "delete");
    del.onclick = () => {
      confirmDanger(`Delete collection '${c.name}'?`,
        "Only the index is removed - your original files are untouched.",
        "Delete", async () => {
          const r = await fetch(
            "/api/rag/collections/" + encodeURIComponent(c.name), {
              method: "DELETE", headers: authHeaders() });
          if (r.ok) { toast("Deleted " + c.name); refreshKnowledgePage(); }
          // Show the server's reason: a 409 here names the localm process that
          // is writing this collection right now (most often a scheduled
          // re-sync), which turns a dead end into "try again in a moment".
          else toast((await r.json().catch(() => ({}))).detail || "Delete failed",
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

/* ---- Embedding model: status + one-click setup / model picker ---------- *
 * Semantic search needs a small embedding model, loaded separately from the
 * chat model. This panel lets the user pick the built-in one (downloaded once)
 * or any model from their list, sets it up without a terminal, and reports a
 * clear result - the dimension when it works, or a specific reason it did not
 * (so a wrong pick is understood, not silently swapped). */
const INTERNAL_DEFAULT = "bge-small-en-v1.5";
const EMBED_LABELS = {
  "bge-small-en-v1.5": "Internal - bge-small-en-v1.5 (recommended, ~24 MB)",
  "nomic-embed-text-v1.5": "Internal - nomic-embed-text-v1.5 (~140 MB)",
};

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
    statusEl.textContent = "Could not load embedding status: " + e.message;
    return null;
  }
  if (st.status === "ready") {
    // "Ready" only covers indexing FROM NOW ON: it says nothing about a
    // collection indexed before this model was set up, or one whose vector
    // index went stale - those stay BM25 until re-embedded (flagged per-row
    // below), so this must not read as a blanket "you're all set".
    statusEl.textContent = st.dim
      ? `Ready: ${st.model} (${st.dim}-dim) - semantic search is on for new `
        + `indexing. Existing collections may need "re-embed" below.`
      : `Ready: ${st.model} is installed - semantic search is on for new `
        + `indexing. Existing collections may need "re-embed" below.`;
    statusEl.style.color = "var(--green)";
  } else if (st.status === "unknown") {
    // The owner pointed embedding_model at a filesystem path. That path (and
    // whether it exists) is owner-only information, so the server withheld it
    // rather than answering - saying "not installed" here would be a guess
    // presented as a fact, and the "pick a model" advice would be wrong too,
    // since this key cannot change the setting.
    statusEl.textContent =
      "The embedding model is set to a file chosen by the owner. Its status is " +
      "not shown for this key, and only the owner can change it.";
    statusEl.style.color = "var(--text-dim)";
  } else {
    statusEl.textContent =
      `Not set up: '${st.model}' is not installed - indexing is BM25 (lexical) ` +
      `only. Pick a model below and click "Set up / apply".`;
    statusEl.style.color = "var(--yellow)";
  }
  if (st.error) {
    statusEl.textContent += "  Last error: " + st.error;
    statusEl.style.color = "var(--yellow)";
  }
  if (st.gpu_fallback_reason) {
    statusEl.textContent += "  " + st.gpu_fallback_reason;
    statusEl.style.color = "var(--yellow)";
  }
  // Options: the internal keys, then the user's registered models.
  const opts = [];
  for (const key of st.internal || []) {
    opts.push([key, EMBED_LABELS[key] || ("Internal - " + key)]);
  }
  try {
    const md = await (await fetch("/api/models", { headers: authHeaders() })).json();
    for (const m of (md.models || [])) {
      if ((st.internal || []).includes(m.name)) continue;   // already listed
      opts.push([m.name, "From your models - " + m.name]);
    }
  } catch (e) { /* model list is optional for the picker */ }
  // A current custom path/name not otherwise listed still shows (and stays selected).
  if (st.model && !opts.some(([v]) => v === st.model)) {
    opts.unshift([st.model, st.model + " (current)"]);
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
  log.textContent = `Checking '${model}'…\n`;
  if (btn) btn.disabled = true;
  try {
    // FIX3: a dry run first (no config write, no embedder reset yet - see the
    // route's own docstring) so a switch that would drop existing collections
    // to BM25 is confirmed BEFORE it happens, not just reported in the job
    // log after the fact. Skipped when nothing is at risk (no collections
    // currently have embeddings), so the common first-setup case stays one
    // click.
    const dry = await fetch("/api/rag/embedding", {
      method: "POST", headers: authHeaders(),
      body: JSON.stringify({ model }),
    });
    const dryData = await dry.json();
    if (!dry.ok) throw new Error(dryData.detail || dry.statusText);
    if ((dryData.collections || []).length
        && !(await kbConfirmEmbeddingSwitch(model, dryData))) {
      log.textContent += "Cancelled.\n";
      return;   // declined - nothing was written
    }

    log.textContent += `Setting up '${model}'…\n`;
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
    toast(failed ? "Embedding setup did not complete - see the log below"
                 : "Embedding model ready", failed);
    // Q3: never silently swap the user's pick. On failure, offer a one-click
    // switch to the internal default instead (unless that IS what they tried).
    if (failed && model !== INTERNAL_DEFAULT) offerInternalFallback();
    else clearInternalFallback();
    refreshEmbeddingPanel();
  } catch (e) {
    log.textContent += "failed: " + e.message + "\n";
    toast("Setup failed: " + e.message, true);
  } finally {
    if (btn) btn.disabled = false;
  }
}

/** In-page confirm before a model switch that would invalidate existing
 *  collections' semantic search (FIX3) - shown only when the dry-run report
 *  names at least one. Mirrors kbConfirmReembed's shape. */
export function kbConfirmEmbeddingSwitch(model, report) {
  return new Promise((resolve) => {
    openModal(`Switch to '${model}'?`, (body) => {
      body.appendChild(el("p", "", report.note));
      const list = el("ul", "kb-addroots");
      for (const c of report.collections) {
        list.appendChild(el("li", "",
          c.name + (c.built_with ? ` (built with ${c.built_with})` : "")
          + (c.n_chunks != null ? ` - ${c.n_chunks} chunks` : "")));
      }
      body.appendChild(list);
      body.appendChild(el("p", "sub",
        "Re-embed each one afterward (below) to restore semantic search if it "
        + "does turn out to need it."));
      const row = el("div", "actions");
      const cancel = el("button", "btn-secondary", "Cancel");
      cancel.onclick = () => { $("modal").style.display = "none"; resolve(false); };
      const ok = el("button", "btn-secondary btn-primary", "Switch anyway");
      ok.onclick = () => { $("modal").style.display = "none"; resolve(true); };
      row.append(cancel, ok);
      body.appendChild(row);
    });
  });
}

function offerInternalFallback() {
  if ($("kb-embed-fallback")) return;
  const btn = el("button", "", `Use internal instead (${INTERNAL_DEFAULT})`);
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
    if (!model) { toast("Pick an embedding model", true); return; }
    applyEmbeddingModel(model);
  };
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

export async function kbAddDocs(name) {
  // Host access -> browse the SERVER disk (the picker hits host-gated /api/fs/*).
  // A caller WITHOUT host access cannot browse the server, so it uploads from its
  // OWN device instead (kbUploadDocs) - that reads no server path and needs no
  // filesystem access.
  if (caps.fsAccess !== "host") return kbUploadDocs(name);
  // In-page file/folder picker (multi-select) instead of prompt(): mobile/PWA
  // browsers suppress prompt(), and typing a full path by hand was the worst of
  // the old flow. The server's /add takes a paths[] array, so several files and
  // folders can be indexed in one job.
  const paths = await pickPath({
    mode: "multi",
    title: `Add documents to '${name}'`,
    startPath: localStorage.getItem("localm.kbAddPath") || "",
    exts: RAG_EXTS,
    hint: "Folders are indexed recursively. Supported: txt, md, pdf, docx, html, code.",
    confirmLabel: "Add",
  });
  if (!paths || !paths.length) return;
  // Reopen near the last add next time (the containing folder of the first pick).
  const m = paths[0].match(/^(.*)[\\/][^\\/]+[\\/]?$/);
  lsSetScoped("localm.kbAddPath", m ? m[1] : paths[0]);
  // Embeddings are opt-out here: the server defaults embed=true and degrades to
  // BM25-only when unchecked (no embedding-capable model needed). The checkbox
  // lets a user index lexical-only on purpose.
  const embed = $("kb-embed") ? $("kb-embed").checked : true;
  const log = $("kb-log");
  log.style.display = "block";
  const label = paths.length === 1 ? paths[0] : `${paths.length} items`;
  log.textContent = `Indexing ${label} into '${name}'`
    + (embed ? "" : " (BM25 only)") + "…\n";
  await kbRunAdd(name, paths, embed, log);
}

/** POST the add job. If the server replies 409 needs_consent (a whitelist miss,
 *  owner only), offer to add the folders to the allowed list and retry ONCE.
 *  Pass reindex=true to force re-embedding of files that already look
 *  unchanged (mtime/size/hash match) - a plain "add docs" click otherwise
 *  skips them, which is why re-adding a folder after turning on/switching the
 *  embedding model does nothing on its own. */
export async function kbRunAdd(name, paths, embed, log, retried = false, reindex = false) {
  try {
    const r = await fetch(
      `/api/rag/collections/${encodeURIComponent(name)}/add`, {
        method: "POST", headers: authHeaders(),
        body: JSON.stringify({ paths, embed, reindex }),
      });
    // Read the body ONCE. A 409 needs_consent reply, a normal error, and a success
    // all carry a JSON body; reading it twice on the same Response throws "body
    // stream already read", which used to mask the server's real `detail` on a
    // 409 that was NOT needs_consent (it fell through to a second r.json()).
    const data = await r.json().catch(() => ({}));
    if (r.status === 409 && !retried && data.needs_consent) {
      const folders = data.addable || [];
      if (!(await kbConfirmAddRoots(folders))) {
        log.textContent += "Cancelled - folders not added.\n";
        return;
      }
      if (!(await kbAppendAllowedRoots(folders))) return;   // PATCH failed (toasted)
      log.textContent += "Added to your allowed folders. Indexing…\n";
      return kbRunAdd(name, paths, embed, log, true, reindex);   // retry once
    }
    if (!r.ok) throw new Error(data.detail || r.statusText);
    const end = await streamJob(data.job_id, (line) => {
      log.textContent += line + "\n";
      log.scrollTop = log.scrollHeight;
    });
    toast(end.status === "done" ? "Indexing finished"
                                : "Indexing " + jobStatusWord(end.status),
          end.status !== "done");
    refreshKnowledgePage();
  } catch (e) {
    log.textContent += "failed: " + e.message + "\n";
    toast("Indexing failed: " + e.message, true);
  }
}

/** Recompute this collection's vectors with the CURRENT embedding model, from
 *  the chunk text already stored in the collection.
 *
 *  This calls POST /api/rag/collections/<name>/reembed, and it MUST. The
 *  obvious-looking alternative - re-adding the collection's own documents with
 *  force - is what this used to do, and it cannot work for the case the button
 *  exists for. Re-adding re-reads the ORIGINAL FILES, which means:
 *
 *    - it trips the very dimension guard the user is trying to get past (the
 *      server's own reembed docstring says so explicitly);
 *    - it silently EXCLUDED every uploaded document (`!d.uploaded`), because
 *      those have no path on the server disk - so a collection built by
 *      uploading re-embedded nothing at all, and a mixed one was left with
 *      some vectors at the old dimension and some at the new;
 *    - it needed the source folder to still exist and still hold those files.
 *
 *  The endpoint has none of those constraints: it works from chunks.jsonl, so
 *  no source file has to exist, uploads are included like anything else, and
 *  an interrupted run leaves the previous index intact. The error the user gets
 *  on a dimension mismatch names this button BY NAME ("or use 'Re-embed' on the
 *  Knowledge page"), so it has to be the thing that actually recovers them.
 *
 *  There is no `embed` toggle here on purpose: re-embedding without computing
 *  embeddings is not an operation. */
export async function kbReembedCollection(name) {
  if (!(await kbConfirmReembed(name))) return;
  const log = $("kb-log");
  log.style.display = "block";
  log.textContent = `Re-embedding '${name}' with the current embedding model…\n`;
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
    toast(end.status === "done" ? "Re-embedding finished"
                                : "Re-embedding " + jobStatusWord(end.status),
          end.status !== "done");
    refreshKnowledgePage();
  } catch (e) {
    log.textContent += "failed: " + e.message + "\n";
    toast("Re-embedding failed: " + e.message, true);
  }
}

/** In-page confirm before a full re-embed (minutes of model work on a large
 *  collection). No document count and no uploaded-document caveat: the server
 *  works from stored chunk text, so every document is covered including
 *  uploads, and nothing is skipped. */
export function kbConfirmReembed(name) {
  return new Promise((resolve) => {
    openModal(`Re-embed '${name}'?`, (body) => {
      body.appendChild(el("p", "",
        `This recomputes every vector in '${name}' with the embedding model `
        + `currently set in Settings, using the text already stored in the `
        + `collection. Nothing is deleted, no original files are needed, and `
        + `uploaded documents are included.`));
      body.appendChild(el("p", "sub",
        `Use this after switching the embedding model, or any time this `
        + `collection shows BM25 when you expected hybrid. On a large `
        + `collection it can take several minutes; the previous index stays `
        + `in place until the new one is complete.`));
      const row = el("div", "actions");
      const cancel = el("button", "btn-secondary", "Cancel");
      cancel.onclick = () => { $("modal").style.display = "none"; resolve(false); };
      const ok = el("button", "btn-secondary btn-primary", "Re-embed");
      ok.onclick = () => { $("modal").style.display = "none"; resolve(true); };
      row.append(cancel, ok);
      body.appendChild(row);
    });
  });
}

/** Rebuild a damaged collection's index from its indexed server-side files
 *  (POST .../repair) - the GUI answer to the "index damaged" badge / info
 *  modal warning. Mirrors kbReembedCollection's log/streamJob shape.
 *
 *  Two-step like applyEmbeddingModel: POST once with no body: the route
 *  itself decides (from whether an embedder is actually available and
 *  whether this collection currently has vectors) if repairing would drop
 *  existing embeddings, and if so answers `needs_confirm` with the exact
 *  reason instead of starting a job - shown here, then re-posted with
 *  `confirm: true`. Never guessed client-side, so this stays correct however
 *  many collections/embedder states exist. */
function kbConfirmRepair(name, detail) {
  return new Promise((resolve) => {
    openModal(`Repair '${name}'?`, (body) => {
      body.appendChild(el("p", "", detail));
      const row = el("div", "actions");
      const cancel = el("button", "btn-secondary", "Cancel");
      cancel.onclick = () => { $("modal").style.display = "none"; resolve(false); };
      const ok = el("button", "btn-secondary btn-primary", "Repair anyway");
      ok.onclick = () => { $("modal").style.display = "none"; resolve(true); };
      row.append(cancel, ok);
      body.appendChild(row);
    });
  });
}

export async function kbRepairCollection(name) {
  const log = $("kb-log");
  log.style.display = "block";
  log.textContent = `Repairing '${name}'…\n`;
  try {
    const post = (body) => fetch(
      `/api/rag/collections/${encodeURIComponent(name)}/repair`,
      { method: "POST", headers: authHeaders(), body: JSON.stringify(body) });
    let r = await post({});
    let data = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(data.detail || r.statusText);
    if (data.needs_confirm) {
      if (!(await kbConfirmRepair(name, data.detail))) {
        log.textContent += "Cancelled.\n";
        return;
      }
      r = await post({ confirm: true });
      data = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(data.detail || r.statusText);
    }
    const end = await streamJob(data.job_id, (line) => {
      log.textContent += line + "\n";
      log.scrollTop = log.scrollHeight;
    });
    toast(end.status === "done" ? "Repair finished"
                                : "Repair " + jobStatusWord(end.status),
          end.status !== "done");
    refreshKnowledgePage();
  } catch (e) {
    log.textContent += "failed: " + e.message + "\n";
    toast("Repair failed: " + e.message, true);
  }
}

/** In-page confirm (window.confirm is suppressed in some PWAs) asking whether to
 *  add the out-of-whitelist folders to the allowed list and continue. */
export function kbConfirmAddRoots(folders) {
  return new Promise((resolve) => {
    openModal("Add to allowed folders?", (body) => {
      body.appendChild(el("p", "", folders.length === 1
        ? "This folder is outside your allowed indexing folders:"
        : "These folders are outside your allowed indexing folders:"));
      const list = el("ul", "kb-addroots");
      for (const f of folders) list.appendChild(el("li", "", f));
      body.appendChild(list);
      body.appendChild(el("p", "sub",
        "Add " + (folders.length === 1 ? "it" : "them") + " to your allowed "
        + "folders and index? You can change this later in Settings › Knowledge."));
      const row = el("div", "actions");
      const cancel = el("button", "btn-secondary", "Cancel");
      cancel.onclick = () => { $("modal").style.display = "none"; resolve(false); };
      const ok = el("button", "btn-secondary btn-primary", "Add and index");
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
      toast(e.detail || "Could not update allowed folders", true);
      return false;
    }
    return true;
  } catch (e) {
    toast("Could not update allowed folders: " + e.message, true);
    return false;
  }
}

/** Per-device upload: pick files from the user's OWN device (the browser's file
 *  input - no server browsing) and POST their bytes to /upload for indexing. Used
 *  when the caller lacks host filesystem access, so it reads no server path. */
export async function uploadAndIndexFiles(name, files) {
  const MAX = 30 * 1024 * 1024;                 // mirror the server per-file cap
  const tooBig = files.filter((f) => f.size > MAX);
  if (tooBig.length) {
    toast("Too large (max 30 MB each): " + tooBig.map((f) => f.name).join(", "), true);
    return;
  }
  const embed = $("kb-embed") ? $("kb-embed").checked : true;
  const log = $("kb-log");
  log.style.display = "block";
  const label = files.length === 1 ? files[0].name : `${files.length} files`;
  log.textContent = `Uploading ${label} to '${name}'`
    + (embed ? "" : " (BM25 only)") + "…\n";
  let payload;
  try {
    payload = await Promise.all(files.map(async (f) => ({
      filename: f.name, content_b64: await fileToB64(f),
    })));
  } catch (e) {
    log.textContent += "failed to read files: " + e.message + "\n";
    toast("Could not read the selected files", true);
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
    toast(end.status === "done" ? "Upload indexed" : "Upload " + jobStatusWord(end.status),
          end.status !== "done");
    refreshKnowledgePage();
  } catch (e) {
    log.textContent += "failed: " + e.message + "\n";
    toast("Upload failed: " + e.message, true);
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
    // A cancelled dialog fires no 'change', only a window focus. Resolve empty on
    // the next focus (after a beat, so a real 'change' wins) so a dismissed picker
    // does not leave the promise pending forever.
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
    reader.onerror = () => reject(new Error("could not read " + file.name));
    reader.readAsDataURL(file);
  });
}

export async function kbInfoModal(name) {
  const r = await fetch("/api/rag/collections/" + encodeURIComponent(name),
                        { headers: authHeaders() });
  const data = await r.json();
  if (!r.ok) { toast(data.detail || "Load failed", true); return; }
  openModal("Collection - " + name, (body) => {
    body.appendChild(el("div", "sub",
      `${data.n_docs} documents · ${data.n_chunks} chunks · ` +
      (data.has_vectors ? "hybrid retrieval (BM25 + embeddings)"
                        : "lexical retrieval (BM25)")));
    // Surface on-disk index damage (meta.json, chunks.jsonl, or the roots map -
    // see store.py's three self.corrupt = True sites) rather than only via the
    // CLI's "(corrupt index ...)" listing marker (AGENTS rule 5). Names the
    // malformed-line count when the server can (chunks.jsonl only - see the
    // table badge above), falls back to generic wording for the other two.
    if (data.corrupt) {
      const warn = el("div", "sub",
        "⚠ " + (data.chunks_bad_lines > 0
          ? `${data.chunks_bad_lines} malformed chunk line(s) in this collection's index.`
          : "Part of this collection's index is corrupt or malformed on disk."));
      warn.style.color = "var(--red)";
      body.appendChild(warn);
      const repair = el("button", "btn-secondary btn-danger", "Repair");
      repair.style.marginTop = "4px";
      repair.onclick = () => kbRepairCollection(name);
      body.appendChild(repair);
    }
    // Surface a degraded semantic index instead of silently answering lexically
    // (AGENTS rule 5). The server sets this when vectors are corrupt/stale/mismatched.
    if (data.vector_degrade_reason) {
      const warn = el("div", "sub",
        "⚠ Semantic search fell back to BM25: " + data.vector_degrade_reason);
      warn.style.color = "var(--yellow)";
      body.appendChild(warn);
    }
    // Same best-effort signal as the collections table's "re-embed needed"
    // badge (#1078 post-merge review) - a collection can report has_vectors
    // and still silently drop to BM25 at query time if its stored vectors
    // were built under a different embedding model than the one active now.
    if (data.dim_mismatch === true) {
      const warn = el("div", "sub",
        "⚠ Stored vectors were built under a different embedding model than "
        + "the one currently active - queries here fall back to BM25 until "
        + "you click 're-embed'.");
      warn.style.color = "var(--yellow)";
      body.appendChild(warn);
    }
    // A re-sync flags documents whose source file has vanished rather than
    // deleting them, so the index can be ahead of the disk. Say so here, where
    // the documents are listed - otherwise the only place it surfaces is the
    // job's run output (AGENTS rule 5).
    if (data.n_missing) {
      const gone = el("div", "sub",
        `⚠ ${data.n_missing} document(s) are no longer on disk. They are still ` +
        "indexed and searchable, flagged rather than deleted, so nothing is " +
        "lost if that was temporary. Remove one below when you are sure.");
      gone.style.color = "var(--yellow)";
      body.appendChild(gone);
    }
    for (const d of data.docs) {
      const row = el("div", "log-entry");
      row.appendChild(iconEl("file", "ic ic-doc log-ic"));
      row.appendChild(el("span", "t", `${d.chunks} chunks`));
      if (d.missing) {
        const tag = el("span", "t", "file missing");
        tag.style.color = "var(--yellow)";
        row.appendChild(tag);
      }
      row.appendChild(document.createTextNode(d.path + " "));
      const rm = el("button", "action", "remove");
      rm.onclick = async () => {
        const rr = await fetch(
          `/api/rag/collections/${encodeURIComponent(name)}/remove-doc`, {
            method: "POST", headers: authHeaders(),
            body: JSON.stringify({ path: d.path }),
          });
        if (rr.ok) { toast("Removed"); kbInfoModal(name); refreshKnowledgePage(); }
        else toast((await rr.json().catch(() => ({}))).detail || "Remove failed",
                   true);
      };
      row.appendChild(rm);
      body.appendChild(row);
    }
    if (!data.docs.length) {
      body.appendChild(el("div", "sub", "(empty - use add docs)"));
    }
  });
}

export function kbSearchModal(name) {
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

$("gui-clear-convs").onclick = () => {
  const where = chat.persist
    ? "from this browser AND the server store" : "from this browser";
  confirmDanger(`Delete all saved conversations ${where}?`,
    "This can't be undone.", "Delete", async () => {
      if (chat.persist) {
        // Server store is the source of truth - clear it too or they come back.
        await Promise.allSettled(chat.conversations.map((c) =>
          fetch("/api/conversations/" + encodeURIComponent(c.id), {
            method: "DELETE", headers: authHeaders(),
          })));
      }
      localStorage.removeItem("localm.conversations");
      location.reload();
    });
};

