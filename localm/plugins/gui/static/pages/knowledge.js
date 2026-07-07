// SPDX-License-Identifier: AGPL-3.0-or-later
/* localm GUI - Knowledge page (split from pages.js). Classic script: it
   shares the one global lexical environment with app.js and the other
   page scripts, so the helpers it uses ($, el, authHeaders, toast, ...)
   resolve by bare name exactly as before. */
"use strict";

// --- ES module imports (auto-generated boundary; bodies unchanged) ---
import { chat } from "../app/chat.js";
import { $, authHeaders, el, openModal, streamJob, toast } from "../app/helpers.js";
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
  refreshEmbeddingPanel();   // embedding-model status + picker
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
  if (!statusEl || !sel) return;
  let st;
  try {
    const r = await fetch("/api/rag/embedding", { headers: authHeaders() });
    st = await r.json();
    if (!r.ok) throw new Error(st.detail || r.statusText);
  } catch (e) {
    statusEl.textContent = "Could not load embedding status: " + e.message;
    return;
  }
  if (st.status === "ready") {
    statusEl.textContent = st.dim
      ? `Ready: ${st.model} (${st.dim}-dim) - semantic search is on.`
      : `Ready: ${st.model} is installed - semantic search is on.`;
    statusEl.style.color = "var(--green)";
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
}

async function applyEmbeddingModel(model) {
  const log = $("kb-embed-log");
  const btn = $("kb-embed-apply");
  log.style.display = "block";
  log.textContent = `Setting up '${model}'…\n`;
  if (btn) btn.disabled = true;
  try {
    const r = await fetch("/api/rag/embedding", {
      method: "POST", headers: authHeaders(),
      body: JSON.stringify({ model }),
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
  if (!chat.privacy) {
    const m = paths[0].match(/^(.*)[\\/][^\\/]+[\\/]?$/);
    localStorage.setItem("localm.kbAddPath", m ? m[1] : paths[0]);
  }
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
 *  owner only), offer to add the folders to the allowed list and retry ONCE. */
export async function kbRunAdd(name, paths, embed, log, retried = false) {
  try {
    const r = await fetch(
      `/api/rag/collections/${encodeURIComponent(name)}/add`, {
        method: "POST", headers: authHeaders(),
        body: JSON.stringify({ paths, embed }),
      });
    if (r.status === 409 && !retried) {
      const info = await r.json().catch(() => ({}));
      if (info && info.needs_consent) {
        const folders = info.addable || [];
        if (!(await kbConfirmAddRoots(folders))) {
          log.textContent += "Cancelled - folders not added.\n";
          return;
        }
        if (!(await kbAppendAllowedRoots(folders))) return;   // PATCH failed (toasted)
        log.textContent += "Added to your allowed folders. Indexing…\n";
        return kbRunAdd(name, paths, embed, log, true);       // retry once
      }
    }
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
    toast(end.status === "done" ? "Upload indexed" : "Upload " + end.status,
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
    // Surface a degraded semantic index instead of silently answering lexically
    // (AGENTS rule 5). The server sets this when vectors are corrupt/stale/mismatched.
    if (data.vector_degrade_reason) {
      const warn = el("div", "sub",
        "⚠ Semantic search fell back to BM25: " + data.vector_degrade_reason);
      warn.style.color = "var(--yellow)";
      body.appendChild(warn);
    }
    for (const d of data.docs) {
      const row = el("div", "log-entry");
      row.appendChild(iconEl("file", "ic ic-doc log-ic"));
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

