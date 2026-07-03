// SPDX-License-Identifier: AGPL-3.0-or-later
/* localm GUI - Knowledge page (split from pages.js). Classic script: it
   shares the one global lexical environment with app.js and the other
   page scripts, so the helpers it uses ($, el, authHeaders, toast, ...)
   resolve by bare name exactly as before. */
"use strict";

// --- ES module imports (auto-generated boundary; bodies unchanged) ---
import { chat } from "../app/chat.js";
import { $, authHeaders, el, openModal, streamJob, toast } from "../app/helpers.js";
import { pickPath } from "../app/picker.js";
import { refreshKbSelect } from "../app/settings-perf.js";

// Selectable file types, kept in step with rag/extract.py EXTRACTABLE_SUFFIXES:
// files outside this set are shown greyed in the picker (the server would refuse
// them anyway). Folders are indexed recursively, picking up any of these.
const RAG_EXTS = [
  ".txt", ".md", ".markdown", ".rst", ".log", ".csv", ".tsv", ".json", ".jsonl",
  ".yaml", ".yml", ".toml", ".ini", ".cfg", ".py", ".js", ".ts", ".jsx", ".tsx",
  ".java", ".c", ".h", ".cpp", ".hpp", ".cs", ".go", ".rs", ".rb", ".php",
  ".swift", ".kt", ".sh", ".ps1", ".bat", ".sql", ".r", ".lua", ".xml", ".css",
  ".pdf", ".docx", ".html", ".htm", ".ipynb",
];

/* ================================================================ */
/*  Knowledge page                                                   */
/* ================================================================ */

export async function refreshKnowledgePage() {
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

export async function kbAddDocs(name) {
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
  try {
    const r = await fetch(
      `/api/rag/collections/${encodeURIComponent(name)}/add`, {
        method: "POST", headers: authHeaders(),
        body: JSON.stringify({ paths, embed }),
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

