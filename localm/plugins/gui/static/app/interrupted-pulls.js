// SPDX-License-Identifier: AGPL-3.0-or-later
/* localm GUI - downloads that stopped before finishing: Resume / Delete / Dismiss. */
"use strict";

import { authHeaders, confirmDangerAsync, el, fmtBytes, toast } from "./helpers.js";
import { t, tn } from "./i18n.js";

// The last successful read of GET /api/models/pulls/interrupted (dismissed
// downloads included; the views filter them). known is false until then.
let _state = { downloads: [], orphans: [], known: false };
let _showDismissed = false;
// Set by the Models page: (view, start) => Promise, where start() posts the
// resume and resolves to {job_id} or null. Null when the page has not loaded.
let _resumeHandler = null;
const _listeners = new Set();

export function interruptedState() { return _state; }

export function onInterruptedChange(fn) { _listeners.add(fn); }

export function setResumeHandler(fn) { _resumeHandler = fn; }

/** Number of downloads and orphan files to offer (dismissed ones excluded). */
export function interruptedCount() {
  return _state.downloads.filter((d) => !d.dismissed).length + _state.orphans.length;
}

function _changed() {
  for (const fn of _listeners) {
    try { fn(_state); } catch (e) { console.warn("interrupted-downloads listener failed", e); }
  }
}

/** Re-read the interrupted downloads. A key without models:write gets 403 and
 *  sees none; any other failure keeps the last known list. */
export async function loadInterrupted() {
  try {
    const r = await fetch("/api/models/pulls/interrupted?include_dismissed=true",
      { headers: authHeaders() });
    if (r.status === 401 || r.status === 403) {
      _state = { downloads: [], orphans: [], known: true };
    } else if (!r.ok) {
      console.warn("could not list interrupted downloads: HTTP " + r.status);
      return _state;
    } else {
      const data = await r.json();
      _state = { downloads: data.downloads || [], orphans: data.orphans || [], known: true };
    }
  } catch (e) {
    console.warn("could not list interrupted downloads", e);
    return _state;
  }
  _changed();
  return _state;
}

function _post(body) {
  return { method: "POST", headers: authHeaders(), body: JSON.stringify(body || {}) };
}

async function _json(r) {
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.detail || r.statusText);
  return data;
}

/** Post the resume of *view*. When the source changed, asks whether to
 *  download it again from the start. Resolves to the started job's response
 *  ({job_id, spec, note}) or null when nothing was started. */
export async function startResume(view) {
  const id = encodeURIComponent(view.id);
  let data = await _json(await fetch(`/api/models/pulls/interrupted/${id}/resume`,
    _post({ fresh: false })));
  if (data.status === "changed") {
    const again = await confirmDangerAsync(
      t("models.interrupted.changedTitle"),
      t("models.interrupted.changedBody", { spec: view.spec, detail: data.detail }),
      t("models.interrupted.downloadAgain"));
    if (!again) return null;
    data = await _json(await fetch(`/api/models/pulls/interrupted/${id}/resume`,
      _post({ fresh: true })));
  }
  if (data.note) toast(data.note);
  return data.status === "started" ? data : null;
}

async function _resume(view, btn) {
  btn.disabled = true;
  try {
    if (_resumeHandler) {
      await _resumeHandler(view, () => startResume(view));
    } else {
      const data = await startResume(view);
      if (data) toast(t("models.interrupted.resumingToast", { spec: view.spec }));
    }
  } catch (e) {
    toast(t("models.interrupted.resumeFailed", { message: e.message }), true);
  } finally {
    btn.disabled = false;
    loadInterrupted();
  }
}

async function _delete(view, btn) {
  const ok = await confirmDangerAsync(
    t("models.interrupted.deleteTitle"),
    t("models.interrupted.deleteBody", { spec: view.spec, size: fmtBytes(view.bytes_on_disk) }),
    t("models.interrupted.deleteButton"));
  if (!ok) return;
  btn.disabled = true;
  try {
    const data = await _json(await fetch(
      `/api/models/pulls/interrupted/${encodeURIComponent(view.id)}/delete`, _post()));
    toast(t("models.interrupted.deletedToast", { size: fmtBytes(Number(data.removed_bytes) || 0) }));
  } catch (e) {
    toast(t("models.interrupted.deleteFailed", { message: e.message }), true);
  } finally {
    btn.disabled = false;
    loadInterrupted();
  }
}

async function _dismiss(view, btn) {
  btn.disabled = true;
  try {
    await _json(await fetch(`/api/models/pulls/interrupted/${encodeURIComponent(view.id)}/dismiss`,
      _post({ dismissed: !view.dismissed })));
  } catch (e) {
    toast(t("models.interrupted.dismissFailed", { message: e.message }), true);
  } finally {
    btn.disabled = false;
    loadInterrupted();
  }
}

async function _deleteOrphan(orphan, btn) {
  const ok = await confirmDangerAsync(
    t("models.interrupted.deleteTitle"),
    t("models.interrupted.deleteOrphanBody", { file: orphan.file, size: fmtBytes(orphan.size) }),
    t("models.interrupted.deleteButton"));
  if (!ok) return;
  btn.disabled = true;
  try {
    const data = await _json(await fetch(
      `/api/models/pulls/orphans/${encodeURIComponent(orphan.id)}/delete`, _post()));
    toast(t("models.interrupted.deletedToast", { size: fmtBytes(Number(data.removed_bytes) || 0) }));
  } catch (e) {
    toast(t("models.interrupted.deleteFailed", { message: e.message }), true);
  } finally {
    btn.disabled = false;
    loadInterrupted();
  }
}

function _sizeLine(view) {
  return view.expected_size
    ? t("models.interrupted.sizeOf", { have: fmtBytes(view.bytes_on_disk), total: fmtBytes(view.expected_size) })
    : t("models.interrupted.size", { have: fmtBytes(view.bytes_on_disk) });
}

function _downloadRow(view) {
  const row = el("div", "job-row interrupted-row");
  row.dataset.pullId = view.id;
  const head = el("div", "job-head");
  head.appendChild(el("span", "job-name", view.spec));
  if (view.name) head.appendChild(el("span", "sub", t("models.interrupted.asName", { name: view.name })));
  head.appendChild(el("span", "job-state st-cancelled",
    view.dismissed ? t("models.interrupted.dismissedState") : t("models.interrupted.state")));
  row.appendChild(head);
  row.appendChild(el("div", "sub", _sizeLine(view)));
  for (const u of view.urls || []) {
    const line = el("div", "sub interrupted-url", u);
    line.title = u;
    row.appendChild(line);
  }
  if (!view.resumable) row.appendChild(el("div", "sub", t("models.interrupted.notResumable")));
  const actions = el("div", "actions");
  if (view.resumable) {
    const resume = el("button", "btn-primary", t("models.interrupted.resumeButton"));
    resume.onclick = () => _resume(view, resume);
    actions.appendChild(resume);
  }
  const del = el("button", "btn-secondary btn-danger", t("models.interrupted.deleteButton"));
  del.onclick = () => _delete(view, del);
  actions.appendChild(del);
  const dis = el("button", "btn-secondary",
    view.dismissed ? t("models.interrupted.undismissButton") : t("models.interrupted.dismissButton"));
  dis.title = view.dismissed ? "" : t("models.interrupted.dismissTitle");
  dis.onclick = () => _dismiss(view, dis);
  actions.appendChild(dis);
  row.appendChild(actions);
  return row;
}

function _orphanRow(orphan) {
  const row = el("div", "job-row interrupted-row");
  row.dataset.orphanId = orphan.id;
  const head = el("div", "job-head");
  head.appendChild(el("span", "job-name", orphan.file));
  head.appendChild(el("span", "sub", fmtBytes(orphan.size)));
  row.appendChild(head);
  row.appendChild(el("div", "sub", t("models.interrupted.orphanNote")));
  const actions = el("div", "actions");
  const del = el("button", "btn-secondary btn-danger", t("models.interrupted.deleteButton"));
  del.onclick = () => _deleteOrphan(orphan, del);
  actions.appendChild(del);
  row.appendChild(actions);
  return row;
}

/** Render the interrupted downloads into *container* (replacing its
 *  content). Returns the number of rows shown. */
export function renderInterrupted(container) {
  if (!container) return 0;
  const shown = _state.downloads.filter((d) => _showDismissed || !d.dismissed);
  const dismissed = _state.downloads.filter((d) => d.dismissed).length;
  const nodes = shown.map(_downloadRow);
  if (_state.orphans.length) {
    nodes.push(el("div", "sub interrupted-orphans-head", t("models.interrupted.orphansHead")));
    for (const o of _state.orphans) nodes.push(_orphanRow(o));
  }
  if (dismissed) {
    const toggle = el("button", "btn-secondary interrupted-toggle",
      _showDismissed ? t("models.interrupted.hideDismissed")
        : tn("models.interrupted.showDismissed", dismissed));
    toggle.onclick = () => { _showDismissed = !_showDismissed; _changed(); };
    nodes.push(toggle);
  }
  container.replaceChildren(...nodes);
  return shown.length + _state.orphans.length;
}
