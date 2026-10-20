// SPDX-License-Identifier: AGPL-3.0-or-later
/* localm GUI - native workflows: the workflow card of the Images, Video and
   Music pages for the native backend. A workflow is a saved preset of models
   (with download sources) and generation settings. The card lists the
   built-in and saved workflows; Apply fills the page's model picker and
   fields and offers the download of every model that is not registered yet;
   Save, Rename, Delete, Export and Import manage the saved ones. Import takes
   a localm native workflow or a ComfyUI workflow (API or UI format). */

"use strict";

import { authHeaders, checkModelsBeforeGenerate, confirmDangerAsync, el, promptText, toast } from "./helpers.js";
import { t } from "./i18n.js";
import { saveTextFile } from "./settings-perf.js";

const NWF_FORMAT = "localm-native-workflow";

/** The number in input *id*, or null when it is empty or not a number. */
export function nwfNumber(id) {
  const node = document.getElementById(id);
  const v = node ? String(node.value).trim() : "";
  return v === "" || Number.isNaN(Number(v)) ? null : Number(v);
}

/** Set input *id* to *value* (empty for null or undefined). */
export function nwfSetField(id, value) {
  const node = document.getElementById(id);
  if (node) node.value = value ?? "";
}

/** Select *value* in *sel*, adding it as an option when the list lacks it. */
export function nwfSetSelect(sel, value) {
  if (!sel) return;
  if (value && ![...sel.options].some((o) => o.value === value)) {
    const o = document.createElement("option");
    o.value = value;
    o.textContent = value;
    sel.appendChild(o);
  }
  sel.value = value || "";
}

async function nwfJSON(r) {
  let data = null;
  try { data = await r.json(); } catch { data = null; }
  if (!r.ok) throw new Error((data && data.detail) || r.statusText);
  return data;
}

/** The workflow card for *media* ("image", "video" or "music"), rendered into
 *  the element with id *mountId*. *picker* is the page's native model picker
 *  (``choice``, ``setModels``, ``setLoras``); *fields* has ``read()`` (the
 *  page's settings as workflow params) and ``apply(params)``; *logId* is the
 *  page's job log. */
export function createNativeWorkflowCard({ media, mountId, picker, fields, logId }) {
  const base = `/api/${media}/native-workflows`;
  const storeKey = `localm.nativeWorkflow.${media}`;
  let list = [];
  let selected = null;
  let pending = null;
  let applied = null;
  let report = null;
  try { selected = localStorage.getItem(storeKey) || null; } catch { selected = null; }

  const log = () => document.getElementById(logId);
  const url = (id, tail = "") => `${base}/${encodeURIComponent(id)}${tail}`;
  const remember = (id) => {
    selected = id;
    try { if (id) localStorage.setItem(storeKey, id); else localStorage.removeItem(storeKey); } catch { /* storage blocked */ }
  };
  const current = () => list.find((w) => w.id === selected) || null;

  async function refresh() {
    const mount = document.getElementById(mountId);
    if (!mount) return;
    try {
      list = (await nwfJSON(await fetch(base, { headers: authHeaders() }))).workflows || [];
    } catch (e) {
      mount.replaceChildren(el("div", "sub", t("nativeWorkflows.loadFailed", { message: e.message })));
      return;
    }
    if (!current()) remember(list.length ? list[0].id : null);
    render(mount);
  }

  function render(mount) {
    const box = el("div", "native-workflows");
    const sel = document.createElement("select");
    sel.className = "native-workflow-select";
    sel.setAttribute("aria-label", t("nativeWorkflows.choose"));
    const groups = [
      [t("nativeWorkflows.builtinGroup"), list.filter((w) => w.builtin)],
      [t("nativeWorkflows.savedGroup"), list.filter((w) => !w.builtin)],
    ];
    for (const [label, items] of groups) {
      if (!items.length) continue;
      const g = document.createElement("optgroup");
      g.label = label;
      for (const w of items) {
        const o = document.createElement("option");
        o.value = w.id;
        o.textContent = w.error ? t("nativeWorkflows.unreadable", { name: w.name }) : w.name;
        g.appendChild(o);
      }
      sel.appendChild(g);
    }
    sel.value = selected || "";
    sel.onchange = () => { remember(sel.value); render(document.getElementById(mountId)); };
    box.appendChild(sel);

    const w = current();
    if (w) {
      box.appendChild(el("div", "sub native-workflow-desc", w.error || w.description || ""));
    }
    const row = el("div", "actions native-workflow-actions");
    const btn = (key, onClick, disabled) => {
      const b = el("button", "btn-secondary", t(key));
      b.type = "button";
      b.disabled = !!disabled;
      b.onclick = async () => {
        b.disabled = true;
        try { await onClick(); } catch (e) { toast(e.message, true); } finally { b.disabled = !!disabled; }
      };
      row.appendChild(b);
      return b;
    };
    btn("nativeWorkflows.apply", () => apply(w.id), !w || w.error).dataset.action = "apply";
    btn("nativeWorkflows.save", saveCurrent).dataset.action = "save";
    btn("nativeWorkflows.rename", () => rename(w), !w || w.builtin || w.error).dataset.action = "rename";
    btn("nativeWorkflows.delete", () => remove(w), !w || w.builtin).dataset.action = "delete";
    btn("nativeWorkflows.export", () => exportOne(w), !w || w.error).dataset.action = "export";
    const file = document.createElement("input");
    file.type = "file";
    file.accept = ".json,application/json";
    file.hidden = true;
    file.className = "native-workflow-file";
    file.onchange = () => importFile(file);
    btn("nativeWorkflows.import", () => file.click()).dataset.action = "import";
    row.appendChild(file);
    box.appendChild(row);

    if (applied) {
      const status = el("div", "sub native-workflow-status",
        t("nativeWorkflows.applied", { name: applied }));
      if (pending && pending.missing.length) {
        status.appendChild(el("div", "native-workflow-missing", t("nativeWorkflows.stillMissing", {
          files: pending.missing.map((m) => m.filename).join(", "),
        })));
      }
      box.appendChild(status);
    }
    if (report) box.appendChild(renderReport(report));
    mount.replaceChildren(box);
  }

  function renderReport(rep) {
    const wrap = el("details", "native-workflow-report");
    wrap.open = !!(rep.unsupported.length || rep.notes.length);
    wrap.appendChild(el("summary", null, t("nativeWorkflows.report.summary", {
      name: rep.name, mapped: rep.mapped.length, unsupported: rep.unsupported.length,
    })));
    if (rep.unsupported.length) {
      wrap.appendChild(el("div", "sub", t("nativeWorkflows.report.unsupported")));
      const ul = el("ul", "native-workflow-unsupported");
      for (const u of rep.unsupported) {
        ul.appendChild(el("li", null, `${u.class_type} #${u.node}${u.title ? ` (${u.title})` : ""}`));
      }
      wrap.appendChild(ul);
    }
    if (rep.mapped.length) {
      wrap.appendChild(el("div", "sub", t("nativeWorkflows.report.mapped")));
      const ul = el("ul", "native-workflow-mapped");
      for (const m of rep.mapped) ul.appendChild(el("li", null, `${m.class_type} #${m.node}: ${m.to}`));
      wrap.appendChild(ul);
    }
    if (rep.notes.length) {
      wrap.appendChild(el("div", "sub", t("nativeWorkflows.report.notes")));
      const ul = el("ul", "native-workflow-notes");
      for (const n of rep.notes) ul.appendChild(el("li", null, n));
      wrap.appendChild(ul);
    }
    return wrap;
  }

  async function resolveOne(id) {
    return nwfJSON(await fetch(url(id, "/resolve"), { method: "POST", headers: authHeaders() }));
  }

  async function offerDownloads(id) {
    const out = log();
    await checkModelsBeforeGenerate(media, out, { native_workflow: id });
    return resolveOne(id);
  }

  async function pick(res) {
    if (picker.setModels) await picker.setModels(res.native_models || {});
    if (picker.setLoras) picker.setLoras(res.loras || []);
  }

  /** Apply workflow *id*: offer the download of every model that is not
   *  registered, then fill the model picker and the page's fields. */
  async function apply(id) {
    let res = await resolveOne(id);
    if (res.missing.length) res = await offerDownloads(id);
    await pick(res);
    fields.apply(res.params || {});
    applied = res.name;
    pending = res.missing.length ? { id, missing: res.missing } : null;
    for (const n of res.notes || []) toast(n);
    toast(pending ? t("nativeWorkflows.appliedMissing", { name: res.name, count: res.missing.length })
      : t("nativeWorkflows.appliedToast", { name: res.name }), !!pending);
    render(document.getElementById(mountId));
  }

  /** Before a generation: when the applied workflow still has models that are
   *  not registered, offer their download again and put the ones that are
   *  registered now into the picker. Resolves true (never blocks). */
  async function prepare() {
    if (!pending) return true;
    try {
      const res = await offerDownloads(pending.id);
      const before = new Set(pending.missing.map((m) => m.slot));
      const models = { ...((picker.choice && picker.choice().native_models) || {}) };
      let changed = false;
      for (const [slot, name] of Object.entries(res.native_models || {})) {
        if (before.has(slot)) { models[slot] = name; changed = true; }
      }
      if (changed && picker.setModels) await picker.setModels(models);
      if (picker.setLoras && pending.missing.some((m) => m.slot.startsWith("lora"))) {
        picker.setLoras(res.loras || []);
      }
      pending = res.missing.length ? { id: pending.id, missing: res.missing } : null;
      render(document.getElementById(mountId));
    } catch (e) {
      toast(t("nativeWorkflows.checkFailed", { message: e.message }), true);
    }
    return true;
  }

  function currentData(name) {
    const choice = picker.choice ? picker.choice() : {};
    const models = {};
    for (const [slot, v] of Object.entries(choice.native_models || {})) models[slot] = { name: v };
    return {
      format: NWF_FORMAT, version: 1, media, name, description: "",
      models, loras: (choice.loras || []).map((l) => ({ name: l.name, strength: l.strength })),
      params: fields.read(),
    };
  }

  async function saveCurrent() {
    const name = await promptText(t("nativeWorkflows.savePrompt"), applied || "");
    if (!name || !name.trim()) return;
    const data = await nwfJSON(await fetch(base, {
      method: "POST", headers: authHeaders(), body: JSON.stringify({ workflow: currentData(name.trim()) }),
    }));
    list = data.workflows || list;
    remember(data.id);
    report = null;
    toast(t("nativeWorkflows.saved", { name: name.trim() }));
    render(document.getElementById(mountId));
  }

  async function rename(w) {
    const name = await promptText(t("nativeWorkflows.renamePrompt"), w.name);
    if (!name || !name.trim() || name.trim() === w.name) return;
    const data = await nwfJSON(await fetch(url(w.id, "/rename"), {
      method: "POST", headers: authHeaders(), body: JSON.stringify({ name: name.trim() }),
    }));
    list = data.workflows || list;
    render(document.getElementById(mountId));
  }

  async function remove(w) {
    const ok = await confirmDangerAsync(t("nativeWorkflows.deleteTitle"),
      t("nativeWorkflows.deleteBody", { name: w.name }), t("nativeWorkflows.delete"));
    if (!ok) return;
    const data = await nwfJSON(await fetch(url(w.id), { method: "DELETE", headers: authHeaders() }));
    list = data.workflows || [];
    if (pending && pending.id === w.id) pending = null;
    remember(list.length ? list[0].id : null);
    render(document.getElementById(mountId));
  }

  async function exportOne(w) {
    const data = await nwfJSON(await fetch(url(w.id), { headers: authHeaders() }));
    const stem = String(w.name || w.id).replace(/[^A-Za-z0-9._-]+/g, "-").replace(/^-+|-+$/g, "") || "workflow";
    await saveTextFile(`${stem}.${media}.json`, JSON.stringify(data.workflow, null, 2), "application/json");
  }

  async function importFile(input) {
    const f = input.files && input.files[0];
    input.value = "";
    if (!f) return;
    let parsed;
    try { parsed = JSON.parse(await f.text()); } catch (e) {
      toast(t("nativeWorkflows.importNotJson", { message: e.message }), true);
      return;
    }
    const isNative = parsed && parsed.format === NWF_FORMAT;
    const name = isNative ? undefined : f.name.replace(/\.json$/i, "").slice(0, 100);
    try {
      const data = await nwfJSON(await fetch(`${base}/import`, {
        method: "POST", headers: authHeaders(), body: JSON.stringify({ data: parsed, name }),
      }));
      list = data.workflows || list;
      remember(data.id);
      const w = current();
      report = data.report && data.report.kind !== "native"
        ? { ...data.report, name: (w && w.name) || data.id } : null;
      toast(t("nativeWorkflows.imported", { name: (w && w.name) || data.id }));
    } catch (e) {
      toast(t("nativeWorkflows.importFailed", { message: e.message }), true);
    }
    render(document.getElementById(mountId));
  }

  return {
    refresh, apply, prepare,
    get list() { return list; },
    get selected() { return selected; },
    get pending() { return pending; },
  };
}
