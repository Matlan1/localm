// SPDX-License-Identifier: AGPL-3.0-or-later
/* localm GUI - model pickers for the native media backends (Images, Video,
   Music pages). The choice applies to the next generation only and is sent
   as `native_models` (and `loras` for image/video); "Settings default" and
   "Automatic" send nothing for that slot, so Settings and the automatic
   component pick decide. The choice is remembered per page in localStorage. */

"use strict";

import { authHeaders, el } from "./helpers.js";
import { t } from "./i18n.js";

const NM_SLOT_KEYS = {
  clip_l: "nativeModels.slot.clipL",
  clip_g: "nativeModels.slot.clipG",
  t5xxl: "nativeModels.slot.t5xxl",
  llm: "nativeModels.slot.llm",
  vae: "nativeModels.slot.vae",
  clip_vision: "nativeModels.slot.clipVision",
};
const NM_MUSIC_KEYS = {
  dit: "nativeModels.music.dit",
  text_encoder: "nativeModels.music.textEncoder",
  vae: "nativeModels.music.vae",
  lm: "nativeModels.music.lm",
};
const NM_MAX_LORAS = 8;

function nmLoad(key, fallback) {
  try {
    const raw = localStorage.getItem(key);
    const v = raw ? JSON.parse(raw) : null;
    return v && typeof v === "object" ? v : fallback;
  } catch { return fallback; }
}

function nmSave(key, value) {
  try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* storage unavailable */ }
}

function nmOption(value, text) {
  const o = document.createElement("option");
  o.value = value;
  o.textContent = text;
  return o;
}

/** A <select> whose first option (value "") is *firstText*, then *items*
 * ({name, file, family_label}) grouped by family. A *current* value missing
 * from *items* is kept as an option so a saved choice is never dropped. */
export function nativeModelSelect(items, current, firstText) {
  const sel = document.createElement("select");
  sel.appendChild(nmOption("", firstText));
  const groups = new Map();
  for (const it of items) {
    const g = it.family_label || "";
    if (!groups.has(g)) groups.set(g, []);
    groups.get(g).push(it);
  }
  for (const [label, list] of groups) {
    const parent = label ? document.createElement("optgroup") : sel;
    if (label) { parent.label = label; sel.appendChild(parent); }
    for (const it of list) {
      parent.appendChild(nmOption(it.name, it.file && it.file !== it.name
        ? `${it.name} (${it.file})` : it.name));
    }
  }
  if (current && ![...sel.options].some((o) => o.value === current)) {
    sel.appendChild(nmOption(current, t("nativeModels.notRegistered", { name: current })));
  }
  sel.value = current || "";
  return sel;
}

function nmField(labelText, control) {
  const wrap = el("div", "native-model-field");
  const label = el("label", null, labelText);
  wrap.append(label, control);
  return wrap;
}

/** The image/video picker mounted in #mountId. `choice()` returns
 * `{native_models, loras}` for the request body (empty values left out). */
export function createNativeModelPicker({ media, prefix, mountId }) {
  const storeKey = `localm.nativeModels.${media}`;
  let state = nmLoad(storeKey, { models: {}, loras: [] });
  if (!state.models || typeof state.models !== "object") state.models = {};
  if (!Array.isArray(state.loras)) state.loras = [];
  let data = null;
  let seq = 0;

  const persist = () => nmSave(storeKey, state);

  const query = () => {
    const q = new URLSearchParams();
    for (const [k, v] of Object.entries(state.models)) if (v) q.set(k, v);
    return q.toString();
  };

  async function refresh() {
    const mount = document.getElementById(mountId);
    if (!mount) return;
    const mine = ++seq;
    let r;
    try {
      r = await fetch(`/api/${prefix}/native-models?${query()}`, { headers: authHeaders() });
    } catch { return; }
    if (mine !== seq) return;
    if (r.status === 400) {
      state.models = {};
      persist();
      return refresh();
    }
    if (!r.ok) return;
    data = await r.json();
    if (mine !== seq) return;
    render(mount);
  }

  function render(mount) {
    const cur = data.current || {};
    const prev = data.preview || null;
    const box = el("div", "native-models");
    const modelSel = nativeModelSelect(data.models || [], state.models.model || "",
      cur.model ? t("nativeModels.settingsDefault", { name: cur.model })
        : t("nativeModels.settingsDefaultNone"));
    modelSel.onchange = () => {
      state.models = modelSel.value ? { model: modelSel.value } : {};
      persist();
      refresh();
    };
    box.appendChild(nmField(t("nativeModels.model"), modelSel));

    const note = el("div", "sub native-models-note");
    if (prev) {
      if (prev.family_label && prev.needs && prev.needs.length) {
        note.textContent = t("nativeModels.needs", {
          family: prev.family_label,
          parts: prev.needs.map((s) => t(NM_SLOT_KEYS[s] || s)).join(", "),
        });
      } else if (prev.kind === "checkpoint") {
        note.textContent = t("nativeModels.checkpoint", { family: prev.family_label || "" });
      } else if (prev.kind === "unknown") {
        note.textContent = t("nativeModels.unknownFamily");
      }
    }
    if (note.textContent) box.appendChild(note);

    const needs = new Set((prev && prev.needs) || []);
    const slots = data.slots || {};
    const unknown = prev && prev.kind === "unknown";
    const comps = (prev && prev.components) || {};
    for (const slot of Object.keys(slots)) {
      if (!needs.has(slot) && !state.models[slot] && !unknown) continue;
      const picked = comps[slot];
      const autoText = picked && picked.auto
        ? t("nativeModels.automatic", { name: picked.name })
        : (cur[slot] && !state.models.model
          ? t("nativeModels.settingsDefault", { name: cur[slot] })
          : t("nativeModels.automaticNone"));
      const sel = nativeModelSelect(slots[slot], state.models[slot] || "", autoText);
      sel.dataset.slot = slot;
      sel.onchange = () => {
        if (sel.value) state.models[slot] = sel.value;
        else delete state.models[slot];
        persist();
        refresh();
      };
      box.appendChild(nmField(t(NM_SLOT_KEYS[slot] || slot), sel));
    }

    if (prev && prev.error) {
      const err = el("div", "sub");
      err.append(el("span", "job-state st-error", t("nativeModels.cannotRun")), " ", prev.error);
      box.appendChild(err);
    }
    for (const n of (prev && prev.notes) || []) box.appendChild(el("div", "sub", n));

    box.appendChild(renderLoras());
    mount.replaceChildren(box);
  }

  function renderLoras() {
    const wrap = el("div", "native-loras");
    wrap.appendChild(el("label", null, t("nativeModels.loras")));
    const available = (data && data.loras) || [];
    state.loras.forEach((item, i) => {
      const row = el("div", "native-lora-row");
      const sel = nativeModelSelect(available, item.name || "", t("nativeModels.loraPick"));
      sel.onchange = () => { item.name = sel.value; persist(); };
      const strength = document.createElement("input");
      strength.type = "number";
      strength.step = "0.05";
      strength.min = "-4";
      strength.max = "4";
      strength.value = item.strength ?? 1;
      strength.title = t("nativeModels.strength");
      strength.oninput = () => {
        const v = Number(strength.value);
        item.strength = Number.isFinite(v) ? v : 1;
        persist();
      };
      const remove = el("button", "btn-secondary btn-sm", t("nativeModels.removeLora"));
      remove.type = "button";
      remove.onclick = () => { state.loras.splice(i, 1); persist(); render(document.getElementById(mountId)); };
      row.append(sel, strength, remove);
      wrap.appendChild(row);
    });
    if (!available.length) {
      wrap.appendChild(el("div", "sub", t("nativeModels.noLoras")));
    } else if (state.loras.length < NM_MAX_LORAS) {
      const add = el("button", "btn-secondary btn-sm", t("nativeModels.addLora"));
      add.type = "button";
      add.onclick = () => {
        state.loras.push({ name: available[0].name, strength: 1 });
        persist();
        render(document.getElementById(mountId));
      };
      wrap.appendChild(add);
    }
    return wrap;
  }

  function choice() {
    const models = {};
    for (const [k, v] of Object.entries(state.models)) if (v) models[k] = v;
    const loras = state.loras.filter((x) => x.name)
      .map((x) => ({ name: x.name, strength: Number.isFinite(Number(x.strength)) ? Number(x.strength) : 1 }));
    const out = {};
    if (Object.keys(models).length) out.native_models = models;
    if (loras.length) out.loras = loras;
    return out;
  }

  function setLoras(list) {
    state.loras = (list || []).filter((x) => x && x.name).slice(0, NM_MAX_LORAS)
      .map((x) => ({ name: String(x.name), strength: Number(x.strength ?? 1) }));
    persist();
    if (data) render(document.getElementById(mountId));
  }

  function setModels(models) {
    state.models = {};
    for (const [k, v] of Object.entries(models || {})) if (v) state.models[k] = String(v);
    persist();
    return refresh();
  }

  return { refresh, choice, setLoras, setModels, get state() { return state; } };
}

/** The music picker: one select per ACE-Step component. */
export function createMusicModelPicker({ mountId }) {
  const storeKey = "localm.nativeModels.music";
  let state = nmLoad(storeKey, {});
  let data = null;

  async function refresh() {
    const mount = document.getElementById(mountId);
    if (!mount) return;
    try {
      const r = await fetch("/api/music/native-models", { headers: authHeaders() });
      if (!r.ok) return;
      data = await r.json();
    } catch { return; }
    const box = el("div", "native-models");
    const comps = data.components || {};
    const cur = data.current || {};
    for (const key of Object.keys(NM_MUSIC_KEYS)) {
      const items = comps[key] || [];
      const sel = nativeModelSelect(items, state[key] || "",
        cur[key] ? t("nativeModels.settingsDefault", { name: cur[key] })
          : t("nativeModels.musicDefault"));
      sel.dataset.slot = key;
      sel.onchange = () => {
        if (sel.value) state[key] = sel.value; else delete state[key];
        nmSave(storeKey, state);
      };
      box.appendChild(nmField(t(NM_MUSIC_KEYS[key]), sel));
    }
    mount.replaceChildren(box);
  }

  function choice() {
    const models = {};
    for (const [k, v] of Object.entries(state)) if (v && NM_MUSIC_KEYS[k]) models[k] = v;
    return Object.keys(models).length ? { native_models: models } : {};
  }

  return { refresh, choice };
}
