// SPDX-License-Identifier: AGPL-3.0-or-later
/* localm GUI - model pickers for the native media backends (Images, Video,
   Music pages). The choice applies to the next generation only and is sent
   as `native_models` (and `loras` for image/video); "Settings default" and
   "Automatic" send nothing for that slot, so Settings and the automatic
   component pick decide. The choice is remembered per page in localStorage. */

"use strict";

import { authHeaders, el, toast } from "./helpers.js";
import { t } from "./i18n.js";
import { modelCache, refreshModels } from "./models-sidebar.js";

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

function nmStem(value) {
  return String(value || "").trim().toLowerCase().replace(/\.(safetensors|gguf|ckpt|sft|pt)$/, "");
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
  const others = items.filter((it) => it.fits === false);
  const groups = new Map();
  for (const it of items.filter((x) => x.fits !== false)) {
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
  for (const it of others) {
    const text = it.file && it.file !== it.name ? `${it.name} (${it.file})` : it.name;
    sel.appendChild(nmOption(it.name, t("nativeModels.otherFamily", { name: text })));
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
export function createNativeModelPicker({ media, prefix, mountId, onData }) {
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
    if (onData) onData(data);
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

  // The registered model or LoRA whose name or file stem equals *wanted* (an
  // imported image names its model by file or display name), or "".
  async function matchOption(kind, wanted) {
    if (!data) await refresh();
    const list = (data && (kind === "lora" ? data.loras : data.models)) || [];
    const w = nmStem(wanted);
    if (!w) return "";
    const hit = list.find((x) => nmStem(x.name) === w || nmStem(x.file) === w);
    return hit ? hit.name : "";
  }

  return {
    refresh, choice, setLoras, setModels,
    matchModel: (name) => matchOption("model", name),
    matchLora: (name) => matchOption("lora", name),
    // The family of the model a job would load (from the preview), or "".
    family: () => (data && data.preview && data.preview.family) || "",
    familyLabel: () => (data && data.preview && data.preview.family_label) || "",
    get state() { return state; },
  };
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
    const unsupported = data.unsupported || [];
    if (unsupported.length) {
      const list = el("div", "sub native-models-unsupported");
      list.appendChild(el("div", null, t("nativeModels.music.unsupported")));
      for (const u of unsupported) {
        const file = u.file && u.file !== u.name ? ` (${u.file})` : "";
        list.appendChild(el("div", null, `${u.name}${file}: ${u.reason}`));
      }
      box.appendChild(list);
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

/* ================================================================ */
/*  Enhance prompt                                                   */
/* ================================================================ */

// Image families whose text encoders read tag lists; every other image family
// reads natural language.
const PE_TAG_FAMILIES = new Set(["sd1", "sd2", "sdxl"]);
const PE_REPLY_RULES = "Reply with the rewritten prompt only: no preface, no explanation, no quotation marks, " +
  "no markdown. Keep every subject, detail and constraint the user gave; add only what the style needs.";

/** The system prompt that rewrites a short idea for *media* ("image", "video"
 *  or "music") and the model *family* ("" when unknown). */
export function enhanceSystemPrompt(media, family, label) {
  const name = label || family || "";
  if (media === "music") {
    return "You write style tags for an AI music generator (ACE-Step). Turn the user's idea into one line of " +
      "comma-separated tags: genre and sub-genre, mood, instruments, tempo as a bpm number, vocal type " +
      "(or instrumental), era and production style. At most 15 tags, no lyrics. " + PE_REPLY_RULES;
  }
  if (media === "video") {
    const target = name ? ` for the text-to-video model ${name}` : " for a text-to-video model";
    return `You rewrite short video ideas into a prompt${target}. Write one to three sentences in the present ` +
      "tense that state the subject, the motion (use concrete motion verbs), the camera movement, the lighting " +
      "and the setting. " + PE_REPLY_RULES;
  }
  if (PE_TAG_FAMILIES.has(family)) {
    return `You rewrite short image ideas into prompts for ${name || "Stable Diffusion"}, which reads tag-style ` +
      "prompts. Write one line of comma-separated phrases: subject, medium or style, composition, lighting, " +
      "colour palette, then quality terms. Do not write full sentences. " + PE_REPLY_RULES;
  }
  const target = name ? ` for ${name}` : " for a text-to-image model";
  return `You rewrite short image ideas into prompts${target}, which understands natural language. Write one or ` +
    "two descriptive sentences covering the subject, setting, composition, lighting, colour and style. " +
    "Do not use tag lists or quality keywords. " + PE_REPLY_RULES;
}

/** The enhanced prompt text from a chat reply: reasoning blocks and wrapping
 *  quotes or code fences removed. */
export function cleanEnhancedPrompt(raw) {
  let text = String(raw || "").replace(/<think>[\s\S]*?<\/think>/gi, "").trim();
  const fence = text.match(/^```[a-z]*\n([\s\S]*?)\n?```$/i);
  if (fence) text = fence[1].trim();
  for (const [open, close] of [['"', '"'], ["'", "'"], ["“", "”"]]) {
    if (text.length > 1 && text.startsWith(open) && text.endsWith(close)) {
      text = text.slice(1, -1).trim();
      break;
    }
  }
  return text;
}

/** Asks the loaded chat model to rewrite *prompt*. Throws an Error whose
 *  message is for the user when no chat model is loaded, the request fails or
 *  the reply is empty. */
export async function requestEnhancedPrompt({ media, prompt, family = "", familyLabel = "" }) {
  await refreshModels();
  const model = modelCache.active;
  if (!model) throw new Error(t("enhance.noModel"));
  const r = await fetch("/v1/chat/completions", {
    method: "POST",
    headers: authHeaders(),
    body: JSON.stringify({
      model,
      pin_model: true,
      messages: [
        { role: "system", content: enhanceSystemPrompt(media, family, familyLabel) },
        { role: "user", origin: "client", content: prompt },
      ],
      max_tokens: 512,
      temperature: 0.7,
      stream: false,
    }),
  });
  let data = null;
  try { data = await r.json(); } catch { /* reported below by status */ }
  if (!r.ok) {
    const detail = data && data.detail
      ? (typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail)) : r.statusText;
    throw new Error(detail || String(r.status));
  }
  const choice = data && data.choices && data.choices[0];
  const text = cleanEnhancedPrompt(choice && choice.message && choice.message.content);
  if (!text) throw new Error(t("enhance.empty", { model }));
  return text;
}

/** Adds an "Enhance prompt" button under the textarea *fieldId*. *familyOf*
 *  returns `{family, label}` for the model that will generate. The result
 *  replaces the field's text (still editable) and Undo puts the original back.
 *  Returns the row the buttons sit in, or null when the field is missing. */
export function mountEnhanceButton({ media, fieldId, familyOf }) {
  const field = document.getElementById(fieldId);
  if (!field) return null;
  const row = el("div", "enhance-row");
  const button = el("button", "btn-secondary btn-sm", t("enhance.button"));
  button.type = "button";
  button.title = t("enhance.title");
  const undo = el("button", "btn-secondary btn-sm", t("enhance.undo"));
  undo.type = "button";
  undo.hidden = true;
  row.append(button, undo);
  field.insertAdjacentElement("afterend", row);

  let original = "";
  button.onclick = async () => {
    const prompt = field.value.trim();
    if (!prompt) { toast(t("enhance.enterPrompt"), true); return; }
    button.disabled = true;
    button.textContent = t("enhance.working");
    try {
      const fam = (familyOf && familyOf()) || {};
      const text = await requestEnhancedPrompt({
        media, prompt, family: fam.family || "", familyLabel: fam.label || "" });
      original = field.value;
      field.value = text;
      field.dispatchEvent(new Event("input", { bubbles: true }));
      undo.hidden = false;
      toast(t("enhance.done"));
    } catch (e) {
      toast(t("enhance.failed", { message: e.message }), true);
    } finally {
      button.disabled = false;
      button.textContent = t("enhance.button");
    }
  };
  undo.onclick = () => {
    field.value = original;
    field.dispatchEvent(new Event("input", { bubbles: true }));
    undo.hidden = true;
  };
  return row;
}
