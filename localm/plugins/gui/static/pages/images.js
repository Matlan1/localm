// SPDX-License-Identifier: AGPL-3.0-or-later
/* localm GUI - Images page. The library (grid, selection, bulk actions, detail
   modal, rename/move/delete) comes from app/media-gallery.js; this file adds
   the LoRA picker and the image-only detail actions (use as img2img input,
   send to chat, copy to clipboard). */

"use strict";

import { chat, renderAttachChips } from "../app/chat.js";
import { $, authHeaders, cancelJob, checkModelsBeforeGenerate, createJobPreview, createMissingModelOffer, dismissErrorToast, el, fetchImageURL, jobStatusWord, revealFilledAdvanced, streamJob, toast } from "../app/helpers.js";
import { t } from "../app/i18n.js";
import { createImageInputs } from "../app/image-inputs.js";
import { bindReloadToggle, createGallery, createJobFollower, imagePreview, refreshReloadToggle } from "../app/media-gallery.js";
import { createNativeGenControls, createNativeModelPicker, mountEnhanceButton } from "../app/native-models.js";
import { createNativeWorkflowCard, nwfNumber, nwfSetField, nwfSetSelect } from "../app/native-workflows.js";
import { chatPageModelChoice } from "../app/slash.js";
import { showView } from "../app/tabs.js";
import { modelOverrides } from "./workflow.js";

/* ================================================================ */
/*  Image library                                                    */
/* ================================================================ */

const imageJobs = createJobFollower();

const imageGallery = createGallery({
  slug: "imagine",
  listKey: "images",
  itemKey: "images.item",
  gridId: "img-history",
  bulkId: "img-bulk",
  moveDestKey: "localm.imgMoveDest",
  emptyIcon: "image",
  emptyTitleKey: "images.empty.title",
  emptyHintKey: "images.empty.hint",

  buildPreview: imagePreview,
  caption: (item) => (item.meta?.prompt ? item.meta.prompt.slice(0, 60) : item.name),

  buildDetailPreview: (item, body, ctx) => {
    const img = document.createElement("img");
    img.style.maxWidth = "100%";
    img.style.borderRadius = "8px";
    const unavailable = (why) => img.replaceWith(el("div", "sub", why));
    img.addEventListener("error", () => unavailable(t("mediaGallery.previewUnavailable")));
    ctx.fileURL(item.name, img, "load").then((url) => (img.src = url))
      .catch(() => unavailable(t("mediaGallery.imageLoadFailed")));
    body.appendChild(img);
  },

  reuse: (item) => reuseImageSettings(item),

  // Still-image-only actions.
  extraActions: (item, ctx) => {
    const useInput = el("button", "btn-secondary", t("images.useAsInput"));
    useInput.title = t("images.useAsInputTitle");
    useInput.onclick = () => {
      $("img-input").value = item.path || item.name;
      ctx.closeModal();
      toast(t("images.setAsInputToast"));
    };

    const toChat = el("button", "btn-secondary", t("images.sendToChat"));
    toChat.title = t("images.sendToChatTitle");
    toChat.onclick = async () => {
      try {
        const url = await fetchImageURL(
          `/api/imagine/file/${encodeURIComponent(item.name)}`);
        const blob = await (await fetch(url)).blob();
        URL.revokeObjectURL(url);
        const dataUri = await new Promise((resolve, reject) => {
          const reader = new FileReader();
          reader.onload = () => resolve(reader.result);
          reader.onerror = reject;
          reader.readAsDataURL(blob);
        });
        chat.attachments.push({ name: item.name, dataUri });
        renderAttachChips();
        ctx.closeModal();
        showView("chat");
        toast(t("images.attachedToast"));
      } catch (e) {
        toast(t("images.attachFailed", { message: e.message }), true);
      }
    };

    const copyImg = el("button", "btn-secondary", t("images.copyImage"));
    copyImg.title = t("images.copyImageTitle");
    copyImg.onclick = async () => {
      try {
        const url = await fetchImageURL(
          `/api/imagine/file/${encodeURIComponent(item.name)}`);
        const blob = await (await fetch(url)).blob();
        URL.revokeObjectURL(url);
        await navigator.clipboard.write([new ClipboardItem({ "image/png": blob })]);
        toast(t("images.copiedToast"));
      } catch (e) {
        toast(t("images.copyFailed", { message: e.message }), true);
      }
    };

    const actions = [useInput, toChat, copyImg];
    if (imageBackend.active === "native") actions.push(buildUpscaleAction(item, ctx));
    return actions;
  },

  beforeRefresh: () => {
    refreshImageBackend().then(() => { if (imageBackend.active !== "native") refreshLoraPicker(); });
    refreshReloadToggle("image", "img-reload-llm");
    imageJobs.adopt("imagine", followAdoptedImageJob);
  },
});

/* "Reuse settings" on a gallery item: writes its sidecar settings back into the form. */
function reuseImageSettings(item) {
  const m = item.meta || {};
  $("img-prompt").value = m.prompt || "";
  $("img-negative").value = m.negative_prompt || "";
  $("img-seed").value = m.seed ?? "";
  $("img-guidance").value = m.guidance ?? "";
  $("img-denoise").value = m.denoise ?? "";
  $("img-cfg").value = m.cfg ?? "";
  $("img-steps").value = m.backend === "native" ? (m.steps ?? "") : "";
  $("img-clip-skip").value = m.clip_skip ?? "";
  $("img-input").value = m.input_image || "";
  $("img-lora").value = m.lora_name || "";
  $("img-lora-strength-model").value = m.lora_strength_model ?? "";
  $("img-lora-strength-clip").value = m.lora_strength_clip ?? "";
  if (m.backend === "native") {
    $("img-size").value = m.size || "";
    imageModelPicker.setModels({ model: m.model, ...(m.components || {}) });
    imageModelPicker.setLoras(m.loras || []);
    imageGen.set(m);
    imageInputs.setControl(m);
  }
  // Most of those ids live behind this page's Advanced fold; open it when
  // they are filled.
  revealFilledAdvanced($("view-images"));
}

export const refreshImageHistory = imageGallery.refresh;

/* This page's detail-view entry point. */
export const showImageDetail = (item) => imageGallery.showDetail(item);

bindReloadToggle("image", "img-reload-llm");

/* Which backend generates (from /api/imagine/backend). The native backend has
   no ComfyUI workflow, workflow model picks or LoRA picker, so those controls
   are hidden while it is active; the note under the Generate heading says
   which backend runs and which model it uses. */
export const imageBackend = { active: null, choice: null };

/* The steps box shows the picked model family's step count as its placeholder. */
function showImageStepDefault(data) {
  const input = $("img-steps");
  if (!input) return;
  const steps = data && data.preview && data.preview.defaults && data.preview.defaults.steps;
  input.placeholder = steps ? t("images.stepsPlaceholderValue", { value: steps })
    : t("images.stepsPlaceholder");
}

/* Mask, reference images, ControlNet and embeddings (native backend only). */
export const imageInputs = createImageInputs({
  mountId: "img-native-inputs", promptId: "img-prompt", negativeId: "img-negative",
  inputId: "img-input", denoiseId: "img-denoise" });

export const imageModelPicker = createNativeModelPicker(
  { media: "image", prefix: "imagine", mountId: "img-native-models",
    onData: (data) => {
      showImageStepDefault(data); imageInputs.refresh(data);
      imageGen.suggest(data && data.preview && data.preview.hires);
    } });

const imageLivePreview = createJobPreview("img-preview");

const imageGen = createNativeGenControls({ prefix: "img" });

/* The native workflow card: presets of models and the settings below. */
export const imageWorkflowCard = createNativeWorkflowCard({
  media: "image", mountId: "img-native-workflows", picker: imageModelPicker, logId: "img-log",
  fields: {
    read: () => {
      const size = /^(\d+)\s*x\s*(\d+)$/i.exec($("img-size").value.trim());
      return {
        negative_prompt: $("img-negative").value.trim() || null,
        steps: nwfNumber("img-steps"), cfg: nwfNumber("img-cfg"),
        guidance: nwfNumber("img-guidance"), clip_skip: nwfNumber("img-clip-skip"),
        width: size ? Number(size[1]) : null, height: size ? Number(size[2]) : null,
        sample_method: $("img-sampler")?.value || null, scheduler: $("img-scheduler")?.value || null,
      };
    },
    apply: (p) => {
      if (p.prompt) nwfSetField("img-prompt", p.prompt);
      if (p.negative_prompt !== undefined) nwfSetField("img-negative", p.negative_prompt);
      for (const [key, id] of [["steps", "img-steps"], ["cfg", "img-cfg"],
                               ["guidance", "img-guidance"], ["clip_skip", "img-clip-skip"]]) {
        nwfSetField(id, p[key]);
      }
      nwfSetField("img-size", p.width && p.height ? `${p.width}x${p.height}` : "");
      nwfSetSelect($("img-sampler"), p.sample_method || "");
      nwfSetSelect($("img-scheduler"), p.scheduler || "");
      revealFilledAdvanced($("view-images"));
    },
  },
});

/* The detail-view "Upscale" control: a registered upscaler model and a button
   that starts the upscale job (a new gallery image). */
function buildUpscaleAction(item, ctx) {
  const row = el("div", "native-lora-row");
  const sel = document.createElement("select");
  sel.title = t("images.upscaleModel");
  const btn = el("button", "btn-secondary", t("images.upscale"));
  btn.type = "button";
  btn.title = t("images.upscaleTitle");
  btn.disabled = true;
  row.append(sel, btn);
  (async () => {
    let list = [];
    let listed = false;
    try {
      const r = await fetch("/api/imagine/native-models", { headers: authHeaders() });
      if (r.ok) { list = (await r.json()).upscalers || []; listed = true; }
    } catch { /* offline: the button stays disabled */ }
    if (!listed) {
      sel.hidden = true;
      btn.title = t("images.upscaleListFailed");
      return;
    }
    for (const u of list) {
      const o = document.createElement("option");
      o.value = u.name;
      o.textContent = u.file && u.file !== u.name ? `${u.name} (${u.file})` : u.name;
      sel.appendChild(o);
    }
    if (list.length) { btn.disabled = false; } else {
      sel.hidden = true;
      btn.title = t("images.upscaleNone");
    }
  })();
  btn.onclick = () => { ctx.closeModal(); runImageUpscale(item.name, sel.value); };
  return row;
}

export async function refreshImageBackend() {
  let data;
  try {
    const r = await fetch("/api/imagine/backend", { headers: authHeaders() });
    if (!r.ok) return;
    data = await r.json();
  } catch { return; }
  imageBackend.active = data.active || null;
  imageBackend.choice = data.choice || null;
  const native = imageBackend.active === "native";
  const note = $("img-backend-note");
  if (note) {
    let text = "";
    if (native) {
      const n = data.native || {};
      text = n.model
        ? t("images.backendNative", { model: n.model, runtime: n.runtime || t("images.backendRuntimeOnFirstUse") })
        : t("images.backendNativeNoModel", { name: (n.recommended || {}).name || "" });
    } else if (imageBackend.active === "comfy") {
      text = t("images.backendComfy");
    }
    if (data.warning) text = text ? `${text} ${data.warning}` : data.warning;
    note.textContent = text;
    note.hidden = !text;
  }
  for (const id of ["img-lora-field", "img-workflow-card"]) {
    const node = $(id);
    if (node) node.hidden = native;
  }
  const workflows = $("img-native-workflow-card");
  if (workflows) {
    workflows.hidden = !native;
    if (native) imageWorkflowCard.refresh();
  }
  const picker = $("img-native-models");
  if (picker) {
    picker.hidden = !native;
    if (native) imageModelPicker.refresh().then(() => imageGen.populate(imageModelPicker.data));
  }
  for (const node of document.querySelectorAll("#view-images .img-comfy-only")) {
    node.hidden = native;
  }
  const cfg = $("img-cfg");
  if (cfg) {
    cfg.dataset.i18nPlaceholder = native ? "images.cfgPlaceholderNative" : "images.cfgPlaceholder";
    cfg.placeholder = t(cfg.dataset.i18nPlaceholder);
  }
  for (const node of document.querySelectorAll(
    "#view-images .img-native-only, #view-images .img-native-gen, #view-images .img-native-extra")) {
    node.hidden = !native;
  }
}

/* LoRA picker - populated from ComfyUI's live-installed LoRA files via
   /api/imagine/comfy-models. Keeps the current selection across a refresh. */
export async function refreshLoraPicker() {
  const sel = $("img-lora");
  if (!sel) return;
  const previous = sel.value;
  let data;
  try {
    const r = await fetch("/api/imagine/comfy-models", { headers: authHeaders() });
    data = await r.json();
  } catch { return; }
  sel.replaceChildren();
  const none = document.createElement("option");
  none.value = "";
  none.textContent = data.reachable ? t("images.loraNone") : t("images.loraNoneOffline");
  sel.appendChild(none);
  for (const name of (data.loras || [])) {
    const o = document.createElement("option");
    o.value = name;
    o.textContent = name;
    sel.appendChild(o);
  }
  sel.value = [...sel.options].some((o) => o.value === previous) ? previous : "";
}

// Media-generation Stop button: reveal it while a job runs and wire it to
// cancel that job; hide it again when the job ends. Shared by image/music/video.
export function showStop(btnId, jobId) {
  const btn = $(btnId);
  if (!btn) return;
  btn.style.display = "inline-block";
  btn.disabled = false;
  btn.onclick = () => { btn.disabled = true; btn.textContent = t("images.stopping"); cancelJob(jobId); };
}
export function hideStop(btnId) {
  const btn = $(btnId);
  if (!btn) return;
  btn.style.display = "none";
  btn.disabled = false;
  btn.textContent = t("images.stop");
  btn.onclick = null;
}

/* ================================================================ */
/*  Generation                                                       */
/* ================================================================ */

function addResultImage(name, shown) {
  if (shown.has(name)) return;
  shown.add(name);
  const img = document.createElement("img");
  $("img-result").appendChild(img);
  fetchImageURL(`/api/imagine/file/${encodeURIComponent(name)}`)
    .then((url) => { img.src = url; })
    .catch((e) => toast(t("images.resultLoadFailed", { name, message: e.message }), true));
}

/* Follow image job *jobId*: log lines into *log*, each saved image into the
   result area as it arrives, then the end status. */
async function followImageJob(jobId, log) {
  showStop("img-stop", jobId);
  const shown = new Set();
  const missing = createMissingModelOffer(log);
  const end = await streamJob(jobId, (line) => {
    if (!missing.show(line)) return;
    log.textContent += line + "\n";
    log.scrollTop = log.scrollHeight;
  }, undefined, imageLivePreview.show, (ev) => {
    missing.onEvent(ev);
    if (ev.type === "image" && ev.name) addResultImage(ev.name, shown);
  });
  imageLivePreview.clear();
  if (end.status === "done" && end.result) addResultImage(end.result, shown);
  refreshImageHistory();
  if (end.status !== "done") await missing.offer();
  return end;
}

async function runImageUpscale(name, upscaler) {
  if (!upscaler) { toast(t("images.upscaleNone"), true); return; }
  await imageJobs.run(() => upscaleImage(name, upscaler));
}

async function upscaleImage(name, upscaler) {
  $("img-generate").disabled = true;
  const log = $("img-log");
  log.style.display = "block";
  log.textContent = "";
  $("img-result").replaceChildren();
  log.scrollIntoView?.({ block: "nearest" });
  try {
    const r = await fetch(`/api/imagine/file/${encodeURIComponent(name)}/upscale`, {
      method: "POST", headers: authHeaders(), body: JSON.stringify({ upscaler }),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || r.statusText);
    const end = await followImageJob(data.job_id, log);
    if (end.status === "done") toast(t("images.upscaledToast"));
    else toast(t("images.upscaleStatus", { status: jobStatusWord(end.status) }), end.status !== "cancelled");
  } catch (e) {
    toast(t("images.upscaleFailed", { message: e.message }), true);
  } finally {
    $("img-generate").disabled = false;
    hideStop("img-stop");
  }
}

$("img-generate").onclick = async () => {
  const promptText = $("img-prompt").value.trim();
  if (!promptText) { toast(t("images.enterPrompt"), true); return; }
  dismissErrorToast(t("images.enterPrompt"));
  const num = (id) => {
    const v = $(id).value.trim();
    return v === "" ? null : Number(v);
  };
  const body = {
    prompt: promptText,
    negative_prompt: $("img-negative").value.trim() || null,
    seed: num("img-seed"),
    guidance: num("img-guidance"),
    cfg: num("img-cfg"),
    denoise: num("img-denoise"),
    input_image: $("img-input").value.trim() || null,
    size: $("img-size").value.trim() || null,
  };
  const native = imageBackend.active === "native";
  const loraName = native ? "" : $("img-lora").value;
  if (loraName) {
    body.lora_name = loraName;
    body.lora_strength_model = num("img-lora-strength-model");
    body.lora_strength_clip = num("img-lora-strength-clip");
  }
  if (native && imageWorkflowCard.pending) {
    $("img-generate").disabled = true;
    let ready;
    try { ready = await imageWorkflowCard.prepare(); } finally { $("img-generate").disabled = false; }
    if (!ready) return;
  }
  const picked = imageModelFields(native);
  const nativeChoice = native ? picked : {};
  Object.assign(body, picked);
  if (native) {
    const problem = imageInputs.problem();
    if (problem) { toast(problem, true); return; }
    body.steps = num("img-steps");
    body.clip_skip = num("img-clip-skip");
    Object.assign(body, imageGen.body(), imageInputs.body());
  }
  await imageJobs.run(() => generateImage(body, native, loraName, nativeChoice));
};

/* The model fields Generate sends: the native picker's models and LoRAs on the
   native backend, the workflow panel's model picks on ComfyUI (none when
   nothing was picked). */
function imageModelFields(native) {
  if (native) return imageModelPicker.choice();
  const picks = modelOverrides.image;
  return picks && Object.keys(picks).length ? { model_overrides: picks } : {};
}

/* The model fields of this page's pick for the chat's /generate-image, read
   against the active backend. */
export async function imagePageModelChoice() {
  await refreshImageBackend();
  return imageModelFields(imageBackend.active === "native");
}
chatPageModelChoice.image = imagePageModelChoice;

async function generateImage(body, native, loraName, nativeChoice) {
  $("img-generate").disabled = true;
  imageLivePreview.clear();
  const log = $("img-log");
  log.style.display = "block";
  log.textContent = "";
  $("img-result").replaceChildren();
  try {
    const proceed = await checkModelsBeforeGenerate("image", log,
      { model_overrides: native ? undefined : modelOverrides.image, lora_name: loraName || undefined,
        native_models: nativeChoice.native_models });
    if (!proceed) return;
    if (native) refreshImageBackend();
    const r = await fetch("/api/imagine", {
      method: "POST", headers: authHeaders(), body: JSON.stringify(body),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || r.statusText);
    const end = await followImageJob(data.job_id, log);
    if (end.status === "done" && end.result) toast(t("images.generatedToast"));
    else toast(t("images.generationStatus", { status: jobStatusWord(end.status) }), end.status !== "cancelled");
  } catch (e) {
    toast(t("images.generationFailed", { message: e.message }), true);
  } finally {
    $("img-generate").disabled = false;
    imageLivePreview.clear();
    hideStop("img-stop");
  }
}

/* Follow an image job this page did not start (one from the chat's
   /generate-image), into the same log and result area. */
async function followAdoptedImageJob(jobId) {
  $("img-generate").disabled = true;
  const log = $("img-log");
  log.style.display = "block";
  log.textContent = "";
  $("img-result").replaceChildren();
  try {
    const end = await followImageJob(jobId, log);
    if (end.status === "done") toast(t("images.generatedToast"));
    else toast(t("images.generationStatus", { status: jobStatusWord(end.status) }), end.status !== "cancelled");
  } catch (e) {
    toast(t("images.generationFailed", { message: e.message }), true);
  } finally {
    $("img-generate").disabled = false;
    imageLivePreview.clear();
    hideStop("img-stop");
  }
}

/* ================================================================ */
/*  Prompt helpers: enhance, import parameters from an image         */
/* ================================================================ */

const IMPORT_SOURCES = new Set(["localm", "a1111", "comfyui", "sidecar"]);

/* Fills this page from the `{source, params}` the import route returns:
   prompt, negative prompt, seed, guidance, CFG, denoise and size, then (native
   backend) the model and LoRAs matched against the registered ones. A localm
   image or sidecar holds the whole record, so a field it lacks is cleared; a
   foreign image holds only what its tool wrote, so a field it lacks is left
   as it is. Returns the names of the parameters it could not apply. */
export async function applyImportedParams(result) {
  const p = (result && result.params) || {};
  const own = result.source === "localm" || result.source === "sidecar";
  const skipped = [];
  const put = (id, value) => {
    const node = $(id);
    if (!node) return false;
    if (value === undefined || value === null) { if (own) node.value = ""; return true; }
    node.value = value;
    return true;
  };
  put("img-prompt", p.prompt);
  put("img-negative", p.negative_prompt);
  put("img-seed", p.seed);
  put("img-guidance", p.guidance);
  put("img-cfg", p.cfg);
  put("img-denoise", p.denoise);
  put("img-size", p.width && p.height ? `${p.width}x${p.height}` : null);
  if (!own && (p.prompt === undefined || p.prompt === null)) skipped.push("prompt");
  if (p.steps !== undefined && p.steps !== null && !put("img-steps", p.steps)) skipped.push("steps");
  if (!own && p.clip_skip !== undefined && p.clip_skip !== null && !put("img-clip-skip", p.clip_skip)) {
    skipped.push("clip skip");
  }
  if (own) {
    imageGen.set(p);
  } else {
    /* A foreign tool names samplers its own way; only a name this build offers is applied. */
    for (const [value, id, key] of [[p.sample_method || p.sampler, "img-sampler", "sampler"],
                                    [p.scheduler, "img-scheduler", "scheduler"]]) {
      if (!value) continue;
      const node = $(id);
      if (node && [...node.options].some((o) => o.value === value)) node.value = value;
      else skipped.push(key);
    }
  }
  if (imageBackend.active === "native") {
    if (p.model) {
      const name = await imageModelPicker.matchModel(p.model);
      if (name) {
        await imageModelPicker.setModels(own ? { ...(p.components || {}), model: name } : { model: name });
        if (imageModelPicker.state.models.model !== name) skipped.push(`model ${p.model}`);
      } else {
        skipped.push(`model ${p.model}`);
      }
    }
    if (p.loras && p.loras.length) {
      const kept = [];
      for (const item of p.loras) {
        const name = await imageModelPicker.matchLora(item.name);
        if (name) kept.push({ name, strength: item.strength }); else skipped.push(`LoRA ${item.name}`);
      }
      imageModelPicker.setLoras(kept);
    }
  } else {
    if (p.model) skipped.push("model");
    if (p.loras && p.loras.length) skipped.push("LoRA");
  }
  revealFilledAdvanced($("view-images"));
  return skipped;
}

export async function importParamsFromFile(file) {
  const r = await fetch("/api/imagine/import-params", {
    method: "POST",
    headers: authHeaders({ "Content-Type": "application/octet-stream" }),
    body: file,
  });
  let data = null;
  try { data = await r.json(); } catch { /* reported below by status */ }
  if (!r.ok) throw new Error((data && data.detail) || r.statusText);
  const skipped = [...(await applyImportedParams(data)), ...(data.unsupported || [])];
  const source = IMPORT_SOURCES.has(data.source) ? t(`images.import.source.${data.source}`) : data.source;
  toast(skipped.length
    ? t("images.import.doneSkipped", { source, items: skipped.join(", ") })
    : t("images.import.done", { source }));
  return data;
}

const imageHelperRow = mountEnhanceButton({
  media: "image", fieldId: "img-prompt",
  familyOf: () => ({ family: imageModelPicker.family(), label: imageModelPicker.familyLabel() }),
});
if (imageHelperRow) {
  const picker = el("input");
  picker.type = "file";
  picker.accept = ".png,image/png,.json,application/json";
  picker.hidden = true;
  const importBtn = el("button", "btn-secondary btn-sm", t("images.import.button"));
  importBtn.type = "button";
  importBtn.title = t("images.import.title");
  importBtn.dataset.i18n = "images.import.button";
  importBtn.dataset.i18nTitle = "images.import.title";
  importBtn.onclick = () => picker.click();
  picker.onchange = async () => {
    const file = picker.files && picker.files[0];
    picker.value = "";
    if (!file) return;
    try {
      await importParamsFromFile(file);
    } catch (e) {
      toast(t("images.import.failed", { message: e.message }), true);
    }
  };
  imageHelperRow.append(importBtn, picker);
}
