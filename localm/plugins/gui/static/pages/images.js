// SPDX-License-Identifier: AGPL-3.0-or-later
/* localm GUI - Images page. The library (grid, selection, bulk actions, detail
   modal, rename/move/delete) comes from app/media-gallery.js; this file adds
   the LoRA picker and the image-only detail actions (use as img2img input,
   send to chat, copy to clipboard). */

"use strict";

import { chat, renderAttachChips } from "../app/chat.js";
import { $, authHeaders, cancelJob, checkModelsBeforeGenerate, createJobPreview, el, fetchImageURL, jobStatusWord, revealFilledAdvanced, streamJob, toast } from "../app/helpers.js";
import { t } from "../app/i18n.js";
import { bindReloadToggle, createGallery, imagePreview, refreshReloadToggle } from "../app/media-gallery.js";
import { createNativeModelPicker, mountEnhanceButton } from "../app/native-models.js";
import { showView } from "../app/tabs.js";
import { modelOverrides } from "./workflow.js";

/* ================================================================ */
/*  Image library                                                    */
/* ================================================================ */

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
    ctx.fileURL(item.name, img, "load").then((url) => (img.src = url));
    body.appendChild(img);
  },

  reuse: (item) => {
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
      imageModelPicker.setModels({ model: m.model, ...(m.components || {}) });
      imageModelPicker.setLoras(m.loras || []);
    }
    // Most of those ids live behind this page's Advanced fold; open it when
    // they are filled.
    revealFilledAdvanced($("view-images"));
  },

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

    return [useInput, toChat, copyImg];
  },

  beforeRefresh: () => {
    refreshImageBackend().then(() => { if (imageBackend.active !== "native") refreshLoraPicker(); });
    refreshReloadToggle("image", "img-reload-llm");
  },
});

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

export const imageModelPicker = createNativeModelPicker(
  { media: "image", prefix: "imagine", mountId: "img-native-models",
    onData: showImageStepDefault });

const imageLivePreview = createJobPreview("img-preview");

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
    note.textContent = text;
    note.hidden = !text;
  }
  for (const id of ["img-lora-field", "img-workflow-card"]) {
    const node = $(id);
    if (node) node.hidden = native;
  }
  const picker = $("img-native-models");
  if (picker) {
    picker.hidden = !native;
    if (native) imageModelPicker.refresh();
  }
  for (const node of document.querySelectorAll("#view-images .img-comfy-only")) {
    node.hidden = native;
  }
  for (const node of document.querySelectorAll("#view-images .img-native-only")) {
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

$("img-generate").onclick = async () => {
  const promptText = $("img-prompt").value.trim();
  if (!promptText) { toast(t("images.enterPrompt"), true); return; }
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
  if (!native && modelOverrides.image && Object.keys(modelOverrides.image).length) {
    body.model_overrides = modelOverrides.image;
  }
  const nativeChoice = native ? imageModelPicker.choice() : {};
  Object.assign(body, nativeChoice);
  if (native) {
    body.steps = num("img-steps");
    body.clip_skip = num("img-clip-skip");
  }
  $("img-generate").disabled = true;
  imageLivePreview.clear();
  const log = $("img-log");
  log.style.display = "block";
  log.textContent = "";
  $("img-result").replaceChildren();
  try {
    await checkModelsBeforeGenerate("image", log,
      { model_overrides: native ? undefined : modelOverrides.image, lora_name: loraName || undefined,
        native_models: nativeChoice.native_models });
    if (native) refreshImageBackend();
    const r = await fetch("/api/imagine", {
      method: "POST", headers: authHeaders(), body: JSON.stringify(body),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || r.statusText);
    showStop("img-stop", data.job_id);
    const end = await streamJob(data.job_id, (line) => {
      log.textContent += line + "\n";
      log.scrollTop = log.scrollHeight;
    }, undefined, imageLivePreview.show);
    imageLivePreview.clear();
    if (end.status === "done" && end.result) {
      const img = document.createElement("img");
      img.src = await fetchImageURL(
        `/api/imagine/file/${encodeURIComponent(end.result)}`);
      $("img-result").appendChild(img);
      toast(t("images.generatedToast"));
      refreshImageHistory();
    } else {
      toast(t("images.generationStatus", { status: jobStatusWord(end.status) }), end.status !== "cancelled");
    }
  } catch (e) {
    toast(t("images.generationFailed", { message: e.message }), true);
  } finally {
    $("img-generate").disabled = false;
    imageLivePreview.clear();
    hideStop("img-stop");
  }
};

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
  for (const [key, id] of [["steps", "img-steps"], ["sampler", "img-sampler"], ["scheduler", "img-scheduler"]]) {
    if (p[key] === undefined || p[key] === null) continue;
    if (!put(id, p[key])) skipped.push(key);
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
  } else if (p.model || (p.loras && p.loras.length)) {
    skipped.push("model", "LoRA");
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
  const skipped = await applyImportedParams(data);
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
