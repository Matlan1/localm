// SPDX-License-Identifier: AGPL-3.0-or-later
/* localm GUI - the native image backend's extra inputs on the Images page:
   a mask painted over the base image (inpaint, and outpaint by extending the
   canvas), reference images for edit models, a ControlNet with its control
   image, and textual-inversion embeddings. Images stay in this page's memory
   only and go to the server as data URLs with the next generation. */

"use strict";

import { $, el, fetchImageURL, openModal, toast } from "./helpers.js";
import { t } from "./i18n.js";

const IIN_MAX_REFS = 4;
const IIN_MAX_SIDE = 4096;
const IIN_TYPES = new Set(["image/png", "image/jpeg", "image/webp"]);
const IIN_BRUSH_DEFAULT = 48;

/* *file* (a File) as a data URL, or null with a toast when it is not a PNG,
   JPEG or WebP image. */
function iinReadFile(file) {
  if (!file || !IIN_TYPES.has(file.type)) {
    toast(t("imageInputs.badFile", { name: (file && file.name) || "" }), true);
    return Promise.resolve(null);
  }
  return new Promise((resolve) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => { toast(t("imageInputs.badFile", { name: file.name }), true); resolve(null); };
    reader.readAsDataURL(file);
  });
}

/* A hidden <input type=file> for images; *onFiles* receives the chosen Files. */
function iinFilePicker(multiple, onFiles) {
  const input = document.createElement("input");
  input.type = "file";
  input.accept = "image/png,image/jpeg,image/webp";
  input.multiple = multiple;
  input.hidden = true;
  input.onchange = () => {
    const files = [...(input.files || [])];
    input.value = "";
    if (files.length) onFiles(files);
  };
  return input;
}

function iinThumb(url, onRemove) {
  const box = el("div", "iin-thumb");
  const img = document.createElement("img");
  img.src = url;
  img.alt = "";
  box.appendChild(img);
  if (onRemove) {
    const x = el("button", "btn-secondary btn-sm", t("imageInputs.remove"));
    x.type = "button";
    x.onclick = onRemove;
    box.appendChild(x);
  }
  return box;
}

/* Whether *word* appears in *text* as a whole word, ignoring case. */
export function iinHasWord(text, word) {
  if (!word) return false;
  const esc = word.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  return new RegExp(`(^|[^\\w])${esc}($|[^\\w])`, "i").test(String(text || ""));
}

/* An <img> of *url*, loaded. */
function iinLoadImage(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(t("imageInputs.loadFailed")));
    img.src = url;
  });
}

function iinCanvas(w, h) {
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  return c;
}

/* A copy of *base* (a canvas) grown by *pad* ({left, top, right, bottom}),
   the new border filled by stretching the edge pixels outwards. */
export function iinExtendBase(base, pad) {
  const w = base.width, h = base.height;
  const out = iinCanvas(w + pad.left + pad.right, h + pad.top + pad.bottom);
  const g = out.getContext("2d");
  g.drawImage(base, pad.left, pad.top);
  if (pad.left) g.drawImage(base, 0, 0, 1, h, 0, pad.top, pad.left, h);
  if (pad.right) g.drawImage(base, w - 1, 0, 1, h, pad.left + w, pad.top, pad.right, h);
  if (pad.top) g.drawImage(out, 0, pad.top, out.width, 1, 0, 0, out.width, pad.top);
  if (pad.bottom) {
    g.drawImage(out, 0, pad.top + h - 1, out.width, 1, 0, pad.top + h, out.width, pad.bottom);
  }
  return out;
}

/* A copy of *mask* (a canvas) grown by *pad*, the new border white (repaint). */
export function iinExtendMask(mask, pad) {
  const out = iinCanvas(mask.width + pad.left + pad.right, mask.height + pad.top + pad.bottom);
  const g = out.getContext("2d");
  g.fillStyle = "#fff";
  g.fillRect(0, 0, out.width, out.height);
  g.drawImage(mask, pad.left, pad.top);
  return out;
}

/* The mask editor in the page modal. *base* is the image's data or object URL
   and *mask* the current mask's data URL (or null). *onDone* receives
   {base, mask, width, height, extended} as data URLs when the user applies it. */
async function iinOpenMaskEditor({ base, mask, onDone }) {
  let img;
  try { img = await iinLoadImage(base); } catch (e) { toast(e.message, true); return; }
  let baseC = iinCanvas(img.naturalWidth, img.naturalHeight);
  const probe = baseC.getContext("2d");
  if (!probe) { toast(t("imageInputs.noCanvas"), true); return; }
  probe.drawImage(img, 0, 0);
  let maskC = iinCanvas(baseC.width, baseC.height);
  const mg = maskC.getContext("2d");
  mg.fillStyle = "#000";
  mg.fillRect(0, 0, maskC.width, maskC.height);
  if (mask) {
    try {
      mg.drawImage(await iinLoadImage(mask), 0, 0, maskC.width, maskC.height);
    } catch {
      toast(t("imageInputs.maskReset"), true);
    }
  }
  let extended = false;
  let erase = false;
  let size = IIN_BRUSH_DEFAULT;
  let last = null;
  let queued = false;

  openModal(t("imageInputs.editorTitle"), (body) => {
    const view = iinCanvas(baseC.width, baseC.height);
    view.className = "iin-canvas";
    const tint = iinCanvas(baseC.width, baseC.height);

    const draw = () => {
      queued = false;
      if (view.width !== baseC.width || view.height !== baseC.height) {
        view.width = tint.width = baseC.width;
        view.height = tint.height = baseC.height;
      }
      const v = view.getContext("2d");
      v.globalCompositeOperation = "source-over";
      v.globalAlpha = 1;
      v.drawImage(baseC, 0, 0);
      const tg = tint.getContext("2d");
      tg.globalCompositeOperation = "source-over";
      tg.drawImage(maskC, 0, 0);
      tg.globalCompositeOperation = "multiply";
      tg.fillStyle = "#ff3030";
      tg.fillRect(0, 0, tint.width, tint.height);
      v.globalCompositeOperation = "screen";
      v.globalAlpha = 0.6;
      v.drawImage(tint, 0, 0);
      v.globalAlpha = 1;
      v.globalCompositeOperation = "source-over";
    };
    const redraw = () => { if (!queued) { queued = true; requestAnimationFrame(draw); } };

    const at = (ev) => {
      const r = view.getBoundingClientRect();
      return { x: (ev.clientX - r.left) * (view.width / r.width),
               y: (ev.clientY - r.top) * (view.height / r.height) };
    };
    const stroke = (from, to) => {
      const g = maskC.getContext("2d");
      g.strokeStyle = g.fillStyle = erase ? "#000" : "#fff";
      g.lineCap = "round";
      g.lineWidth = size;
      g.beginPath();
      g.moveTo(from.x, from.y);
      g.lineTo(to.x, to.y);
      g.stroke();
      g.beginPath();
      g.arc(to.x, to.y, size / 2, 0, Math.PI * 2);
      g.fill();
      redraw();
    };
    view.addEventListener("pointerdown", (ev) => {
      ev.preventDefault();
      view.setPointerCapture?.(ev.pointerId);
      last = at(ev);
      stroke(last, last);
    });
    view.addEventListener("pointermove", (ev) => {
      if (!last) return;
      const p = at(ev);
      stroke(last, p);
      last = p;
    });
    const end = () => { last = null; };
    view.addEventListener("pointerup", end);
    view.addEventListener("pointercancel", end);

    const tools = el("div", "iin-tools");
    const brush = el("button", "btn-secondary btn-sm", t("imageInputs.brush"));
    const eraser = el("button", "btn-secondary btn-sm", t("imageInputs.eraser"));
    const mark = () => { brush.classList.toggle("active", !erase); eraser.classList.toggle("active", erase); };
    brush.type = eraser.type = "button";
    brush.onclick = () => { erase = false; mark(); };
    eraser.onclick = () => { erase = true; mark(); };
    mark();
    const sizeIn = document.createElement("input");
    sizeIn.type = "range";
    sizeIn.min = "4";
    sizeIn.max = "256";
    sizeIn.value = String(size);
    sizeIn.title = t("imageInputs.brushSize");
    sizeIn.oninput = () => { size = Number(sizeIn.value) || IIN_BRUSH_DEFAULT; };
    const invert = el("button", "btn-secondary btn-sm", t("imageInputs.invert"));
    invert.type = "button";
    invert.onclick = () => {
      const g = maskC.getContext("2d");
      g.globalCompositeOperation = "difference";
      g.fillStyle = "#fff";
      g.fillRect(0, 0, maskC.width, maskC.height);
      g.globalCompositeOperation = "source-over";
      redraw();
    };
    const clear = el("button", "btn-secondary btn-sm", t("imageInputs.clear"));
    clear.type = "button";
    clear.onclick = () => {
      const g = maskC.getContext("2d");
      g.fillStyle = "#000";
      g.fillRect(0, 0, maskC.width, maskC.height);
      redraw();
    };
    tools.append(brush, eraser, el("label", "iin-size", t("imageInputs.brushSize")), sizeIn, invert, clear);

    const ext = el("div", "iin-tools");
    const pads = {};
    for (const [side, label] of [["left", t("imageInputs.padLeft")], ["top", t("imageInputs.padTop")],
                                 ["right", t("imageInputs.padRight")], ["bottom", t("imageInputs.padBottom")]]) {
      const n = document.createElement("input");
      n.type = "number";
      n.min = "0";
      n.step = "8";
      n.placeholder = "0";
      n.className = "iin-pad";
      n.title = label;
      pads[side] = n;
      ext.append(el("label", "iin-size", label), n);
    }
    const grow = el("button", "btn-secondary btn-sm", t("imageInputs.extend"));
    grow.type = "button";
    grow.title = t("imageInputs.extendTitle");
    grow.onclick = () => {
      const pad = {};
      for (const [side, n] of Object.entries(pads)) {
        const v = Math.max(0, Math.round((Number(n.value) || 0) / 8) * 8);
        pad[side] = v;
      }
      const w = baseC.width + pad.left + pad.right;
      const h = baseC.height + pad.top + pad.bottom;
      if (w === baseC.width && h === baseC.height) return;
      if (w > IIN_MAX_SIDE || h > IIN_MAX_SIDE) {
        toast(t("imageInputs.tooBig", { max: IIN_MAX_SIDE }), true);
        return;
      }
      baseC = iinExtendBase(baseC, pad);
      maskC = iinExtendMask(maskC, pad);
      extended = true;
      for (const n of Object.values(pads)) n.value = "";
      dims.textContent = t("imageInputs.size", { w: baseC.width, h: baseC.height });
      redraw();
    };
    ext.appendChild(grow);

    const dims = el("div", "sub", t("imageInputs.size", { w: baseC.width, h: baseC.height }));
    const hint = el("div", "sub", t("imageInputs.editorHint"));
    const actions = el("div", "actions");
    const use = el("button", "btn-primary", t("imageInputs.useMask"));
    use.type = "button";
    use.onclick = () => {
      onDone({ base: baseC.toDataURL("image/png"), mask: maskC.toDataURL("image/png"),
               width: baseC.width, height: baseC.height, extended });
      $("modal").style.display = "none";
    };
    actions.appendChild(use);
    const wrap = el("div", "iin-canvas-wrap");
    wrap.appendChild(view);
    body.append(hint, tools, ext, dims, wrap, actions);
    draw();
  });
}

/** The extra inputs mounted in #mountId. `refresh(data)` re-renders from a
 * `/api/imagine/native-models` response, `problem()` names what stops a job
 * (or ""), `body()` returns the request fields, `clear()` drops the images.
 * *promptId*, *negativeId*, *inputId* and *denoiseId* are the page's fields. */
export function createImageInputs({ mountId, promptId, negativeId, inputId, denoiseId }) {
  const state = { base: null, mask: null, size: null, refs: [],
                  control: { model: "", image: null, strength: "", preprocess: "none" } };
  let data = null;

  const caps = () => (data && data.preview && data.preview.capabilities) || {};

  async function baseImageURL() {
    if (state.base) return state.base;
    const raw = ($(inputId)?.value || "").trim();
    if (!raw) return null;
    const name = raw.split(/[\\/]/).pop();
    try {
      return await fetchImageURL(`/api/imagine/file/${encodeURIComponent(name)}`);
    } catch (e) {
      toast(/[\\/]/.test(raw) ? t("imageInputs.pathNotShown")
        : t("imageInputs.loadFailedNamed", { name: raw, message: e.message }), true);
      return null;
    }
  }

  function openEditor(url) {
    iinOpenMaskEditor({ base: url, mask: state.mask, onDone: (res) => {
      state.base = res.base;
      state.mask = res.mask;
      state.size = { w: res.width, h: res.height };
      const den = $(denoiseId);
      if (res.extended && den && !String(den.value).trim()) {
        den.value = "1";
        toast(t("imageInputs.outpaintDenoise"));
      }
      render();
    } });
  }

  const sections = () => {
    const box = el("div", "iin");
    box.appendChild(maskSection());
    box.appendChild(refSection());
    if (caps().controlnet || state.control.model || (data && (data.controlnets || []).length)) {
      box.appendChild(controlSection());
    }
    if (data && (data.embeddings || []).length) box.appendChild(embeddingSection());
    return box;
  };

  function maskSection() {
    const sec = el("div", "iin-section");
    sec.appendChild(el("label", null, t("imageInputs.maskLabel")));
    const row = el("div", "iin-row");
    const paint = el("button", "btn-secondary btn-sm",
      state.mask ? t("imageInputs.editMask") : t("imageInputs.paintMask"));
    paint.type = "button";
    paint.title = t("imageInputs.paintMaskTitle");
    paint.onclick = async () => {
      const url = await baseImageURL();
      if (url) openEditor(url);
      else if (!state.base) toast(t("imageInputs.needBase"), true);
    };
    const load = el("button", "btn-secondary btn-sm", t("imageInputs.loadBase"));
    load.type = "button";
    const picker = iinFilePicker(false, async (files) => {
      const url = await iinReadFile(files[0]);
      if (url) { state.base = null; state.mask = null; openEditor(url); }
    });
    load.onclick = () => picker.click();
    row.append(paint, load, picker);
    if (state.mask) {
      const drop = el("button", "btn-secondary btn-sm", t("imageInputs.removeMask"));
      drop.type = "button";
      drop.onclick = () => { state.base = state.mask = state.size = null; render(); };
      row.appendChild(drop);
      row.appendChild(el("span", "sub", t("imageInputs.maskSet",
        { w: state.size ? state.size.w : "?", h: state.size ? state.size.h : "?" })));
    }
    sec.appendChild(row);
    if (state.mask && caps().inpaint) sec.appendChild(el("div", "sub", t("imageInputs.inpaintModel")));
    return sec;
  }

  function refSection() {
    const sec = el("div", "iin-section");
    sec.appendChild(el("label", null, t("imageInputs.refsLabel")));
    const row = el("div", "iin-row iin-thumbs");
    state.refs.forEach((url, i) => row.appendChild(iinThumb(url, () => { state.refs.splice(i, 1); render(); })));
    if (state.refs.length < IIN_MAX_REFS) {
      const add = el("button", "btn-secondary btn-sm", t("imageInputs.addRefs"));
      add.type = "button";
      const picker = iinFilePicker(true, async (files) => {
        for (const f of files) {
          if (state.refs.length >= IIN_MAX_REFS) break;
          const url = await iinReadFile(f);
          if (url) state.refs.push(url);
        }
        render();
      });
      add.onclick = () => picker.click();
      row.append(add, picker);
    }
    sec.appendChild(row);
    if (data && data.preview && !caps().references) {
      sec.appendChild(el("div", "sub", t("imageInputs.refsUnsupported")));
    } else {
      sec.appendChild(el("div", "sub", t("imageInputs.refsHint")));
    }
    return sec;
  }

  function controlSection() {
    const sec = el("div", "iin-section");
    sec.appendChild(el("label", null, t("imageInputs.controlLabel")));
    const row = el("div", "iin-row");
    const sel = document.createElement("select");
    sel.id = "img-control-model";
    sel.appendChild(new Option(t("imageInputs.controlNone"), ""));
    for (const c of (data && data.controlnets) || []) {
      const text = c.file && c.file !== c.name ? `${c.name} (${c.file})` : c.name;
      sel.appendChild(new Option(c.fits === false ? t("nativeModels.otherFamily", { name: text }) : text, c.name));
    }
    if (state.control.model && ![...sel.options].some((o) => o.value === state.control.model)) {
      sel.appendChild(new Option(t("nativeModels.notRegistered", { name: state.control.model }), state.control.model));
    }
    sel.value = state.control.model;
    sel.onchange = () => { state.control.model = sel.value; render(); };
    row.appendChild(sel);
    if (state.control.model) {
      const strength = document.createElement("input");
      strength.type = "number";
      strength.id = "img-control-strength";
      strength.step = "0.05";
      strength.min = "0";
      strength.max = "2";
      strength.placeholder = "0.9";
      strength.title = t("imageInputs.controlStrength");
      strength.value = state.control.strength;
      strength.oninput = () => { state.control.strength = strength.value; };
      const pre = document.createElement("select");
      pre.id = "img-control-preprocess";
      pre.title = t("imageInputs.controlPreprocess");
      pre.append(new Option(t("imageInputs.preprocessNone"), "none"),
        new Option(t("imageInputs.preprocessCanny"), "canny"));
      pre.value = state.control.preprocess;
      pre.onchange = () => { state.control.preprocess = pre.value; };
      const pick = el("button", "btn-secondary btn-sm",
        state.control.image ? t("imageInputs.controlReplace") : t("imageInputs.controlImage"));
      pick.type = "button";
      const picker = iinFilePicker(false, async (files) => {
        const url = await iinReadFile(files[0]);
        if (url) { state.control.image = url; render(); }
      });
      pick.onclick = () => picker.click();
      row.append(strength, pre, pick, picker);
    }
    sec.appendChild(row);
    if (state.control.image) {
      sec.appendChild(iinThumb(state.control.image, () => { state.control.image = null; render(); }));
    }
    if (data && data.preview && !caps().controlnet) {
      sec.appendChild(el("div", "sub", t("imageInputs.controlUnsupported")));
    }
    return sec;
  }

  function embeddingSection() {
    const sec = el("div", "iin-section");
    sec.appendChild(el("label", null, t("imageInputs.embeddingsLabel")));
    const list = el("div", "iin-row iin-embeddings");
    for (const e of data.embeddings) {
      const chip = el("span", "iin-chip");
      chip.appendChild(el("span", null, e.trigger || e.name));
      if (e.fits === false) {
        chip.appendChild(el("span", "sub", " " + t("imageInputs.otherFamilyShort")));
      } else {
        for (const [label, id] of [[t("imageInputs.addToPrompt"), promptId],
                                   [t("imageInputs.addToNegative"), negativeId]]) {
          const b = el("button", "btn-secondary btn-sm", label);
          b.type = "button";
          b.onclick = () => {
            const field = $(id);
            if (!field || iinHasWord(field.value, e.trigger)) return;
            field.value = field.value.trim() ? `${field.value.trim()}, ${e.trigger}` : e.trigger;
          };
          chip.appendChild(b);
        }
      }
      list.appendChild(chip);
    }
    sec.append(list, el("div", "sub", t("imageInputs.embeddingsHint")));
    return sec;
  }

  function render() {
    const mount = $(mountId);
    if (mount) mount.replaceChildren(sections());
  }

  function usedEmbeddings() {
    const text = `${$(promptId)?.value || ""} , ${$(negativeId)?.value || ""}`;
    return ((data && data.embeddings) || [])
      .filter((e) => e.fits !== false && iinHasWord(text, e.trigger))
      .map((e) => e.name);
  }

  function problem() {
    if (state.control.model && !state.control.image) return t("imageInputs.needControlImage");
    if (state.refs.length && data && data.preview && !caps().references) return t("imageInputs.refsUnsupported");
    return "";
  }

  function body() {
    const out = {};
    if (state.base && state.mask) {
      out.input_image_data = state.base;
      out.mask = state.mask;
      out.input_image = null;
    }
    if (state.refs.length) out.ref_images = [...state.refs];
    if (state.control.model && state.control.image) {
      const c = { model: state.control.model, image: state.control.image,
                  preprocess: state.control.preprocess || "none" };
      const s = Number(String(state.control.strength).trim());
      if (String(state.control.strength).trim() !== "" && Number.isFinite(s)) c.strength = s;
      out.control = c;
    }
    const emb = usedEmbeddings();
    if (emb.length) out.embeddings = emb;
    return out;
  }

  function refresh(next) {
    data = next || null;
    render();
  }

  function clear() {
    state.base = state.mask = state.size = null;
    state.refs = [];
    state.control.image = null;
    render();
  }

  function setControl(meta) {
    const c = (meta && meta.control) || null;
    state.control.model = c ? String(c.model || "") : "";
    state.control.strength = c && c.strength !== undefined && c.strength !== null ? String(c.strength) : "";
    state.control.preprocess = c && c.preprocess === "canny" ? "canny" : "none";
    render();
  }

  return { refresh, problem, body, clear, render, setControl, get state() { return state; } };
}
