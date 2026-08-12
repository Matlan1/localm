// SPDX-License-Identifier: AGPL-3.0-or-later
/* localm GUI - Images page (split from pages.js). Classic script: it
   shares the one global lexical environment with app.js and the other
   page scripts, so the helpers it uses ($, el, authHeaders, toast, ...)
   resolve by bare name exactly as before. */
"use strict";

// --- ES module imports (auto-generated boundary; bodies unchanged) ---
import { chat, lsSetScoped, renderAttachChips } from "../app/chat.js";
import { pickDirectory } from "../app/picker.js";
import { $, authHeaders, cancelJob, checkModelsBeforeGenerate, confirmDanger, el, fetchImageURL, jobStatusWord, openModal, streamJob, toast } from "../app/helpers.js";
import { emptyState } from "../app/icons.js";
import { showView } from "../app/tabs.js";
import { modelOverrides } from "./workflow.js";

/* ================================================================ */
/*  Images page                                                      */
/* ================================================================ */

export const imgState = {
  items: [],
  selected: new Set(),   // names selected for bulk actions
  showAll: false,        // grid shows 24 by default
};

export const IMG_GRID_DEFAULT = 24;

export async function imgApiDelete(name) {
  const r = await fetch(`/api/imagine/file/${encodeURIComponent(name)}`, {
    method: "DELETE", headers: authHeaders() });
  if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
}

export async function imgApiMove(name, dest) {
  const r = await fetch(`/api/imagine/file/${encodeURIComponent(name)}/move`, {
    method: "POST", headers: authHeaders(),
    body: JSON.stringify({ dest }),
  });
  const data = await r.json();
  if (!r.ok) throw new Error(data.detail || r.statusText);
  return data.path;
}

export async function imgDownload(name) {
  const url = await fetchImageURL(`/api/imagine/file/${encodeURIComponent(name)}`);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
}

export async function refreshImageHistory() {
  refreshReloadToggle();
  refreshLoraPicker();
  try {
    const r = await fetch("/api/imagine/history", { headers: authHeaders() });
    if (!r.ok) {
      // Show a visible message instead of a blank grid (mirrors music/video),
      // so a transient non-200 does not look like a broken/half-loaded page.
      $("img-history").replaceChildren(
        el("div", "sub", `Could not load images (HTTP ${r.status}).`));
      return;
    }
    imgState.items = (await r.json()).images;
  } catch (e) {
    $("img-history").replaceChildren(
      el("div", "sub", `Could not load images: ${(e && e.message) || e}`));
    return;
  }
  // Drop selections for images that no longer exist
  const names = new Set(imgState.items.map((i) => i.name));
  for (const n of imgState.selected) if (!names.has(n)) imgState.selected.delete(n);
  renderImageGrid();
}

export function renderImageGrid() {
  const grid = $("img-history");
  grid.replaceChildren();
  renderImgBulkBar();
  if (!imgState.items.length) {
    grid.appendChild(emptyState("image", "No images yet",
      "Generate one above; your results appear here."));
    return;
  }
  const shown = imgState.showAll
    ? imgState.items : imgState.items.slice(0, IMG_GRID_DEFAULT);
  for (const item of shown) {
    const thumb = el("div", "thumb");
    const img = document.createElement("img");
    fetchImageURL(`/api/imagine/file/${encodeURIComponent(item.name)}`)
      .then((url) => (img.src = url))
      .catch(() => thumb.remove());
    thumb.appendChild(img);

    // selection checkbox (top-left) - selected thumbs stay marked
    const sel = document.createElement("input");
    sel.type = "checkbox";
    sel.className = "thumb-sel";
    sel.checked = imgState.selected.has(item.name);
    thumb.classList.toggle("selected", sel.checked);
    sel.onclick = (e) => {
      e.stopPropagation();
      if (sel.checked) imgState.selected.add(item.name);
      else imgState.selected.delete(item.name);
      thumb.classList.toggle("selected", sel.checked);
      renderImgBulkBar();
    };
    thumb.appendChild(sel);

    // hover quick actions (top-right) - inline SVG icons (no emoji glyphs)
    const acts = el("div", "thumb-acts");
    const dl = el("button");
    dl.appendChild(iconEl("download"));
    dl.title = "Download";
    dl.onclick = (e) => {
      e.stopPropagation();
      imgDownload(item.name).catch((err) => toast("Download failed: " + err.message, true));
    };
    const del = el("button", "danger");
    del.appendChild(iconEl("trash"));
    del.title = "Delete from disk";
    del.onclick = (e) => {
      e.stopPropagation();
      confirmDanger(`Delete "${item.name}"?`, "This removes the file from disk.",
        "Delete", async () => {
          try {
            await imgApiDelete(item.name);
            toast("Deleted " + item.name);
            refreshImageHistory();
          } catch (err) { toast("Delete failed: " + err.message, true); }
        });
    };
    acts.append(dl, del);
    thumb.appendChild(acts);

    const capText = item.meta?.prompt
      ? item.meta.prompt.slice(0, 60) : item.name;
    thumb.appendChild(el("div", "cap", capText));
    thumb.onclick = () => showImageDetail(item);
    grid.appendChild(thumb);
  }
  if (imgState.items.length > IMG_GRID_DEFAULT) {
    const toggle = el("button", "btn-secondary img-show-all",
      imgState.showAll ? "show fewer"
                       : `show all (${imgState.items.length})`);
    toggle.onclick = () => { imgState.showAll = !imgState.showAll; renderImageGrid(); };
    grid.appendChild(toggle);
  }
}

/** Bulk actions bar - appears above the grid while a selection exists. */
export function renderImgBulkBar() {
  const bar = $("img-bulk");
  const n = imgState.selected.size;
  bar.style.display = n ? "flex" : "none";
  if (!n) { bar.replaceChildren(); return; }
  bar.replaceChildren();
  bar.appendChild(el("span", "count", `${n} selected`));

  const move = el("button", "btn-secondary", "move to folder…");
  move.onclick = async () => {
    const dest = await pickDirectory(`Move ${n} image(s) to…`,
      localStorage.getItem("localm.imgMoveDest") || "");
    if (!dest) return;
    let moved = 0, failed = 0;
    for (const name of [...imgState.selected]) {
      try { await imgApiMove(name, dest); moved++; }
      catch (e) { failed++; toast(`${name}: ${e.message}`, true); }
    }
    lsSetScoped("localm.imgMoveDest", dest);
    toast(`Moved ${moved} image(s)` + (failed ? `, ${failed} failed` : ""));
    imgState.selected.clear();
    refreshImageHistory();
  };

  const del = el("button", "danger", "delete");
  del.onclick = () => {
    confirmDanger(`Delete ${n} image(s)?`, "This removes the files from disk.",
      "Delete", async () => {
        let deleted = 0, failed = 0;
        for (const name of [...imgState.selected]) {
          try { await imgApiDelete(name); deleted++; }
          catch (e) { failed++; toast(`${name}: ${e.message}`, true); }
        }
        toast(`Deleted ${deleted} image(s)` + (failed ? `, ${failed} failed` : ""));
        imgState.selected.clear();
        refreshImageHistory();
      });
  };

  const clear = el("button", "btn-secondary", "clear selection");
  clear.onclick = () => {
    imgState.selected.clear();
    renderImageGrid();
  };

  bar.append(move, del, clear);
}

export function closeModal() { $("modal").style.display = "none"; }

export function showImageDetail(item) {
  openModal(item.name, (body) => {
    const img = document.createElement("img");
    img.style.maxWidth = "100%";
    img.style.borderRadius = "8px";
    fetchImageURL(`/api/imagine/file/${encodeURIComponent(item.name)}`)
      .then((url) => (img.src = url));
    body.appendChild(img);

    const actions = el("div", "actions");

    const useInput = el("button", "btn-secondary", "use as input");
    useInput.title = "Use this image as the img2img input";
    useInput.onclick = () => {
      $("img-input").value = item.path || item.name;
      closeModal();
      toast("Set as img2img input - adjust denoise and generate");
    };
    actions.appendChild(useInput);

    const toChat = el("button", "btn-secondary", "send to chat");
    toChat.title = "Attach this image to the chat composer";
    toChat.onclick = async () => {
      try {
        const url = await fetchImageURL(
          `/api/imagine/file/${encodeURIComponent(item.name)}`);
        const blob = await (await fetch(url)).blob();
        const dataUri = await new Promise((resolve, reject) => {
          const reader = new FileReader();
          reader.onload = () => resolve(reader.result);
          reader.onerror = reject;
          reader.readAsDataURL(blob);
        });
        chat.attachments.push({ name: item.name, dataUri });
        renderAttachChips();
        closeModal();
        showView("chat");
        toast("Image attached - type your message");
      } catch (e) {
        toast("Attach failed: " + e.message, true);
      }
    };
    actions.appendChild(toChat);

    if (item.meta?.prompt) {
      const reuse = el("button", "btn-secondary", "reuse settings");
      reuse.title = "Fill the generation form with this image's prompt, seed, and settings";
      reuse.onclick = () => {
        $("img-prompt").value = item.meta.prompt || "";
        $("img-negative").value = item.meta.negative_prompt || "";
        $("img-seed").value = item.meta.seed ?? "";
        $("img-guidance").value = item.meta.guidance ?? "";
        $("img-denoise").value = item.meta.denoise ?? "";
        $("img-input").value = item.meta.input_image || "";
        $("img-lora").value = item.meta.lora_name || "";
        $("img-lora-strength-model").value = item.meta.lora_strength_model ?? "";
        $("img-lora-strength-clip").value = item.meta.lora_strength_clip ?? "";
        closeModal();
        toast("Settings restored - tweak and generate");
      };
      actions.appendChild(reuse);
    }

    const dl = el("button", "btn-secondary", "download");
    dl.onclick = () =>
      imgDownload(item.name).catch((e) => toast("Download failed: " + e.message, true));
    actions.appendChild(dl);

    const copyImg = el("button", "btn-secondary", "copy image");
    copyImg.title = "Copy the image to the clipboard";
    copyImg.onclick = async () => {
      try {
        const url = await fetchImageURL(
          `/api/imagine/file/${encodeURIComponent(item.name)}`);
        const blob = await (await fetch(url)).blob();
        await navigator.clipboard.write([new ClipboardItem({ "image/png": blob })]);
        toast("Image copied to clipboard");
      } catch (e) {
        toast("Copy failed: " + e.message, true);
      }
    };
    actions.appendChild(copyImg);

    const copyPath = el("button", "btn-secondary", "copy path");
    copyPath.title = item.path || "";
    copyPath.onclick = async () => {
      try {
        await navigator.clipboard.writeText(item.path || item.name);
        toast("Path copied");
      } catch (e) { toast("Copy failed: " + e.message, true); }
    };
    actions.appendChild(copyPath);

    const rename = el("button", "btn-secondary", "rename…");
    rename.onclick = async () => {
      const newName = prompt("New name:", item.name);
      if (!newName || newName.trim() === item.name) return;
      const r = await fetch(
        `/api/imagine/file/${encodeURIComponent(item.name)}/rename`, {
          method: "POST", headers: authHeaders(),
          body: JSON.stringify({ new_name: newName.trim() }),
        });
      const data = await r.json();
      if (r.ok) {
        toast("Renamed to " + data.name);
        closeModal();
        refreshImageHistory();
      } else {
        toast(data.detail || "Rename failed", true);
      }
    };
    actions.appendChild(rename);

    const move = el("button", "btn-secondary", "move to folder…");
    move.onclick = async () => {
      const dest = await pickDirectory("Move image to…",
        localStorage.getItem("localm.imgMoveDest") || "");
      if (!dest) return;
      try {
        const path = await imgApiMove(item.name, dest);
        lsSetScoped("localm.imgMoveDest", dest);
        toast("Moved to " + path);
        refreshImageHistory();
      } catch (e) {
        toast("Move failed: " + e.message, true);
      }
    };
    actions.appendChild(move);

    const del = el("button", "danger", "delete");
    del.onclick = () => {
      confirmDanger(`Delete "${item.name}"?`, "This removes the file from disk.",
        "Delete", async () => {
          try {
            await imgApiDelete(item.name);
            toast("Deleted " + item.name);
            closeModal();
            refreshImageHistory();
          } catch (e) {
            toast("Delete failed: " + e.message, true);
          }
        });
    };
    actions.appendChild(del);

    body.appendChild(actions);

    if (item.meta && Object.keys(item.meta).length) {
      for (const [k, v] of Object.entries(item.meta)) {
        const row = el("div", "log-entry");
        row.appendChild(el("span", "t", k));
        row.appendChild(document.createTextNode(
          typeof v === "string" ? v : JSON.stringify(v)));
        body.appendChild(row);
      }
    }
  });
}

/* reload-after-generation toggle - mirrors the reload_llm_after_imagine
   server config key */

export async function refreshReloadToggle() {
  try {
    const r = await fetch("/v1/config", { headers: authHeaders() });
    if (!r.ok) return;
    const cfg = await r.json();
    $("img-reload-llm").checked = cfg.reload_llm_after_imagine !== false;
  } catch (e) { /* server unreachable */ }
}

/* LoRA picker - populated from ComfyUI's live-installed LoRA files (the same
   /api/imagine/comfy-models call the Workflow panel's model picker uses, see
   comfyModelPicker in workflow.js; loras is enumerated independently there
   since a LoraLoader node is not normally in the active workflow graph).
   Keeps the current selection across a refresh rather than resetting it back
   to "None" every time the panel reloads. */
export async function refreshLoraPicker() {
  const sel = $("img-lora");
  if (!sel) return;
  const previous = sel.value;
  let data;
  try {
    const r = await fetch("/api/imagine/comfy-models", { headers: authHeaders() });
    data = await r.json();
  } catch (e) { return; }
  sel.replaceChildren();
  const none = document.createElement("option");
  none.value = "";
  none.textContent = data.reachable ? "None" : "None (ComfyUI not running)";
  sel.appendChild(none);
  for (const name of (data.loras || [])) {
    const o = document.createElement("option");
    o.value = name;
    o.textContent = name;
    sel.appendChild(o);
  }
  sel.value = [...sel.options].some((o) => o.value === previous) ? previous : "";
}

$("img-reload-llm").onchange = async () => {
  const value = $("img-reload-llm").checked;
  const r = await fetch("/v1/config", {
    method: "PATCH", headers: authHeaders(),
    body: JSON.stringify({ reload_llm_after_imagine: value }),
  });
  if (r.ok) {
    toast(value ? "Chat model reloads after each generation"
                : "ComfyUI stays loaded - chat model reloads on next message");
  } else {
    toast("Could not save setting", true);
    $("img-reload-llm").checked = !value;
  }
};

// Media-generation Stop button: reveal it while a job runs and wire it to
// cancel that job; hide it again when the job ends. Shared by image/music/video.
export function showStop(btnId, jobId) {
  const btn = $(btnId);
  if (!btn) return;
  btn.style.display = "inline-block";
  btn.disabled = false;
  btn.onclick = () => { btn.disabled = true; btn.textContent = "Stopping…"; cancelJob(jobId); };
}
export function hideStop(btnId) {
  const btn = $(btnId);
  if (!btn) return;
  btn.style.display = "none";
  btn.disabled = false;
  btn.textContent = "Stop";
  btn.onclick = null;
}

$("img-generate").onclick = async () => {
  const promptText = $("img-prompt").value.trim();
  if (!promptText) { toast("Enter a prompt", true); return; }
  const num = (id) => {
    const v = $(id).value.trim();
    return v === "" ? null : Number(v);
  };
  const body = {
    prompt: promptText,
    negative_prompt: $("img-negative").value.trim() || null,
    seed: num("img-seed"),
    guidance: num("img-guidance"),
    denoise: num("img-denoise"),
    input_image: $("img-input").value.trim() || null,
  };
  const loraName = $("img-lora").value;
  if (loraName) {
    body.lora_name = loraName;
    body.lora_strength_model = num("img-lora-strength-model");
    body.lora_strength_clip = num("img-lora-strength-clip");
  }
  if (modelOverrides.image && Object.keys(modelOverrides.image).length) {
    body.model_overrides = modelOverrides.image;
  }
  $("img-generate").disabled = true;
  const log = $("img-log");
  log.style.display = "block";
  log.textContent = "";
  $("img-result").replaceChildren();
  try {
    await checkModelsBeforeGenerate("image", log,
      { model_overrides: modelOverrides.image, lora_name: loraName || undefined });
    const r = await fetch("/api/imagine", {
      method: "POST", headers: authHeaders(), body: JSON.stringify(body),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || r.statusText);
    showStop("img-stop", data.job_id);
    const end = await streamJob(data.job_id, (line) => {
      log.textContent += line + "\n";
      log.scrollTop = log.scrollHeight;
    });
    if (end.status === "done" && end.result) {
      const img = document.createElement("img");
      img.src = await fetchImageURL(
        `/api/imagine/file/${encodeURIComponent(end.result)}`);
      $("img-result").appendChild(img);
      toast("Image generated");
      refreshImageHistory();
    } else {
      toast("Generation " + jobStatusWord(end.status), end.status !== "cancelled");
    }
  } catch (e) {
    toast("Generation failed: " + e.message, true);
  } finally {
    $("img-generate").disabled = false;
    hideStop("img-stop");
  }
};

