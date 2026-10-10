// SPDX-License-Identifier: AGPL-3.0-or-later
/* localm GUI - Video page. The library (grid, selection, bulk actions, detail
   modal, rename/move/delete) comes from app/media-gallery.js; this file adds
   the medium-specific bits. */

"use strict";

import { MIB, $, authHeaders, checkModelsBeforeGenerate, fetchImageURL, fmtBytes, jobStatusWord, revealFilledAdvanced, streamJob, toast } from "../app/helpers.js";
import { t } from "../app/i18n.js";
import { bindReloadToggle, createGallery, playerDetail, reportMediaLoadFailure, videoPreview, refreshReloadToggle } from "../app/media-gallery.js";
import { hideStop, showStop } from "./images.js";
import { modelOverrides } from "./workflow.js";

/* ================================================================ */
/*  Video library                                                    */
/* ================================================================ */

const videoGallery = createGallery({
  slug: "video",
  listKey: "videos",
  itemKey: "video.item",
  gridId: "video-history",
  bulkId: "video-bulk",
  moveDestKey: "localm.videoMoveDest",
  emptyIcon: "video",
  emptyTitleKey: "video.empty.title",
  emptyHintKey: "video.empty.hint",

  beforeRefresh: () => {
    refreshVideoBackend();
    refreshReloadToggle("video", "video-reload-llm");
  },

  buildPreview: videoPreview,
  buildDetailPreview: playerDetail("video", "clip"),
  caption: (item) => (item.meta?.prompt
    ? item.meta.prompt.slice(0, 60)
    : `${item.name} · ${(item.size_bytes / MIB).toFixed(1)} MB`),

  reuse: (item) => {
    const m = item.meta || {};
    $("video-prompt").value = m.prompt || "";
    $("video-negative").value = m.negative_prompt || "";
    $("video-image").value = m.input_image || "";
    $("video-seconds").value = m.seconds ?? "";
    $("video-fps").value = m.fps ?? "";
    $("video-width").value = m.width ?? "";
    $("video-height").value = m.height ?? "";
    $("video-seed").value = m.seed ?? "";
    $("video-steps").value = m.steps ?? "";
    $("video-cfg").value = m.cfg ?? "";
    // Most of these fields live behind this page's Advanced fold; open it when
    // they are filled.
    revealFilledAdvanced($("view-video"));
  },
});

export const refreshVideoHistory = videoGallery.refresh;

bindReloadToggle("video", "video-reload-llm");

/* Which backend generates (from /api/video/backend). The native backend takes
   no workflow model picks; the note under the Generate heading says which
   backend runs and which model it uses, and the size and CFG placeholders
   show the defaults of the backend that will run. */
export const videoBackend = { active: null, choice: null };

const COMFY_PLACEHOLDERS = [["video-width", "video.widthPlaceholder"],
                            ["video-height", "video.heightPlaceholder"],
                            ["video-cfg", "video.cfgPlaceholder"]];

export async function refreshVideoBackend() {
  let data;
  try {
    const r = await fetch("/api/video/backend", { headers: authHeaders() });
    if (!r.ok) return;
    data = await r.json();
  } catch { return; }
  videoBackend.active = data.active || null;
  videoBackend.choice = data.choice || null;
  const native = videoBackend.active === "native";
  const n = data.native || {};
  const rec = n.recommended || {};
  const note = $("video-backend-note");
  if (note) {
    let text = "";
    if (native) {
      text = n.model
        ? t("video.backendNative", { model: n.model, runtime: n.runtime || t("images.backendRuntimeOnFirstUse") })
        : t("video.backendNativeNoModel", { name: rec.name || "", size: fmtBytes(rec.size_bytes || 0) });
    } else if (videoBackend.active === "comfy") {
      text = t("video.backendComfy");
    }
    note.textContent = text;
    note.hidden = !text;
  }
  for (const [id, key] of COMFY_PLACEHOLDERS) {
    const input = $(id);
    if (input) input.placeholder = t(key);
  }
  if (native) {
    $("video-width").placeholder = t("video.defaultPlaceholder", { value: rec.width });
    $("video-height").placeholder = t("video.defaultPlaceholder", { value: rec.height });
    if (!n.model || n.model === rec.name) {
      $("video-cfg").placeholder = t("video.defaultPlaceholder", { value: rec.cfg_scale });
    }
  }
}

/* ================================================================ */
/*  Generation                                                       */
/* ================================================================ */

$("video-generate").onclick = async () => {
  const promptText = $("video-prompt").value.trim();
  if (!promptText) { toast(t("video.enterPrompt"), true); return; }
  const body = { prompt: promptText };
  const negative = $("video-negative").value.trim();
  if (negative) body.negative_prompt = negative;
  const image = $("video-image").value.trim();
  if (image) body.input_image = image;
  for (const [field, id] of [["seconds", "video-seconds"], ["fps", "video-fps"],
                             ["width", "video-width"], ["height", "video-height"],
                             ["seed", "video-seed"], ["steps", "video-steps"],
                             ["cfg", "video-cfg"]]) {
    const v = $(id).value.trim();
    if (v !== "" && !Number.isNaN(Number(v))) body[field] = Number(v);
  }
  const native = videoBackend.active === "native";
  if (!native && modelOverrides.video && Object.keys(modelOverrides.video).length) {
    body.model_overrides = modelOverrides.video;
  }

  $("video-generate").disabled = true;
  const log = $("video-log");
  log.style.display = "block";
  log.textContent = "";
  $("video-result").replaceChildren();
  try {
    await checkModelsBeforeGenerate("video", log,
      { model_overrides: native ? undefined : modelOverrides.video });
    if (native) refreshVideoBackend();
    const r = await fetch("/api/video", {
      method: "POST", headers: authHeaders(), body: JSON.stringify(body),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || r.statusText);
    showStop("video-stop", data.job_id);
    const end = await streamJob(data.job_id, (line) => {
      log.textContent += line + "\n";
      log.scrollTop = log.scrollHeight;
    });
    if (end.status === "done" && end.result) {
      toast(t("video.generatedToast"));
      const player = document.createElement("video");
      player.controls = true;
      player.style.width = "100%";
      reportMediaLoadFailure(player, t("video.mediaWhat"));
      player.src = await fetchImageURL(
        "/api/video/file/" + encodeURIComponent(end.result));
      $("video-result").appendChild(player);
      refreshVideoHistory();
    } else {
      toast(t("video.generationStatus", { status: jobStatusWord(end.status) }), end.status !== "cancelled");
    }
  } catch (e) {
    toast(t("video.generationFailed", { message: e.message }), true);
  } finally {
    $("video-generate").disabled = false;
    hideStop("video-stop");
  }
};
