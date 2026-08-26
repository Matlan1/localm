// SPDX-License-Identifier: AGPL-3.0-or-later
/* localm GUI - Video page. The library (grid, selection, bulk actions, detail
   modal, rename/move/delete) comes from app/media-gallery.js; this file adds
   the medium-specific bits. */

"use strict";

import { MIB, $, authHeaders, checkModelsBeforeGenerate, fetchImageURL, jobStatusWord, revealFilledAdvanced, streamJob, toast } from "../app/helpers.js";
import { bindReloadToggle, createGallery, playerDetail, reportMediaLoadFailure, videoPreview, refreshReloadToggle } from "../app/media-gallery.js";
import { hideStop, showStop } from "./images.js";
import { modelOverrides } from "./workflow.js";

/* ================================================================ */
/*  Video library                                                    */
/* ================================================================ */

const videoGallery = createGallery({
  slug: "video",
  listKey: "videos",
  noun: "clip",
  plural: "clips",
  gridId: "video-history",
  bulkId: "video-bulk",
  moveDestKey: "localm.videoMoveDest",
  emptyIcon: "video",
  emptyTitle: "No clips yet",
  emptyHint: "Generate one above; your clips appear here.",

  beforeRefresh: () => refreshReloadToggle("video", "video-reload-llm"),

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

/* ================================================================ */
/*  Generation                                                       */
/* ================================================================ */

$("video-generate").onclick = async () => {
  const promptText = $("video-prompt").value.trim();
  if (!promptText) { toast("Enter a prompt first", true); return; }
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
  if (modelOverrides.video && Object.keys(modelOverrides.video).length) {
    body.model_overrides = modelOverrides.video;
  }

  $("video-generate").disabled = true;
  const log = $("video-log");
  log.style.display = "block";
  log.textContent = "";
  $("video-result").replaceChildren();
  try {
    await checkModelsBeforeGenerate("video", log, { model_overrides: modelOverrides.video });
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
      toast("Clip finished");
      const player = document.createElement("video");
      player.controls = true;
      player.style.width = "100%";
      reportMediaLoadFailure(player, "the clip");
      player.src = await fetchImageURL(
        "/api/video/file/" + encodeURIComponent(end.result));
      $("video-result").appendChild(player);
      refreshVideoHistory();
    } else {
      toast("Generation " + jobStatusWord(end.status), end.status !== "cancelled");
    }
  } catch (e) {
    toast("Video generation failed: " + e.message, true);
  } finally {
    $("video-generate").disabled = false;
    hideStop("video-stop");
  }
};
