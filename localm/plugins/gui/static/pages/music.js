// SPDX-License-Identifier: AGPL-3.0-or-later
/* localm GUI - Music page. The library (grid, selection, bulk actions, detail
   modal, rename/move/delete) comes from app/media-gallery.js; this file adds
   the medium-specific bits. Cards are text-forward, with no thumbnail. */

"use strict";

import { $, authHeaders, checkModelsBeforeGenerate, fetchImageURL, fmtBytes, jobStatusWord, revealFilledAdvanced, streamJob, toast } from "../app/helpers.js";
import { t } from "../app/i18n.js";
import { bindReloadToggle, createGallery, musicPreview, playerDetail, reportMediaLoadFailure, refreshReloadToggle } from "../app/media-gallery.js";
import { hideStop, showStop } from "./images.js";
import { modelOverrides } from "./workflow.js";

/* ================================================================ */
/*  Music library                                                    */
/* ================================================================ */

const musicGallery = createGallery({
  slug: "music",
  listKey: "tracks",
  itemKey: "music.item",
  gridId: "music-history",
  bulkId: "music-bulk",
  moveDestKey: "localm.musicMoveDest",
  emptyIcon: "music",
  emptyTitleKey: "music.empty.title",
  emptyHintKey: "music.empty.hint",
  cardClass: "thumb-track",

  beforeRefresh: () => {
    refreshMusicBackend();
    refreshReloadToggle("music", "music-reload-llm");
  },

  buildPreview: musicPreview,
  buildDetailPreview: playerDetail("audio", "track"),
  caption: (item) => item.name,

  reuse: (item) => {
    const m = item.meta || {};
    $("music-tags").value = m.tags || "";
    $("music-lyrics").value = m.lyrics || "";
    $("music-duration").value = m.duration_seconds ?? "";
    $("music-seed").value = m.seed ?? "";
    $("music-steps").value = m.steps ?? "";
    $("music-cfg").value = m.cfg ?? "";
    // Most of these fields live behind this page's Advanced fold; open it when
    // they are filled.
    revealFilledAdvanced($("view-music"));
  },
});

export const refreshMusicHistory = musicGallery.refresh;

bindReloadToggle("music", "music-reload-llm");

/* Which backend generates (from /api/music/backend). The native backend has no
   ComfyUI workflow, so the Workflow card is hidden and its model picks are not
   sent while it is active; the note under the Generate heading says which
   backend runs. */
export const musicBackend = { active: null, choice: null };

export async function refreshMusicBackend() {
  let data;
  try {
    const r = await fetch("/api/music/backend", { headers: authHeaders() });
    if (!r.ok) return;
    data = await r.json();
  } catch { return; }
  musicBackend.active = data.active || null;
  musicBackend.choice = data.choice || null;
  const native = musicBackend.active === "native";
  const note = $("music-backend-note");
  if (note) {
    let text = "";
    if (native) {
      const n = data.native || {};
      const missing = n.missing || [];
      if (missing.length) {
        const size = missing.reduce((sum, m) => sum + (m.size_bytes || 0), 0);
        text = t("music.backendNativeMissing", { size: fmtBytes(size) });
      } else {
        const rt = n.runtime || {};
        text = t("music.backendNative", {
          runtime: rt.installed ? (rt.backend || rt.choice) : t("music.backendRuntimeOnFirstUse"),
        });
      }
    } else if (musicBackend.active === "comfy") {
      text = t("music.backendComfy");
    }
    note.textContent = text;
    note.hidden = !text;
  }
  const card = $("music-workflow-card");
  if (card) card.hidden = native;
}

/* ================================================================ */
/*  Generation                                                       */
/* ================================================================ */

$("music-generate").onclick = async () => {
  const tags = $("music-tags").value.trim();
  if (!tags) { toast(t("music.enterTags"), true); return; }
  const body = { tags };
  const lyrics = $("music-lyrics").value.trim();
  if (lyrics) body.lyrics = lyrics;
  const duration = Number($("music-duration").value);
  if (duration > 0) body.duration_seconds = duration;
  for (const [field, id] of [["seed", "music-seed"], ["steps", "music-steps"],
                             ["cfg", "music-cfg"]]) {
    const v = $(id).value.trim();
    if (v !== "" && !Number.isNaN(Number(v))) body[field] = Number(v);
  }
  if (musicBackend.active === null) await refreshMusicBackend();
  const native = musicBackend.active === "native";
  if (!native && modelOverrides.music && Object.keys(modelOverrides.music).length) {
    body.model_overrides = modelOverrides.music;
  }

  $("music-generate").disabled = true;
  const log = $("music-log");
  log.style.display = "block";
  log.textContent = "";
  $("music-result").replaceChildren();
  try {
    await checkModelsBeforeGenerate("music", log,
      native ? {} : { model_overrides: modelOverrides.music });
    const r = await fetch("/api/music", {
      method: "POST", headers: authHeaders(), body: JSON.stringify(body),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || r.statusText);
    showStop("music-stop", data.job_id);
    const end = await streamJob(data.job_id, (line) => {
      log.textContent += line + "\n";
      log.scrollTop = log.scrollHeight;
    });
    if (end.status === "done" && end.result) {
      toast(t("music.generatedToast"));
      const player = document.createElement("audio");
      player.controls = true;
      player.style.width = "100%";
      reportMediaLoadFailure(player, t("music.mediaWhat"));
      player.src = await fetchImageURL(
        "/api/music/file/" + encodeURIComponent(end.result));
      $("music-result").appendChild(player);
      refreshMusicHistory();
    } else {
      toast(t("music.generationStatus", { status: jobStatusWord(end.status) }), end.status !== "cancelled");
    }
  } catch (e) {
    toast(t("music.generationFailed", { message: e.message }), true);
  } finally {
    $("music-generate").disabled = false;
    hideStop("music-stop");
  }
};
