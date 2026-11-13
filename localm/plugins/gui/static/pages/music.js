// SPDX-License-Identifier: AGPL-3.0-or-later
/* localm GUI - Music page. The library (grid, selection, bulk actions, detail
   modal, rename/move/delete) comes from app/media-gallery.js; this file adds
   the medium-specific bits. Cards are text-forward, with no thumbnail. */

"use strict";

import { $, authHeaders, checkModelsBeforeGenerate, createMissingModelOffer, dismissErrorToast, fetchImageURL, fmtBytes, jobStatusWord, revealFilledAdvanced, streamJob, toast } from "../app/helpers.js";
import { t } from "../app/i18n.js";
import { bindReloadToggle, createGallery, createJobFollower, musicPreview, playerDetail, reportMediaLoadFailure, refreshReloadToggle } from "../app/media-gallery.js";
import { createMusicModelPicker, mountEnhanceButton } from "../app/native-models.js";
import { createNativeWorkflowCard, nwfNumber, nwfSetField } from "../app/native-workflows.js";
import { chatPageModelChoice } from "../app/slash.js";
import { hideStop, showStop } from "./images.js";
import { modelOverrides } from "./workflow.js";

/* ================================================================ */
/*  Music library                                                    */
/* ================================================================ */

const musicJobs = createJobFollower();

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
    musicJobs.adopt("music", (jobId) => runMusicJob(async () => jobId));
  },

  buildPreview: musicPreview,
  buildDetailPreview: playerDetail("audio", "music.mediaWhat"),
  caption: (item) => item.name,

  reuse: (item) => {
    const m = item.meta || {};
    $("music-tags").value = m.tags || "";
    $("music-lyrics").value = m.lyrics || "";
    $("music-duration").value = m.duration_seconds ?? "";
    $("music-seed").value = m.seed ?? "";
    $("music-steps").value = m.steps ?? "";
    $("music-cfg").value = m.cfg ?? "";
    if ($("music-shift")) $("music-shift").value = m.shift ?? "";
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

export const musicModelPicker = createMusicModelPicker({ mountId: "music-native-models" });

/* The native workflow card: presets of the ACE-Step parts and the settings below. */
export const musicWorkflowCard = createNativeWorkflowCard({
  media: "music", mountId: "music-native-workflows", picker: musicModelPicker, logId: "music-log",
  fields: {
    read: () => ({
      duration_seconds: nwfNumber("music-duration"), steps: nwfNumber("music-steps"),
      cfg: nwfNumber("music-cfg"), shift: nwfNumber("music-shift"),
    }),
    apply: (p) => {
      if (p.tags) nwfSetField("music-tags", p.tags);
      if (p.lyrics !== undefined) nwfSetField("music-lyrics", p.lyrics);
      if (p.duration_seconds !== undefined) nwfSetField("music-duration", p.duration_seconds);
      for (const [key, id] of [["steps", "music-steps"], ["cfg", "music-cfg"],
                               ["shift", "music-shift"]]) {
        nwfSetField(id, p[key]);
      }
      revealFilledAdvanced($("view-music"));
    },
  },
});

mountEnhanceButton({ media: "music", fieldId: "music-tags" });

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
        const all = Object.keys(n.models || {}).length;
        text = missing.length < all
          ? t("music.backendNativeSomeMissing",
            { count: missing.length, total: all, size: fmtBytes(size) })
          : t("music.backendNativeMissing", { size: fmtBytes(size) });
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
  const workflows = $("music-native-workflow-card");
  if (workflows) {
    workflows.hidden = !native;
    if (native) musicWorkflowCard.refresh();
  }
  for (const node of document.querySelectorAll("#view-music .music-native-only")) {
    node.hidden = !native;
  }
  const picker = $("music-native-models");
  if (picker) {
    picker.hidden = !native;
    if (native) musicModelPicker.refresh();
  }
  for (const [id, key] of [["music-steps", "music.stepsPlaceholder"],
                           ["music-cfg", "music.cfgPlaceholder"]]) {
    const input = $(id);
    if (!input) continue;
    input.dataset.i18nPlaceholder = native ? `${key}Native` : key;
    input.placeholder = t(input.dataset.i18nPlaceholder);
  }
}

/* ================================================================ */
/*  Generation                                                       */
/* ================================================================ */

$("music-generate").onclick = async () => {
  const tags = $("music-tags").value.trim();
  if (!tags) { toast(t("music.enterTags"), true); return; }
  dismissErrorToast(t("music.enterTags"));
  const body = { tags };
  const lyrics = $("music-lyrics").value.trim();
  if (lyrics) body.lyrics = lyrics;
  const durationEl = $("music-duration");
  const durationRaw = durationEl.value.trim();
  const duration = Number(durationRaw);
  const minDuration = Number(durationEl.min) || 1, maxDuration = Number(durationEl.max) || 3600;
  if (durationRaw === "" || !Number.isFinite(duration) || duration < minDuration || duration > maxDuration) {
    toast(t("music.durationRange", { min: minDuration, max: maxDuration }), true);
    return;
  }
  dismissErrorToast(t("music.durationRange", { min: minDuration, max: maxDuration }));
  body.duration_seconds = duration;
  for (const [field, id] of [["seed", "music-seed"], ["steps", "music-steps"],
                             ["cfg", "music-cfg"]]) {
    const v = $(id).value.trim();
    if (v !== "" && !Number.isNaN(Number(v))) body[field] = Number(v);
  }
  if (musicBackend.active === null) await refreshMusicBackend();
  const native = musicBackend.active === "native";
  if (native && musicWorkflowCard.pending) {
    $("music-generate").disabled = true;
    let ready;
    try { ready = await musicWorkflowCard.prepare(); } finally { $("music-generate").disabled = false; }
    if (!ready) return;
  }
  const picked = musicModelFields(native);
  const nativeChoice = native ? picked : {};
  Object.assign(body, picked);
  if (native) {
    const shift = nwfNumber("music-shift");
    if (shift !== null) body.shift = shift;
  }
  await musicJobs.run(() => runMusicJob(async (log) => {
    const proceed = await checkModelsBeforeGenerate("music", log,
      native ? nativeChoice : { model_overrides: modelOverrides.music });
    if (!proceed) return null;
    const r = await fetch("/api/music", {
      method: "POST", headers: authHeaders(), body: JSON.stringify(body),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || r.statusText);
    return data.job_id;
  }));
};

/* The model fields Generate sends: the native picker's components on the
   native backend, the workflow panel's model picks on ComfyUI (none when
   nothing was picked). */
function musicModelFields(native) {
  if (native) return musicModelPicker.choice();
  const picks = modelOverrides.music;
  return picks && Object.keys(picks).length ? { model_overrides: picks } : {};
}

/* The model fields of this page's pick for the chat's /generate-music, read
   against the active backend and the registered models; null when this page's Generate would refuse
   (an applied workflow's main model is still not registered). */
export async function musicPageModelChoice() {
  await refreshMusicBackend();
  const native = musicBackend.active === "native";
  if (native) {
    await musicModelPicker.refresh();
    if (musicWorkflowCard.pending && !(await musicWorkflowCard.prepare())) return null;
  }
  return musicModelFields(native);
}
chatPageModelChoice.music = musicPageModelChoice;

/* Run one music job on this page: *start(log)* resolves to the job id to
   follow (null when nothing was started); its log lines and the track fill
   the page, and the end status is toasted. */
async function runMusicJob(start) {
  $("music-generate").disabled = true;
  const log = $("music-log");
  log.style.display = "block";
  log.textContent = "";
  $("music-result").replaceChildren();
  try {
    const jobId = await start(log);
    if (!jobId) return;
    showStop("music-stop", jobId);
    const missing = createMissingModelOffer(log);
    const end = await streamJob(jobId, (line) => {
      if (!missing.show(line)) return;
      log.textContent += line + "\n";
      log.scrollTop = log.scrollHeight;
    }, undefined, undefined, missing.onEvent);
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
      await missing.offer();
    }
  } catch (e) {
    toast(t("music.generationFailed", { message: e.message }), true);
  } finally {
    $("music-generate").disabled = false;
    hideStop("music-stop");
  }
}
