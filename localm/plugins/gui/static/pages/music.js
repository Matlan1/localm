// SPDX-License-Identifier: AGPL-3.0-or-later
/* localm GUI - Music page (split from pages.js). Classic script: it
   shares the one global lexical environment with app.js and the other
   page scripts, so the helpers it uses ($, el, authHeaders, toast, ...)
   resolve by bare name exactly as before. */
"use strict";

// --- ES module imports (auto-generated boundary; bodies unchanged) ---
import { $, authHeaders, fetchImageURL, streamJob, toast } from "../app/helpers.js";
import { hideStop, showStop } from "./images.js";
import { refreshMusicHistory } from "./video.js";

/* ================================================================ */
/*  Music page                                                       */
/* ================================================================ */

$("music-generate").onclick = async () => {
  const tags = $("music-tags").value.trim();
  if (!tags) { toast("Enter style tags first", true); return; }
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

  $("music-generate").disabled = true;
  const log = $("music-log");
  log.style.display = "block";
  log.textContent = "";
  $("music-result").replaceChildren();
  try {
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
      toast("Track finished");
      const player = document.createElement("audio");
      player.controls = true;
      player.style.width = "100%";
      const url = await fetchImageURL(
        "/api/music/file/" + encodeURIComponent(end.result));
      player.src = url;
      $("music-result").appendChild(player);
      refreshMusicHistory();
    } else {
      toast("Generation " + end.status, end.status !== "cancelled");
    }
  } catch (e) {
    toast("Music generation failed: " + e.message, true);
  } finally {
    $("music-generate").disabled = false;
    hideStop("music-stop");
  }
};

