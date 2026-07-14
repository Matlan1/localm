// SPDX-License-Identifier: AGPL-3.0-or-later
/* localm GUI - Video page (split from pages.js). Classic script: it
   shares the one global lexical environment with app.js and the other
   page scripts, so the helpers it uses ($, el, authHeaders, toast, ...)
   resolve by bare name exactly as before. */
"use strict";

// --- ES module imports (auto-generated boundary; bodies unchanged) ---
import { chat } from "../app/chat.js";
import { $, MIB, authHeaders, checkModelsBeforeGenerate, confirmDanger, el, fetchImageURL, streamJob, toast } from "../app/helpers.js";
import { emptyState } from "../app/icons.js";
import { hideStop, showStop } from "./images.js";
import { pickDirectory } from "../app/picker.js";

/* ================================================================ */
/*  Video page                                                       */
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

  $("video-generate").disabled = true;
  const log = $("video-log");
  log.style.display = "block";
  log.textContent = "";
  $("video-result").replaceChildren();
  try {
    await checkModelsBeforeGenerate("video", log);
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
      const url = await fetchImageURL(
        "/api/video/file/" + encodeURIComponent(end.result));
      player.src = url;
      $("video-result").appendChild(player);
      refreshVideoHistory();
    } else {
      toast("Generation " + end.status, end.status !== "cancelled");
    }
  } catch (e) {
    toast("Video generation failed: " + e.message, true);
  } finally {
    $("video-generate").disabled = false;
    hideStop("video-stop");
  }
};

export async function refreshVideoHistory() {
  const box = $("video-history");
  box.replaceChildren();
  let data;
  try {
    const r = await fetch("/api/video/history", { headers: authHeaders() });
    if (!r.ok) throw new Error(r.statusText);
    data = await r.json();
  } catch (e) {
    box.appendChild(el("div", "sub", "Could not load history: " + e.message));
    return;
  }
  if (!data.videos.length) {
    box.appendChild(emptyState("video", "No clips yet",
      "Generate one above; your clips appear here."));
    return;
  }
  for (const item of data.videos) {
    const row = el("div", "disc-repo");
    const head = el("div", "head");
    head.appendChild(iconEl("video", "ic ic-video"));
    head.appendChild(el("span", "name", item.name));
    const bits = [];
    if (item.meta?.prompt) bits.push(item.meta.prompt.slice(0, 60));
    if (item.meta?.seconds) bits.push(`${item.meta.seconds}s`);
    bits.push(`${(item.size_bytes / MIB).toFixed(1)} MB`);
    head.appendChild(el("span", "meta", bits.join(" · ")));

    const play = el("button", "btn-secondary", "play");
    let player = null;
    play.onclick = async () => {
      if (player) { player.remove(); player = null; play.textContent = "play"; return; }
      play.disabled = true;
      try {
        const url = await fetchImageURL(
          "/api/video/file/" + encodeURIComponent(item.name));
        player = document.createElement("video");
        player.controls = true;
        player.autoplay = true;
        player.style.width = "100%";
        player.src = url;
        row.appendChild(player);
        play.textContent = "hide";
      } catch (e) {
        toast("Could not load clip: " + e.message, true);
      } finally {
        play.disabled = false;
      }
    };
    head.appendChild(play);

    const move = el("button", "btn-secondary", "move…");
    move.onclick = async () => {
      // pickDirectory (in-page browser modal) instead of prompt(): mobile/PWA
      // browsers suppress window.prompt(), which left the move button dead there
      // - the image page already uses this picker.
      const dest = await pickDirectory("Move video to…",
        localStorage.getItem("localm.videoMoveDest") || "");
      if (!dest) return;
      const r = await fetch(
        `/api/video/file/${encodeURIComponent(item.name)}/move`, {
          method: "POST", headers: authHeaders(),
          body: JSON.stringify({ dest }),
        });
      const data2 = await r.json();
      if (r.ok) {
        if (!chat.privacy) localStorage.setItem("localm.videoMoveDest", dest);
        toast("Moved to " + data2.path);
        refreshVideoHistory();
      } else {
        toast(data2.detail || "Move failed", true);
      }
    };
    head.appendChild(move);

    const del = el("button", "btn-secondary btn-danger", "delete");
    del.onclick = () => {
      confirmDanger(`Delete "${item.name}"?`, "This removes the file from disk.",
        "Delete", async () => {
          const r = await fetch("/api/video/file/" + encodeURIComponent(item.name),
                                { method: "DELETE", headers: authHeaders() });
          if (r.ok) { toast("Deleted"); refreshVideoHistory(); }
          else toast("Delete failed", true);
        });
    };
    head.appendChild(del);

    row.appendChild(head);
    box.appendChild(row);
  }
}

export async function refreshMusicHistory() {
  const box = $("music-history");
  box.replaceChildren();
  let data;
  try {
    const r = await fetch("/api/music/history", { headers: authHeaders() });
    if (!r.ok) throw new Error(r.statusText);
    data = await r.json();
  } catch (e) {
    box.appendChild(el("div", "sub", "Could not load history: " + e.message));
    return;
  }
  if (!data.tracks.length) {
    box.appendChild(emptyState("music", "No tracks yet",
      "Generate one above; your tracks appear here."));
    return;
  }
  for (const item of data.tracks) {
    const row = el("div", "disc-repo");
    const head = el("div", "head");
    head.appendChild(iconEl("music", "ic ic-audio"));
    head.appendChild(el("span", "name", item.name));
    const bits = [];
    if (item.meta?.tags) bits.push(item.meta.tags.slice(0, 60));
    if (item.meta?.duration_seconds) bits.push(`${item.meta.duration_seconds}s`);
    bits.push(`${(item.size_bytes / MIB).toFixed(1)} MB`);
    head.appendChild(el("span", "meta", bits.join(" · ")));

    const play = el("button", "btn-secondary", "play");
    let player = null;
    play.onclick = async () => {
      if (player) { player.remove(); player = null; play.textContent = "play"; return; }
      play.disabled = true;
      try {
        const url = await fetchImageURL(
          "/api/music/file/" + encodeURIComponent(item.name));
        player = document.createElement("audio");
        player.controls = true;
        player.autoplay = true;
        player.style.width = "100%";
        player.src = url;
        row.appendChild(player);
        play.textContent = "hide";
      } catch (e) {
        toast("Could not load track: " + e.message, true);
      } finally {
        play.disabled = false;
      }
    };
    head.appendChild(play);

    const move = el("button", "btn-secondary", "move…");
    move.onclick = async () => {
      // pickDirectory (in-page browser modal) instead of prompt(): mobile/PWA
      // browsers suppress window.prompt(), which left the move button dead there
      // - the image page already uses this picker.
      const dest = await pickDirectory("Move track to…",
        localStorage.getItem("localm.musicMoveDest") || "");
      if (!dest) return;
      const r = await fetch(
        `/api/music/file/${encodeURIComponent(item.name)}/move`, {
          method: "POST", headers: authHeaders(),
          body: JSON.stringify({ dest }),
        });
      const data2 = await r.json();
      if (r.ok) {
        if (!chat.privacy) localStorage.setItem("localm.musicMoveDest", dest);
        toast("Moved to " + data2.path);
        refreshMusicHistory();
      } else {
        toast(data2.detail || "Move failed", true);
      }
    };
    head.appendChild(move);

    const del = el("button", "btn-secondary btn-danger", "delete");
    del.onclick = () => {
      confirmDanger(`Delete "${item.name}"?`, "This removes the file from disk.",
        "Delete", async () => {
          const r = await fetch("/api/music/file/" + encodeURIComponent(item.name),
                                { method: "DELETE", headers: authHeaders() });
          if (r.ok) { toast("Deleted"); refreshMusicHistory(); }
          else toast("Delete failed", true);
        });
    };
    head.appendChild(del);

    row.appendChild(head);
    box.appendChild(row);
  }
}

