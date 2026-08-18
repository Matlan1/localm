// SPDX-License-Identifier: AGPL-3.0-or-later
/* localm GUI - Models / Images / Plugins / Settings pages.
   Relies on helpers from app.js ($, el, authHeaders, toast, streamJob,
   fetchImageURL, openModal, refreshModels, modelCache, switchModel).
   Untrusted strings only ever reach the DOM via textContent. */

"use strict";

// --- ES module imports (auto-generated boundary; bodies unchanged) ---
import { populateSetupModels } from "../app/coder.js";
import { $, authHeaders } from "../app/helpers.js";
import { refreshPerfEstimate, refreshPluginCommands } from "../app/settings-perf.js";
import { refreshImageHistory } from "./images.js";
import { refreshKnowledgePage } from "./knowledge.js";
import { refreshModelsPage, refreshUploadsList } from "./models.js";
import { refreshPluginsPage, renderCatalogPlugins } from "./plugins.js";
import { refreshSettingsPage } from "./settings.js";
import { refreshMusicHistory } from "./music.js";
import { refreshVideoHistory } from "./video.js";
import { refreshWorkflowPanel } from "./workflow.js";

/** Fire-and-forget prime of the backend ComfyUI readiness cache on media-module
 *  open (one of 5 check points - see comfy_client.py). Result not read here. */
function warmComfyStatus() {
  fetch("/v1/comfy/status", { headers: authHeaders() }).catch(() => {});
}

window.onViewShown = (name) => {
  // Re-sync the plugin command catalog on entering a composer so a plugin
  // toggled elsewhere (CLI, another tab) updates the slash hints without a
  // full reload (refreshPluginCommands lives in app.js, shared global scope).
  if (name === "chat" || name === "coder") refreshPluginCommands();
  if (name === "coder") { populateSetupModels(); presetCoderMode(); }
  if (name === "models") refreshModelsPage();
  if (name === "images") { refreshImageHistory(); refreshWorkflowPanel("image"); warmComfyStatus(); }
  if (name === "music") { refreshMusicHistory(); refreshWorkflowPanel("music"); warmComfyStatus(); }
  if (name === "video") { refreshVideoHistory(); refreshWorkflowPanel("video"); warmComfyStatus(); }
  if (name === "knowledge") refreshKnowledgePage();
  if (name === "plugins") { renderCatalogPlugins(); refreshPluginsPage(); }
  if (name === "settings") {
    refreshSettingsPage();
    refreshUploadsList();
    // The model could have been switched from the model dropdown or the Models
    // page while Settings was not on screen - re-fetch the VRAM estimate so it
    // always reflects the currently active model on (re-)entering the tab.
    refreshPerfEstimate();
  }
};

/** Pre-select the configured coder session mode in the setup form. */
export async function presetCoderMode() {
  try {
    const r = await fetch("/v1/config", { headers: authHeaders() });
    if (!r.ok) return;
    const cfg = await r.json();
    const sel = $("setup-mode");
    if (sel && cfg.effective_coder_mode) sel.value = cfg.effective_coder_mode;
  } catch (e) { /* keep form default */ }
}

