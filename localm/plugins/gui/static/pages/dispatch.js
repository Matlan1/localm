// SPDX-License-Identifier: AGPL-3.0-or-later
/* localm GUI - Models / Images / Plugins / Settings pages.
   Relies on helpers from app.js ($, el, authHeaders, toast, streamJob,
   fetchImageURL, openModal, refreshModels, modelCache, switchModel).
   Untrusted strings only ever reach the DOM via textContent. */

"use strict";

// --- ES module imports (auto-generated boundary; bodies unchanged) ---
import { populateSetupModels, refreshDormant } from "../app/coder.js";
import { $, authHeaders } from "../app/helpers.js";
import { refreshPerfEstimate, refreshPluginCommands } from "../app/settings-perf.js";
import { refreshImageHistory } from "./images.js";
import { refreshKnowledgePage } from "./knowledge.js";
import { refreshModelsPage, refreshUploadsList, runtimeUpdateCheck } from "./models.js";
import { refreshPluginsPage, renderCatalogPlugins } from "./plugins.js";
import { refreshDiagnosticsCard, refreshSettingsPage } from "./settings.js";
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
  // refreshDormant on ARRIVAL, not only when the directory field changes.
  // Without this the rail reads "No sessions yet" for someone who has past
  // work in several projects, until they happen to type a path - which is a
  // false statement about their own history, and the exact case the rail
  // exists to serve. Found in a browser; every jsdom test called
  // refreshDormant() itself and so could not see the missing trigger.
  if (name === "coder") { populateSetupModels(); presetCoderMode(); refreshDormant(); }
  if (name === "models") refreshModelsPage();
  if (name === "images") { refreshImageHistory(); refreshWorkflowPanel("image"); warmComfyStatus(); }
  if (name === "music") { refreshMusicHistory(); refreshWorkflowPanel("music"); warmComfyStatus(); }
  if (name === "video") { refreshVideoHistory(); refreshWorkflowPanel("video"); warmComfyStatus(); }
  if (name === "knowledge") refreshKnowledgePage();
  if (name === "plugins") { renderCatalogPlugins(); refreshPluginsPage(); }
  if (name === "settings") {
    refreshSettingsPage();
    refreshUploadsList();
    // Paints the last diagnostics report, and REATTACHES to a run still in
    // flight - one started in another tab, or in this one before a reload
    // (ADR-0008: a background operation the server knows about must not be
    // undiscoverable just because you did not start it here).
    refreshDiagnosticsCard();
    // The model could have been switched from the model dropdown or the Models
    // page while Settings was not on screen - re-fetch the VRAM estimate so it
    // always reflects the currently active model on (re-)entering the tab.
    refreshPerfEstimate();
    // Inference runtime: report what is provisioned WITHOUT the user first
    // pressing "Check for runtime update". The case that needs it is the one
    // where nothing is installed at all - a user with no runtime should not
    // have to press a button labelled "check for an UPDATE" to discover they
    // can install one. Cheap by default: check_runtime_update() answers from
    // the on-disk marker and a constant, and only reaches the network on an
    // install that opted into tracking upstream.
    runtimeUpdateCheck();
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

