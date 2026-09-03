// SPDX-License-Identifier: AGPL-3.0-or-later
/* localm GUI - Models / Images / Plugins / Settings pages.
   Untrusted strings reach the DOM only via textContent. */

"use strict";

// --- ES module imports ---
import { populateSetupModels, refreshDormant } from "../app/coder.js";
import { $, authHeaders } from "../app/helpers.js";
import { refreshPerfEstimate, refreshPluginCommands } from "../app/settings-perf.js";
import { refreshImageHistory } from "./images.js";
import { refreshKnowledgePage } from "./knowledge.js";
import { refreshInstancesCard, refreshModelsPage, refreshUploadsList, runtimeUpdateCheck } from "./models.js";
import { refreshPluginsPage, renderCatalogPlugins } from "./plugins.js";
import { refreshDiagnosticsCard, refreshSettingsPage } from "./settings.js";
import { refreshMusicHistory } from "./music.js";
import { refreshVideoHistory } from "./video.js";
import { refreshWorkflowPanel } from "./workflow.js";

/** Fire-and-forget prime of the backend ComfyUI readiness cache on media-module
 *  open. The result is not read here. */
function warmComfyStatus() {
  fetch("/v1/comfy/status", { headers: authHeaders() }).catch(() => {});
}

window.onViewShown = (name) => {
  // Re-sync the plugin command catalog on entering a composer, so the slash
  // hints pick up a plugin toggled from the CLI or another tab.
  if (name === "chat" || name === "coder") refreshPluginCommands();
  // refreshDormant runs on ARRIVAL, not only when the directory field changes.
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
    refreshInstancesCard();
    // Paints the last diagnostics report and reattaches to a run still in
    // flight, including one started in another tab or before a reload.
    refreshDiagnosticsCard();
    // Re-fetch the VRAM estimate so it reflects the currently active model.
    refreshPerfEstimate();
    // Report the provisioned inference runtime without waiting for "Check for
    // runtime update". check_runtime_update() answers from the on-disk marker
    // and only reaches the network on an install that tracks upstream.
    runtimeUpdateCheck();
  }
  // Only reached by a deliberate click on the Setup nav button (or the
  // generic "restore the view you were last on" boot path every core view
  // gets) - never auto-triggered by an unprovisioned runtime or zero models.
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

