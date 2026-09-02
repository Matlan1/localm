// SPDX-License-Identifier: AGPL-3.0-or-later
/* localm GUI - Setup: a guided desktop flow for the llama.cpp runtime and the
 * first model (ADR-0001). Reached ONLY by a deliberate click on the Setup
 * nav button - refreshSetupPage runs solely from onViewShown("setup"), which
 * itself only fires from that click or from the generic "restore the view
 * you were last on" boot path every core view already gets. Nothing here
 * auto-triggers on an unprovisioned runtime, an empty model list, or a red
 * diagnostics report.
 *
 * Reuses rather than duplicates: postRuntimeUpdate (models.js) is the same
 * POST /api/runtime/update caller the Settings "Inference runtime" card
 * uses, and the model-add step links to the Models page's own discover/pull
 * flow instead of embedding a second copy of it.
 */
"use strict";

// --- ES module imports ---
import { $, authHeaders, streamJob } from "../app/helpers.js";
import { t, tn } from "../app/i18n.js";
import { showView } from "../app/tabs.js";
import { postRuntimeUpdate } from "./models.js";

/** Paint the runtime status card from a fresh GET /api/backend read - the
 *  same read-only, single-flighted endpoint Settings > Live tuning already
 *  polls (settings-perf.js refreshBackendInfo). Never provisions anything. */
async function refreshSetupRuntime() {
  const status = $("onb-runtime-status"), warn = $("onb-runtime-warning");
  const btn = $("onb-runtime-install");
  if (!status) return;
  status.textContent = t("setup.runtime.checking");
  try {
    const r = await fetch("/api/backend", { headers: authHeaders() });
    if (!r.ok) throw new Error("HTTP " + r.status);
    const d = await r.json();
    if (warn) {
      warn.textContent = d.warning || "";
      warn.hidden = !d.warning;
    }
    if (d.installed) {
      status.textContent = t("setup.runtime.installed", { backend: d.installed });
      if (btn) btn.hidden = true;
    } else {
      status.textContent = d.recommended
        ? t("setup.runtime.recommended", { backend: d.recommended })
        : t("setup.runtime.notInstalled");
      if (btn) btn.hidden = false;
    }
  } catch (e) {
    status.textContent = t("setup.runtime.checkFailed", { message: e.message });
    if (warn) warn.hidden = true;
  }
}

/** Provision the runtime. No backend/tag is ever sent from Setup - the route
 *  falls back to whatever is already installed, else "auto" (hardware
 *  detection), the same default a bare `localm setup-llama` would pick. The
 *  backend/tag/rollback picker stays a Settings-only, power-user control. */
async function installSetupRuntime() {
  const btn = $("onb-runtime-install"), status = $("onb-runtime-status");
  const log = $("onb-runtime-log");
  if (btn) btn.disabled = true;
  if (status) status.textContent = t("setup.runtime.working");
  if (log) { log.style.display = ""; log.textContent = ""; }
  let jobId;
  try {
    jobId = await postRuntimeUpdate("", "", false);
  } catch (e) {
    if (status) status.textContent = t("setup.runtime.failed", { message: e.message });
    if (btn) btn.disabled = false;
    if (log) log.style.display = "none";
    return;
  }
  const tail = [];
  const end = await streamJob(jobId, (line) => {
    if (log) { log.textContent += line + "\n"; log.scrollTop = log.scrollHeight; }
    // A short tail, not just the last line: setup-llama's failure messages
    // wrap across several printed lines (mirrors runtimeProvision).
    if (line && line.trim()) { tail.push(line.trim()); if (tail.length > 6) tail.shift(); }
  });
  const ok = !!(end && end.status === "done");
  if (btn) btn.disabled = false;
  if (ok) {
    await refreshSetupRuntime();
  } else if (status) {
    status.textContent = tail.join(" ").trim() || t("setup.runtime.notFinished");
  }
}

/** Paint the "add a model" card from a fresh model count. Read-only; the
 *  actual add/search flow lives on the Models page and is never duplicated
 *  here. */
async function refreshSetupModels() {
  const status = $("onb-models-status");
  if (!status) return;
  try {
    const r = await fetch("/api/models", { headers: authHeaders() });
    if (!r.ok) throw new Error("HTTP " + r.status);
    const d = await r.json();
    const count = Array.isArray(d.models) ? d.models.length : 0;
    status.textContent = count ? tn("setup.models.count", count) : t("setup.models.none");
  } catch (e) {
    status.textContent = t("setup.models.checkFailed", { message: e.message });
  }
}

/** (Re)paint the whole Setup view. Called only from onViewShown("setup"). */
export function refreshSetupPage() {
  refreshSetupRuntime();
  refreshSetupModels();
}

if ($("onb-runtime-install")) $("onb-runtime-install").onclick = installSetupRuntime;
if ($("onb-models-go")) {
  $("onb-models-go").onclick = () => {
    showView("models");
    const q = $("disc-query");
    if (q) q.focus();
  };
}
if ($("onb-go-chat")) $("onb-go-chat").onclick = () => showView("chat");
