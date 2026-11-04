// SPDX-License-Identifier: AGPL-3.0-or-later
// The server's English note about a model's own code not running reaches the user in the
// GUI's language, in the toast and with the rest of the message intact.
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import assert from "node:assert/strict";

import { loadApp, runScript } from "./harness.mjs";

const here = dirname(fileURLToPath(import.meta.url));
const staticDir = join(here, "..", "localm", "plugins", "gui", "static");
const de = JSON.parse(readFileSync(join(staticDir, "i18n", "de.json"), "utf-8"));

test("the note is shown in the GUI's language and the failure text stays", async () => {
  const { window } = loadApp();
  window.fetch = (url) => Promise.resolve({
    ok: true, status: 200, json: async () => (String(url).includes("/i18n/de.json") ? de : {}), text: async () => "",
  });
  await runScript(window, "window.__apply = applyLanguage; window.__en = I18N_EN; window.__toast = toast;");
  await window.__apply("de");
  const note = window.__en["models.bundledCodeNote"];
  window.__toast(`It broke. ${note}`, true);
  const text = window.document.getElementById("toast").textContent;
  assert.equal(text, `It broke. ${de["models.bundledCodeNote"]}`);
  assert.notEqual(de["models.bundledCodeNote"], note);
});

test("the switch question names the disk when the experts are read from it", async () => {
  const { window } = loadApp();
  runScript(window, "window.__ask = confirmLoadText;");
  const asked = { status: "confirm_required", reason: "degraded_load", model: "m", plan_mb: 1, full_mb: 2, free_mb: 0 };
  assert.match(window.__ask({ ...asked, experts_on_disk: true }), /keeps its routed experts on disk/);
  assert.match(window.__ask({ ...asked, experts_on_disk: false }), /on the CPU/);
});
