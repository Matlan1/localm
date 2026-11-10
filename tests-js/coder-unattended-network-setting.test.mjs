// SPDX-License-Identifier: AGPL-3.0-or-later
// The "Network tools in unattended coder runs" setting (coder_unattended_network)
// renders in Settings > Coder as a deny/allow dropdown with its catalog label and
// help, and saving it sends the chosen value.

import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { loadAppWithPages, runScript } from "./harness.mjs";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const I18N_DIR = join(ROOT, "localm", "plugins", "gui", "static", "i18n");
const LABEL = "settings.field.coderUnattendedNetwork.label";
const HELP = "settings.field.coderUnattendedNetwork.help";
const EN_SRC = readFileSync(
  join(ROOT, "localm", "plugins", "gui", "static", "app", "i18n-en.js"), "utf-8");
const I18N_EN = Object.fromEntries([LABEL, HELP].map((key) => {
  const m = EN_SRC.match(new RegExp(`"${key.replaceAll(".", "\\.")}":\\s*"([^"]*)"`));
  return [key, m ? m[1] : undefined];
}));

const SCHEMA = {
  fields: [
    { key: "coder_untrusted_provenance", widget: "toggle", label: "server label",
      help: "server help", group: "Coder", owner: "coder", default: true,
      shipped_default: true },
    { key: "coder_unattended_network", widget: "select", label: "server label",
      help: "server help", group: "Coder", owner: "coder",
      options: ["deny", "allow"], default: "deny", shipped_default: "deny" },
  ],
};

function makeFetch(patches) {
  return async (url, opts = {}) => {
    if (url === "/v1/config/schema") {
      return { ok: true, status: 200, json: async () => SCHEMA, text: async () => "" };
    }
    if (url === "/v1/config" && (opts.method || "GET") === "PATCH") {
      patches.push(JSON.parse(opts.body));
      return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
    }
    return {
      ok: true, status: 200, text: async () => "",
      json: async () => ({ models: [], active: "", conversations: [], plugins: [] }),
    };
  };
}

async function render(win) {
  runScript(win, "refreshSettingsPage();");
  await new Promise((r) => setTimeout(r, 0));
}

test("the control renders as a deny/allow dropdown with its catalog label and help", async () => {
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch([]) });
  await render(win);
  const sel = win.document.querySelector('select[data-key="coder_unattended_network"]');
  assert.ok(sel, "the dropdown is rendered");
  assert.deepEqual([...sel.options].map((o) => o.value), ["deny", "allow"]);
  assert.equal(sel.value, "deny");
  const wrap = sel.closest('[data-field-key="coder_unattended_network"]');
  assert.equal(wrap.querySelector("label").textContent, I18N_EN[LABEL]);
  assert.equal(wrap.querySelector(".sub").textContent, I18N_EN[HELP]);
  assert.notEqual(I18N_EN[LABEL], "server label");
});

test("choosing allow and saving sends coder_unattended_network=allow", async () => {
  const patches = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(patches) });
  await render(win);
  const sel = win.document.querySelector('select[data-key="coder_unattended_network"]');
  sel.value = "allow";
  sel.dispatchEvent(new win.Event("change"));
  runScript(win, 'saveSettingsSection([...document.querySelectorAll(".settings-section")]' +
    '.find((s) => s.querySelector(\'[data-key="coder_unattended_network"]\')).dataset.sec);');
  await new Promise((r) => setTimeout(r, 0));
  assert.equal(patches.length, 1);
  assert.equal(patches[0].coder_unattended_network, "allow");
});

test("every locale catalog carries the label and help", () => {
  const files = readdirSync(I18N_DIR).filter((f) => f.endsWith(".json"));
  assert.ok(files.length >= 3);
  for (const f of files) {
    const cat = JSON.parse(readFileSync(join(I18N_DIR, f), "utf-8"));
    for (const key of [LABEL, HELP]) {
      assert.equal(typeof cat[key], "string", `${f} lacks ${key}`);
      assert.notEqual(cat[key], I18N_EN[key], `${f} ${key} is not translated`);
    }
  }
});
