// SPDX-License-Identifier: AGPL-3.0-or-later
// Where each schema group's card lands on the Settings page: its nav group and its
// heading. tests/test_settings_section_placement.py pins the same mapping for every
// real schema field; this file renders it through refreshSettingsPage.
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

// One visible field per schema group the generic form renders, with the nav group
// and heading its card must get.
const CASES = [
  ["Engine", "n_ctx", "number", "model", "Runtime & GPU"],
  ["Timeouts", "gguf_load_timeout_s", "number", "model", "Timeouts & limits"],
  ["Chat", "temperature", "number", "model", "Generation defaults"],
  ["Models", "confirm_remove", "toggle", "model", "Model Library & Sources"],
  ["Embeddings", "embedding_pooling", "select", "model", "Embeddings"],
  ["Server", "bind_host", "text", "server", "Server"],
  ["Network", "net_mode", "select", "server", "Outbound access"],
  ["Security", "allow_network_drives", "toggle", "security", "Access"],
  ["Plugins", "suggest_plugins", "toggle", "plugins", "Plugin management"],
  ["Coder", "coder_rail_side", "select", "plugins", "Coder"],
  ["Knowledge", "rag_rerank", "toggle", "plugins", "Knowledge (RAG)"],
  ["Voice", "voice_stt_language", "text", "plugins", "Voice"],
  ["Privacy", "mode", "select", "privacy", "Session persistence"],
  ["Memory", "memory_enabled", "toggle", "privacy", "Memory"],
  ["Diagnostics", "keep_diagnostics", "toggle", "privacy", "Crash diagnostics"],
  ["General", "temp_location", "text", "system", "General"],
  ["Desktop", "desktop_window_mode", "select", "system", "Desktop app"],
];

function field([group, key, widget]) {
  const f = { key, widget, label: key, help: "", group, owner: "core", applies: "live" };
  if (widget === "select") { f.options = ["a", "b"]; f.default = "a"; }
  else if (widget === "toggle") f.default = false;
  else if (widget === "number") f.default = 1;
  else f.default = "";
  return f;
}

function makeFetch(fields) {
  return async (url) => {
    if (url === "/v1/config/schema") {
      return { ok: true, status: 200, json: async () => ({ fields }), text: async () => "" };
    }
    return {
      ok: true, status: 200, text: async () => "",
      json: async () => ({ models: [], active: "", conversations: [], plugins: [] }),
    };
  };
}

async function render(fields) {
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(fields) });
  runScript(win, "refreshSettingsPage();");
  for (let i = 0; i < 10; i++) await new Promise((r) => setTimeout(r, 0));
  return win.document;
}

function heading(sec) {
  const h = sec.querySelector(":scope > .card-head h3");
  return h ? h.textContent.trim() : "";
}

test("each schema group's card lands in its nav group under its own heading", async () => {
  const doc = await render(CASES.map(field));
  for (const [group, key, , nav, head] of CASES) {
    const node = doc.querySelector(`[data-field-key="${key}"]`);
    assert.ok(node, `${key} renders`);
    const sec = node.closest(".settings-section");
    assert.equal(sec.id, "settings-sec-core-" + group, `${key} sits in the ${group} card`);
    assert.equal(sec.dataset.group, nav, `the ${group} card is in the ${nav} nav group`);
    assert.equal(heading(sec), head, `the ${group} card is headed "${head}"`);
    assert.equal(sec.querySelector(".settings-section-save").textContent, "Save " + head,
      `the ${group} card's save button names its own heading`);
  }
});

test("Temporary files is not shown under an Appearance heading", async () => {
  const doc = await render(CASES.map(field));
  const sec = doc.querySelector('[data-field-key="temp_location"]').closest(".settings-section");
  assert.notEqual(heading(sec), "Appearance");
  assert.notEqual(sec, doc.getElementById("sec-appearance"));
});

test("no two Settings cards share a heading (search shows every group at once)", async () => {
  const doc = await render(CASES.map(field));
  const seen = new Map();
  let n = 0;
  for (const sec of doc.querySelectorAll("#settings-content .settings-section")) {
    const h = heading(sec);
    if (!h) continue;
    n++;
    assert.ok(!seen.has(h), `"${h}" heads both #${seen.get(h)} and #${sec.id}`);
    seen.set(h, sec.id);
  }
  assert.ok(n > CASES.length + 10, `expected the static cards too, saw ${n} headed cards`);
});
