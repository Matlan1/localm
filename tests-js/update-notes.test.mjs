// SPDX-License-Identifier: AGPL-3.0-or-later
// Settings > Updates: the offered release's notes render under "Update
// available" through the real vendored marked + DOMPurify (renderMarkdown), not
// as raw Markdown in the status line, and remote HTML in them is inert.

import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import { loadAppWithPages } from "./harness.mjs";

const VENDOR = new URL("../localm/plugins/gui/static/vendor/", import.meta.url);

function makeFetch(check) {
  return async (url) => {
    const u = String(url);
    if (u.includes("/api/update/check")) {
      return { ok: true, status: 200, text: async () => "", json: async () => check };
    }
    return { ok: true, status: 200, text: async () => "",
      json: async () => ({ models: [], active: "", conversations: [], plugins: [] }) };
  };
}

async function flush() {
  for (let i = 0; i < 4; i++) await new Promise((r) => setTimeout(r, 0));
}

/** Boot the app with pages, then swap the harness's identity stubs for the real
 *  vendored marked and DOMPurify (globals resolved at call time). */
function setup(check) {
  const { window } = loadAppWithPages({ fetchImpl: makeFetch(check) });
  for (const f of ["marked.min.js", "purify.min.js"]) {
    const tag = window.document.createElement("script");
    tag.textContent = fs.readFileSync(new URL(f, VENDOR), "utf8");
    window.document.head.appendChild(tag);
  }
  assert.equal(typeof window.DOMPurify.sanitize, "function", "real DOMPurify did not load");
  window.marked.setOptions({ breaks: true });
  return window;
}

const NOTES = "This is a release candidate.\n\n**New since rc1**\n\n" +
  "- Error messages in the GUI\n- Faster model loads";

test("release notes render as Markdown (bold, list), not raw asterisks in the status line", async () => {
  const window = setup({ available: true, newer: true, current: "0.3.1",
    latest: "v0.3.2rc2", notes: NOTES, asset: { id: 7 } });
  const doc = window.document;
  doc.getElementById("update-check").click();
  await flush();

  const status = doc.getElementById("update-status");
  assert.match(status.textContent, /Update available: v0\.3\.2rc2/);
  assert.doesNotMatch(status.textContent, /\*\*|New since rc1/,
    "the notes are not appended to the status line as raw text");

  const notes = doc.getElementById("update-notes");
  assert.equal(notes.hidden, false, "the notes box is shown");
  const strong = notes.querySelector("strong");
  assert.ok(strong, "**New since rc1** renders as bold");
  assert.equal(strong.textContent, "New since rc1");
  const items = [...notes.querySelectorAll("ul > li")].map((li) => li.textContent.trim());
  assert.deepEqual(items, ["Error messages in the GUI", "Faster model loads"],
    "the '- ' lines render as a list");
  assert.doesNotMatch(notes.textContent, /\*\*/, "no literal asterisks survive");
});

test("script and onerror in release notes are inert", async () => {
  const payload = "Notes <script>window.__xss=1</script>\n\n" +
    '<img src=x onerror="window.__xss=2">\n\n[c](javascript:window.__xss=3)';
  const window = setup({ available: true, newer: true, current: "0.3.1",
    latest: "v0.3.2", notes: payload, asset: { id: 7 } });
  const doc = window.document;
  doc.getElementById("update-check").click();
  await flush();

  const notes = doc.getElementById("update-notes");
  assert.equal(notes.hidden, false);
  assert.equal(notes.querySelector("script"), null, "no <script> element survives");
  const withHandler = [...notes.querySelectorAll("*")]
    .filter((n) => [...n.attributes].some((a) => /^on/i.test(a.name)));
  assert.deepEqual(withHandler.map((n) => n.tagName), [], "no on* handler attribute survives");
  const jsLinks = [...notes.querySelectorAll("a[href]")]
    .filter((a) => /^\s*javascript:/i.test(a.getAttribute("href")));
  assert.equal(jsLinks.length, 0, "no javascript: link survives");
  assert.equal(window.__xss, undefined, "nothing in the notes executed");
});

test("long notes start collapsed behind 'Show all' and expand on click", async () => {
  const long = Array.from({ length: 12 }, (_, i) => `- change number ${i + 1}`).join("\n");
  const window = setup({ available: true, newer: true, current: "0.3.1",
    latest: "v0.3.2", notes: long, asset: { id: 7 } });
  const doc = window.document;
  doc.getElementById("update-check").click();
  await flush();

  const notes = doc.getElementById("update-notes");
  const more = doc.getElementById("update-notes-more");
  assert.equal(notes.querySelectorAll("li").length, 12, "every item is rendered");
  assert.ok(notes.classList.contains("collapsed"), "long notes start collapsed");
  assert.equal(doc.getElementById("update-notes-actions").hidden, false);
  assert.equal(more.textContent, "Show all");
  more.click();
  assert.ok(!notes.classList.contains("collapsed"), "Show all expands");
  assert.equal(more.textContent, "Show less");
  more.click();
  assert.ok(notes.classList.contains("collapsed"), "Show less collapses again");
});

test("short notes show in full with no 'Show all', and no notes hide the box", async () => {
  const window = setup({ available: true, newer: true, current: "0.3.1",
    latest: "v0.3.2", notes: "Small fix.", asset: { id: 7 } });
  const doc = window.document;
  doc.getElementById("update-check").click();
  await flush();
  const notes = doc.getElementById("update-notes");
  assert.equal(notes.hidden, false);
  assert.ok(!notes.classList.contains("collapsed"));
  assert.equal(doc.getElementById("update-notes-actions").hidden, true);

  window.renderUpdateNotes("");
  assert.equal(notes.hidden, true, "empty notes hide the box");
  assert.equal(notes.textContent, "", "and clear the previous render");
});
