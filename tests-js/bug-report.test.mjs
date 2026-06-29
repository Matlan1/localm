// SPDX-License-Identifier: AGPL-3.0-or-later
// R47: a Settings "Report a bug" control that POSTs a description (and an
// optional "attach recent log" flag) to /api/bug-report, which saves an
// editable report file the user can send to the maintainer.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages } from "./harness.mjs";

function makeFetch(posts, bugResponse) {
  return async (url, opts = {}) => {
    const u = String(url);
    if (u === "/api/bug-report") {
      posts.push(JSON.parse(opts.body || "{}"));
      return {
        ok: true, status: 200, text: async () => "",
        json: async () => (bugResponse || { saved: true, filename: "bug-x.md",
          path: "/home/bug-reports/bug-x.md", maintainer: "owner@example.com" }),
      };
    }
    return { ok: true, status: 200, text: async () => "",
      json: async () => ({ models: [], active: "", conversations: [], plugins: [] }) };
  };
}

async function flush() {
  await new Promise((r) => setTimeout(r, 0));
  await new Promise((r) => setTimeout(r, 0));
}

test("R47: Save report POSTs description + include_log and shows the saved path", async () => {
  const posts = [];
  const { window } = loadAppWithPages({ fetchImpl: makeFetch(posts) });
  const doc = window.document;
  doc.getElementById("bug-desc").value = "Mic button does nothing";
  doc.getElementById("bug-include-log").checked = true;
  doc.getElementById("bug-send").click();
  await flush();
  assert.equal(posts.length, 1, "one bug-report POST");
  assert.equal(posts[0].description, "Mic button does nothing");
  assert.equal(posts[0].include_log, true);
  const out = doc.getElementById("bug-result");
  assert.equal(out.hidden, false, "result line is shown");
  assert.match(out.textContent, /bug-x\.md/);
  assert.match(out.textContent, /owner@example\.com/);
  assert.equal(doc.getElementById("bug-desc").value, "", "textarea cleared after send");
});

test("R47: a blank description does not POST", async () => {
  const posts = [];
  const { window } = loadAppWithPages({ fetchImpl: makeFetch(posts) });
  window.document.getElementById("bug-desc").value = "   ";
  window.document.getElementById("bug-send").click();
  await flush();
  assert.equal(posts.length, 0, "no POST for an empty description");
});

test("R47: the POST carries a browser client context (UA, page, viewport, console errors)", async () => {
  const posts = [];
  const { window } = loadAppWithPages({ fetchImpl: makeFetch(posts) });
  const doc = window.document;
  // A JS error captured by app.js's always-on client error ring buffer.
  window.__localmClientLog.push("12:00:00  TypeError: render is not a function");
  doc.getElementById("bug-desc").value = "studio page went blank";
  doc.getElementById("bug-send").click();
  await flush();
  assert.equal(posts.length, 1, "one bug-report POST");
  const client = posts[0].client;
  assert.ok(client, "a client context block is attached");
  assert.equal(typeof client.userAgent, "string");
  assert.ok("page" in client && "viewport" in client, "page + viewport present");
  assert.ok(Array.isArray(client.console), "console errors sent as a list");
  assert.ok(
    client.console.some((l) => /TypeError: render is not a function/.test(l)),
    "the captured console error is included");
});

test("R47: Send to maintainer POSTs upload:true and shows the tracking issue URL", async () => {
  const posts = [];
  const { window } = loadAppWithPages({ fetchImpl: makeFetch(posts, {
    saved: true, uploaded: true, path: "/home/bug-reports/bug-y.md",
    issue_url: "https://github.com/Matlan1/localm/issues/42",
    maintainer: "owner@example.com",
  }) });
  const doc = window.document;
  doc.getElementById("bug-desc").value = "studio froze";
  // The button is revealed by capabilities in the app; click it directly here.
  doc.getElementById("bug-upload").click();
  await flush();
  assert.equal(posts.length, 1, "one bug-report POST");
  assert.equal(posts[0].upload, true, "upload flag set");
  const out = doc.getElementById("bug-result");
  assert.equal(out.hidden, false);
  assert.match(out.textContent, /issues\/42/);
});

test("R47: a 2xx upload with no issue_url still reports 'Sent' (matches the toast)", async () => {
  const posts = [];
  const { window } = loadAppWithPages({ fetchImpl: makeFetch(posts, {
    saved: true, uploaded: true, path: "/home/bug-reports/bug-z.md",
    maintainer: "owner@example.com",  // note: proxy returned no issue_url
  }) });
  const doc = window.document;
  doc.getElementById("bug-desc").value = "no url returned";
  doc.getElementById("bug-upload").click();
  await flush();
  const out = doc.getElementById("bug-result");
  assert.match(out.textContent, /^Sent\./, "shows Sent, not the Saved fallback");
});
