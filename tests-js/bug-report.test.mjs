// SPDX-License-Identifier: AGPL-3.0-or-later
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

test("#958: the include-log checkbox defaults to checked, and its value is honored unset", async () => {
  const posts = [];
  const { window } = loadAppWithPages({ fetchImpl: makeFetch(posts) });
  const doc = window.document;
  assert.equal(doc.getElementById("bug-include-log").checked, true,
    "checkbox must default to checked");
  doc.getElementById("bug-desc").value = "did not touch the checkbox";
  doc.getElementById("bug-send").click();
  await flush();
  assert.equal(posts[0].include_log, true, "unset checkbox POSTs its checked default");
});

test("R47: a blank description does not POST", async () => {
  const posts = [];
  const { window } = loadAppWithPages({ fetchImpl: makeFetch(posts) });
  window.document.getElementById("bug-desc").value = "   ";
  window.document.getElementById("bug-send").click();
  await flush();
  assert.equal(posts.length, 0, "no POST for an empty description");
});

test("#958: What I expected / What happened POST as their own fields, and clear on save", async () => {
  const posts = [];
  const { window } = loadAppWithPages({ fetchImpl: makeFetch(posts) });
  const doc = window.document;
  doc.getElementById("bug-desc").value = "clicked generate";
  doc.getElementById("bug-expected").value = "a picture of a cat";
  doc.getElementById("bug-happened").value = "a blank grey square appeared";
  doc.getElementById("bug-send").click();
  await flush();
  assert.equal(posts.length, 1, "one bug-report POST");
  assert.equal(posts[0].description, "clicked generate");
  assert.equal(posts[0].what_i_expected, "a picture of a cat");
  assert.equal(posts[0].what_happened, "a blank grey square appeared");
  assert.equal(doc.getElementById("bug-desc").value, "", "description cleared after save");
  assert.equal(doc.getElementById("bug-expected").value, "", "expected cleared after save");
  assert.equal(doc.getElementById("bug-happened").value, "", "happened cleared after save");
});

test("#958: a blank description with 'what happened' filled in still POSTs", async () => {
  const posts = [];
  const { window } = loadAppWithPages({ fetchImpl: makeFetch(posts) });
  const doc = window.document;
  doc.getElementById("bug-desc").value = "   ";
  doc.getElementById("bug-happened").value = "it crashed on startup";
  doc.getElementById("bug-send").click();
  await flush();
  assert.equal(posts.length, 1, "what_happened alone is enough to send");
  assert.equal(posts[0].what_happened, "it crashed on startup");
});

test("#958: blank description and blank 'what happened' still refuses to POST", async () => {
  const posts = [];
  const { window } = loadAppWithPages({ fetchImpl: makeFetch(posts) });
  const doc = window.document;
  doc.getElementById("bug-desc").value = "";
  doc.getElementById("bug-expected").value = "something, but not what happened";
  doc.getElementById("bug-happened").value = "   ";
  doc.getElementById("bug-send").click();
  await flush();
  assert.equal(posts.length, 0, "expected alone (no description/happened) does not POST");
});

test("R47: the POST carries a browser client context (UA, page, viewport, console errors)", async () => {
  const posts = [];
  const { window } = loadAppWithPages({ fetchImpl: makeFetch(posts) });
  const doc = window.document;
  // a JS error in app.js's client error ring buffer
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
  // the app reveals this button via capabilities; click it directly
  doc.getElementById("bug-upload").click();
  await flush();
  assert.equal(posts.length, 1, "one bug-report POST");
  assert.equal(posts[0].upload, true, "upload flag set");
  const out = doc.getElementById("bug-result");
  assert.equal(out.hidden, false);
  assert.match(out.textContent, /issues\/42/);
});

test("R47: a failed upload shows WHERE it failed, reveals Retry + Download, keeps the text", async () => {
  const posts = [];
  const { window } = loadAppWithPages({ fetchImpl: makeFetch(posts, {
    saved: true, uploaded: false,
    upload_stage: "offline_or_dns", upload_message: "You may be offline.",
    upload_error: "could not reach the bug-report server: dns",
    report_markdown: "# localm bug report\nbody", filename: "bug-f.md",
    path: "/home/bug-reports/bug-f.md", maintainer: "owner@example.com",
  }) });
  const doc = window.document;
  doc.getElementById("bug-desc").value = "it will not send";
  doc.getElementById("bug-upload").click();
  await flush();
  const out = doc.getElementById("bug-result");
  assert.match(out.textContent, /Could not send: You may be offline\./,
    "shows the diagnosed 'where it failed' message");
  assert.equal(doc.getElementById("bug-retry").hidden, false, "Retry revealed");
  assert.equal(doc.getElementById("bug-download").hidden, false, "Download revealed");
  assert.equal(doc.getElementById("bug-desc").value, "it will not send",
    "the description is kept so Retry can re-use it");
});

test("R47: Retry re-sends the report and Download hands back the saved markdown", async () => {
  const posts = [];
  const { window } = loadAppWithPages({ fetchImpl: makeFetch(posts, {
    saved: true, uploaded: false, upload_message: "The server may be down.",
    upload_error: "unreachable", report_markdown: "# report\nbody",
    filename: "bug-g.md", path: "/p/bug-g.md", maintainer: "owner@example.com",
  }) });
  const doc = window.document;
  const blobs = [];
  window.URL.createObjectURL = (b) => { blobs.push(b); return "blob:x"; };
  window.URL.revokeObjectURL = () => {};
  // jsdom does not implement download-anchor navigation; no-op the click
  window.HTMLAnchorElement.prototype.click = function () {};
  doc.getElementById("bug-desc").value = "please send";
  doc.getElementById("bug-upload").click();
  await flush();
  assert.equal(posts.length, 1);
  doc.getElementById("bug-retry").click();          // retry re-attempts the send
  await flush();
  assert.equal(posts.length, 2, "Retry re-POSTed the report");
  assert.equal(posts[1].upload, true, "retry keeps the upload flag");
  doc.getElementById("bug-download").click();        // download the saved report
  assert.equal(blobs.length, 1, "download built a blob from the saved report");
});

test("R47: a 2xx upload with no issue_url still reports 'Sent' (matches the toast)", async () => {
  const posts = [];
  const { window } = loadAppWithPages({ fetchImpl: makeFetch(posts, {
    saved: true, uploaded: true, path: "/home/bug-reports/bug-z.md",
    maintainer: "owner@example.com",  // proxy returned no issue_url
  }) });
  const doc = window.document;
  doc.getElementById("bug-desc").value = "no url returned";
  doc.getElementById("bug-upload").click();
  await flush();
  const out = doc.getElementById("bug-result");
  assert.match(out.textContent, /^Sent\./, "shows Sent, not the Saved fallback");
});
