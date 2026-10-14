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

const SAVED = {
  saved: true, filename: "bug-y.md", path: "/home/bug-reports/bug-y.md",
  maintainer: "owner@example.com",
  report_markdown: "# localm bug report: studio froze\n\nthe whole text",
  report_sha256: "a".repeat(64),
};

function makeSendFetch(calls, sendResponses) {
  const queue = [...sendResponses];
  return async (url, opts = {}) => {
    const u = String(url);
    if (u === "/api/bug-report" || u === "/api/bug-report/send") {
      calls.push({ url: u, body: JSON.parse(opts.body || "{}") });
      const data = u === "/api/bug-report" ? SAVED : (queue.shift() || {});
      return { ok: true, status: 200, text: async () => "", json: async () => data };
    }
    return { ok: true, status: 200, text: async () => "",
      json: async () => ({ models: [], active: "", conversations: [], plugins: [] }) };
  };
}

test("Send to maintainer saves and shows the whole report; nothing is sent until confirmed", async () => {
  const calls = [];
  const { window } = loadAppWithPages({ fetchImpl: makeSendFetch(calls, [
    { uploaded: true, issue_url: "https://github.com/Matlan1/localm/issues/42",
      maintainer: "owner@example.com" },
  ]) });
  const doc = window.document;
  doc.getElementById("bug-upload").hidden = false;
  doc.getElementById("bug-desc").value = "studio froze";
  doc.getElementById("bug-upload").click();
  await flush();
  assert.deepEqual(calls.map((c) => c.url), ["/api/bug-report"], "only the save was posted");
  assert.equal("upload" in calls[0].body, false, "the save carries no upload flag");
  assert.equal(doc.getElementById("bug-review").hidden, false, "review panel shown");
  assert.equal(doc.getElementById("bug-preview").value, SAVED.report_markdown,
    "the preview is the saved text");
  assert.equal(doc.getElementById("bug-upload").hidden, true,
    "Send to maintainer is hidden while the confirm button is on screen");
  assert.equal(doc.getElementById("bug-desc").value, "studio froze", "text kept until sent");

  doc.getElementById("bug-confirm-send").click();
  await flush();
  assert.deepEqual(calls.map((c) => c.url), ["/api/bug-report", "/api/bug-report/send"]);
  assert.deepEqual(calls[1].body, { filename: "bug-y.md", sha256: SAVED.report_sha256 },
    "the send names the reviewed file by its digest");
  const out = doc.getElementById("bug-result");
  assert.match(out.textContent, /issues\/42/);
  assert.equal(doc.getElementById("bug-review").hidden, true, "review panel closed");
  assert.equal(doc.getElementById("bug-upload").hidden, false, "Send to maintainer restored");
  assert.equal(doc.getElementById("bug-desc").value, "", "cleared once sent");
});

test("Don't send closes the review without sending", async () => {
  const calls = [];
  const { window } = loadAppWithPages({ fetchImpl: makeSendFetch(calls, []) });
  const doc = window.document;
  doc.getElementById("bug-desc").value = "never mind";
  doc.getElementById("bug-upload").click();
  await flush();
  doc.getElementById("bug-cancel-send").click();
  await flush();
  assert.deepEqual(calls.map((c) => c.url), ["/api/bug-report"], "no send was posted");
  assert.equal(doc.getElementById("bug-review").hidden, true);
  assert.match(doc.getElementById("bug-result").textContent, /Not sent\..*bug-y\.md/);
  doc.getElementById("bug-retry").click();
  await flush();
  assert.equal(calls.length, 1, "nothing left to retry after Don't send");
});

test("a report that cannot be loaded for review is not offered for sending", async () => {
  const calls = [];
  const unreadable = { saved: true, filename: "bug-n.md", path: "/p/bug-n.md",
    maintainer: "owner@example.com" };
  const { window } = loadAppWithPages({ fetchImpl: async (url, opts = {}) => {
    if (String(url).startsWith("/api/bug-report")) {
      calls.push(String(url));
      return { ok: true, status: 200, text: async () => "", json: async () => unreadable };
    }
    return { ok: true, status: 200, text: async () => "",
      json: async () => ({ models: [], active: "", conversations: [], plugins: [] }) };
  } });
  const doc = window.document;
  doc.getElementById("bug-desc").value = "x";
  doc.getElementById("bug-upload").click();
  await flush();
  assert.deepEqual(calls, ["/api/bug-report"]);
  assert.equal(doc.getElementById("bug-review").hidden, true);
  assert.match(doc.getElementById("bug-result").textContent, /was not sent/);
});

test("R47: a failed send shows WHERE it failed, reveals Retry + Download, keeps the text", async () => {
  const calls = [];
  const { window } = loadAppWithPages({ fetchImpl: makeSendFetch(calls, [{
    uploaded: false, upload_stage: "offline_or_dns", upload_message: "You may be offline.",
    upload_error: "could not reach the bug-report server: dns",
    maintainer: "owner@example.com",
  }]) });
  const doc = window.document;
  doc.getElementById("bug-desc").value = "it will not send";
  doc.getElementById("bug-upload").click();
  await flush();
  doc.getElementById("bug-confirm-send").click();
  await flush();
  const out = doc.getElementById("bug-result");
  assert.match(out.textContent, /Could not send: You may be offline\./,
    "shows the diagnosed 'where it failed' message");
  assert.equal(doc.getElementById("bug-retry").hidden, false, "Retry revealed");
  assert.equal(doc.getElementById("bug-download").hidden, false, "Download revealed");
  assert.equal(doc.getElementById("bug-desc").value, "it will not send",
    "the description is kept");
});

test("R47: Retry re-sends the SAME reviewed report and Download hands back its markdown", async () => {
  const calls = [];
  const { window } = loadAppWithPages({ fetchImpl: makeSendFetch(calls, [
    { uploaded: false, upload_message: "The server may be down.", upload_error: "unreachable" },
    { uploaded: true },
  ]) });
  const doc = window.document;
  const blobs = [];
  window.URL.createObjectURL = (b) => { blobs.push(b); return "blob:x"; };
  window.URL.revokeObjectURL = () => {};
  // jsdom does not implement download-anchor navigation; no-op the click
  window.HTMLAnchorElement.prototype.click = function () {};
  doc.getElementById("bug-desc").value = "please send";
  doc.getElementById("bug-upload").click();
  await flush();
  doc.getElementById("bug-confirm-send").click();
  await flush();
  assert.equal(doc.getElementById("bug-download").hidden, false, "Download offered after a failed send");
  doc.getElementById("bug-download").click();
  assert.equal(blobs.length, 1, "download built a blob from the saved report");
  assert.equal(await blobs[0].text(), SAVED.report_markdown, "the download is the reviewed text");
  doc.getElementById("bug-retry").click();
  await flush();
  assert.deepEqual(calls.map((c) => c.url),
    ["/api/bug-report", "/api/bug-report/send", "/api/bug-report/send"],
    "Retry re-sent without saving a fresh report");
  assert.deepEqual(calls[2].body, calls[1].body, "the retry names the same reviewed file");
  assert.match(doc.getElementById("bug-result").textContent, /^Sent\./);
  assert.equal(doc.getElementById("bug-download").hidden, true, "Download hidden once sent");
  assert.equal(doc.getElementById("bug-retry").hidden, true, "Retry hidden once sent");
});

test("R47: a 2xx send with no issue_url still reports 'Sent' (matches the toast)", async () => {
  const calls = [];
  const { window } = loadAppWithPages({ fetchImpl: makeSendFetch(calls, [
    { uploaded: true, maintainer: "owner@example.com" },
  ]) });
  const doc = window.document;
  doc.getElementById("bug-desc").value = "no url returned";
  doc.getElementById("bug-upload").click();
  await flush();
  doc.getElementById("bug-confirm-send").click();
  await flush();
  assert.match(doc.getElementById("bug-result").textContent, /^Sent\./,
    "shows Sent, not the Saved fallback");
});

test("Download after a plain Save hands back the saved markdown", async () => {
  const calls = [];
  const { window } = loadAppWithPages({ fetchImpl: makeSendFetch(calls, []) });
  const doc = window.document;
  const blobs = [];
  window.URL.createObjectURL = (b) => { blobs.push(b); return "blob:x"; };
  window.URL.revokeObjectURL = () => {};
  window.HTMLAnchorElement.prototype.click = function () {};
  doc.getElementById("bug-desc").value = "keep it local";
  doc.getElementById("bug-send").click();
  await flush();
  assert.deepEqual(calls.map((c) => c.url), ["/api/bug-report"]);
  assert.equal(doc.getElementById("bug-review").hidden, true, "Save shows no send prompt");
  doc.getElementById("bug-download").click();
  assert.equal(blobs.length, 1);
});

test("a rate-limited send is retried once, automatically, with the SAME confirmed report", async () => {
  const calls = [];
  const { window } = loadAppWithPages({ fetchImpl: makeSendFetch(calls, [
    { uploaded: false, rate_limited: true, retry_after: 1, upload_error: "rate limited" },
    { uploaded: true, issue_url: "https://github.com/Matlan1/localm/issues/7" },
  ]) });
  const doc = window.document;
  doc.getElementById("bug-upload").hidden = false;
  doc.getElementById("bug-desc").value = "slow down";
  doc.getElementById("bug-upload").click();
  await flush();
  doc.getElementById("bug-confirm-send").click();
  await flush();
  assert.match(doc.getElementById("bug-result").textContent, /retrying in 1s/);
  for (const id of ["bug-send", "bug-upload", "bug-confirm-send", "bug-cancel-send"]) {
    assert.equal(doc.getElementById(id).disabled, true, `${id} is locked during the countdown`);
  }
  await new Promise((r) => setTimeout(r, 1300));
  await flush();
  assert.deepEqual(calls.map((c) => c.url),
    ["/api/bug-report", "/api/bug-report/send", "/api/bug-report/send"]);
  assert.deepEqual(calls[2].body, calls[1].body, "the retry sends the confirmed report");
  assert.match(doc.getElementById("bug-result").textContent, /issues\/7/);
  assert.equal(doc.getElementById("bug-send").disabled, false, "buttons unlocked afterwards");
});

test("the automatic retry sends nothing once the confirmed report is no longer on review", async () => {
  const calls = [];
  const { window } = loadAppWithPages({ fetchImpl: makeSendFetch(calls, [
    { uploaded: false, rate_limited: true, retry_after: 1, upload_error: "rate limited" },
    { uploaded: true },
  ]) });
  const doc = window.document;
  doc.getElementById("bug-desc").value = "first report";
  doc.getElementById("bug-upload").click();
  await flush();
  doc.getElementById("bug-confirm-send").click();
  await flush();
  // A new review replaces the confirmed one during the countdown.
  doc.getElementById("bug-upload").disabled = false;
  doc.getElementById("bug-upload").click();
  await flush();
  await new Promise((r) => setTimeout(r, 1300));
  await flush();
  assert.deepEqual(calls.map((c) => c.url),
    ["/api/bug-report", "/api/bug-report/send", "/api/bug-report"],
    "nothing was sent for the report that was never confirmed");
  assert.match(doc.getElementById("bug-result").textContent, /Not sent\./);
});
