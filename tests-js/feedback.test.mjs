// SPDX-License-Identifier: AGPL-3.0-or-later
// Settings > Feedback and Suggestion: two cards, one text box each, saved and
// reviewed before anything is sent, and the sidebar link that opens them.
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages } from "./harness.mjs";

const SAVED = { saved: true, filename: "feedback-1.md", path: "/home/bug-reports/feedback-1.md",
  maintainer: "owner@example.com", report_markdown: "# hi\n\nhi\n", report_sha256: "ab".repeat(32) };

function boot({ saved = SAVED, sends = [], saveStatus = 200, gate = null, upload = true } = {}) {
  const calls = [];
  const fetchImpl = async (url, opts = {}) => {
    const u = String(url);
    const reply = (body, status = 200) => ({ ok: status < 400, status, statusText: "x",
      text: async () => "", json: async () => body });
    if (u === "/api/feedback") {
      calls.push({ url: u, body: JSON.parse(opts.body) });
      if (gate) await gate;
      return reply(saveStatus < 400 ? saved : { detail: "disk full" }, saveStatus);
    }
    if (u.startsWith("/api/bug-report")) {
      calls.push({ url: u, body: JSON.parse(opts.body) });
      return reply(sends.shift() || { uploaded: true });
    }
    return reply({ models: [], active: "", conversations: [], plugins: [], bugreport_upload: upload });
  };
  const { window } = loadAppWithPages({ fetchImpl });
  window.setFeedbackUpload(upload);
  const $ = (id) => window.document.getElementById(id);
  return { window, $, calls };
}

async function flush() {
  for (let i = 0; i < 4; i++) await new Promise((r) => setTimeout(r, 0));
}

async function review($, kind, text = "hi") {
  $(`${kind}-text`).value = text;
  $(`${kind}-upload`).click();
  await flush();
}

for (const kind of ["feedback", "suggestion"]) {
  test(`${kind}: the text is saved, reviewed, and sent only when confirmed`, async () => {
    const { $, calls } = boot({ sends: [{ uploaded: true, issue_url: "https://example.invalid/7" }] });
    await review($, kind, "  hello there ");
    assert.deepEqual(calls, [{ url: "/api/feedback", body: { kind, text: "hello there" } }]);
    assert.equal($(`${kind}-review`).hidden, false);
    assert.equal($(`${kind}-preview`).value, SAVED.report_markdown);
    assert.equal($(`${kind}-upload`).hidden, true, "one primary button at a time");
    assert.equal($(`${kind}-text`).readOnly, true, "the text under review cannot drift from it");
    $(`${kind}-confirm`).click();
    await flush();
    assert.deepEqual(calls[1], { url: "/api/bug-report/send",
      body: { filename: SAVED.filename, sha256: SAVED.report_sha256 } });
    assert.equal(calls.length, 2);
    assert.match($(`${kind}-result`).textContent, /Sent\..*example\.invalid\/7/);
    assert.equal($(`${kind}-text`).value, "");
    assert.equal($(`${kind}-text`).readOnly, false);
    assert.equal($(`${kind}-review`).hidden, true);
    assert.equal($(`${kind}-upload`).hidden, false);
  });
}

test("the two cards do not share state", async () => {
  const { $, calls } = boot();
  await review($, "feedback", "one");
  $("suggestion-text").value = "two";
  assert.equal($("suggestion-review").hidden, true);
  assert.equal($("suggestion-confirm").disabled, false);
  $("suggestion-confirm").click();
  await flush();
  assert.equal(calls.length, 1, "the other card's hidden Send sends nothing");
  $("feedback-cancel").click();
  assert.equal($("suggestion-text").value, "two");
  assert.equal($("feedback-text").value, "one", "the text is kept when it is not sent");
});

test("empty text is not saved", async () => {
  const { $, calls } = boot();
  await review($, "feedback", "   ");
  assert.deepEqual(calls, []);
});

test("Don't send closes the review and nothing is sent afterwards", async () => {
  const { $, calls } = boot();
  await review($, "feedback");
  $("feedback-cancel").click();
  assert.equal($("feedback-review").hidden, true);
  assert.match($("feedback-result").textContent, /Not sent\..*feedback-1\.md/);
  $("feedback-confirm").click();
  await flush();
  assert.equal(calls.length, 1);
});

test("a failed save says so and keeps the text", async () => {
  const { $, calls, window } = boot({ saveStatus: 500 });
  await review($, "feedback", "keep me");
  assert.equal(calls.length, 1);
  assert.equal($("feedback-review").hidden, true);
  assert.equal($("feedback-text").value, "keep me");
  assert.match(window.document.getElementById("toast").textContent, /disk full/);
  assert.equal($("feedback-upload").disabled, false);
});

test("a text that cannot be shown for review is not offered for sending", async () => {
  const { $, calls } = boot({ saved: { saved: true, filename: "feedback-1.md", path: "/p/feedback-1.md" } });
  await review($, "feedback");
  assert.equal($("feedback-review").hidden, true);
  assert.match($("feedback-result").textContent, /feedback-1\.md/);
  assert.equal(calls.length, 1);
});

test("a failed send keeps the review and can be sent again", async () => {
  const { $, calls } = boot({ sends: [
    { uploaded: false, upload_stage: "dns", upload_message: "no network", maintainer: "o@example.com" },
    { uploaded: true }] });
  await review($, "feedback");
  $("feedback-confirm").click();
  await flush();
  assert.match($("feedback-result").textContent, /no network/);
  assert.equal($("feedback-review").hidden, false);
  assert.equal($("feedback-anyway").hidden, true);
  $("feedback-confirm").click();
  await flush();
  assert.equal(calls.length, 3);
  assert.equal($("feedback-review").hidden, true);
});

test("a rate limit says to wait and keeps the review", async () => {
  const { $ } = boot({ sends: [{ uploaded: false, rate_limited: true, retry_after: 30 }] });
  await review($, "feedback");
  $("feedback-confirm").click();
  await flush();
  assert.match($("feedback-result").textContent, /rate limited/);
  assert.equal($("feedback-review").hidden, false);
});

test("Send anyway is offered only when the server offers it, and is passed on", async () => {
  const off = { uploaded: false, upload_stage: "network_off", upload_message: "Network access is off" };
  const { $, calls } = boot({ sends: [off, { ...off, can_send_anyway: true }, { uploaded: true }] });
  await review($, "feedback");
  $("feedback-confirm").click();
  await flush();
  assert.equal($("feedback-anyway").hidden, true);
  $("feedback-confirm").click();
  await flush();
  assert.equal($("feedback-anyway").hidden, false);
  $("feedback-send-anyway").click();
  await flush();
  assert.equal(calls[3].body.send_anyway, true);
  assert.equal("send_anyway" in calls[2].body, false);
});

test("without an upload endpoint the cards only save", async () => {
  const { $, calls, window } = boot({ upload: false });
  window.setFeedbackUpload(false);
  assert.equal($("feedback-upload").hidden, true);
  assert.equal($("feedback-save").hidden, false);
  $("feedback-text").value = "hi";
  $("feedback-save").click();
  await flush();
  assert.equal(calls.length, 1);
  assert.equal($("feedback-review").hidden, true);
  assert.match($("feedback-result").textContent, /feedback-1\.md.*owner@example\.com/);
  window.setFeedbackUpload(true);
  assert.equal($("feedback-save").hidden, true);
  assert.equal($("suggestion-upload").hidden, false);
});

test("the server's capability decides between sending and saving, whichever loads first", async () => {
  for (const available of [true, false]) {
    const { window, $ } = boot({ upload: available });
    window.setFeedbackUpload(!available);
    await window.refreshPluginCommands();
    for (const kind of ["feedback", "suggestion"]) {
      assert.equal($(`${kind}-upload`).hidden, !available, kind);
      assert.equal($(`${kind}-save`).hidden, available, kind);
    }
    const { window: late } = loadAppWithPages({ fetchImpl: async () => ({ ok: true, status: 200,
      text: async () => "", json: async () => ({}) }),
    beforePages: (w) => { w.__localmFeedbackUpload = available; } });
    assert.equal(late.document.getElementById("feedback-upload").hidden, !available);
  }
});

test("losing the capability during a review leaves the review as it is", async () => {
  const { $, calls, window } = boot();
  await review($, "feedback", "under review");
  window.setFeedbackUpload(false);
  assert.equal($("feedback-save").hidden, true);
  $("feedback-save").click();
  await flush();
  assert.equal(calls.length, 1, "no second file");
  assert.equal($("feedback-text").value, "under review");
  assert.equal($("feedback-review").hidden, false);
});

test("the same failure twice is shown twice", async () => {
  const off = { uploaded: false, upload_stage: "dns", upload_message: "no network" };
  const { $ } = boot({ sends: [off, off] });
  await review($, "feedback");
  $("feedback-confirm").click();
  await flush();
  const first = $("feedback-result").textContent;
  assert.doesNotMatch(first, /download/);
  $("feedback-confirm").click();
  assert.equal($("feedback-result").textContent, "", "cleared while sending");
  await flush();
  assert.equal($("feedback-result").textContent, first);
});

test("the buttons and the text are locked while the text is saved", async () => {
  let release;
  const gate = new Promise((r) => { release = r; });
  const { $, calls } = boot({ gate });
  await review($, "feedback");
  assert.equal($("feedback-upload").disabled, true);
  assert.equal($("feedback-text").readOnly, true);
  $("feedback-upload").click();
  assert.equal(calls.length, 1, "a second press saves nothing");
  release();
  await flush();
  assert.equal($("feedback-upload").disabled, false);
});

test("the sidebar link opens the report cards in Settings", async () => {
  const { window, $ } = boot();
  const seen = [];
  window.showView = (v) => seen.push(v);
  window.gotoSettingsSection = (s) => seen.push(s);
  $("feedback-link").click();
  assert.deepEqual(seen, ["settings", "sec-bugreport"]);
});

test("the sidebar link is hidden when Settings is hidden for this key", () => {
  const { window, $ } = boot();
  window.applyCoreTabVisibility({ models: true, plugins: true, settings: false });
  assert.equal($("feedback-link").hidden, true);
  window.applyCoreTabVisibility({ models: true, plugins: true, settings: true });
  assert.equal($("feedback-link").hidden, false);
});

test("Report this error still fills the bug form only", async () => {
  const { window, $ } = boot();
  $("feedback-text").value = "half written";
  window.toast("it broke", true);
  await flush();
  window.document.querySelector("#recent-errors-list .recent-error .actions button:nth-child(2)").click();
  assert.equal($("bug-happened").value, "it broke");
  assert.equal($("feedback-text").value, "half written");
  assert.equal($("suggestion-text").value, "");
});
