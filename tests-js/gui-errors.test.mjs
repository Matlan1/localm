// SPDX-License-Identifier: AGPL-3.0-or-later
// Error toasts stay until dismissed and stack; every GUI error is recorded in the
// Settings "Recent errors" list (memory only in privacy mode) and POSTed once to
// /api/gui-error; "Report this error" fills the bug report form and sends nothing.
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

function makeFetch(log, { mode = "privacy", guiError } = {}) {
  return async (url, opts = {}) => {
    const u = String(url);
    log.push({ url: u, opts });
    if (u === "/api/gui-error") {
      if (guiError) return guiError();
      return { ok: true, status: 200, json: async () => ({ logged: true }), text: async () => "" };
    }
    if (u === "/v1/config") {
      return { ok: true, status: 200, text: async () => "",
        json: async () => ({ effective_mode: mode, instance_id: "inst-1" }) };
    }
    return { ok: true, status: 200, text: async () => "",
      json: async () => ({ models: [], active: "", conversations: [], plugins: [] }) };
  };
}

const flush = async (n = 4) => {
  for (let i = 0; i < n; i++) await new Promise((r) => setTimeout(r, 0));
};
const guiPosts = (log) => log.filter((c) => c.url === "/api/gui-error")
  .map((c) => JSON.parse(c.opts.body));
const stackTexts = (doc) => [...doc.querySelectorAll("#toast-errors .toast-error-text")]
  .map((n) => n.textContent);

async function boot(opts = {}) {
  const log = [];
  const app = loadAppWithPages({ fetchImpl: makeFetch(log, opts), ...opts.load });
  await flush(8);
  await app.window.refreshCtxLimit();
  await flush();
  log.length = 0;
  return { ...app, log, doc: app.window.document };
}

test("an error toast stays past the info-toast timeout; a later info toast does not remove it", async () => {
  const { window, doc } = await boot();
  window.toast("Could not load model x", true);
  window.toast("Saved");
  assert.deepEqual(stackTexts(doc), ["Could not load model x"]);
  assert.equal(doc.getElementById("toast").className, "show");
  await new Promise((r) => setTimeout(r, 3700));
  assert.deepEqual(stackTexts(doc), ["Could not load model x"], "the error is still on screen");
  assert.equal(doc.getElementById("toast-errors").hidden, false);
  assert.equal(doc.getElementById("toast").className, "", "the info toast auto-hid");
});

test("#toast still names the latest message and flags an error", async () => {
  const { window, doc } = await boot();
  window.toast("disk full", true);
  assert.equal(doc.getElementById("toast").textContent, "disk full");
  assert.ok(doc.getElementById("toast").className.includes("error"));
});

test("the close button dismisses an error, and Escape dismisses the focused one", async () => {
  const { window, doc } = await boot();
  window.toast("first", true);
  window.toast("second", true);
  const closes = doc.querySelectorAll("#toast-errors .toast-error-close");
  assert.equal(closes.length, 2);
  for (const c of closes) {
    assert.equal(c.tagName, "BUTTON");
    assert.equal(c.getAttribute("aria-label"), "Dismiss this error");
  }
  closes[0].click();
  assert.deepEqual(stackTexts(doc), ["second"]);
  const close = doc.querySelector("#toast-errors .toast-error-close");
  close.focus();
  close.dispatchEvent(new window.KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
  assert.deepEqual(stackTexts(doc), []);
  assert.equal(doc.getElementById("toast-errors").hidden, true);
});

test("only the error just added is announced, and focus stays on a close button across a new error", async () => {
  const { window, doc } = await boot();
  window.toast("first", true);
  doc.querySelector("#toast-errors .toast-error-close").focus();
  window.toast("second", true);
  const alerts = [...doc.querySelectorAll("#toast-errors [role=alert] .toast-error-text")].map((n) => n.textContent);
  assert.deepEqual(alerts, ["second"]);
  assert.equal(doc.activeElement.closest(".toast-error").dataset.text, "first");
});

test("an identical repeated error collapses into one toast with a count", async () => {
  const { window, doc } = await boot();
  window.toast("same failure", true);
  window.toast("other failure", true);
  window.toast("same failure", true);
  assert.deepEqual(stackTexts(doc), ["other failure", "same failure"]);
  const counts = [...doc.querySelectorAll("#toast-errors .toast-error-count")].map((n) => n.textContent);
  assert.deepEqual(counts, ["(x2)"]);
});

test("the stack shows the newest three and a +N more row with Dismiss all", async () => {
  const { window, doc } = await boot();
  for (const m of ["e1", "e2", "e3", "e4", "e5"]) window.toast(m, true);
  assert.deepEqual(stackTexts(doc), ["e3", "e4", "e5"]);
  const more = doc.querySelector("#toast-errors .toast-error-more-btn");
  assert.equal(more.textContent, "+2 more");
  let opened = 0;
  window.openRecentErrors = () => { opened++; };
  more.click();
  assert.equal(opened, 1);
  doc.querySelector("#toast-errors .toast-error-dismiss-all").click();
  assert.deepEqual(stackTexts(doc), []);
});

test("the Recent errors list keeps exactly the last five, newest first", async () => {
  const { window, doc } = await boot();
  for (let i = 1; i <= 7; i++) window.toast(`err ${i}`, true);
  const items = [...doc.querySelectorAll("#recent-errors-list .recent-error-text")].map((n) => n.textContent);
  assert.deepEqual(items, ["err 7", "err 6", "err 5", "err 4", "err 3"]);
  assert.equal(doc.getElementById("recent-errors-empty").hidden, true);
  doc.getElementById("recent-errors-clear").click();
  assert.equal(doc.querySelectorAll("#recent-errors-list li").length, 0);
  assert.equal(doc.getElementById("recent-errors-empty").hidden, false);
});

test("an error already in the list moves to the top with a count instead of a new row", async () => {
  const { window, doc } = await boot();
  window.toast("load failed for x", true);
  runScript(window, `setStatus("err", "load failed")`);
  window.toast("load failed for x", true);
  const rows = [...doc.querySelectorAll("#recent-errors-list .recent-error")];
  assert.deepEqual(rows.map((r) => r.querySelector(".recent-error-text").textContent),
    ["load failed for x", "load failed"]);
  assert.match(rows[0].querySelector(".recent-error-meta").textContent, /\(x2\)/);
});

test("privacy mode keeps the list in memory only: nothing reaches any browser storage", async () => {
  const { window, doc } = await boot({ mode: "privacy" });
  const before = { ls: window.localStorage.length, ss: window.sessionStorage.length };
  const lsKeys = () => Object.keys({ ...window.localStorage });
  for (let i = 0; i < 6; i++) window.toast(`private error ${i}`, true);
  window.dispatchEvent(new window.ErrorEvent("error", { message: "thrown in privacy" }));
  assert.equal(doc.querySelectorAll("#recent-errors-list li").length, 5, "shown in memory");
  assert.equal(window.localStorage.getItem("localm.recentErrors"), null);
  assert.ok(!lsKeys().some((k) => /error/i.test(k)), `no error key in localStorage: ${lsKeys()}`);
  for (let i = 0; i < window.localStorage.length; i++) {
    assert.ok(!/private error|thrown in privacy/.test(window.localStorage.getItem(window.localStorage.key(i))));
  }
  assert.equal(window.sessionStorage.length, before.ss, "sessionStorage untouched");
  assert.equal(window.localStorage.length, before.ls, "localStorage gained no key");
});

test("outside privacy mode the list persists, and turning privacy on wipes the stored copy", async () => {
  const log = [];
  let mode = "full";
  const fetchImpl = async (url, opts) => {
    if (String(url) === "/v1/config") {
      return { ok: true, status: 200, text: async () => "",
        json: async () => ({ effective_mode: mode, instance_id: "inst-1" }) };
    }
    return makeFetch(log)(url, opts);
  };
  const { window } = loadAppWithPages({ fetchImpl });
  await flush(8);
  await window.refreshCtxLimit();
  window.toast("kept error", true);
  const stored = JSON.parse(window.localStorage.getItem("localm.recentErrors"));
  assert.equal(stored[0].text, "kept error");
  mode = "privacy";
  await window.refreshCtxLimit();
  assert.equal(window.localStorage.getItem("localm.recentErrors"), null);
  window.toast("private one", true);
  assert.equal(window.localStorage.getItem("localm.recentErrors"), null);
  mode = "full";
  await window.refreshCtxLimit();
  window.toast("after privacy", true);
  assert.equal(JSON.parse(window.localStorage.getItem("localm.recentErrors"))[0].text, "after privacy");
});

test("a stored list is restored once the server mode is known (not privacy)", async () => {
  const seed = JSON.stringify([{ text: "from last time", time: 1, page: "#models", source: "toast", count: 1 }]);
  const { window, doc } = await boot({ mode: "full", load: {
    seedLocalStorage: { "localm.instanceId": "inst-1", "localm.recentErrors": seed } } });
  const items = [...doc.querySelectorAll("#recent-errors-list .recent-error-text")].map((n) => n.textContent);
  assert.deepEqual(items, ["from last time"]);
  window.toast("new one", true);
  assert.deepEqual(JSON.parse(window.localStorage.getItem("localm.recentErrors")).map((e) => e.text),
    ["new one", "from last time"]);
});

test("Report this error fills the bug report form and sends nothing", async () => {
  const { window, doc, log } = await boot();
  window.toast("Model load failed: out of memory", true);
  await flush();
  log.length = 0;
  doc.querySelector("#recent-errors-list .recent-error .actions button:nth-child(2)").click();
  await flush();
  assert.equal(doc.getElementById("bug-happened").value, "Model load failed: out of memory");
  assert.match(doc.getElementById("bug-desc").value, /The GUI showed this error/);
  assert.equal(doc.getElementById("bug-review").hidden, true, "no review opened");
  assert.deepEqual(log.filter((c) => c.url.startsWith("/api/bug-report")), [], "nothing sent");
});

test("a toast error, an uncaught error and a rejection each POST exactly once", async () => {
  const { window, log } = await boot();
  window.toast("toast boom", true);
  window.dispatchEvent(new window.ErrorEvent("error", { message: "uncaught boom", filename: "x.js", lineno: 3 }));
  const ev = new window.Event("unhandledrejection");
  ev.reason = new Error("rejected boom");
  window.dispatchEvent(ev);
  await flush();
  const posts = guiPosts(log);
  assert.equal(posts.length, 3);
  assert.deepEqual(posts.map((p) => p.source), ["toast", "window", "promise"]);
  assert.equal(posts[0].message, "toast boom");
  assert.match(posts[1].message, /^uncaught boom {2}\(x\.js:3\)$/);
  assert.equal(posts[2].message, "Unhandled rejection: rejected boom");
  for (const c of log.filter((x) => x.url === "/api/gui-error")) {
    assert.equal(c.opts.method, "POST");
    assert.equal(c.opts.headers["Content-Type"], "application/json");
  }
});

test("the report carries the open-mode shell token", async () => {
  const { window, log } = await boot({ load: { shellToken: "shell-abc" } });
  window.toast("needs auth", true);
  await flush();
  const call = log.find((c) => c.url === "/api/gui-error");
  assert.equal(call.opts.headers.Authorization, "Bearer shell-abc");
});

test("a failing report POST does not toast, record or report again", async () => {
  for (const guiError of [
    () => Promise.reject(new TypeError("Failed to fetch")),
    () => ({ ok: false, status: 500, json: async () => ({}), text: async () => "" }),
    () => { throw new Error("sync throw"); },
  ]) {
    const { window, doc, log } = await boot({ guiError });
    window.toast("original", true);
    await flush(8);
    assert.equal(guiPosts(log).length, 1);
    assert.deepEqual(stackTexts(doc), ["original"]);
    assert.equal(doc.querySelectorAll("#recent-errors-list li").length, 1);
  }
});

test("the same error is sent at most once per five seconds", async () => {
  const { window, log } = await boot();
  for (let i = 0; i < 4; i++) window.toast("storm", true);
  await flush();
  assert.equal(guiPosts(log).length, 1);
});

test("a status-pill error is recorded when it appears, not on every repeat", async () => {
  const { window, log } = await boot();
  runScript(window, `setStatus("err", "Server unreachable"); setStatus("err", "Server unreachable");`);
  await flush();
  assert.deepEqual(guiPosts(log).map((p) => [p.source, p.message]), [["status", "Server unreachable"]]);
  const items = [...window.document.querySelectorAll("#recent-errors-list li")];
  assert.equal(items.length, 1);
  assert.doesNotMatch(items[0].querySelector(".recent-error-meta").textContent, /\(x\d/,
    "a repeat of the shown status error is not counted again");
});
