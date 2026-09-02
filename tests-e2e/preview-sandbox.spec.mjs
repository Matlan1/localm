// SPDX-License-Identifier: AGPL-3.0-or-later
// The preview (artifacts canvas) containment boundary, driven in a REAL browser.
//
// The jsdom suite can only assert that the sandbox attribute and the CSP text are
// PRESENT: jsdom enforces neither, so "the artifact cannot reach the app" is
// unfalsifiable there. Here the artifact's own script tries the escapes and
// reports what it got, and an UNSANDBOXED control frame runs the identical probe
// so a blocked result is read against a reachable one rather than assumed.
//
// What each arm actually measures, because the two mechanisms are different and
// the shell's own CSP is inherited by both frames:
//   cookie / localStorage / parent  - the SANDBOX (no allow-same-origin, so the
//                                     frame is an opaque origin)
//   fetch("/api/...")              - the ARTIFACT's injected CSP. The shell's own
//                                     policy allows same-origin connect, so this
//                                     pair discriminates.
// A cross-origin beacon is deliberately NOT asserted: the shell's connect-src
// would block it in BOTH arms, so it could not tell the arms apart.

import { test, expect } from "@playwright/test";
import { BASE_URL } from "./_env.mjs";

const SECRET = "LOCALM-E2E-PREVIEW-COOKIE-7Q4M";

/** The escape probe. Reports every outcome to the embedder via postMessage,
 *  which is allowed cross-origin and is the one channel a sandboxed opaque
 *  origin keeps. */
const PROBE = `<!doctype html><html><head><meta charset="utf-8"></head><body>
<script>
(async function () {
  const out = { ran: true };
  try { out.cookie = document.cookie || "(empty)"; }
  catch (e) { out.cookie = "THREW:" + e.name; }
  try { window.localStorage.setItem("probe", "1"); out.storage = "written"; }
  catch (e) { out.storage = "THREW:" + e.name; }
  try { out.parentHref = String(parent.location.href); }
  catch (e) { out.parentHref = "THREW:" + e.name; }
  try { out.parentDoc = parent.document ? "reachable" : "null"; }
  catch (e) { out.parentDoc = "THREW:" + e.name; }
  try {
    // Escalate the way a real escape would: borrow the embedder's own
    // credentials. Unreachable from an opaque origin, so this degrades to an
    // unauthenticated call there and the fetch itself is what must fail.
    let headers = {};
    try { headers = parent.authHeaders ? parent.authHeaders() : {}; }
    catch (e) { headers = {}; }
    const r = await fetch("/api/capabilities", { headers: headers });
    out.api = "status:" + r.status;
    if (r.ok) { const j = await r.json(); out.apiRead = Object.keys(j).length > 0; }
  } catch (e) { out.api = "THREW:" + e.name; }
  parent.postMessage(JSON.stringify(out), "*");
})();
<\/script>
</body></html>`;

/** Plant a cookie in the shell, so "the artifact could not read the cookie" is a
 *  reading rather than a vacuous pass on a page that has no cookie at all. */
async function boot(page) {
  await page.goto(BASE_URL);
  await page.waitForFunction(() => typeof window.openArtifact === "function"
                                && typeof window.renderMarkdown === "function");
  await page.evaluate((s) => { document.cookie = "localm_e2e_probe=" + s + "; path=/"; }, SECRET);
  expect(await page.evaluate(() => document.cookie),
    "the shell itself can read the planted cookie").toContain(SECRET);
}

/** Run the probe and return its report. `arm` selects the frame it runs in. */
function runProbe(page, arm) {
  return page.evaluate(async ([probe, which]) => {
    const got = new Promise((resolve) => {
      window.addEventListener("message", (e) => resolve(e.data), { once: true });
    });
    if (which === "artifact") {
      window.openArtifact(probe, "html");
    } else {
      // The control: same markup, same shell CSP, but NO sandbox attribute and
      // none of the artifact's injected CSP. Its inline script still needs the
      // shell nonce, which openArtifact's own path stamps for the other arm.
      const n = window.__LOCALM_CSP_NONCE__ || "";
      const f = document.createElement("iframe");
      f.id = "control-frame";
      f.srcdoc = probe.replace("<script>", '<script nonce="' + n + '">');
      document.body.appendChild(f);
    }
    const timeout = new Promise((r) => setTimeout(() => r(null), 8000));
    return Promise.race([got, timeout]);
  }, [PROBE, arm]);
}

test("the CONTROL frame reaches everything, so the probe can report success", async ({ page }) => {
  await boot(page);
  const raw = await runProbe(page, "control");
  expect(raw, "the control frame reported back at all").not.toBeNull();
  const r = JSON.parse(raw);
  expect(r.ran).toBe(true);
  expect(r.cookie, "an unsandboxed same-origin frame READS the shell cookie").toContain(SECRET);
  expect(r.storage, "and writes the shell's localStorage").toBe("written");
  expect(r.parentDoc, "and reaches the embedding document").toBe("reachable");
  expect(r.parentHref, "and reads the embedder's URL").toContain("127.0.0.1");
  expect(r.api, "and calls localm's own API with the embedder's credentials")
    .toMatch(/^status:2\d\d$/);
  expect(r.apiRead, "and reads the response body back").toBe(true);
});

test("the ARTIFACT frame is contained: no cookie, no storage, no parent, no API",
  async ({ page }) => {
    await boot(page);
    const raw = await runProbe(page, "artifact");
    expect(raw, "the artifact's own script ran and reported").not.toBeNull();
    const r = JSON.parse(raw);
    expect(r.ran, "the probe executed, so the blocks below are real results").toBe(true);

    expect(r.cookie, "the shell cookie must not be readable").not.toContain(SECRET);
    expect(r.storage, "localStorage must not be writable").toMatch(/^THREW:/);
    expect(r.parentDoc, "the embedding document must be unreachable").toMatch(/^THREW:/);
    expect(r.parentHref, "the embedder's URL must be unreadable").toMatch(/^THREW:/);
    expect(r.api, "localm's own API must be unreachable").toMatch(/^THREW:/);
  });

test("a withdrawn preview means the canvas button is not even offered", async ({ page }) => {
  await boot(page);
  const state = await page.evaluate(() => {
    const box = document.createElement("div");
    box.id = "probe-md";
    document.body.appendChild(box);
    window.renderMarkdown(box, "reply\n\n```html\n<b>hi</b>\n```\n");
    const before = !!box.querySelector(".canvas-btn");
    window.setPreviewAllowed(false);
    window.refreshPreviewButtons();
    const after = !!box.querySelector(".canvas-btn");
    return { before, after };
  });
  expect(state.before, "offered while allowed").toBe(true);
  expect(state.after, "withdrawn once the server says no").toBe(false);
});
