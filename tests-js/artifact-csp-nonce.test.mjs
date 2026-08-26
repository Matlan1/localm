// SPDX-License-Identifier: AGPL-3.0-or-later
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp } from "./harness.mjs";

const NONCE = "test-nonce-abc123";

function withNonce(win, value) {
  win.__LOCALM_CSP_NONCE__ = value;
}

test("artifact scripts are stamped with the shell nonce in all three shapes", () => {
  const { window: win } = loadApp();
  withNonce(win, NONCE);

  // fragment
  const frag = win.artifactSrcdoc("<button>go</button><script>x=1</script>", "html");
  assert.match(frag, new RegExp(`<script nonce="${NONCE}"`),
    "a fragment artifact's script must carry the nonce");

  // full document
  const full = win.artifactSrcdoc(
    "<!doctype html><html><head><title>t</title></head>"
    + "<body><script>x=1</script></body></html>", "html");
  assert.match(full, new RegExp(`<script nonce="${NONCE}"`),
    "a full-document artifact's script must carry the nonce");
  // our CSP precedes the artifact's own head
  assert.ok(full.indexOf("Content-Security-Policy") < full.indexOf("</head>"));

  // svg
  const svg = win.artifactSrcdoc("<svg><script>x=1</script></svg>", "svg");
  assert.match(svg, new RegExp(`<script nonce="${NONCE}"`),
    "an svg artifact's script must carry the nonce");
});

test("stamping covers EVERY script, not just the first", () => {
  const { window: win } = loadApp();
  withNonce(win, NONCE);
  const out = win.artifactSrcdoc(
    "<script>a=1</script><p>x</p><script type='module'>b=2</script>", "html");
  const stamped = out.match(new RegExp(`<script nonce="${NONCE}"`, "g")) || [];
  assert.equal(stamped.length, 2,
    "both scripts must be stamped; one uncovered script is a dead artifact");
  // existing attributes survive the rewrite
  assert.match(out, /type='module'/);
});

test("an existing nonce is not double-stamped", () => {
  const { window: win } = loadApp();
  withNonce(win, NONCE);
  const out = win.artifactSrcdoc('<script nonce="already">a=1</script>', "html");
  assert.equal((out.match(/nonce=/g) || []).length, 1, out);
  assert.match(out, /nonce="already"/);
});

test("no shell nonce means the content is passed through untouched", () => {
  const { window: win } = loadApp();
  withNonce(win, "");
  const out = win.artifactSrcdoc("<script>a=1</script>", "html");
  assert.ok(!/nonce=/.test(out), out);
  assert.match(out, /<script>a=1<\/script>/);
});

test("the unsubstituted placeholder is never treated as a real nonce", () => {
  // index.html ships the literal placeholder; only the server replaces it.
  const { window: win } = loadApp();
  assert.ok(!win.__LOCALM_CSP_NONCE__,
    "an unsubstituted shell must expose no nonce, got "
    + JSON.stringify(win.__LOCALM_CSP_NONCE__));
});
