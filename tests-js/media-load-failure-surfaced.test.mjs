// SPDX-License-Identifier: AGPL-3.0-or-later
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages } from "./harness.mjs";

// A media element does NOT throw and does NOT reject when its src is refused -
// it fires an `error` EVENT on itself. So a try/catch wrapped around
// `player.src = url` is structurally unable to see the failure.
//
// That is not hypothetical. Found live 2026-08-13: the GUI's own CSP had no
// media-src directive, so every blob: URL fed to a <video> was refused with
// "Media load rejected by URL safety check". The success path still ran to the
// end - the button still flipped to "hide" - and the user got a dead black
// player, no toast, and no clue. A step that failed while reporting success.
//
// The CSP itself is fixed and guarded server-side by
// tests/test_security_headers.py::test_media_and_fetch_of_blob_urls_are_permitted.
// This is the OTHER half: whatever the cause - CSP, a truncated file, a codec
// the browser will not take - a media load that fails must SAY SO.
//
// Asserts on the TOAST TEXT rather than on "no exception was thrown", because
// "no exception" was true both before and after the fix and would pass either
// way.

test("a media element whose src is refused produces a user-visible toast", () => {
  const { window } = loadAppWithPages({});
  const fn = window.reportMediaLoadFailure;
  assert.equal(typeof fn, "function",
    "reportMediaLoadFailure must be reachable from the page scripts");

  const toasts = [];
  const realToast = window.toast;
  window.toast = (msg, isErr) => { toasts.push({ msg: String(msg), isErr }); };
  try {
    const player = window.document.createElement("video");
    fn(player, "the clip");
    // Stand in for what the browser reports on a refused source. jsdom does not
    // implement media loading, so the element's own error path is driven
    // directly - the contract under test is "an error event becomes a toast",
    // not jsdom's media stack.
    Object.defineProperty(player, "error", {
      value: { code: 4, message: "MEDIA_ELEMENT_ERROR: Media load rejected by URL safety check" },
      configurable: true,
    });
    player.dispatchEvent(new window.Event("error"));

    assert.equal(toasts.length, 1, "exactly one toast for one failed load");
    assert.equal(toasts[0].isErr, true, "it is reported as an error, not a notice");
    assert.match(toasts[0].msg, /the clip/,
      "the message names what failed to play");
    assert.match(toasts[0].msg, /refused/i,
      "code 4 is reported as a refused source, not as a generic failure");
  } finally {
    window.toast = realToast;
  }
});

test("the failed player is torn down so the button does not lie about state", () => {
  const { window } = loadAppWithPages({});
  const toasts = [];
  const realToast = window.toast;
  window.toast = (msg) => { toasts.push(String(msg)); };
  try {
    let cleanedUp = false;
    const player = window.document.createElement("audio");
    window.reportMediaLoadFailure(player, "the track", () => { cleanedUp = true; });
    Object.defineProperty(player, "error", {
      value: { code: 2, message: "network" }, configurable: true,
    });
    player.dispatchEvent(new window.Event("error"));

    assert.equal(cleanedUp, true,
      "the caller's cleanup runs, so the play button can go back to 'play' "
      + "instead of sitting on 'hide' over a dead element");
    assert.equal(toasts.length, 1);
    assert.match(toasts[0], /network/i, "code 2 is reported as a network error");
  } finally {
    window.toast = realToast;
  }
});
