// SPDX-License-Identifier: AGPL-3.0-or-later
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages } from "./harness.mjs";

// A media element fires an `error` event on itself when its src is refused; it
// does not throw or reject. reportMediaLoadFailure listens for that event and
// turns it into a toast, then runs the caller's cleanup.

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
    // jsdom does not implement media loading, so the element's error state is
    // set and the error event dispatched directly.
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
