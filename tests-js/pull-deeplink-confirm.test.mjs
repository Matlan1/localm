// SPDX-License-Identifier: AGPL-3.0-or-later
// A `?pull=<spec>` deep link redeems its `pull_token` via
// POST /api/models/pull-token/redeem and otherwise asks in the in-page dialog,
// never the browser's own confirm().
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

function makeFetch(calls, { redeemOk = true } = {}) {
  return async (url, opts = {}) => {
    if (url === "/api/models/pull-token/redeem") {
      calls.push({ url, body: opts.body ? JSON.parse(opts.body) : null });
      return redeemOk
        ? { ok: true, status: 200, json: async () => ({ ok: true }), text: async () => "" }
        : { ok: false, status: 403, json: async () => ({ detail: "bad token" }), text: async () => "" };
    }
    if (url === "/api/models/pull") {
      calls.push({ url });
      return { ok: true, status: 200, json: async () => ({ job_id: "j1" }), text: async () => "" };
    }
    return {
      ok: true, status: 200, text: async () => "",
      json: async () => ({ models: [], active: "", conversations: [], plugins: [] }),
    };
  };
}

const SPEC = "owner/repo:m.gguf";
const deepLinkUrl = (extra = "") =>
  `http://localhost:8642/?view=models&pull=${encodeURIComponent(SPEC)}${extra}`;

const drain = async (n = 4) => { for (let i = 0; i < n; i++) await new Promise((r) => setTimeout(r, 0)); };
const modalOpen = (w) => w.document.getElementById("modal").style.display !== "none";
const modalText = (w) => w.document.getElementById("modal-body").textContent;
const answer = (w, accept) => w.document.querySelector(
  `#modal-body .actions button${accept ? ".btn-danger" : ":not(.btn-danger)"}`).click();

function boot(calls, { redeemOk, extra = "" } = {}) {
  const { window: win } = loadAppWithPages({
    fetchImpl: makeFetch(calls, redeemOk === undefined ? {} : { redeemOk }),
    url: deepLinkUrl(extra),
  });
  runScript(win, `
    window.confirm = () => { globalThis.__nativeConfirm = true; return true; };
    streamJob = async () => ({ status: "done" });
  `);
  return win;
}

// --------------------------------------------------------------------------- //
//  No token (or an unrecognised link): the in-page confirmation               //
// --------------------------------------------------------------------------- //

test("no pull_token: declining the in-page dialog does NOT start the pull", async () => {
  const calls = [];
  const win = boot(calls);
  await drain();
  assert.ok(modalOpen(win), "the in-page dialog asks first");
  answer(win, false);
  await drain();

  assert.deepEqual(calls, [], "no /api/models/pull request was made without explicit confirmation");
  assert.equal(win.document.getElementById("pull-spec").value, SPEC,
    "the spec is still pre-filled so the user can start it manually");
  assert.equal(win.__nativeConfirm, undefined, "the browser's own confirm() is never used");
});

test("no pull_token: accepting the in-page dialog starts the pull", async () => {
  const calls = [];
  const win = boot(calls);
  await drain();
  assert.deepEqual(calls, [], "nothing starts before the answer");
  answer(win, true);
  await drain();

  assert.deepEqual(calls.map((c) => c.url), ["/api/models/pull"],
    "confirming the dialog kicks off the pull");
  assert.equal(win.__nativeConfirm, undefined, "the browser's own confirm() is never used");
});

test("no pull_token: the dialog names the spec so the user can verify it", async () => {
  const win = boot([]);
  await drain();
  assert.ok(modalOpen(win), "the dialog is shown");
  assert.match(modalText(win), /owner\/repo:m\.gguf/, "the dialog shows the exact spec being pulled");
});

// --------------------------------------------------------------------------- //
//  A valid pull_token: zero-click auto-start                                  //
// --------------------------------------------------------------------------- //

test("a redeemed pull_token starts the pull WITHOUT asking", async () => {
  const calls = [];
  const win = boot(calls, { redeemOk: true, extra: "&pull_token=real-cli-minted-token" });
  await drain();

  assert.ok(!modalOpen(win) || !/owner\/repo:m\.gguf/.test(modalText(win)),
    "no confirmation is shown for a token-authorised deep link");
  assert.deepEqual(calls.map((c) => c.url),
    ["/api/models/pull-token/redeem", "/api/models/pull"],
    "the token is redeemed server-side, then the pull starts automatically");
  assert.deepEqual(calls[0].body, { spec: SPEC, token: "real-cli-minted-token" },
    "the redeem call carries the exact spec and token from the URL");
});

// --------------------------------------------------------------------------- //
//  A forged/expired/already-used pull_token: falls back to the dialog         //
// --------------------------------------------------------------------------- //

test("an invalid pull_token falls back to the in-page dialog, not a silent start", async () => {
  const calls = [];
  const win = boot(calls, { redeemOk: false, extra: "&pull_token=forged-or-expired" });
  await drain();
  assert.ok(modalOpen(win), "the in-page dialog asks");
  answer(win, false);
  await drain();

  assert.deepEqual(calls.map((c) => c.url), ["/api/models/pull-token/redeem"],
    "the redeem attempt failed and no pull was ever started");
});

test("an invalid pull_token still lets the user proceed by accepting the dialog", async () => {
  const calls = [];
  const win = boot(calls, { redeemOk: false, extra: "&pull_token=forged-or-expired" });
  await drain();
  answer(win, true);
  await drain();

  assert.deepEqual(calls.map((c) => c.url),
    ["/api/models/pull-token/redeem", "/api/models/pull"],
    "the user's explicit confirmation still starts the pull after a failed redeem");
  assert.equal(win.__nativeConfirm, undefined, "the browser's own confirm() is never used");
});
