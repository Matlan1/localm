// SPDX-License-Identifier: AGPL-3.0-or-later
// A `?pull=<spec>` deep link redeems its `pull_token` via
// POST /api/models/pull-token/redeem and otherwise falls back to confirm().
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

// --------------------------------------------------------------------------- //
//  No token (or an unrecognised link): the confirm() gate                     //
// --------------------------------------------------------------------------- //

test("no pull_token: declining the confirm() dialog does NOT start the pull", async () => {
  const calls = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(calls), url: deepLinkUrl() });
  runScript(win, `
    window.confirm = () => false;
    streamJob = async () => ({ status: "done" });
  `);
  await new Promise((r) => setTimeout(r, 0));
  await new Promise((r) => setTimeout(r, 0));

  assert.deepEqual(calls, [], "no /api/models/pull request was made without explicit confirmation");
  assert.equal(win.document.getElementById("pull-spec").value, SPEC,
    "the spec is still pre-filled so the user can start it manually");
});

test("no pull_token: accepting the confirm() dialog starts the pull", async () => {
  const calls = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(calls), url: deepLinkUrl() });
  runScript(win, `
    window.confirm = () => true;
    streamJob = async () => ({ status: "done" });
  `);
  await new Promise((r) => setTimeout(r, 0));
  await new Promise((r) => setTimeout(r, 0));

  assert.deepEqual(calls.map((c) => c.url), ["/api/models/pull"],
    "confirming the dialog kicks off the pull");
});

test("no pull_token: the confirm() prompt names the spec so the user can verify it", async () => {
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch([]), url: deepLinkUrl() });
  runScript(win, `
    window.confirm = (msg) => { globalThis.__msg = msg; return false; };
    streamJob = async () => ({ status: "done" });
  `);
  await new Promise((r) => setTimeout(r, 0));
  await new Promise((r) => setTimeout(r, 0));

  assert.ok(win.__msg, "confirm() was called");
  assert.match(win.__msg, /owner\/repo:m\.gguf/, "the prompt shows the exact spec being pulled");
});

// --------------------------------------------------------------------------- //
//  A valid pull_token: zero-click auto-start                                  //
// --------------------------------------------------------------------------- //

test("a redeemed pull_token starts the pull WITHOUT showing confirm()", async () => {
  const calls = [];
  const { window: win } = loadAppWithPages({
    fetchImpl: makeFetch(calls, { redeemOk: true }),
    url: deepLinkUrl("&pull_token=real-cli-minted-token"),
  });
  runScript(win, `
    window.confirm = () => { globalThis.__confirmCalled = true; return false; };
    streamJob = async () => ({ status: "done" });
  `);
  await new Promise((r) => setTimeout(r, 0));
  await new Promise((r) => setTimeout(r, 0));

  assert.equal(win.__confirmCalled, undefined,
    "confirm() must never be shown for a token-authorised deep link");
  assert.deepEqual(calls.map((c) => c.url),
    ["/api/models/pull-token/redeem", "/api/models/pull"],
    "the token is redeemed server-side, then the pull starts automatically");
  assert.deepEqual(calls[0].body, { spec: SPEC, token: "real-cli-minted-token" },
    "the redeem call carries the exact spec and token from the URL");
});

// --------------------------------------------------------------------------- //
//  A forged/expired/already-used pull_token: falls back to confirm()          //
// --------------------------------------------------------------------------- //

test("an invalid pull_token falls back to the confirm() gate, not a silent start", async () => {
  const calls = [];
  const { window: win } = loadAppWithPages({
    fetchImpl: makeFetch(calls, { redeemOk: false }),
    url: deepLinkUrl("&pull_token=forged-or-expired"),
  });
  runScript(win, `
    window.confirm = () => false;
    streamJob = async () => ({ status: "done" });
  `);
  await new Promise((r) => setTimeout(r, 0));
  await new Promise((r) => setTimeout(r, 0));

  assert.deepEqual(calls.map((c) => c.url), ["/api/models/pull-token/redeem"],
    "the redeem attempt failed and no pull was ever started");
});

test("an invalid pull_token still lets the user proceed by accepting confirm()", async () => {
  const calls = [];
  const { window: win } = loadAppWithPages({
    fetchImpl: makeFetch(calls, { redeemOk: false }),
    url: deepLinkUrl("&pull_token=forged-or-expired"),
  });
  runScript(win, `
    window.confirm = () => true;
    streamJob = async () => ({ status: "done" });
  `);
  await new Promise((r) => setTimeout(r, 0));
  await new Promise((r) => setTimeout(r, 0));

  assert.deepEqual(calls.map((c) => c.url),
    ["/api/models/pull-token/redeem", "/api/models/pull"],
    "the user's explicit confirmation still starts the pull after a failed redeem");
});
