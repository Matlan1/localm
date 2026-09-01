// SPDX-License-Identifier: AGPL-3.0-or-later
// R-NET: the Kokoro voice model is fetched by the BROWSER directly from
// Hugging Face, so localm's server-side net_mode enforcement (netpolicy.py)
// never sees the request - net_mode=off did not stop the download at all.
// These tests cover the client-side gate added to
// localm/plugins/builtin/tts/static/{tts-util.js,tts.js}:
//   planModelFetch()        pure decision (allow / refuse / confirm)
//   requestDownloadConsent() the one-time confirmation dialog itself
//   register()/load()       the gate actually wired into the load path

import { test } from "node:test";
import assert from "node:assert/strict";
import { JSDOM } from "jsdom";
import { fileURLToPath, pathToFileURL } from "node:url";
import { dirname, join } from "node:path";

const HERE = dirname(fileURLToPath(import.meta.url));
const UTIL = join(HERE, "..", "localm", "plugins", "builtin", "tts", "static", "tts-util.js");
const TTS_JS = join(HERE, "..", "localm", "plugins", "builtin", "tts", "static", "tts.js");
const { planModelFetch, NetGateError, shouldWarmPassively } = await import(pathToFileURL(UTIL).href);

// ---- planModelFetch: pure decision ------------------------------------- //

test("planModelFetch: an already-cached model needs no gate, in any mode", () => {
  assert.equal(planModelFetch("off", true), "allow");
  assert.equal(planModelFetch("ask", true), "allow");
  assert.equal(planModelFetch("allow", true), "allow");
  assert.equal(planModelFetch(undefined, true), "allow");
});

test("planModelFetch: net_mode=allow proceeds uncached with no prompt", () => {
  assert.equal(planModelFetch("allow", false), "allow");
});

test("planModelFetch: net_mode=off refuses outright by default (no override)", () => {
  assert.equal(planModelFetch("off", false), "refuse");
  assert.equal(planModelFetch("off", false, false), "refuse");
});

test("planModelFetch: net_mode=off proceeds when allowDownloadsWhenOff is set", () => {
  assert.equal(planModelFetch("off", false, true), "allow");
});

test("planModelFetch: net_mode=ask requires a one-time confirmation", () => {
  assert.equal(planModelFetch("ask", false), "confirm");
});

test("planModelFetch: an unresolved/unrecognised mode fails toward asking, never toward a silent fetch", () => {
  assert.equal(planModelFetch(undefined, false), "confirm");
  assert.equal(planModelFetch("", false), "confirm");
  assert.equal(planModelFetch("bogus", false), "confirm");
});

test("NetGateError is a real Error subclass carrying its own name", () => {
  const e = new NetGateError("refused");
  assert.ok(e instanceof Error);
  assert.equal(e.name, "NetGateError");
  assert.equal(e.message, "refused");
});

// ---- shouldWarmPassively: pure decision --------------------------------- //
test("shouldWarmPassively: an already-cached model may always warm, in any mode", () => {
  assert.equal(shouldWarmPassively(true, "ask"), true);
  assert.equal(shouldWarmPassively(true, "off"), true);
  assert.equal(shouldWarmPassively(true, "allow"), true);
  assert.equal(shouldWarmPassively(true, undefined), true);
});

test("shouldWarmPassively: net_mode=allow may warm uncached, no prompt needed", () => {
  assert.equal(shouldWarmPassively(false, "allow"), true);
});

test("shouldWarmPassively: net_mode=ask/off must NOT warm uncached - that needs a prompt", () => {
  assert.equal(shouldWarmPassively(false, "ask"), false);
  assert.equal(shouldWarmPassively(false, "off"), false);
  assert.equal(shouldWarmPassively(false, undefined), false);
});

// ---- requestDownloadConsent: the confirmation dialog itself ------------ //

// Minimal, faithful stand-ins for app/helpers.js's $ / el / openModal, backed
// by a real jsdom document - not spies, because requestDownloadConsent()
// needs to actually render buttons and have them clicked. Mirrors the real
// implementations exactly: $ = document.getElementById, el = createElement
// (+className/textContent), openModal fills #modal-title/#modal-body and
// sets #modal's display to "flex" (helpers.js's own shape).
function installModalShell() {
  const dom = new JSDOM(
    `<!DOCTYPE html><html><body>
       <div id="modal" style="display:none">
         <div id="modal-title"></div>
         <div id="modal-body"></div>
       </div>
     </body></html>`,
    { url: "http://localhost:8642/" });
  const win = dom.window;
  win.$ = (id) => win.document.getElementById(id);
  win.el = (tag, cls, text) => {
    const n = win.document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined) n.textContent = text;
    return n;
  };
  win.openModal = (title, bodyBuilder) => {
    win.$("modal-title").textContent = title;
    const body = win.$("modal-body");
    body.replaceChildren();
    bodyBuilder(body);
    win.$("modal").style.display = "flex";
  };
  global.window = win;
  global.document = win.document;
  delete global.confirm;                 // force the with-shell path
  // register() unconditionally builds its playback queue's `new Audio()`,
  // which jsdom does not implement - a no-op stub, since these tests never
  // reach speak()/playNext(), which are the only things that touch it.
  win.Audio = class {};
  global.Audio = win.Audio;
  return win;
}

function clickButtonNamed(win, text) {
  const btn = Array.from(win.document.querySelectorAll("#modal-body button"))
    .find((b) => b.textContent === text);
  assert.ok(btn, `no "${text}" button rendered in the modal`);
  btn.onclick();
}

async function importFresh(path) {
  return import(pathToFileURL(path).href + `?t=${Date.now()}_${Math.random()}`);
}

test("requestDownloadConsent: clicking Download resolves true and closes the modal", async () => {
  const win = installModalShell();
  const { requestDownloadConsent } = await importFresh(TTS_JS);
  const p = requestDownloadConsent();
  assert.equal(win.$("modal-title").textContent, "Download voice model?");
  assert.match(win.$("modal-body").textContent, /huggingface\.co/);
  clickButtonNamed(win, "Download");
  assert.equal(await p, true);
  assert.equal(win.$("modal").style.display, "none");
});

test("requestDownloadConsent: clicking Not now resolves false", async () => {
  const win = installModalShell();
  const { requestDownloadConsent } = await importFresh(TTS_JS);
  const p = requestDownloadConsent();
  clickButtonNamed(win, "Not now");
  assert.equal(await p, false);
});

test("requestDownloadConsent: dismissing via the shared modal chrome (backdrop/x) resolves false", async () => {
  const win = installModalShell();
  const { requestDownloadConsent } = await importFresh(TTS_JS);
  const p = requestDownloadConsent();
  // Neither button was clicked - simulate the shared chrome's own dismiss,
  // which only ever sets display:none (see helpers.js's modal-close wiring).
  win.$("modal").style.display = "none";
  assert.equal(await p, false);
});

test("requestDownloadConsent: no GUI shell falls back to native confirm(), matching jobs.js's own fallback", async () => {
  delete global.window;
  delete global.document;
  global.confirm = (msg) => { global.confirm.lastMessage = msg; return true; };
  const { requestDownloadConsent } = await importFresh(TTS_JS);
  assert.equal(await requestDownloadConsent(), true);
  assert.match(global.confirm.lastMessage, /huggingface\.co/);
  delete global.confirm;
});

// ---- load()/register(): the gate actually wired in ---------------------- //

function makeCtx() {
  const calls = { toasts: [], registerTTS: [] };
  const ctx = {
    authHeaders: () => ({}),
    toast: (msg, isError) => calls.toasts.push({ msg, isError }),
    registerTTS: (provider) => calls.registerTTS.push(provider),
  };
  return { ctx, calls };
}

// register() itself fetches /api/tts/config and vendor/voices.json; load()
// (triggered by provider.ready()) re-fetches /api/tts/config for a fresh
// net_mode read. jsdom defines no global `caches`, so modelCached() resolves
// false in every case here - exactly the "never downloaded before" state
// these tests target.
function installFetchEnv(win, { netMode, allowDownloadsWhenOff = false }) {
  win.fetch = async (url) => {
    const u = String(url);
    if (u.includes("/api/tts/config")) {
      return { ok: true, status: 200, json: async () => (
        { net_mode: netMode, net_allow_model_downloads: allowDownloadsWhenOff }) };
    }
    if (u.includes("voices.json")) {
      return { ok: false, status: 404, json: async () => ({}) };
    }
    return { ok: false, status: 404, json: async () => ({}) };
  };
  global.fetch = win.fetch;
}

test("load(): net_mode=off refuses without ever offering the dialog", async () => {
  const win = installModalShell();
  installFetchEnv(win, { netMode: "off" });
  const { ctx, calls } = makeCtx();
  const { register } = await importFresh(TTS_JS);
  await register(ctx);

  const provider = calls.registerTTS[0];
  assert.ok(provider, "register() must call ctx.registerTTS with a provider");
  await assert.rejects(provider.ready());

  assert.equal(win.$("modal").style.display, "none",
    "off (with no override) has no bypass - no confirmation dialog is offered");
  const errToast = calls.toasts.find((t) => t.isError);
  assert.ok(errToast, "a toast must explain the refusal");
  assert.match(errToast.msg, /off/i);
  // A browser has no CLI to run: the message must point at Settings, never
  // at a terminal command it cannot act on.
  assert.doesNotMatch(errToast.msg, /localm config/);
  assert.match(errToast.msg, /Settings/);
  assert.equal(calls.registerTTS.at(-1), null,
    "a refused load must revert to the browser voice fallback");
});

test("load(): net_mode=off with net_allow_model_downloads never refuses at the net_mode gate", async () => {
  // The override makes planModelFetch return "allow" (same as net_mode=allow),
  // so load() proceeds straight to importing the real Kokoro vendor bundle -
  // which jsdom cannot execute (no `self` global; a real jsdom limitation,
  // not a gate defect). This test stops at the boundary planModelFetch's own
  // direct unit tests already cover in full: that the gate itself does not
  // refuse. It does not attempt the vendor import.
  const win = installModalShell();
  installFetchEnv(win, { netMode: "off", allowDownloadsWhenOff: true });
  const { ctx, calls } = makeCtx();
  const { register } = await importFresh(TTS_JS);
  await register(ctx);

  const provider = calls.registerTTS[0];
  // Swallow whatever load() rejects with past the gate (the jsdom-incapable
  // vendor import) - only the toast, if any, is inspected below.
  await provider.ready().catch(() => {});
  assert.equal(win.$("modal").style.display, "none",
    "an exempted off proceeds like allow - no confirmation needed");
  const gateToast = calls.toasts.find(
    (t) => t.isError && /Voice model download is off/.test(t.msg));
  assert.equal(gateToast, undefined,
    "must not fail on the net_mode gate when the override is set");
});

test("load(): net_mode=ask offers the dialog; declining reverts to the browser voice with an actionable toast", async () => {
  const win = installModalShell();
  installFetchEnv(win, { netMode: "ask" });
  const { ctx, calls } = makeCtx();
  const { register } = await importFresh(TTS_JS);
  await register(ctx);

  const provider = calls.registerTTS[0];
  const readyPromise = provider.ready();
  // Let the gate's own fetch + modelCached microtasks settle before the
  // dialog is expected to be open.
  await new Promise((r) => setTimeout(r, 0));
  assert.equal(win.$("modal-title").textContent, "Download voice model?",
    "net_mode=ask must offer the one-time confirmation before fetching");

  clickButtonNamed(win, "Not now");
  await assert.rejects(readyPromise);

  const errToast = calls.toasts.find((t) => t.isError);
  assert.ok(errToast);
  assert.match(errToast.msg, /net_mode=ask/);
  assert.match(errToast.msg, /not granted/);
  assert.equal(calls.registerTTS.at(-1), null);
});

// ---- ready({ passive: true }): the proactive warm-up ---------------------- //
// jsdom defines no global `caches`, so modelCached() resolves false here in
// every case - exactly the "never downloaded before" state that makes the
// net_mode branch the only thing under test.

test("ready({passive:true}): net_mode=ask and uncached never opens the dialog, never fetches", async () => {
  const win = installModalShell();
  installFetchEnv(win, { netMode: "ask" });
  const { ctx, calls } = makeCtx();
  const { register } = await importFresh(TTS_JS);
  await register(ctx);

  const provider = calls.registerTTS[0];
  const result = await provider.ready({ passive: true });
  assert.equal(result, null, "declined silently - not an error the caller must catch");
  assert.equal(win.$("modal").style.display, "none",
    "a passive warm-up must never surprise the user with a consent prompt");
  assert.equal(calls.toasts.length, 0, "a declined passive warm-up is not a failure");
  assert.equal(calls.registerTTS.length, 1,
    "must not fall back to the browser voice - the Kokoro provider is still fine");
});

test("ready({passive:true}): net_mode=off and uncached never opens the dialog, never fetches", async () => {
  const win = installModalShell();
  installFetchEnv(win, { netMode: "off" });
  const { ctx, calls } = makeCtx();
  const { register } = await importFresh(TTS_JS);
  await register(ctx);

  const provider = calls.registerTTS[0];
  const result = await provider.ready({ passive: true });
  assert.equal(result, null);
  assert.equal(win.$("modal").style.display, "none");
  assert.equal(calls.toasts.length, 0);
});

test("ready({passive:true}): net_mode=allow and uncached proceeds straight to a real load attempt", async () => {
  const win = installModalShell();
  installFetchEnv(win, { netMode: "allow" });
  const { ctx, calls } = makeCtx();
  const { register } = await importFresh(TTS_JS);
  await register(ctx);

  const provider = calls.registerTTS[0];
  // Same boundary as the net_mode=off+allowDownloadsWhenOff test above: proceeding
  // means attempting the real vendor import, which jsdom cannot execute. Reaching
  // that failure (rather than a gate refusal) is exactly what proves it proceeded.
  await provider.ready({ passive: true }).catch(() => {});
  assert.equal(win.$("modal").style.display, "none",
    "allow never needs a prompt, passive or not");
});

test("ready({passive:true}): a real click already loading is reused, not raced", async () => {
  const win = installModalShell();
  installFetchEnv(win, { netMode: "ask" });
  const { ctx, calls } = makeCtx();
  const { register } = await importFresh(TTS_JS);
  await register(ctx);

  const provider = calls.registerTTS[0];
  const clickPromise = provider.ready();               // a real click: net_mode=ask offers the dialog
  await new Promise((r) => setTimeout(r, 0));
  assert.equal(win.$("modal-title").textContent, "Download voice model?");

  // A passive warm-up firing while that real load is in flight must not open a
  // second dialog or start a second fetch - it just rides the same promise.
  const passivePromise = provider.ready({ passive: true });
  clickButtonNamed(win, "Not now");
  await assert.rejects(clickPromise);
  await assert.rejects(passivePromise);
});
