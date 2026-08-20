// SPDX-License-Identifier: AGPL-3.0-or-later
// The Settings "Inference runtime" block: the GUI's whole form of
// `localm setup-llama`. Check surfaces what is provisioned; the action button
// installs, switches backend, or moves to a named build, and says which of
// those it is about to do before you press it.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

function makeFetch(routes) {
  return async (url, opts = {}) => {
    const u = String(url);
    for (const [frag, resp] of routes) {
      if (u.includes(frag)) {
        const body = typeof resp === "function" ? resp(opts) : resp;
        return { ok: true, status: 200, text: async () => "", json: async () => body };
      }
    }
    return { ok: true, status: 200, text: async () => "",
      json: async () => ({ models: [], active: "", conversations: [], plugins: [] }) };
  };
}

async function flush() {
  await new Promise((r) => setTimeout(r, 0));
  await new Promise((r) => setTimeout(r, 0));
}

const INSTALLED_VULKAN = { installed: true, backend: "vulkan", current: "b10300",
  target: "b10361", newer: true, pinned: null, previous: null };
const NOTHING_INSTALLED = { installed: false, backend: null, current: null,
  target: null, newer: false, pinned: null, previous: null };

/** Stub the job stream so a provision completes without a real subprocess.
 *  `status` is what the job ends with. */
function stubJobStream(window, status = "done", lines = ["Fetching release b10361 ..."]) {
  runScript(window, `streamJob = (id, onLine) => {
    ${JSON.stringify(lines)}.forEach(onLine);
    return Promise.resolve({ status: ${JSON.stringify(status)} });
  };`);
}

/** Stub the in-page confirm. Records the call and (by default) accepts, so a
 *  test can assert BOTH that a confirm was demanded and what happened after. */
function stubConfirm(window, { accept = true } = {}) {
  window.__confirmCall = null;
  runScript(window, `confirmDanger = (title, message, label, onConfirm) => {
    window.__confirmCall = { title, message, label };
    if (${accept ? "true" : "false"}) onConfirm();
  };`);
}

function pick(doc, window, backend) {
  const sel = doc.getElementById("runtime-backend");
  sel.value = backend;
  sel.dispatchEvent(new window.Event("change"));
}

function typeTag(doc, window, tag) {
  const el = doc.getElementById("runtime-tag");
  el.value = tag;
  el.dispatchEvent(new window.Event("input"));
}

// --------------------------------------------------------------------------- //
//  Check                                                                       //
// --------------------------------------------------------------------------- //

test("Runtime: check surfaces a different build and reveals 'Update runtime'", async () => {
  const { window } = loadAppWithPages({ fetchImpl: makeFetch([
    ["/api/runtime/check", INSTALLED_VULKAN],
  ]) });
  const doc = window.document;
  doc.getElementById("runtime-update-check").click();
  await flush();
  assert.match(doc.getElementById("runtime-update-status").textContent, /b10361/);
  const btn = doc.getElementById("runtime-update-apply");
  assert.equal(btn.hidden, false);
  assert.equal(btn.textContent, "Update runtime");
});

test("Runtime: up to date hides the button", async () => {
  const { window } = loadAppWithPages({ fetchImpl: makeFetch([
    ["/api/runtime/check", { ...INSTALLED_VULKAN, current: "b10361", newer: false }],
  ]) });
  const doc = window.document;
  doc.getElementById("runtime-update-check").click();
  await flush();
  assert.match(doc.getElementById("runtime-update-status").textContent, /Up to date/);
  assert.equal(doc.getElementById("runtime-update-apply").hidden, true);
});

test("Runtime: a pin is reported on BOTH the up-to-date and the newer reading", async () => {
  // The pin suffix is the one branch of this status line no other test covers,
  // and it is shared by both readings - a user who pinned a build away from a
  // broken release must be able to see that is why they are being offered what
  // they are being offered.
  for (const [payload, want] of [
    [{ ...INSTALLED_VULKAN, pinned: "b10300" }, /Pinned to b10300\./],
    [{ ...INSTALLED_VULKAN, current: "b10361", newer: false, pinned: "b10361" }, /Pinned to b10361\./],
  ]) {
    const { window } = loadAppWithPages({ fetchImpl: makeFetch([
      ["/api/runtime/check", payload],
    ]) });
    const doc = window.document;
    doc.getElementById("runtime-update-check").click();
    await flush();
    assert.match(doc.getElementById("runtime-update-status").textContent, want);
  }
});

test("Runtime: nothing installed INVITES an install rather than sending the user to a terminal", async () => {
  // This is the behaviour the whole unit exists for. It used to read "run setup
  // first" and hide the button, because the route refused to provision anything
  // on a box with no runtime - which is exactly the box that needs it, and the
  // one whose user has no terminal open.
  const { window } = loadAppWithPages({ fetchImpl: makeFetch([
    ["/api/runtime/check", NOTHING_INSTALLED],
  ]) });
  const doc = window.document;
  doc.getElementById("runtime-update-check").click();
  await flush();
  const status = doc.getElementById("runtime-update-status").textContent;
  assert.match(status, /No llama\.cpp runtime is installed yet/);
  assert.doesNotMatch(status, /run setup first/);
  const btn = doc.getElementById("runtime-update-apply");
  assert.equal(btn.hidden, false, "the install action must be offered");
  assert.equal(btn.textContent, "Install runtime");
});

test("Runtime: opening Settings reports the state without pressing 'Check'", async () => {
  // A user with NO runtime should not have to press a button labelled "check
  // for an UPDATE" to find out they can install one. dispatch.js drives this.
  const { window } = loadAppWithPages({ fetchImpl: makeFetch([
    ["/api/runtime/check", NOTHING_INSTALLED],
  ]) });
  const doc = window.document;
  assert.equal(doc.getElementById("runtime-update-status").hidden, true,
    "nothing is claimed before the page is opened");
  window.onViewShown("settings");
  await flush();
  assert.match(doc.getElementById("runtime-update-status").textContent,
    /No llama\.cpp runtime is installed yet/);
});

// --------------------------------------------------------------------------- //
//  The button says what it will do                                             //
// --------------------------------------------------------------------------- //

test("Runtime: the label follows the selection, with no round trip", async () => {
  const { window } = loadAppWithPages({ fetchImpl: makeFetch([
    ["/api/runtime/check", INSTALLED_VULKAN],
  ]) });
  const doc = window.document;
  doc.getElementById("runtime-update-check").click();
  await flush();
  const btn = doc.getElementById("runtime-update-apply");

  pick(doc, window, "cuda");
  assert.equal(btn.textContent, "Switch to cuda");
  assert.equal(btn.hidden, false);

  pick(doc, window, "vulkan");   // the one already installed
  assert.equal(btn.textContent, "Reinstall vulkan",
    "re-provisioning the SAME backend is a repair, not a switch");

  pick(doc, window, "auto");
  assert.equal(btn.textContent, "Re-detect and reinstall");

  pick(doc, window, "");         // back to "Keep the installed backend"
  assert.equal(btn.textContent, "Update runtime");

  typeTag(doc, window, "b10355");
  assert.equal(btn.textContent, "Install build b10355");
  assert.equal(btn.hidden, false);
});

test("Runtime: an explicit choice stays actionable even when the CHECK failed", async () => {
  // The recoverability property. A box whose runtime is broken or absent is
  // exactly the one whose check cannot answer - if a failed read hid the
  // button, the only route to a repair would vanish precisely when needed.
  const { window } = loadAppWithPages({ fetchImpl: async (url) => {
    if (String(url).includes("/api/runtime/check")) {
      return { ok: false, status: 500, statusText: "Internal Server Error",
        json: async () => ({ detail: "probe exploded" }) };
    }
    return makeFetch([])(url);
  } });
  const doc = window.document;
  doc.getElementById("runtime-update-check").click();
  await flush();
  assert.match(doc.getElementById("runtime-update-status").textContent, /probe exploded/);
  const btn = doc.getElementById("runtime-update-apply");
  assert.equal(btn.hidden, true, "with no selection there is nothing to promise");

  pick(doc, window, "vulkan");
  assert.equal(btn.hidden, false, "an explicit pick is still actionable");
  assert.equal(btn.textContent, "Install vulkan",
    "and it must not claim to 'switch', which is not knowable here");
});

// --------------------------------------------------------------------------- //
//  Provisioning                                                                //
// --------------------------------------------------------------------------- //

test("Runtime: an untouched card still POSTs the plain re-provision it always did", async () => {
  const bodies = [];
  const { window } = loadAppWithPages({ fetchImpl: makeFetch([
    ["/api/runtime/check", INSTALLED_VULKAN],
    ["/api/runtime/update", (opts) => { bodies.push(opts.body); return { job_id: "j1" }; }],
  ]) });
  stubJobStream(window);
  const doc = window.document;
  doc.getElementById("runtime-update-check").click();
  await flush();
  doc.getElementById("runtime-update-apply").click();
  await flush();
  assert.deepEqual(JSON.parse(bodies[0]), {},
    "no backend, no tag: the body must not invent a request the user did not make");
  assert.match(doc.getElementById("runtime-update-log").textContent, /Fetching release/);
});

test("Runtime: a first install POSTs the chosen backend and does NOT confirm", async () => {
  const bodies = [];
  const { window } = loadAppWithPages({ fetchImpl: makeFetch([
    ["/api/runtime/check", NOTHING_INSTALLED],
    ["/api/runtime/update", (opts) => { bodies.push(opts.body); return { job_id: "j1" }; }],
  ]) });
  stubJobStream(window);
  stubConfirm(window);
  const doc = window.document;
  doc.getElementById("runtime-update-check").click();
  await flush();
  pick(doc, window, "cuda");
  doc.getElementById("runtime-update-apply").click();
  await flush();
  assert.deepEqual(JSON.parse(bodies[0]), { backend: "cuda" });
  assert.equal(window.__confirmCall, null,
    "there is nothing to lose on a box with no runtime, so nothing to confirm");
});

test("Runtime: switching backend on a PROVISIONED box confirms first", async () => {
  const bodies = [];
  const { window } = loadAppWithPages({ fetchImpl: makeFetch([
    ["/api/runtime/check", INSTALLED_VULKAN],
    ["/api/runtime/update", (opts) => { bodies.push(opts.body); return { job_id: "j1" }; }],
  ]) });
  stubJobStream(window);
  stubConfirm(window);
  const doc = window.document;
  doc.getElementById("runtime-update-check").click();
  await flush();
  pick(doc, window, "cuda");
  doc.getElementById("runtime-update-apply").click();
  await flush();
  assert.ok(window.__confirmCall, "a switch replaces a working runtime - confirm it");
  assert.match(window.__confirmCall.message, /vulkan/);
  assert.match(window.__confirmCall.message, /cuda/);
  assert.deepEqual(JSON.parse(bodies[0]), { backend: "cuda" });
});

test("Runtime: declining the switch confirm sends nothing", async () => {
  const bodies = [];
  const { window } = loadAppWithPages({ fetchImpl: makeFetch([
    ["/api/runtime/check", INSTALLED_VULKAN],
    ["/api/runtime/update", (opts) => { bodies.push(opts.body); return { job_id: "j1" }; }],
  ]) });
  stubJobStream(window);
  stubConfirm(window, { accept: false });
  const doc = window.document;
  doc.getElementById("runtime-update-check").click();
  await flush();
  pick(doc, window, "cuda");
  doc.getElementById("runtime-update-apply").click();
  await flush();
  assert.ok(window.__confirmCall);
  assert.deepEqual(bodies, [], "cancelling must not provision anything");
});

test("Runtime: a same-backend reinstall does not confirm", async () => {
  const bodies = [];
  const { window } = loadAppWithPages({ fetchImpl: makeFetch([
    ["/api/runtime/check", INSTALLED_VULKAN],
    ["/api/runtime/update", (opts) => { bodies.push(opts.body); return { job_id: "j1" }; }],
  ]) });
  stubJobStream(window);
  stubConfirm(window);
  const doc = window.document;
  doc.getElementById("runtime-update-check").click();
  await flush();
  pick(doc, window, "vulkan");
  doc.getElementById("runtime-update-apply").click();
  await flush();
  assert.equal(window.__confirmCall, null,
    "this is what the button has always done - do not start nagging about it");
  assert.deepEqual(JSON.parse(bodies[0]), { backend: "vulkan" });
});

test("Runtime: a build tag is sent, and the selection is cleared once carried out", async () => {
  const bodies = [];
  const { window } = loadAppWithPages({ fetchImpl: makeFetch([
    ["/api/runtime/check", INSTALLED_VULKAN],
    ["/api/runtime/update", (opts) => { bodies.push(opts.body); return { job_id: "j1" }; }],
  ]) });
  stubJobStream(window);
  const doc = window.document;
  doc.getElementById("runtime-update-check").click();
  await flush();
  typeTag(doc, window, " b10355 ");
  doc.getElementById("runtime-update-apply").click();
  await flush();
  assert.deepEqual(JSON.parse(bodies[0]), { tag: "b10355" }, "trimmed, as the route expects");
  assert.equal(doc.getElementById("runtime-tag").value, "",
    "a request that has been carried out must stop being offered");
  assert.equal(doc.getElementById("runtime-backend").value, "");
});

test("Runtime: a failed provision shows the job's own reason and keeps the retry", async () => {
  const { window } = loadAppWithPages({ fetchImpl: makeFetch([
    ["/api/runtime/check", INSTALLED_VULKAN],
    ["/api/runtime/update", { job_id: "j2" }],
  ]) });
  stubJobStream(window, "failed",
    ["Cannot replace the installed runtime: it is in use."]);
  const doc = window.document;
  doc.getElementById("runtime-update-check").click();
  await flush();
  doc.getElementById("runtime-update-apply").click();
  await flush();
  assert.match(doc.getElementById("runtime-update-status").textContent, /it is in use/);
  const btn = doc.getElementById("runtime-update-apply");
  assert.equal(btn.hidden, false, "the user has to be able to retry");
  assert.equal(btn.disabled, false);
});

test("Runtime: a 400 from the route is reported, and no job is streamed", async () => {
  // The route validates backend and tag against setup_llama's own definitions,
  // so a refusal arrives BEFORE a job exists. Nothing may claim otherwise.
  let streamed = 0;
  const { window } = loadAppWithPages({ fetchImpl: async (url, opts = {}) => {
    if (String(url).includes("/api/runtime/update")) {
      return { ok: false, status: 400, statusText: "Bad Request",
        json: async () => ({ detail: "Unknown backend 'rocm'. One of: auto, vulkan, ..." }) };
    }
    return makeFetch([["/api/runtime/check", INSTALLED_VULKAN]])(url, opts);
  } });
  runScript(window, "streamJob = () => { window.__streamed = (window.__streamed || 0) + 1; " +
    "return Promise.resolve({ status: 'done' }); };");
  const doc = window.document;
  doc.getElementById("runtime-update-check").click();
  await flush();
  doc.getElementById("runtime-update-apply").click();
  await flush();
  assert.match(doc.getElementById("runtime-update-status").textContent, /Unknown backend/);
  streamed = window.__streamed || 0;
  assert.equal(streamed, 0, "a 400 means no job was created - do not stream one");
  assert.equal(doc.getElementById("runtime-update-log").style.display, "none");
});

test("Runtime: a 409 (already running) is reported without a job stream", async () => {
  const { window } = loadAppWithPages({ fetchImpl: makeFetch([]) });
  window.fetch = async (url, opts = {}) => {
    if (String(url).includes("/api/runtime/update")) {
      return { ok: false, status: 409, statusText: "Conflict",
        json: async () => ({ detail: "A runtime update is already running." }) };
    }
    return makeFetch([])(url, opts);
  };
  const doc = window.document;
  doc.getElementById("runtime-update-apply").click();
  await flush();
  assert.match(doc.getElementById("runtime-update-status").textContent,
    /A runtime update is already running/);
});

// --------------------------------------------------------------------------- //
//  Roll back (S12: the GUI form of `setup-llama --rollback`)                   //
// --------------------------------------------------------------------------- //

const INSTALLED_WITH_PREVIOUS = { ...INSTALLED_VULKAN, current: "b10361",
  target: "b10361", newer: false, previous: "b10300" };

test("Runtime: a recorded previous build reveals Roll back and names it", async () => {
  const { window } = loadAppWithPages({ fetchImpl: makeFetch([
    ["/api/runtime/check", INSTALLED_WITH_PREVIOUS],
  ]) });
  const doc = window.document;
  doc.getElementById("runtime-update-check").click();
  await flush();
  const btn = doc.getElementById("runtime-rollback");
  assert.equal(btn.hidden, false);
  assert.match(doc.getElementById("runtime-rollback-status").textContent, /b10300/);
});

test("Runtime: no previous build keeps Roll back hidden", async () => {
  const { window } = loadAppWithPages({ fetchImpl: makeFetch([
    ["/api/runtime/check", INSTALLED_VULKAN],   // previous: null
  ]) });
  const doc = window.document;
  doc.getElementById("runtime-update-check").click();
  await flush();
  assert.equal(doc.getElementById("runtime-rollback").hidden, true);
  assert.equal(doc.getElementById("runtime-rollback-status").hidden, true);
});

test("Runtime: nothing installed keeps Roll back hidden even if 'previous' were set", async () => {
  // installed:false must win over any stray previous value - there is no
  // backend to roll back FOR.
  const { window } = loadAppWithPages({ fetchImpl: makeFetch([
    ["/api/runtime/check", { ...NOTHING_INSTALLED, previous: "b10300" }],
  ]) });
  const doc = window.document;
  doc.getElementById("runtime-update-check").click();
  await flush();
  assert.equal(doc.getElementById("runtime-rollback").hidden, true);
});

test("Runtime: Roll back confirms first, names the backend and target build", async () => {
  const bodies = [];
  const { window } = loadAppWithPages({ fetchImpl: makeFetch([
    ["/api/runtime/check", INSTALLED_WITH_PREVIOUS],
    ["/api/runtime/update", (opts) => { bodies.push(opts.body); return { job_id: "j1" }; }],
  ]) });
  stubJobStream(window, "done", ["Rolling back the vulkan runtime to llama.cpp b10300 ..."]);
  stubConfirm(window);
  const doc = window.document;
  doc.getElementById("runtime-update-check").click();
  await flush();
  doc.getElementById("runtime-rollback").click();
  await flush();
  assert.ok(window.__confirmCall, "a rollback replaces a working runtime - confirm it");
  assert.match(window.__confirmCall.message, /vulkan/);
  assert.match(window.__confirmCall.message, /b10300/);
  assert.deepEqual(JSON.parse(bodies[0]), { rollback: true },
    "no backend/tag: the button rolls back whatever is installed");
  assert.match(doc.getElementById("runtime-update-log").textContent, /Rolling back/);
});

test("Runtime: declining the rollback confirm sends nothing", async () => {
  const bodies = [];
  const { window } = loadAppWithPages({ fetchImpl: makeFetch([
    ["/api/runtime/check", INSTALLED_WITH_PREVIOUS],
    ["/api/runtime/update", (opts) => { bodies.push(opts.body); return { job_id: "j1" }; }],
  ]) });
  stubJobStream(window);
  stubConfirm(window, { accept: false });
  const doc = window.document;
  doc.getElementById("runtime-update-check").click();
  await flush();
  doc.getElementById("runtime-rollback").click();
  await flush();
  assert.ok(window.__confirmCall);
  assert.deepEqual(bodies, [], "cancelling must not roll back anything");
});

test("Runtime: a failed rollback shows the job's reason and keeps the button", async () => {
  const { window } = loadAppWithPages({ fetchImpl: makeFetch([
    ["/api/runtime/check", INSTALLED_WITH_PREVIOUS],
    ["/api/runtime/update", { job_id: "j2" }],
  ]) });
  stubJobStream(window, "failed", ["Cannot replace the installed runtime: it is in use."]);
  stubConfirm(window);
  const doc = window.document;
  doc.getElementById("runtime-update-check").click();
  await flush();
  doc.getElementById("runtime-rollback").click();
  await flush();
  assert.match(doc.getElementById("runtime-update-status").textContent, /it is in use/);
  const btn = doc.getElementById("runtime-rollback");
  assert.equal(btn.hidden, false, "the user has to be able to retry");
  assert.equal(btn.disabled, false);
});
