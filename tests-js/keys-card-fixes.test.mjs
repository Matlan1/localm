// SPDX-License-Identifier: AGPL-3.0-or-later
// Settings > Keys & devices, Owner key and Sign in: expiry and last-used tags,
// refusals shown with their reason as errors, the chat-only key line, the
// server's grant warnings, the first key's owner key, the Copy buttons, why
// owner-only scopes are disabled, the sign-in refusal reason, and the reason a
// configured bind address is not in effect.
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

const settle = (ms = 0) => new Promise((r) => setTimeout(r, ms));
async function waitFor(fn, timeout = 2000) {
  const end = Date.now() + timeout;
  while (Date.now() < end) { if (fn()) return true; await settle(15); }
  return false;
}

function router(routes, calls = []) {
  return async (url, opts = {}) => {
    const method = (opts.method || "GET").toUpperCase();
    const path = String(url).replace(/^https?:\/\/[^/]+/, "");
    calls.push(`${method} ${path}`);
    for (const [key, fn] of Object.entries(routes)) {
      const [m, p] = key.split(" ");
      if (m === method && (path === p || (p.endsWith("*") && path.startsWith(p.slice(0, -1))))) {
        const res = fn(path, opts);
        return { ok: res.status < 400, status: res.status,
                 json: async () => res.body || {}, text: async () => res.text || "" };
      }
    }
    return { ok: true, status: 200,
             json: async () => ({ models: [], active: "" }), text: async () => "" };
  };
}

async function open(routes, calls) {
  const { window } = loadAppWithPages({ fetchImpl: router(routes, calls) });
  assert.ok(await waitFor(() => window.__localmLocked === false), "the shell never unlocked");
  const toasts = [];
  window.toast = (msg, isError) => toasts.push({ msg: String(msg), isError: !!isError });
  return { window, toasts };
}

function setClipboard(window, writeText) {
  Object.defineProperty(window.navigator, "clipboard", {
    configurable: true, value: writeText ? { writeText } : undefined,
  });
}

test("expiry tag: a fresh 24 hour / 7 day / 30 day key reads 1d / 7d / 30d", () => {
  const { window } = loadAppWithPages({});
  const now = Date.now() / 1000;
  const label = (s) => window.keyExpiryLabel(now + s - 2);
  assert.equal(label(86400), "expires in 1d");
  assert.equal(label(7 * 86400), "expires in 7d");
  assert.equal(label(30 * 86400), "expires in 30d");
  assert.equal(label(3 * 3600), "expires in 3h");
  assert.equal(label(600), "expires in 1h");
  assert.equal(window.keyExpiryLabel(now - 5), "expired");
});

test("last-used tag: a key used a second ago reads used just now", () => {
  const { window } = loadAppWithPages({});
  const now = Date.now() / 1000;
  assert.equal(window.keyLastUsedLabel(now - 1), "used just now");
  assert.equal(window.keyLastUsedLabel(now + 30), "used just now");
  assert.equal(window.keyLastUsedLabel(now - 300), "used recently");
});

test("a refused revoke shows the server's reason as an error", async () => {
  const { window, toasts } = await open({
    "GET /v1/keys": () => ({ status: 200, body: { keys: [{ id: "k1", name: "phone", scopes: [] }],
                                                  is_owner: true } }),
    "DELETE /v1/keys/k1": () => ({ status: 409, body: { detail: "This is the last key of a networked server." } }),
  });
  await window.refreshKeysPanel();
  const revoke = [...window.document.querySelectorAll("#keys-list button")]
    .find((b) => b.textContent === "Revoke");
  revoke.onclick();
  window.document.querySelector("#modal-body .btn-danger").click();
  assert.ok(await waitFor(() => toasts.length > 0));
  assert.deepEqual(toasts[0], { msg: "This is the last key of a networked server.", isError: true });
});

test("a refused create and a missing name are error toasts", async () => {
  const { window, toasts } = await open({
    "GET /v1/keys": () => ({ status: 200, body: { keys: [], is_owner: true } }),
    "POST /v1/keys": () => ({ status: 403, body: { detail: "Refusing to grant scope(s) admin" } }),
  });
  await window.refreshKeysPanel();
  await window.document.getElementById("key-create").onclick();
  assert.deepEqual(toasts.pop(), { msg: "Enter a key name", isError: true });
  window.document.getElementById("key-name").value = "phone";
  [...window.document.querySelectorAll(".key-scope-cb")].find((c) => c.value === "chat").checked = true;
  await window.document.getElementById("key-create").onclick();
  assert.deepEqual(toasts.pop(), { msg: "Refusing to grant scope(s) admin", isError: true });
});

async function createKey(made, extraRoutes = {}, calls = []) {
  const ctx = await open({
    "GET /v1/keys": () => ({ status: 200, body: { keys: [], is_owner: true } }),
    "POST /v1/keys": () => ({ status: 200, body: made }),
    ...extraRoutes,
  }, calls);
  await ctx.window.refreshKeysPanel();
  ctx.window.document.getElementById("key-name").value = made.name;
  [...ctx.window.document.querySelectorAll(".key-scope-cb")]
    .find((c) => c.value === "chat").checked = true;
  await ctx.window.document.getElementById("key-create").onclick();
  return ctx;
}

test("a chat-only key's line has no empty brackets", async () => {
  const { window } = await createKey({ id: "n", name: "phone", scopes: [], key: "K" });
  const text = window.document.getElementById("key-secret").textContent;
  assert.match(text, /New chat-only key "phone"/);
  assert.doesNotMatch(text, /\(\)/);
});

test("the server's grant warnings are shown under the new key", async () => {
  const { window } = await createKey({ id: "n", name: "phone", scopes: ["image"], key: "K",
    warnings: ["scope 'image': the image plugin is not installed"] });
  const warn = window.document.querySelector("#key-secret .key-warn");
  assert.ok(warn, "a warning line is drawn");
  assert.equal(warn.textContent, "scope 'image': the image plugin is not installed");
});

test("the first key says an owner key was set too and re-reads the Owner key card", async () => {
  const calls = [];
  const { window } = await createKey(
    { id: "n", name: "phone", scopes: ["chat"], key: "K", owner_key_seeded: true }, {}, calls);
  assert.match(window.document.getElementById("key-secret").textContent,
    /now also has an owner key/);
  assert.ok(await waitFor(() => !window.document.getElementById("owner-key-card")
    .classList.contains("sec-hidden")), "the Owner key card is shown without a reload");
});

test("a later key does not claim an owner key was set", async () => {
  const { window } = await createKey({ id: "n", name: "phone", scopes: ["chat"], key: "K" });
  assert.doesNotMatch(window.document.getElementById("key-secret").textContent, /owner key/);
});

test("Copy says Copied only when the clipboard took the key", async () => {
  const { window, toasts } = await createKey({ id: "n", name: "phone", scopes: ["chat"], key: "K" });
  const copy = [...window.document.querySelectorAll("#key-secret button")]
    .find((b) => b.textContent === "Copy");
  setClipboard(window, () => Promise.reject(new Error("denied")));
  assert.equal(await copy.onclick(), false);
  assert.equal(copy.textContent, "Copy");
  assert.ok(toasts.some((x) => x.isError), "a blocked copy is an error toast");
  const written = [];
  setClipboard(window, (s) => { written.push(s); return Promise.resolve(); });
  assert.equal(await copy.onclick(), true);
  assert.equal(copy.textContent, "Copied");
  assert.deepEqual(written, ["K"]);
});

test("the Owner key card's Copy reports a blocked clipboard", async () => {
  const { window, toasts } = await open({
    "GET /v1/keys": () => ({ status: 200, body: { keys: [], is_owner: true } }),
    "POST /api/auth/key/rotate": () => ({ status: 200, body: { rotated: true, key: "OWN" } }),
  });
  await window.refreshOwnerKeyPanel();
  window.document.getElementById("owner-key-roll").onclick();
  window.document.querySelector("#modal-body .btn-danger").click();
  assert.ok(await waitFor(() => window.document.querySelector("#owner-key-secret button")));
  const copy = window.document.querySelector("#owner-key-secret button");
  setClipboard(window, undefined);
  assert.equal(await copy.onclick(), false);
  assert.ok(toasts.some((x) => x.isError), "a missing clipboard is an error toast");
});

test("owner-only scopes say why they are disabled: open mode, a non-owner device", async () => {
  for (const [body, want] of [
    [{ keys: [], is_owner: false, open_mode: true }, /has no key yet/],
    [{ keys: [], is_owner: false, open_mode: false }, /signed in with the owner key/],
  ]) {
    const { window } = await open({ "GET /v1/keys": () => ({ status: 200, body }) });
    await window.refreshKeysPanel();
    const note = window.document.getElementById("key-owner-note");
    assert.ok(note, "a note says why owner-only scopes are disabled");
    assert.match(note.textContent, want);
    const admin = [...window.document.querySelectorAll(".key-scope-cb")].find((c) => c.value === "admin");
    assert.equal(admin.disabled, true);
    assert.equal(admin.closest(".key-scope").title, "Owner-only: needs the owner key");
  }
  const { window } = await open({ "GET /v1/keys": () => ({ status: 200, body: { keys: [], is_owner: true } }) });
  await window.refreshKeysPanel();
  assert.equal(window.document.getElementById("key-owner-note"), null, "the owner sees no note");
  const admin = [...window.document.querySelectorAll(".key-scope-cb")].find((c) => c.value === "admin");
  assert.equal(admin.closest(".key-scope").getAttribute("title"), null);
});

test("Save key on a keyless server shows the server's reason", async () => {
  const { window, toasts } = await open({
    "POST /api/session": () => ({ status: 400,
      body: { detail: "This server runs in open mode (no API key configured); there is nothing to log into." } }),
  });
  window.document.getElementById("gui-api-key").value = "some-key";
  window.document.getElementById("gui-key-save").click();
  assert.ok(await waitFor(() => toasts.length > 0));
  assert.deepEqual(toasts[0], { msg: "This server runs in open mode (no API key configured); "
    + "there is nothing to log into.", isError: true });
});

test("a configured bind address that is not in effect says so under its field", async () => {
  const reason = "The configured bind address (0.0.0.0) was not applied: no API key is set.";
  const { window } = await open({
    "GET /api/companion": () => ({ status: 200, body: { network_bind: false, bind_fallback: reason } }),
  });
  runScript(window, 'globalThis.__f = document.createElement("div");'
    + ' __f.setAttribute("data-field-key", "bind_host"); document.body.appendChild(__f);');
  await window.refreshCompanion();
  const note = window.document.querySelector('[data-field-key="bind_host"] #bind-fallback-note');
  assert.ok(note, "the note is drawn under the Bind address field");
  assert.equal(note.textContent, `Not in effect right now: ${reason}`);
  window.showBindFallbackNote("");
  assert.equal(window.document.getElementById("bind-fallback-note"), null);
});
