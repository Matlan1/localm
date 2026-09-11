// SPDX-License-Identifier: AGPL-3.0-or-later
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

// Settings -> API keys panel (pages.js refreshKeysPanel): mint named, scope-limited
// keys, list them, revoke them - backed by the owner-gated /v1/keys API. Owner-only:
// the card hides when /v1/keys is forbidden.

// Polls until fn() is true. The timeout is a failure bound, not a delay: the
// panel's own handlers are awaited directly; only fire-and-forget saves
// (saveKeyPresets) need a wait, and it sits on their PATCH.
const settle = (ms = 0) => new Promise((r) => setTimeout(r, ms));
async function waitFor(fn, timeout = 2000) {
  const end = Date.now() + timeout;
  while (Date.now() < end) { if (fn()) return true; await settle(15); }
  return false;
}
// The load-time bootAuthProbe() unlocked the shell (unlockUI sets the flag), so
// the pages' auth-gated refreshes may run.
async function bootSettled(window) {
  assert.ok(await waitFor(() => window.__localmLocked === false),
    "the load-time bootAuthProbe() never unlocked the shell");
}

function router(routes) {
  return async (url, opts = {}) => {
    const method = (opts.method || "GET").toUpperCase();
    const path = String(url).replace(/^https?:\/\/[^/]+/, "");
    for (const [key, fn] of Object.entries(routes)) {
      const [m, p] = key.split(" ");
      const hit = m === method
        && (path === p || (p.endsWith("*") && path.startsWith(p.slice(0, -1))));
      if (hit) {
        const res = fn(path, opts);
        return { ok: res.status < 400, status: res.status,
                 json: async () => res.body || {}, text: async () => res.text || "" };
      }
    }
    // Default for unmatched routes: a benign models shape for app.js's boot
    // helpers, which read modelCache.models.
    return { ok: true, status: 200,
             json: async () => ({ models: [], active: "" }), text: async () => "" };
  };
}

test("keys panel: hides the card for a non-owner (/v1/keys 403)", async () => {
  const { window } = loadAppWithPages({
    fetchImpl: router({ "GET /v1/keys": () => ({ status: 403 }) }),
  });
  await bootSettled(window);
  await window.refreshKeysPanel();
  // The card is hidden via the sec-hidden class, not an inline display style.
  assert.ok(window.document.getElementById("keys-card").classList.contains("sec-hidden"));
});

test("keys panel: renders the scope checkboxes and lists existing keys", async () => {
  const { window } = loadAppWithPages({
    fetchImpl: router({
      "GET /v1/keys": () => ({ status: 200,
        body: { keys: [{ id: "abc", name: "phone", scopes: ["coder", "models:read"] }] } }),
    }),
  });
  await bootSettled(window);
  await window.refreshKeysPanel();
  assert.ok(!window.document.getElementById("keys-card").classList.contains("sec-hidden"));
  assert.ok(window.document.querySelectorAll(".key-scope-cb").length >= 5);
  const list = window.document.getElementById("keys-list").textContent;
  assert.match(list, /phone/);
  assert.match(list, /coder/);
});

test("keys panel: create posts {name, scopes} and shows the secret once", async () => {
  let posted = null;
  const { window } = loadAppWithPages({
    fetchImpl: router({
      "GET /v1/keys": () => ({ status: 200, body: { keys: [] } }),
      "POST /v1/keys": (_p, opts) => {
        posted = JSON.parse(opts.body);
        return { status: 200,
                 body: { id: "new1", name: posted.name, scopes: posted.scopes, key: "secret-xyz" } };
      },
    }),
  });
  await bootSettled(window);
  await window.refreshKeysPanel();
  window.document.getElementById("key-name").value = "phone";
  const cb = [...window.document.querySelectorAll(".key-scope-cb")]
    .find((c) => c.value === "coder");
  cb.checked = true;
  await window.document.getElementById("key-create").onclick();
  assert.deepEqual(posted, { name: "phone", scopes: ["coder"] });
  const box = window.document.getElementById("key-secret");
  assert.notEqual(box.style.display, "none");
  assert.match(box.textContent, /shown only once/i);
  assert.equal(window.document.querySelector(".key-secret-value").value, "secret-xyz");
});

test("keys panel: create with no scope checked does not POST", async () => {
  let posts = 0;
  const { window } = loadAppWithPages({
    fetchImpl: router({
      "GET /v1/keys": () => ({ status: 200, body: { keys: [] } }),
      "POST /v1/keys": () => { posts++; return { status: 200, body: {} }; },
    }),
  });
  await bootSettled(window);
  await window.refreshKeysPanel();
  window.document.getElementById("key-name").value = "phone";   // name but no scope
  await window.document.getElementById("key-create").onclick();
  assert.equal(posts, 0, "no scope checked (and no confirmation): nothing is minted");
  // Positive control, same window: with a scope checked the same click POSTs.
  [...window.document.querySelectorAll(".key-scope-cb")]
    .find((c) => c.value === "chat").checked = true;
  await window.document.getElementById("key-create").onclick();
  assert.equal(posts, 1, "positive control: a scoped create reaches the stub as a POST");
});

test("keys panel: presets (from /v1/keys) populate checkboxes; coder vs coder:full distinct", async () => {
  const { window } = loadAppWithPages({
    fetchImpl: router({
      "GET /v1/keys": () => ({ status: 200, body: { keys: [], is_owner: true,
        presets: [{ name: "Companion", scopes: ["chat", "image"] }] } }),
    }),
  });
  await bootSettled(window);
  await window.refreshKeysPanel();
  const cbs = [...window.document.querySelectorAll(".key-scope-cb")].map((c) => c.value);
  assert.ok(cbs.includes("coder") && cbs.includes("coder:full"),
            "coder and coder:full are separate checkboxes");
  const btn = [...window.document.querySelectorAll(".key-preset-btn")]
    .find((b) => b.textContent.startsWith("Companion"));
  assert.ok(btn, "the Companion preset button rendered");
  btn.onclick();
  const checked = [...window.document.querySelectorAll(".key-scope-cb")]
    .filter((c) => c.checked).map((c) => c.value).sort();
  assert.deepEqual(checked, ["chat", "image"]);   // preset set exactly its scopes
});

test("keys panel: all five privileged scopes are disabled for a non-owner", async () => {
  const { window } = loadAppWithPages({
    fetchImpl: router({
      "GET /v1/keys": () => ({ status: 200, body: { keys: [], is_owner: false, presets: [] } }),
    }),
  });
  await bootSettled(window);
  await window.refreshKeysPanel();
  const byVal = {};
  for (const c of window.document.querySelectorAll(".key-scope-cb")) byVal[c.value] = c;
  for (const priv of ["admin", "coder:full", "keys:admin", "plugins:admin", "config:write"]) {
    assert.equal(byVal[priv].disabled, true, `${priv} must be disabled for a non-owner`);
    assert.ok(byVal[priv].closest(".key-scope").classList.contains("key-scope-danger"),
      `${priv} must render with the danger styling`);
  }
  assert.equal(byVal["coder"].disabled, false);     // plain coder stays available
  assert.equal(byVal["chat"].disabled, false);
  assert.equal(byVal["config:read"].disabled, false);   // read is not privileged
  assert.equal(window.document.querySelector(".key-preset-save"), null);  // no edit for non-owner
});

test("keys panel: the owner CAN check the three newly-offered privileged scopes", async () => {
  let posted = null;
  const { window } = loadAppWithPages({
    fetchImpl: router({
      "GET /v1/keys": () => ({ status: 200, body: { keys: [], is_owner: true, presets: [] } }),
      "POST /v1/keys": (_p, opts) => {
        posted = JSON.parse(opts.body);
        return { status: 200, body: { id: "n", name: posted.name, scopes: posted.scopes, key: "K" } };
      },
    }),
  });
  await bootSettled(window);
  await window.refreshKeysPanel();
  window.document.getElementById("key-name").value = "admin-device";
  for (const priv of ["keys:admin", "plugins:admin", "config:write"]) {
    const cb = [...window.document.querySelectorAll(".key-scope-cb")].find((c) => c.value === priv);
    assert.equal(cb.disabled, false, `${priv} must be mintable by the owner`);
    cb.checked = true;
  }
  await window.document.getElementById("key-create").onclick();
  assert.deepEqual(posted.scopes.sort(),
    ["config:write", "keys:admin", "plugins:admin"]);
});

test("keys panel: owner can save and delete a preset (PATCH /v1/config)", async () => {
  let patched = null;
  const { window } = loadAppWithPages({
    fetchImpl: router({
      "GET /v1/keys": () => ({ status: 200, body: { keys: [], is_owner: true,
        presets: [{ name: "Old", scopes: ["chat"] }] } }),
      "PATCH /v1/config": (_p, opts) => { patched = JSON.parse(opts.body); return { status: 200, body: {} }; },
    }),
  });
  await bootSettled(window);
  await window.refreshKeysPanel();
  const del = window.document.querySelector(".key-preset-del");
  assert.ok(del, "owner sees a delete affordance");
  del.onclick({ stopPropagation() {} });
  // Deletion confirms via the in-page confirmDanger modal; click its danger
  // button.
  const ok = window.document.querySelector("#modal-body .btn-danger");
  assert.ok(ok, "delete-preset confirm modal shown");
  ok.click();
  assert.ok(await waitFor(() => patched !== null), "the delete PATCHed /v1/config");
  assert.equal(patched.key_presets.length, 0);      // "Old" removed
  patched = null;
  runScript(window, 'promptText = async () => "Phone";');
  [...window.document.querySelectorAll(".key-scope-cb")].find((c) => c.value === "chat").checked = true;
  window.document.querySelector(".key-preset-save").onclick();
  assert.ok(await waitFor(() => patched !== null), "the save PATCHed /v1/config");
  assert.ok(patched.key_presets.some((p) => p.name === "Phone" && p.scopes.includes("chat")));
});

test("keys panel: create threads expires and requests a pairing QR for the new key", async () => {
  let posted = null, qrKey = null;
  const { window } = loadAppWithPages({
    fetchImpl: router({
      "GET /v1/keys": () => ({ status: 200, body: { keys: [] } }),
      "POST /v1/keys": (_p, opts) => {
        posted = JSON.parse(opts.body);
        return { status: 200,
                 body: { id: "n", name: posted.name, scopes: posted.scopes, key: "K9" } };
      },
      "POST /api/pairing/qr": (_p, opts) => {
        qrKey = JSON.parse(opts.body).key;
        return { status: 200, text: '<svg viewBox="0 0 10 10"></svg>' };
      },
    }),
  });
  await bootSettled(window);
  await window.refreshKeysPanel();
  window.document.getElementById("key-name").value = "phone";
  [...window.document.querySelectorAll(".key-scope-cb")]
    .find((c) => c.value === "chat").checked = true;
  window.document.getElementById("key-expiry").value = "86400";   // in 24 hours
  await window.document.getElementById("key-create").onclick();
  assert.equal(posted.name, "phone");
  assert.deepEqual(posted.scopes, ["chat"]);
  assert.equal(posted.expires_in, 86400);          // relative TTL threaded (server-clock)
  assert.equal(qrKey, "K9");                        // QR requested for the minted key
});
