// SPDX-License-Identifier: AGPL-3.0-or-later
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

// Settings -> Owner key (settings.js refreshOwnerKeyPanel), over
// POST /api/auth/key/rotate.

const tick = () => new Promise((r) => setTimeout(r, 50));

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
    return { ok: true, status: 200,
             json: async () => ({ models: [], active: "" }), text: async () => "" };
  };
}

const OWNER = { status: 200, body: { keys: [], is_owner: true, presets: [] } };

// both buttons go through confirmDanger; auto-confirm it
const autoConfirm = (win) =>
  runScript(win, "confirmDanger = (t, m, l, onConfirm) => onConfirm();");

function toasts(win) {
  const seen = [];
  runScript(win, "window.__toasts = []; toast = (m, isErr) => "
    + "window.__toasts.push({ msg: String(m), isErr: !!isErr });");
  return { seen, all: () => win.__toasts || [] };
}

async function ownerPanel(routes) {
  const { window } = loadAppWithPages({
    fetchImpl: router({ "GET /v1/keys": () => OWNER, ...routes }),
  });
  await tick();
  return window;
}

test("owner key: card hides when /v1/keys is forbidden", async () => {
  const { window } = loadAppWithPages({
    fetchImpl: router({ "GET /v1/keys": () => ({ status: 403 }) }),
  });
  await tick();
  await window.refreshOwnerKeyPanel();
  assert.ok(window.document.getElementById("owner-key-card")
                  .classList.contains("sec-hidden"));
});

test("owner key: card hides for a keys:admin device that is not the owner", async () => {
  // /v1/keys answers 200 for a keys:admin key; is_owner is the discriminator
  const { window } = loadAppWithPages({
    fetchImpl: router({
      "GET /v1/keys": () => ({ status: 200, body: { keys: [], is_owner: false } }),
    }),
  });
  await tick();
  await window.refreshOwnerKeyPanel();
  assert.ok(window.document.getElementById("owner-key-card")
                  .classList.contains("sec-hidden"));
});

test("owner key: card shows for the owner", async () => {
  const win = await ownerPanel({});
  await win.refreshOwnerKeyPanel();
  assert.ok(!win.document.getElementById("owner-key-card")
                 .classList.contains("sec-hidden"));
});

test("owner key: Generate posts an empty body and shows the new key once", async () => {
  let posted = null;
  const win = await ownerPanel({
    "POST /api/auth/key/rotate": (_p, opts) => {
      posted = JSON.parse(opts.body);
      return { status: 200,
               body: { rotated: true, active: true, key: "fresh-key-123", warnings: [] } };
    },
  });
  await win.refreshOwnerKeyPanel();
  autoConfirm(win);
  const t = toasts(win);

  win.document.getElementById("owner-key-roll").click();
  await tick();

  assert.deepEqual(posted, {}, "Generate must not send a key of its own");
  const box = win.document.getElementById("owner-key-secret");
  assert.equal(box.style.display, "");
  assert.equal(box.querySelector(".key-secret-value").value, "fresh-key-123");
  assert.match(box.textContent, /shown only once/i);
  assert.ok(t.all().some((x) => /updated/i.test(x.msg) && !x.isErr));
});

test("owner key: Set posts the pasted key", async () => {
  let posted = null;
  const win = await ownerPanel({
    "POST /api/auth/key/rotate": (_p, opts) => {
      posted = JSON.parse(opts.body);
      return { status: 200,
               body: { rotated: true, active: true, key: posted.key, warnings: [] } };
    },
  });
  await win.refreshOwnerKeyPanel();
  autoConfirm(win);

  win.document.getElementById("owner-key-value").value = "  chosen-key-value  ";
  win.document.getElementById("owner-key-set").click();
  await tick();

  assert.deepEqual(posted, { key: "chosen-key-value" }, "and it is trimmed");
  assert.equal(win.document.getElementById("owner-key-value").value, "",
               "the input is cleared so the key is not left on screen");
});

test("owner key: Set with an empty box does not call the server", async () => {
  // an empty value would make the server generate a random key; the client
  // refuses before it gets there
  let calls = 0;
  const win = await ownerPanel({
    "POST /api/auth/key/rotate": () => {
      calls += 1;
      return { status: 200, body: { rotated: true, active: true, key: "x", warnings: [] } };
    },
  });
  await win.refreshOwnerKeyPanel();
  autoConfirm(win);

  win.document.getElementById("owner-key-value").value = "   ";
  win.document.getElementById("owner-key-set").click();
  await tick();

  assert.equal(calls, 0, "an empty box must not silently generate a random key");
});

test("owner key: rotated:false is NOT reported as a completed rotation", async () => {
  // 200 + rotated:false = written to disk, not in effect
  const warning = "LOCALM_API_KEY is set in the server's environment and overrides "
    + "the stored key, so the server still accepts the environment's key.";
  const win = await ownerPanel({
    "POST /api/auth/key/rotate": () => ({
      status: 200,
      body: { rotated: false, active: false, key: "written-but-dead",
              warnings: [warning] },
    }),
  });
  await win.refreshOwnerKeyPanel();
  autoConfirm(win);
  const t = toasts(win);

  win.document.getElementById("owner-key-roll").click();
  await tick();

  const box = win.document.getElementById("owner-key-secret");
  assert.match(box.textContent, /not currently in effect/i,
    "the user must be told the key they are looking at is not the live one");
  assert.match(box.textContent, /LOCALM_API_KEY/,
    "and WHY, which only the server's warning can say");
  assert.ok(box.querySelector(".key-warn"), "the warning is styled as a warning");
  const msgs = t.all();
  assert.ok(msgs.length, "a rotation that did not take must still say something");
  assert.ok(msgs.every((x) => !/updated/i.test(x.msg)),
    "'Owner key updated' here is the exact rule-5 lie: the old key still works");
  assert.ok(msgs.some((x) => x.isErr), "and it is surfaced as a problem, not a success");
});

test("owner key: a refused rotation surfaces the server's reason", async () => {
  const win = await ownerPanel({
    "POST /api/auth/key/rotate": () => ({
      status: 400, body: { detail: "API key must be at least 16 characters long." },
    }),
  });
  await win.refreshOwnerKeyPanel();
  autoConfirm(win);
  const t = toasts(win);

  win.document.getElementById("owner-key-value").value = "short";
  win.document.getElementById("owner-key-set").click();
  await tick();

  assert.ok(t.all().some((x) => /at least 16 characters/.test(x.msg) && x.isErr),
    "the server explains what is wrong with the key; do not replace it with 'failed'");
  assert.equal(win.document.getElementById("owner-key-secret").style.display, "none",
    "and nothing is presented as a new key");
});
