// SPDX-License-Identifier: AGPL-3.0-or-later
// Settings > Updates: when an update restarts or ends other processes of LocaLM,
// the server names them (needs_confirmation "processes") and the same request again
// is the agreement. The GUI asks in a confirm dialog, never shows it as a failure,
// and sends the request again only after the confirm.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages } from "./harness.mjs";

const NAMED = {
  applied: false, needs_confirmation: "processes",
  processes: [{ key: "9-9", pid: 9, purpose: "chat" }],
  error: "The update to 0.3.5 restarts or ends these processes of localm: chat (pid 9).",
};

function makeFetch(path, replies, calls) {
  return async (url, opts = {}) => {
    const u = String(url);
    const method = (opts.method || "GET").toUpperCase();
    if (u.includes(path) && method === "POST") {
      calls.push(method);
      const body = replies[Math.min(calls.length - 1, replies.length - 1)];
      return { ok: true, status: 200, text: async () => "", json: async () => body };
    }
    if (u.includes("/api/update/rollback")) {
      return { ok: true, status: 200, text: async () => "",
        json: async () => ({ available: true, version: "0.3.3", current: "0.3.4" }) };
    }
    return { ok: true, status: 200, text: async () => "",
      json: async () => ({ models: [], active: "", conversations: [], plugins: [] }) };
  };
}

async function flush() {
  for (let i = 0; i < 6; i++) await new Promise((r) => setTimeout(r, 0));
}

for (const agree of [true, false]) {
  test(`Update now that names processes asks first (${agree ? "agreed" : "declined"})`, async () => {
    const calls = [];
    const { window } = loadAppWithPages({ fetchImpl: makeFetch("/api/update/apply",
      [NAMED, { applied: true, version: "0.3.5", restarting: true }], calls) });
    const doc = window.document;
    doc.getElementById("update-apply").click();
    await flush();
    const body = doc.getElementById("modal-body");
    assert.match(body.textContent, /chat \(pid 9\)/, "the dialog names the processes");
    const status = doc.getElementById("update-status").textContent;
    assert.doesNotMatch(status, /Update failed/, "naming processes is not a failure");
    assert.equal(calls.length, 1, "nothing is sent again before the confirm");
    const button = agree ? body.querySelector(".btn-danger")
      : [...body.querySelectorAll("button")].find((b) => !b.classList.contains("btn-danger"));
    button.click();
    await flush();
    assert.equal(calls.length, agree ? 2 : 1);
    assert.match(doc.getElementById("update-status").textContent,
      agree ? /0\.3\.5/ : /Not installed/);
  });
}

test("Roll back that names processes asks, then sends the rollback again", async () => {
  const calls = [];
  const { window } = loadAppWithPages({ fetchImpl: makeFetch("/api/update/rollback",
    [NAMED, { applied: true, version: "0.3.3", restarting: true }], calls) });
  const doc = window.document;
  await window.__localmRollbackCheck();
  await flush();
  doc.getElementById("update-rollback").click();
  await flush();
  doc.querySelector("#modal-body .btn-danger").click();   // the rollback's own confirm
  await flush();
  assert.equal(calls.length, 1);
  assert.match(doc.getElementById("modal-body").textContent, /chat \(pid 9\)/);
  assert.doesNotMatch(doc.getElementById("update-rollback-status").textContent,
    /Update now/, "the rollback never tells the user to press Update now");
  doc.querySelector("#modal-body .btn-danger").click();
  await flush();
  assert.equal(calls.length, 2);
  assert.match(doc.getElementById("update-rollback-status").textContent, /0\.3\.3/);
});
