// SPDX-License-Identifier: AGPL-3.0-or-later
// R34: the per-chat Web-access and Speak-aloud toggles used to reset to OFF on
// every load, so the user had to re-enable them in every session, and the global
// allow/ask/deny policy never auto-enabled web. They now restore the saved choice
// and, for web with no saved choice, follow the global net policy.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp, runScript } from "./harness.mjs";

function appWithChat() {
  const { window } = loadApp();
  runScript(window, "window.chatState = chat;");
  window.localStorage.clear();
  window.document.getElementById("p-web").checked = false;
  window.document.getElementById("p-speak").checked = false;
  return window;
}

test("R34: net_mode=allow auto-enables web when there is no saved choice", () => {
  const w = appWithChat();
  w.hydrateChatToggles({ net_mode: "allow" });
  assert.equal(w.document.getElementById("p-web").checked, true);
});

test("R34: net_mode=ask leaves web off so consent still applies", () => {
  const w = appWithChat();
  w.hydrateChatToggles({ net_mode: "ask" });
  assert.equal(w.document.getElementById("p-web").checked, false);
});

test("R34: a saved web choice overrides the global policy", () => {
  const w = appWithChat();
  w.localStorage.setItem("localm.webAccess", "0");
  w.hydrateChatToggles({ net_mode: "allow" });
  assert.equal(w.document.getElementById("p-web").checked, false, "saved OFF beats allow");
});

test("R34: speak-aloud is restored from the saved choice", () => {
  const w = appWithChat();
  w.localStorage.setItem("localm.speakAloud", "1");
  w.hydrateChatToggles({});
  assert.equal(w.document.getElementById("p-speak").checked, true);
});

test("R34: toggling persists the choice across loads", () => {
  const w = appWithChat();
  const web = w.document.getElementById("p-web");
  web.checked = true;
  web.dispatchEvent(new w.Event("change"));
  assert.equal(w.localStorage.getItem("localm.webAccess"), "1", "the choice was persisted");
});

test("R34: privacy mode neither restores nor persists toggle state", () => {
  const w = appWithChat();
  w.localStorage.setItem("localm.webAccess", "1");
  w.chatState.privacy = true;

  // Does not restore from storage under privacy.
  w.hydrateChatToggles({ net_mode: "ask" });
  assert.equal(w.document.getElementById("p-web").checked, false,
    "privacy does not restore the toggle from storage");

  // Does not write a new trace under privacy.
  w.localStorage.removeItem("localm.webAccess");
  const web = w.document.getElementById("p-web");
  web.checked = true;
  web.dispatchEvent(new w.Event("change"));
  assert.equal(w.localStorage.getItem("localm.webAccess"), null,
    "privacy leaves no toggle trace");
});
